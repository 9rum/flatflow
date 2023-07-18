// Copyright 2022 Sogang University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:generate protoc --proto_path=proto/ --go_out=communicator/ --go_opt=paths=source_relative --go-grpc_out=communicator/ --go-grpc_opt=paths=source_relative --experimental_allow_proto3_optional communicator.proto

// Package main implements the communicator server. The initialization and
// termination of the server may be invoked by the sampler, and the types of
// scheduler and data set are provided upon the initialization.
package main

import (
	"flag"
	"fmt"
	"net"
	"os"

	"github.com/9rum/chronica/communicator"
	"github.com/golang/glog"
	grpc_recovery "github.com/grpc-ecosystem/go-grpc-middleware/recovery"
	"google.golang.org/grpc"
)

func main() {
	port := flag.Int("p", 50051, "The server port")
	worldSize := flag.Int("w", 0, "Number of processes participating in distributed training")
	flag.Parse()

	if err := serve(*port, *worldSize); err != nil {
		glog.Fatalf("failed to serve: %v", err)
	}
}

func serve(port, worldSize int) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}

	server := newServer(worldSize)
	glog.Infof("server listening at %v", lis.Addr())

	return server.Serve(lis)
}

func newServer(worldSize int) *grpc.Server {
	server := grpc.NewServer(
		grpc.ChainUnaryInterceptor(
			grpc_recovery.UnaryServerInterceptor(),
		),
	)
	done := make(chan os.Signal)

	go func(done <-chan os.Signal, server *grpc.Server) {
		<-done
		server.GracefulStop()
	}(done, server)

	communicator.RegisterCommunicatorServer(server, communicator.NewCommunicatorServer(done, worldSize))

	return server
}
