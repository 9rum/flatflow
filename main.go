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

//go:generate protoc --proto_path=proto/ --go_out=scheduler/ --go_opt=paths=source_relative --go-grpc_out=scheduler/ --go-grpc_opt=paths=source_relative --experimental_allow_proto3_optional scheduler.proto

// Package main implements the scheduler server. The initialization and
// termination of the server may be invoked by the sampler, and the type of
// scheduler and dataset is provided by the sampler.
package main

import (
	"flag"
	"fmt"
	"net"
	"os"

	"github.com/9rum/chronica/scheduler"
	"github.com/golang/glog"
	grpc_recovery "github.com/grpc-ecosystem/go-grpc-middleware/recovery"
	"google.golang.org/grpc"
)

func main() {
	port := flag.Int("p", 50051, "The server port")
	flag.Parse()

	if err := serve(*port); err != nil {
		glog.Fatalf("failed to serve: %v", err)
	}
}

func serve(port int) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}

	server := newServer()
	glog.Infof("server listening at %v", lis.Addr())

	return server.Serve(lis)
}

func newServer() *grpc.Server {
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

	scheduler.RegisterSchedulerServer(server, scheduler.NewSchedulerServer(done))

	return server
}
