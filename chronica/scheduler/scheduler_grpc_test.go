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

package scheduler

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestSchedulerServer(t *testing.T) {
	const (
		datasetSize = 1 << 10
		worldSize   = 1 << 2
		batchSize   = 1 << 5
	)
	port := flag.Int("p", 50051, "The server port")
	flag.Parse()

	conn, err := grpc.Dial(fmt.Sprintf("localhost:%d", *port), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	c := NewSchedulerClient(conn)

	if _, err = c.Init(context.Background(), &Arguments{WorldSize: worldSize, BatchSize: batchSize, Sizes: cast[int, int64](rand.Perm(datasetSize))}); err != nil {
		t.Fatalf("could not init: %v", err)
	}

	var wg sync.WaitGroup
	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				wg.Add(1)
				go func(rank int) {
					defer wg.Done()
					r, err := c.Bcast(context.Background(), &Feedback{Rank: int64(rank)})
					if err != nil {
						t.Errorf("could not bcast: %v", err)
					}
					t.Logf("step: %d rank: %d got: %v", step, rank, r.GetIndices())
				}(rank)
			}
			wg.Wait()
		}
		if _, err = c.Reset(context.Background(), new(emptypb.Empty)); err != nil {
			t.Fatalf("could not reset: %v", err)
		}
	}

	if _, err = c.Finalize(context.Background(), new(emptypb.Empty)); err != nil {
		t.Fatalf("could not finalize: %v", err)
	}
}
