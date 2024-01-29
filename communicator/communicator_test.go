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

package communicator

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

// cast casts the given slice.
func cast(slice []int) []int64 {
	out := make([]int64, len(slice))
	stride := func(numerator, denominator int) int {
		if numerator%denominator == 0 {
			return numerator / denominator
		}
		return numerator/denominator + 1
	}(len(slice), runtime.NumCPU())

	var wg sync.WaitGroup
	for base := 0; base < len(slice); base += stride {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			limit := min(base+stride, len(slice))
			for index := base; index < limit; index++ {
				out[index] = int64(slice[index])
			}
		}(base)
	}
	wg.Wait()

	return out
}

func TestCommunicatorServer(t *testing.T) {
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

	c := NewCommunicatorClient(conn)

	var wg sync.WaitGroup
	for rank := 0; rank < worldSize; rank++ {
		wg.Add(1)
		go func(rank int64) {
			defer wg.Done()
			if _, err = c.Init(context.Background(), &InitRequest{Rank: rank, BatchSize: batchSize, Sizes: cast(rand.Perm(datasetSize))}); err != nil {
				t.Errorf("could not init: %v", err)
			}
		}(int64(rank))
	}
	wg.Wait()

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := 0; rank < worldSize; rank++ {
			wg.Add(1)
			go func(rank int64) {
				defer wg.Done()
				r, err := c.Bcast(context.Background(), &BcastRequest{Epoch: int64(epoch), Rank: rank})
				if err != nil {
					t.Errorf("could not bcast: %v", err)
				}
				t.Logf("rank: %d got: %v", rank, r.GetIndices())
			}(int64(rank))
		}
		wg.Wait()
	}

	if _, err = c.Finalize(context.Background(), new(emptypb.Empty)); err != nil {
		t.Fatalf("could not finalize: %v", err)
	}
}
