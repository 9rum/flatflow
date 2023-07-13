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

package data

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func init() {
	seed := time.Now().Unix()
	fmt.Println(seed)
	rand.Seed(seed)
}

func TestShardedDataset(t *testing.T) {
	const (
		datasetSize = 10000
		batchSize   = 10
	)
	sizes := rand.Perm(datasetSize)
	dataset, err := NewShardedDataset(sizes)
	if err != nil {
		t.Fatal(err)
	}

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				if _, size := dataset.Getitem(sizes[index], sizes[index]); size != sizes[index] {
					t.Fatalf("did not find %d", sizes[index])
				}
			}
		}
		dataset.OnEpochEnd(int64(epoch))
	}
	dataset.OnTrainEnd()

	for size := range sizes {
		sizes[size] = size
	}
	dataset, err = NewShardedDataset(sizes)
	if err != nil {
		t.Fatal(err)
	}

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				if _, size := dataset.Getitem(sizes[index], sizes[index]); size != sizes[index] {
					t.Fatalf("did not find %d", sizes[index])
				}
			}
		}
		dataset.OnEpochEnd(int64(epoch))
	}
	dataset.OnTrainEnd()
}

func TestPartitionedDataset(t *testing.T) {
	const (
		datasetSize = 10000
		batchSize   = 40
		worldSize   = 4
	)
	groups := make([]int, worldSize)
	partitions := make([][]int, worldSize)
	for rank := range partitions {
		groups[rank] = rank
		partitions[rank] = rand.Perm(datasetSize / worldSize)
	}
	dataset, err := NewPartitionedDataset(groups, partitions)
	if err != nil {
		t.Fatal(err)
	}

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					if _, size := dataset.Getitem(rank, partitions[rank][index]); size != partitions[rank][index] {
						t.Fatalf("did not find %d", partitions[rank][index])
					}
				}
			}
		}
		dataset.OnEpochEnd(int64(epoch))
	}
	dataset.OnTrainEnd()

	for _, partition := range partitions {
		for size := range partition {
			partition[size] = size
		}
	}
	dataset, err = NewPartitionedDataset(groups, partitions)
	if err != nil {
		t.Fatal(err)
	}

	for epoch := 0; epoch < 10; epoch++ {
		for step := 0; step < datasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					if _, size := dataset.Getitem(rank, partitions[rank][index]); size != partitions[rank][index] {
						t.Fatalf("did not find %d", partitions[rank][index])
					}
				}
			}
		}
		dataset.OnEpochEnd(int64(epoch))
	}
	dataset.OnTrainEnd()
}

const benchmarkDatasetSize = 10000

func BenchmarkShardedDataset(b *testing.B) {
	b.StopTimer()
	const batchSize = 10
	sizes := rand.Perm(benchmarkDatasetSize)
	b.StartTimer()
	dataset, err := NewShardedDataset(sizes)
	if err != nil {
		b.Fatal(err)
	}

	for epoch := 0; epoch < b.N; epoch++ {
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				dataset.Getitem(sizes[index], sizes[index])
			}
		}
		dataset.OnEpochEnd(int64(epoch))
	}
	dataset.OnTrainEnd()
}

func BenchmarkPartitionedDataset(b *testing.B) {
	b.StopTimer()
	const (
		batchSize = 40
		worldSize = 4
	)
	groups := make([]int, worldSize)
	partitions := make([][]int, worldSize)
	for rank := range partitions {
		groups[rank] = rank
		partitions[rank] = rand.Perm(benchmarkDatasetSize / worldSize)
	}
	b.StartTimer()
	dataset, err := NewPartitionedDataset(groups, partitions)
	if err != nil {
		b.Fatal(err)
	}

	for epoch := 0; epoch < b.N; epoch++ {
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					dataset.Getitem(rank, partitions[rank][index])
				}
			}
		}
		dataset.OnEpochEnd(int64(epoch))
	}
	dataset.OnTrainEnd()
}
