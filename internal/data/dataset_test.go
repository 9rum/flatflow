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
	"math/rand"
	"testing"
)

func TestShardedDataset(t *testing.T) {
	const (
		datasetSize = 10000
		batchSize   = 10
	)
	sizes := rand.Perm(datasetSize)
	dataset := NewShardedDataset(sizes)

	for epoch := int64(0); epoch < 10; epoch++ {
		dataset.OnEpochEnd(epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				if _, size := dataset.Getitem(0, sizes[index]); size != sizes[index] {
					t.Fatalf("did not find %d", sizes[index])
				}
			}
		}
	}
	dataset.OnTrainEnd()

	for size := range sizes {
		sizes[size] = size
	}
	dataset = NewShardedDataset(sizes)

	for epoch := int64(0); epoch < 10; epoch++ {
		dataset.OnEpochEnd(epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				if _, size := dataset.Getitem(0, sizes[index]); size != sizes[index] {
					t.Fatalf("did not find %d", sizes[index])
				}
			}
		}
	}
	dataset.OnTrainEnd()
}

func TestPartitionedDataset(t *testing.T) {
	const (
		datasetSize = 10000
		batchSize   = 40
		worldSize   = 4
	)
	sizes := rand.Perm(datasetSize)
	groups := make([]int, 0, worldSize)
	for len(groups) < cap(groups) {
		groups = append(groups, len(groups))
	}
	dataset := NewPartitionedDataset(sizes, groups)

	for epoch := int64(0); epoch < 10; epoch++ {
		dataset.OnEpochEnd(epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					if _, size := dataset.Getitem(rank, sizes[datasetSize/worldSize*rank+index]); size != sizes[datasetSize/worldSize*rank+index] {
						t.Fatalf("did not find %d", sizes[datasetSize/worldSize*rank+index])
					}
				}
			}
		}
	}
	dataset.OnTrainEnd()

	for size := range sizes {
		sizes[size] = size
	}
	dataset = NewPartitionedDataset(sizes, groups)

	for epoch := int64(0); epoch < 10; epoch++ {
		dataset.OnEpochEnd(epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					if _, size := dataset.Getitem(rank, sizes[datasetSize/worldSize*rank+index]); size != sizes[datasetSize/worldSize*rank+index] {
						t.Fatalf("did not find %d", sizes[datasetSize/worldSize*rank+index])
					}
				}
			}
		}
	}
	dataset.OnTrainEnd()
}

const benchmarkDatasetSize = 10000

func BenchmarkShardedDataset(b *testing.B) {
	b.StopTimer()
	const batchSize = 10
	sizes := rand.Perm(benchmarkDatasetSize)
	b.StartTimer()
	dataset := NewShardedDataset(sizes)

	for epoch := int64(0); epoch < int64(b.N); epoch++ {
		dataset.OnEpochEnd(epoch)
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			for index := step * batchSize; index < (step+1)*batchSize; index++ {
				dataset.Getitem(0, sizes[index])
			}
		}
	}
	dataset.OnTrainEnd()
}

func BenchmarkPartitionedDataset(b *testing.B) {
	b.StopTimer()
	const (
		batchSize = 40
		worldSize = 4
	)
	sizes := rand.Perm(benchmarkDatasetSize)
	groups := make([]int, 0, worldSize)
	for len(groups) < cap(groups) {
		groups = append(groups, len(groups))
	}
	b.StartTimer()
	dataset := NewPartitionedDataset(sizes, groups)

	for epoch := int64(0); epoch < int64(b.N); epoch++ {
		dataset.OnEpochEnd(epoch)
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				for index := step * batchSize / worldSize; index < (step+1)*batchSize/worldSize; index++ {
					dataset.Getitem(rank, sizes[benchmarkDatasetSize/worldSize*rank+index])
				}
			}
		}
	}
	dataset.OnTrainEnd()
}
