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
	"runtime"
	"sync"
	"testing"
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

func TestShardedDataset(t *testing.T) {
	const (
		datasetSize = 10000
		batchSize   = 10
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	dataset := NewShardedDataset(sizes, seed)

	for epoch := 0; epoch < 10; epoch++ {
		dataset.OnEpochEnd(int64(epoch))
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
		sizes[size] = int64(size)
	}
	dataset = NewShardedDataset(sizes, seed)

	for epoch := 0; epoch < 10; epoch++ {
		dataset.OnEpochEnd(int64(epoch))
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
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	groups := make([]int64, 0, worldSize)
	for len(groups) < cap(groups) {
		groups = append(groups, int64(len(groups)))
	}
	dataset := NewPartitionedDataset(sizes, groups, seed)

	for epoch := 0; epoch < 10; epoch++ {
		dataset.OnEpochEnd(int64(epoch))
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
		sizes[size] = int64(size)
	}
	dataset = NewPartitionedDataset(sizes, groups, seed)

	for epoch := 0; epoch < 10; epoch++ {
		dataset.OnEpochEnd(int64(epoch))
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
	const (
		batchSize = 10
		seed      = 0
	)
	sizes := cast(rand.Perm(benchmarkDatasetSize))
	b.StartTimer()
	dataset := NewShardedDataset(sizes, seed)

	for epoch := 0; epoch < b.N; epoch++ {
		dataset.OnEpochEnd(int64(epoch))
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
		seed      = 0
	)
	sizes := cast(rand.Perm(benchmarkDatasetSize))
	groups := make([]int64, 0, worldSize)
	for len(groups) < cap(groups) {
		groups = append(groups, int64(len(groups)))
	}
	b.StartTimer()
	dataset := NewPartitionedDataset(sizes, groups, seed)

	for epoch := 0; epoch < b.N; epoch++ {
		dataset.OnEpochEnd(int64(epoch))
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
