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
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/9rum/chronica/internal/btree"
	"github.com/9rum/chronica/internal/data"
)

func init() {
	seed := time.Now().Unix()
	fmt.Println(seed)
	rand.Seed(seed)
}

// reduce extracts all rank-wise sums of the selected data sample size.
func reduce(indices [][]int, sizes []int) []int {
	sums := make([]int, len(indices))
	for rank, indexes := range indices {
		sum := 0
		for _, index := range indexes {
			sum += sizes[index]
		}
		sums[rank] = sum
	}
	return sums
}

// mean returns the mean of the given sizes.
func mean(sizes []int) float64 {
	sum := 0
	for _, size := range sizes {
		sum += size
	}
	return float64(sum) / float64(len(sizes))
}

func TestStaticScheduler(t *testing.T) {
	const (
		datasetSize = 1 << 10
		worldSize   = 1 << 2
		batchSize   = 1 << 5
	)
	var (
		sizes                  = rand.Perm(datasetSize)
		dataset   data.Dataset = data.NewShardedDataset[*btree.ItemBase](sizes)
		scheduler Scheduler    = NewStaticScheduler(dataset, worldSize, batchSize, int(math.Round(mean(sizes)*batchSize/worldSize)))
	)
	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			indices := scheduler.Schedule()
			t.Logf("step: %d got: %v", step, reduce(indices, sizes))
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()
}

const benchmarkDatasetSize = 1 << 14

func BenchmarkStaticScheduler(b *testing.B) {
	b.StopTimer()
	const (
		worldSize = 1 << 3
		batchSize = 1 << 7
	)
	var (
		sizes                  = rand.Perm(benchmarkDatasetSize)
		dataset   data.Dataset = data.NewShardedDataset[*btree.ItemBase](sizes)
		scheduler Scheduler    = NewStaticScheduler(dataset, worldSize, batchSize, int(math.Round(mean(sizes)*batchSize/worldSize)))
	)
	b.StartTimer()
	for epoch := 0; epoch < b.N; epoch++ {
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			scheduler.Schedule()
		}
		dataset.OnEpochEnd()
	}
	dataset.OnTrainEnd()
}
