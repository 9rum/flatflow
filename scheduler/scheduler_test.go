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

// sum returns the sum of the selected data samples size.
func sum(indices, sizes []int) (sum int) {
	for _, index := range indices {
		sum += sizes[index]
	}
	return
}

func TestStaticScheduler(t *testing.T) {
	const (
		datasetSize = 1 << 10
		worldSize   = 1 << 2
		batchSize   = 1 << 5
	)
	sizes := rand.Perm(datasetSize)
	dataset, err := data.NewShardedDataset[*btree.ItemBase](sizes)
	if err != nil {
		t.Fatal(err)
	}
	scheduler := NewStaticScheduler(dataset, worldSize, batchSize, binSize(sizes, worldSize, batchSize), datasetSize/batchSize)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			sums := make([]int, 0, worldSize)
			for _, indices := range scheduler.Schedule() {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
		scheduler.OnEpochEnd()
	}
	scheduler.OnTrainEnd()
}

func TestDynamicScheduler(t *testing.T) {
	const (
		datasetSize = 1 << 10
		worldSize   = 1 << 2
		batchSize   = 1 << 5
	)
	sizes := rand.Perm(datasetSize)
	dataset, err := data.NewShardedDataset[*btree.ItemBase](sizes)
	if err != nil {
		t.Fatal(err)
	}
	scheduler := NewDynamicScheduler(dataset, worldSize, batchSize)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for step := 0; step < datasetSize/batchSize; step++ {
			sums := make([]int, 0, worldSize)
			for rank := range sums {
				scheduler.OnBatchEnd(rank, 1., 0.)
			}
			for _, indices := range scheduler.Schedule() {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
		scheduler.OnEpochEnd()
	}
	scheduler.OnTrainEnd()
}

const benchmarkDatasetSize = 1 << 14

func BenchmarkStaticScheduler(b *testing.B) {
	b.StopTimer()
	const (
		worldSize = 1 << 3
		batchSize = 1 << 7
	)
	sizes := rand.Perm(benchmarkDatasetSize)
	dataset, _ := data.NewShardedDataset[*btree.ItemBase](sizes)
	scheduler := NewStaticScheduler(dataset, worldSize, batchSize, binSize(sizes, worldSize, batchSize), benchmarkDatasetSize/batchSize)
	b.StartTimer()

	for epoch := 0; epoch < b.N; epoch++ {
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			scheduler.Schedule()
		}
		scheduler.OnEpochEnd()
	}
	scheduler.OnTrainEnd()
}

func BenchmarkDynamicScheduler(b *testing.B) {
	b.StopTimer()
	const (
		worldSize = 1 << 3
		batchSize = 1 << 7
	)
	sizes := rand.Perm(benchmarkDatasetSize)
	dataset, _ := data.NewShardedDataset[*btree.ItemBase](sizes)
	scheduler := NewDynamicScheduler(dataset, worldSize, batchSize)
	b.StartTimer()

	for epoch := 0; epoch < b.N; epoch++ {
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			for rank := 0; rank < worldSize; rank++ {
				scheduler.OnBatchEnd(rank, 1., 0.)
			}
			scheduler.Schedule()
		}
		scheduler.OnEpochEnd()
	}
	scheduler.OnTrainEnd()
}
