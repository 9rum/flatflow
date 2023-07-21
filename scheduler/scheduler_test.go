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
	"math/rand"
	"testing"

	"github.com/9rum/chronica/internal/data"
)

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
	dataset := data.NewShardedDataset(sizes)
	scheduler := NewStaticScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := int64(0); epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := int64(0); rank < worldSize; rank++ {
			scheduler.OnEpochEnd(epoch, rank, 0., 0.)
		}
		for step := 0; step < datasetSize/batchSize; step++ {
			sums := make([]int, 0, worldSize)
			for _, indices := range scheduler.Schedule(step) {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
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
	dataset := data.NewShardedDataset(sizes)
	scheduler := NewDynamicScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := int64(0); epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := int64(0); rank < worldSize; rank++ {
			scheduler.OnEpochEnd(epoch, rank, 1., 0.)
		}
		for step := 0; step < datasetSize/batchSize; step++ {
			sums := make([]int, 0, worldSize)
			for _, indices := range scheduler.Schedule(step) {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
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
	dataset := data.NewShardedDataset(sizes)
	scheduler := NewStaticScheduler(dataset, worldSize, batchSize, sizes)
	b.StartTimer()

	for epoch := int64(0); epoch < int64(b.N); epoch++ {
		for rank := int64(0); rank < worldSize; rank++ {
			scheduler.OnEpochEnd(epoch, rank, 0., 0.)
		}
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			scheduler.Schedule(step)
		}
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
	dataset := data.NewShardedDataset(sizes)
	scheduler := NewDynamicScheduler(dataset, worldSize, batchSize, sizes)
	b.StartTimer()

	for epoch := int64(0); epoch < int64(b.N); epoch++ {
		for rank := int64(0); rank < worldSize; rank++ {
			scheduler.OnEpochEnd(epoch, rank, 1., 0.)
		}
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			scheduler.Schedule(step)
		}
	}
	scheduler.OnTrainEnd()
}
