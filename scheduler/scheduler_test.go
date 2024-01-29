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
	"runtime"
	"sync"
	"testing"

	"github.com/9rum/chronica/internal/data"
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

// sum returns the sum of the selected data samples size.
func sum(indices, sizes []int64) (sum int64) {
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
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewStaticScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 0., 0.)
		}
		for step := 0; step < datasetSize/batchSize; step++ {
			sums := make([]int64, 0, worldSize)
			for _, indices := range scheduler.Schedule(step) {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
	}
	scheduler.OnTrainEnd()
}

func TestStaticSchedulerWithRemainder(t *testing.T) {
	const (
		datasetSize = 1<<10 + 1<<3
		worldSize   = 1 << 2
		batchSize   = 1 << 5
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewStaticScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 0., 0.)
		}
		for step := 0; step < datasetSize/batchSize+1; step++ {
			sums := make([]int64, 0, worldSize)
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
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewDynamicScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 1., 0.)
		}
		for step := 0; step < datasetSize/batchSize; step++ {
			sums := make([]int64, 0, worldSize)
			for _, indices := range scheduler.Schedule(step) {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
	}
	scheduler.OnTrainEnd()
}

func TestDynamicSchedulerWithRemainder(t *testing.T) {
	const (
		datasetSize = 1<<10 + 1<<3
		worldSize   = 1 << 2
		batchSize   = 1 << 5
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewDynamicScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 1., 0.)
		}
		for step := 0; step < datasetSize/batchSize+1; step++ {
			sums := make([]int64, 0, worldSize)
			for _, indices := range scheduler.Schedule(step) {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
	}
	scheduler.OnTrainEnd()
}

func TestGuidedScheduler(t *testing.T) {
	const (
		datasetSize = 1 << 10
		worldSize   = 1 << 2
		batchSize   = 1 << 5
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewGuidedScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 0., 0.)
		}
		for step := 0; step < datasetSize/batchSize; step++ {
			sums := make([]int64, 0, worldSize)
			for _, indices := range scheduler.Schedule(step) {
				sums = append(sums, sum(indices, sizes))
			}
			t.Logf("step: %d got: %v", step, sums)
		}
	}
	scheduler.OnTrainEnd()
}

func TestGuidedSchedulerWithRemainder(t *testing.T) {
	const (
		datasetSize = 1<<10 + 1<<3
		worldSize   = 1 << 2
		batchSize   = 1 << 5
		seed        = 0
	)
	sizes := cast(rand.Perm(datasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewGuidedScheduler(dataset, worldSize, batchSize, sizes)

	for epoch := 0; epoch < 10; epoch++ {
		t.Logf("epoch: %d", epoch)
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 0., 0.)
		}
		for step := 0; step < datasetSize/batchSize+1; step++ {
			sums := make([]int64, 0, worldSize)
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
		seed      = 0
	)
	sizes := cast(rand.Perm(benchmarkDatasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewStaticScheduler(dataset, worldSize, batchSize, sizes)
	b.StartTimer()

	for epoch := 0; epoch < b.N; epoch++ {
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 0., 0.)
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
		seed      = 0
	)
	sizes := cast(rand.Perm(benchmarkDatasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewDynamicScheduler(dataset, worldSize, batchSize, sizes)
	b.StartTimer()

	for epoch := 0; epoch < b.N; epoch++ {
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 1., 0.)
		}
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			scheduler.Schedule(step)
		}
	}
	scheduler.OnTrainEnd()
}

func BenchmarkGuidedScheduler(b *testing.B) {
	b.StopTimer()
	const (
		worldSize = 1 << 3
		batchSize = 1 << 7
		seed      = 0
	)
	sizes := cast(rand.Perm(benchmarkDatasetSize))
	dataset := data.NewShardedDataset(sizes, seed)
	scheduler := NewGuidedScheduler(dataset, worldSize, batchSize, sizes)
	b.StartTimer()

	for epoch := 0; epoch < b.N; epoch++ {
		for rank := 0; rank < worldSize; rank++ {
			scheduler.OnEpochEnd(int64(epoch), int64(rank), 0., 0.)
		}
		for step := 0; step < benchmarkDatasetSize/batchSize; step++ {
			scheduler.Schedule(step)
		}
	}
	scheduler.OnTrainEnd()
}
