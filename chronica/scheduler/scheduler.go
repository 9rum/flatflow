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

//go:build goexperiment.arenas

// Package scheduler provides primitives for scheduling imbalanced data.
// In addition to static scheduling that reduces the load imbalance,
// it supports a feedback-directed optimization that adaptively adjusts
// the workload on each worker.
package scheduler

import (
	"arena"
	"math/rand"
	"time"

	"github.com/9rum/chronica/internal/data"
)

// Scheduler represents the data scheduler.
// All implementations must embed SchedulerBase for forward compatibility.
type Scheduler interface {
	// Schedule selects data samples for the next mini-batch.
	Schedule() [][]int

	// OnBatchEnd is called at the end of a training batch.
	OnBatchEnd(rank int, coefficient, intercept float64)

	// OnEpochEnd is called at the end of an epoch during training.
	OnEpochEnd()

	// OnTrainEnd terminates the training environment.
	OnTrainEnd()
}

// SchedulerBase must be embedded to have forward compatible implementations.
type SchedulerBase struct {
}

func (SchedulerBase) Schedule() [][]int {
	return nil
}
func (SchedulerBase) OnBatchEnd(rank int, coefficient, intercept float64) {}
func (SchedulerBase) OnEpochEnd()                                         {}
func (SchedulerBase) OnTrainEnd()                                         {}

// StaticScheduler provides balanced workload to each of the workers while
// limiting the peak device memory usage; this allows for larger batch size,
// reducing the communication overheads and thereby improving the scalability.
type StaticScheduler struct {
	SchedulerBase
	dataset   data.Dataset
	worldSize int
	batchSize int
	binSize   int
	step      int
	indices   [][][]int
	arena     *arena.Arena
}

// NewStaticScheduler creates a new static scheduler with the given arguments.
func NewStaticScheduler(dataset data.Dataset, worldSize, batchSize, binSize, steps int) Scheduler {
	// We use memory arenas to reduce GC overhead.
	scheduler := &StaticScheduler{
		dataset:   dataset,
		worldSize: worldSize,
		batchSize: batchSize,
		binSize:   binSize,
		arena:     arena.NewArena(),
	}
	scheduler.indices = arena.MakeSlice[[][]int](scheduler.arena, 0, steps)

	return scheduler
}

// Schedule returns the next mini-batch. It selects data samples in a
// first-fit-decreasing manner in the first epoch, while the training sequence
// is randomized by shuffling the batch indices in the subsequent epochs.
func (s *StaticScheduler) Schedule() [][]int {
	defer func() {
		s.step++
	}()

	if s.dataset != nil {
		s.indices = append(s.indices, s.schedule())
	}
	return s.indices[s.step]
}

// schedule assigns the next mini-batch to each of the workers.  It adopts
// first-fit-decreasing (FFD), which is an approximately-optimal heuristic for
// bin packing.
// FFD paper: https://dspace.mit.edu/bitstream/handle/1721.1/57819/17595570-MIT.pdf?sequence=2
// Python implementation: https://github.com/erelsgl/prtpy/blob/main/prtpy/packing/first_fit.py
func (s StaticScheduler) schedule() [][]int {
	bins := make([]int, s.worldSize)
	indices := arena.MakeSlice[[]int](s.arena, s.worldSize, s.worldSize)
	for rank := range indices {
		indices[rank] = arena.MakeSlice[int](s.arena, 0, s.batchSize/s.worldSize)
	}

	// pack the bins in a first-fit-decreasing fashion
	for rank := range indices {
		for step := 0; step < s.batchSize/s.worldSize; step++ {
			if s.dataset.Len(rank) <= 0 {
				break
			}
			index, size := s.dataset.Getitem(rank, s.binSize-bins[rank])
			indices[rank] = append(indices[rank], index)
			bins[rank] += size
		}
	}

	return indices
}

func (s StaticScheduler) OnBatchEnd(rank int, coefficient, intercept float64) {
	if s.dataset != nil {
		s.dataset.OnBatchEnd(rank)
	}
}

// OnEpochEnd randomizes the training sequence.
func (s *StaticScheduler) OnEpochEnd() {
	if s.dataset != nil {
		s.dataset.OnTrainEnd()
		s.dataset = nil
	}

	rand.Seed(time.Now().Unix())
	rand.Shuffle(len(s.indices), func(i, j int) {
		s.indices[i], s.indices[j] = s.indices[j], s.indices[i]
	})

	s.step = 0
}

// OnTrainEnd frees the allocated memory arena.
func (s StaticScheduler) OnTrainEnd() {
	s.arena.Free()
}
