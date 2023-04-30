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

// Package scheduler provides primitives for scheduling imbalanced data.
// In addition to static scheduling that reduces the load imbalance,
// it supports a feedback-directed optimization that adaptively adjusts
// the workload on each worker.
package scheduler

import (
	"sync"

	"github.com/9rum/chronica/internal/data"
)

// Scheduler represents the data scheduler.
type Scheduler interface {
	// WorldSize provides a primitive for the total number of workers in the group.
	WorldSize() int

	// BatchSize provides a primitive for the number of data samples to select
	// at each step.
	BatchSize() int

	// Schedule provides a mechanism for selecting data samples to be assigned
	// to each worker.
	Schedule() []map[int]struct{}
}

// SchedulerBase aims to reduce the load imbalance between workers.  It provides
// static data scheduling, which can be effective in a group of workers with
// similar performance.
type SchedulerBase struct {
	dataset   data.Dataset
	worldSize int
	batchSize int
}

// NewSchedulerBase creates a new base scheduler with the given arguments.
func NewSchedulerBase(dataset data.Dataset, worldSize, batchSize int) *SchedulerBase {
	return &SchedulerBase{
		dataset:   dataset,
		worldSize: worldSize,
		batchSize: batchSize,
	}
}

// WorldSize returns the total number of workers in the group.
func (s SchedulerBase) WorldSize() int {
	return s.worldSize
}

// BatchSize returns the batch size.
func (s SchedulerBase) BatchSize() int {
	return s.batchSize
}

// Schedule assigns the next mini-batch to each of the workers.  It utilizes the
// best-fit bin packing, but with the random first pivots.
// Best-fit bin packing paper: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=26899b37abf3df0c1d2adaeb51acad2ab44b9c43
func (s *SchedulerBase) Schedule() []map[int]struct{} {
	binSize := 0
	bins := make([]int, s.worldSize)
	indices := make([]map[int]struct{}, s.worldSize)
	for rank := range indices {
		indices[rank] = make(map[int]struct{})
	}

	// assign random first pivots
	for rank := range indices {
		if 0 < s.dataset.Len(rank) {
			index, size := s.dataset.Rand(rank)
			indices[rank][index] = struct{}{}
			bins[rank] = size
			if binSize < bins[rank] {
				binSize = bins[rank]
			}
		}
	}

	// select data samples iteratively in a best-fit fashion
	for step := 1; step < s.batchSize/s.worldSize; step++ {
		for rank := range indices {
			if 0 < s.dataset.Len(rank) {
				index, size := s.dataset.Getitem(rank, binSize-bins[rank])
				indices[rank][index] = struct{}{}
				bins[rank] += size
				if binSize < bins[rank] {
					binSize = bins[rank]
				}
			}
		}
	}

	return indices
}

// SizedScheduler provides balanced workload to each of the workers while
// limiting the peak device memory usage; this allows for larger batch size,
// improving the scalability by reducing the overhead of communications.
type SizedScheduler struct {
	*SchedulerBase
	binSize int
}

// NewSizedScheduler creates a new sized scheduler with the given arguments.
func NewSizedScheduler(dataset data.Dataset, worldSize, batchSize, binSize int) *SizedScheduler {
	return &SizedScheduler{
		SchedulerBase: NewSchedulerBase(dataset, worldSize, batchSize),
		binSize:       binSize,
	}
}

// Schedule returns the indices of the scheduled data samples.  It adopts
// first-fit-decreasing (FFD), which is an approximately-optimal heuristic for
// bin packing, but with the random first pivots.
// FFD paper: https://dspace.mit.edu/bitstream/handle/1721.1/57819/17595570-MIT.pdf?sequence=2
// Python implementation: https://github.com/erelsgl/prtpy/blob/main/prtpy/packing/first_fit.py
func (s *SizedScheduler) Schedule() []map[int]struct{} {
	bins := make([]int, s.worldSize)
	indices := make([]map[int]struct{}, s.worldSize)
	for rank := range indices {
		indices[rank] = make(map[int]struct{})
	}

	// assign random first pivots
	for rank := range indices {
		if 0 < s.dataset.Len(rank) {
			index, size := s.dataset.Rand(rank)
			indices[rank][index] = struct{}{}
			bins[rank] = size
		}
	}

	// pack the bins concurrently in a first-fit-decreasing fashion
	var (
		wg sync.WaitGroup
		mu sync.Mutex
	)
	for rank := range indices {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			for step := 1; step < s.batchSize/s.worldSize; step++ {
				mu.Lock()
				if s.dataset.Len(rank) <= 0 {
					break
				}
				index, size := s.dataset.Getitem(rank, s.binSize-bins[rank])
				mu.Unlock()
				indices[rank][index] = struct{}{}
				bins[rank] += size
			}
		}(rank)
	}

	wg.Wait()

	return indices
}
