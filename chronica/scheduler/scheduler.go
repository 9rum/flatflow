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

import "github.com/9rum/chronica/internal/data"

// Scheduler represents the data scheduler.
type Scheduler interface {
	// Schedule selects data samples for the next mini-batch.
	Schedule() [][]int
}

// StaticScheduler provides balanced workload to each of the workers while
// limiting the peak device memory usage; this allows for larger batch size,
// improving the scalability by reducing the overhead of communications.
type StaticScheduler struct {
	dataset   data.Dataset
	worldSize int
	batchSize int
	binSize   int
}

// NewStaticScheduler creates a new static scheduler with the given arguments.
func NewStaticScheduler(dataset data.Dataset, worldSize, batchSize, binSize int) *StaticScheduler {
	return &StaticScheduler{
		dataset:   dataset,
		worldSize: worldSize,
		batchSize: batchSize,
		binSize:   binSize,
	}
}

// Schedule assigns the next mini-batch to each of the workers.  It adopts
// first-fit-decreasing (FFD), which is an approximately-optimal heuristic for
// bin packing, but with the random first pivots.
// FFD paper: https://dspace.mit.edu/bitstream/handle/1721.1/57819/17595570-MIT.pdf?sequence=2
// Python implementation: https://github.com/erelsgl/prtpy/blob/main/prtpy/packing/first_fit.py
func (s *StaticScheduler) Schedule() [][]int {
	bins := make([]int, s.worldSize)
	indices := make([][]int, s.worldSize)
	for rank := range indices {
		indices[rank] = make([]int, 0, s.batchSize/s.worldSize)
	}

	// assign random first pivots
	for rank := range indices {
		if 0 < s.dataset.Len(rank) {
			index, size := s.dataset.Rand(rank)
			indices[rank] = append(indices[rank], index)
			bins[rank] = size
		}
	}

	// pack the bins in a first-fit-decreasing fashion
	for rank := range indices {
		for step := 1; step < s.batchSize/s.worldSize; step++ {
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
