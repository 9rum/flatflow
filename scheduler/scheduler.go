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
	"math"
	"math/rand"

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
	epoch     int64
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
	rand.Seed(scheduler.epoch)

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

	s.step = 0
	s.epoch++
	rand.Seed(s.epoch)
	rand.Shuffle(len(s.indices), func(i, j int) {
		s.indices[i], s.indices[j] = s.indices[j], s.indices[i]
	})
}

// OnTrainEnd frees the allocated memory arena.
func (s StaticScheduler) OnTrainEnd() {
	s.arena.Free()
}

// DynamicScheduler provides a feedback-directed optimization. It adaptively
// adjusts the workload on each worker, which can be useful in heterogeneous
// clusters where the workers have different compute capabilities.
type DynamicScheduler struct {
	SchedulerBase
	dataset      data.Dataset
	worldSize    int
	batchSize    int
	coefficients []float64
	intercepts   []float64
}

// NewDynamicScheduler creates a new dynamic scheduler with the given arguments.
func NewDynamicScheduler(dataset data.Dataset, worldSize, batchSize int) Scheduler {
	return &DynamicScheduler{
		dataset:      dataset,
		worldSize:    worldSize,
		batchSize:    batchSize,
		coefficients: make([]float64, worldSize),
		intercepts:   make([]float64, worldSize),
	}
}

// Schedule assigns the next mini-batch to each of the workers based on their
// performance indicators. It adopts best-fit with a random first pivot to
// equalize the estimated training time while randomizing the training sequence.
// This is a revised version of our original solution for straggler mitigation
// against imbalanced data, which has been proposed in the 23rd IEEE/ACM
// International Symposium on Cluster, Cloud and Internet Computing (CCGrid).
// Chronica paper: https://discos.sogang.ac.kr/file/2023/intl_conf/ccgrid23_chronica.pdf
func (s DynamicScheduler) Schedule() [][]int {
	binSize := 0.
	bins := make([]float64, s.worldSize)
	indices := make([][]int, s.worldSize)
	for rank := range indices {
		indices[rank] = make([]int, 0, s.batchSize/s.worldSize)
	}

	// assign a random first pivot
	if 0 < s.dataset.Len(0) {
		index, size := s.dataset.Rand(0)
		indices[0] = append(indices[0], index)
		bins[0] = s.coefficients[0]*float64(size) + s.intercepts[0]
		if binSize < bins[0] {
			binSize = bins[0]
		}
	}

	// select data samples iteratively in a best-fit fashion
	for rank := 1; rank < s.worldSize; rank++ {
		if 0 < s.dataset.Len(rank) {
			if s.coefficients[rank] == 0. {
				index, _ := s.dataset.Rand(rank)
				indices[rank] = append(indices[rank], index)
				bins[rank] = s.intercepts[rank]
			} else {
				index, size := s.dataset.Getitem(rank, int(math.Round((binSize-bins[rank]-s.intercepts[rank])/s.coefficients[rank])))
				indices[rank] = append(indices[rank], index)
				bins[rank] = s.coefficients[rank]*float64(size) + s.intercepts[rank]
			}
			if binSize < bins[rank] {
				binSize = bins[rank]
			}
		}
	}

	for step := 1; step < s.batchSize/s.worldSize; step++ {
		for rank := range indices {
			if 0 < s.dataset.Len(rank) {
				if s.coefficients[rank] == 0. {
					index, _ := s.dataset.Rand(rank)
					indices[rank] = append(indices[rank], index)
					bins[rank] += s.intercepts[rank]
				} else {
					index, size := s.dataset.Getitem(rank, int(math.Round((binSize-bins[rank]-s.intercepts[rank])/s.coefficients[rank])))
					indices[rank] = append(indices[rank], index)
					bins[rank] += s.coefficients[rank]*float64(size) + s.intercepts[rank]
				}
				if binSize < bins[rank] {
					binSize = bins[rank]
				}
			}
		}
	}

	return indices
}

// OnBatchEnd updates the worker profile with the given feedback.
func (s *DynamicScheduler) OnBatchEnd(rank int, coefficient, intercept float64) {
	s.coefficients[rank], s.intercepts[rank] = coefficient, intercept
	s.dataset.OnBatchEnd(rank)
}

func (s DynamicScheduler) OnEpochEnd() {
	s.dataset.OnEpochEnd()
}

func (s *DynamicScheduler) OnTrainEnd() {
	s.dataset.OnTrainEnd()
	s.dataset = nil
}
