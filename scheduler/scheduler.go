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
	"math"
	"math/rand"
	"sync"

	"github.com/9rum/chronica/internal/data"
)

const (
	STATIC = iota
	DYNAMIC
)

// Scheduler represents the data scheduler.
// All implementations must embed SchedulerBase for forward compatibility.
type Scheduler interface {
	// Schedule selects data samples for the next mini-batch.
	Schedule() [][]int

	// Len returns the number of required steps for each training epoch.
	Len() int

	// OnEpochEnd is called at the end of an epoch during training.
	OnEpochEnd(epoch, rank int64, coefficient, intercept float64)

	// OnTrainEnd terminates the training environment.
	OnTrainEnd()
}

// SchedulerBase must be embedded to have forward compatible implementations.
type SchedulerBase struct {
}

func (SchedulerBase) Schedule() (_ [][]int) {
	return
}
func (SchedulerBase) Len() (_ int) {
	return
}
func (SchedulerBase) OnEpochEnd(epoch, rank int64, coefficient, intercept float64) {}
func (SchedulerBase) OnTrainEnd()                                                  {}

// New creates a new scheduler with the given arguments.
func New[T ~int32](dataset data.Dataset, worldSize, batchSize int, sizes []int, typ T) Scheduler {
	steps := func(numerator, denominator int) int {
		if numerator%denominator == 0 {
			return numerator / denominator
		}
		return numerator/denominator + 1
	}(len(sizes), batchSize)

	switch typ {
	case STATIC:
		return NewStaticScheduler(dataset, worldSize, batchSize, steps, sizes)
	case DYNAMIC:
		return NewDynamicScheduler(dataset, worldSize, batchSize, steps)
	default:
		panic("invalid type")
	}
}

// Next returns mini-batches for the next training epoch. This returns a matrix
// of shape (world size, # of samples).
func Next(scheduler Scheduler) [][]int {
	indices := make([][][]int, 0, scheduler.Len())
	for len(indices) < cap(indices) {
		indices = append(indices, scheduler.Schedule())
	}

	// store the number of scheduled data samples considering
	// insufficient data samples before shuffling
	samples := (len(indices)-1)*len(indices[0][0]) + len(indices[len(indices)-1][0])

	// shuffle between batches
	if samples%len(indices[0][0]) == 0 {
		rand.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	} else {
		rand.Shuffle(len(indices)-1, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	return transpose(indices, samples)
}

// transpose returns a corresponding two-dimensional tensor (i.e., a matrix) of
// the given three-dimensional tensor.  This converts the given tensor of shape
// (# of steps, world size, local batch size) to a tensor of shape
// (world size, # of samples). samples stands for the number of scheduled data
// samples to each of the workers.
func transpose(tensor [][][]int, samples int) [][]int {
	matrix := make([][]int, 0, len(tensor[0]))
	for len(matrix) < cap(matrix) {
		matrix = append(matrix, make([]int, samples))
	}

	var wg sync.WaitGroup
	for rank := range matrix {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			base := 0
			for _, m := range tensor {
				base += copy(matrix[rank][base:], m[rank])
			}
		}(rank)
	}
	wg.Wait()

	return matrix
}

// StaticScheduler provides balanced workload to each of the workers while
// limiting the peak device memory usage; this allows for larger batch size,
// reducing the communication overheads and thereby improving the scalability.
type StaticScheduler struct {
	SchedulerBase
	dataset   data.Dataset
	worldSize int
	batchSize int
	binSize   int
	steps     int
}

// NewStaticScheduler creates a new static scheduler with the given arguments.
func NewStaticScheduler(dataset data.Dataset, worldSize, batchSize, steps int, sizes []int) *StaticScheduler {
	return &StaticScheduler{
		dataset:   dataset,
		worldSize: worldSize,
		batchSize: batchSize,
		binSize:   int(math.Round(mean(sizes) * float64(batchSize/worldSize))),
		steps:     steps,
	}
}

// mean averages the given sizes.
func mean(sizes []int) float64 {
	sum := func() (sum int) {
		for _, size := range sizes {
			sum += size
		}
		return
	}()
	return float64(sum) / float64(len(sizes))
}

// Schedule assigns the next mini-batch to each of the workers.  It adopts
// first-fit-decreasing (FFD), which is an approximately-optimal heuristic for
// bin packing.
// FFD paper: https://dspace.mit.edu/bitstream/handle/1721.1/57819/17595570-MIT.pdf
// Python implementation: https://github.com/erelsgl/prtpy/blob/main/prtpy/packing/first_fit.py
func (s *StaticScheduler) Schedule() [][]int {
	bins := make([]int, s.worldSize)
	indices := make([][]int, 0, s.worldSize)
	for len(indices) < cap(indices) {
		indices = append(indices, make([]int, 0, s.batchSize/s.worldSize))
	}

	// pack the bins in a first-fit-decreasing fashion
	for rank := range indices {
		for len(indices[rank]) < cap(indices[rank]) {
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

func (s *StaticScheduler) Len() int {
	return s.steps
}

func (s *StaticScheduler) OnEpochEnd(epoch, rank int64, coefficient, intercept float64) {
	if rank == 0 {
		s.dataset.OnEpochEnd(epoch)
	}
}

func (s *StaticScheduler) OnTrainEnd() {
	s.dataset.OnTrainEnd()
	s.dataset = nil
}

// DynamicScheduler provides a feedback-directed optimization. It adaptively
// adjusts the workload on each worker, which can be useful in heterogeneous
// clusters where the workers have different compute capabilities.
type DynamicScheduler struct {
	SchedulerBase
	dataset      data.Dataset
	worldSize    int
	batchSize    int
	steps        int
	coefficients []float64
	intercepts   []float64
}

// NewDynamicScheduler creates a new dynamic scheduler with the given arguments.
func NewDynamicScheduler(dataset data.Dataset, worldSize, batchSize, steps int) *DynamicScheduler {
	return &DynamicScheduler{
		dataset:      dataset,
		worldSize:    worldSize,
		batchSize:    batchSize,
		steps:        steps,
		coefficients: make([]float64, worldSize),
		intercepts:   make([]float64, worldSize),
	}
}

// Schedule assigns the next mini-batch to each of the workers based on their
// performance indicators. It adopts best-fit with a random first pivot to
// equalize the estimated training time while randomizing the training sequence.
// This is a revised version of our original scheme for straggler mitigation
// against imbalanced data, which has been proposed in the 23rd IEEE/ACM
// International Symposium on Cluster, Cloud and Internet Computing (CCGrid).
// Chronica paper: https://ieeexplore.ieee.org/document/10171495
func (s *DynamicScheduler) Schedule() [][]int {
	binSize := 0.
	bins := make([]float64, s.worldSize)
	indices := make([][]int, 0, s.worldSize)
	for len(indices) < cap(indices) {
		indices = append(indices, make([]int, 0, s.batchSize/s.worldSize))
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

	for len(indices[0]) < cap(indices[0]) {
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

func (s *DynamicScheduler) Len() int {
	return s.steps
}

// OnEpochEnd updates the worker profile with the given feedback.
func (s *DynamicScheduler) OnEpochEnd(epoch, rank int64, coefficient, intercept float64) {
	s.coefficients[rank], s.intercepts[rank] = coefficient, intercept
	if rank == 0 {
		s.dataset.OnEpochEnd(epoch)
	}
}

func (s *DynamicScheduler) OnTrainEnd() {
	s.dataset.OnTrainEnd()
	s.dataset = nil
}
