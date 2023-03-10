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

// Package scheduler provides data-imbalance-aware scheduling primitives
// for distributed deep learning.
// In addition to static scheduling that reduces the load imbalance,
// it supports a feedback-directed optimization that adaptively adjusts
// the workload to each worker.
package scheduler

// Scheduler represents the data scheduler.
type Scheduler interface {
	// WorldSize provides a primitive for the number of workers in the group.
	WorldSize() int

	// BatchSize provides a primitive for the number of data samples to select
	// at each step.
	BatchSize() int

	// Schedule provides a mechanism for selecting which data samples to assign
	// to the worker at each step.
	Schedule(rank int) map[int]struct{}
}

// SchedulerBase is a static data scheduler that aims to reduce
// the load imbalance between workers.
type SchedulerBase struct {
	worldSize int
	batchSize int
}

// WorldSize returns the number of workers in the group.
func (sched SchedulerBase) WorldSize() int {
	return sched.worldSize
}

// BatchSize returns the batch size.
func (sched SchedulerBase) BatchSize() int {
	return sched.batchSize
}

// Schedule assigns the next mini-batch to the worker.
