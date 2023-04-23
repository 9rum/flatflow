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

// Scheduler represents the data scheduler.
type Scheduler interface {
	// WorldSize provides a primitive for the total number of workers in the group.
	WorldSize() int

	// Schedule provides a mechanism for selecting data samples to assign
	// to each worker.
	Schedule() []map[int]struct{}
}

// SchedulerBase aims to reduce the load imbalance between workers.
// It provides static data scheduling, which is effective in a group of workers
// with similar performance.
type SchedulerBase struct {
	worldSize int
}

// WorldSize returns the total number of workers in the group.
func (s SchedulerBase) WorldSize() int {
	return s.worldSize
}

// Schedule assigns the next mini-batch to the worker.
func (s *SchedulerBase) Schedule() []map[int]struct{}
