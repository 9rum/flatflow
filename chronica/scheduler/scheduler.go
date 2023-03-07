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

// Package scheduler provides a data-imbalance-aware scheduling mechanism
// for distributed deep learning.
// In addition to static scheduling that reduces the load imbalance,
// it also provides a feedback-directed optimization that adjusts workload
// based on the feedback of each worker.
package scheduler

// Scheduler represents the data scheduler.
type Scheduler interface {
	// Schedule assigns the mini-batch for the next step to the worker.
	Schedule(rank int) map[int]struct{}

	// WorldSize returns the number of workers in the group.
	WorldSize() int

	// BatchSize returns the batch size.
	BatchSize() int

	// Len returns the number of data samples to schedule
	// within each training epoch.
	Len() int
}

// StaticScheduler is a static data scheduler
// that aims to reduce the load imbalance between workers.
type StaticScheduler struct {
}

// DynamicScheduler is a dynamic data scheduler
// that optimizes workload based on the feedback of each worker.
type DynamicScheduler struct {
}
