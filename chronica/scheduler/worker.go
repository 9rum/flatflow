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

// Worker represents a single process in the group.
type Worker interface {
	// Rank provides a primitive for other processes
	// to identify a particular process.
	Rank() int
}

// WorkerBase meets the minimum requirement for identifying the process.
// It is used for static scheduling in a group of workers
// with similar performance.
type WorkerBase struct {
	rank int
}

// Rank returns the unique ID of the worker.
func (w WorkerBase) Rank() int {
	return w.rank
}

// LinearWorker is a worker with linear time complexity
// for the size of each data sample.
// It is used for feedback-directed optimization in a group of workers
// with dynamic performance.
type LinearWorker struct {
	WorkerBase
	coefficient float64
	intercept   float64
}
