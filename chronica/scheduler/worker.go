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
	// Rank provides a mechanism for other processes
	// to identify a particular process.
	Rank() int
}

// StaticWorker is a worker used for static scheduling
// in a group of workers with similar performance
// (e.g., homogeneous cluster).
type StaticWorker struct {
	rank int
}

// Rank returns the unique ID of the worker.
func (worker StaticWorker) Rank() int {
	return worker.rank
}

// DynamicWorker is a worker used for dynamic scheduling
// in a group of workers with different performances
// (e.g., heterogeneous cluster).
type DynamicWorker struct {
	StaticWorker
	coefficient float64
	intercept   float64
}
