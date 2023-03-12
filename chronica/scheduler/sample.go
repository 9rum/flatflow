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

// Sample represents a single data sample in the dataset.
type Sample interface {
	// Index provides a primitive to identify a particular data sample
	// in the dataset.
	Index() int

	// Size provides a primitive for the user-defined relative size
	// of the data sample.
	Size() int

	// Less provides a primitive for ordering data samples.
	// This must provide a strict weak ordering.
	Less(than Sample) bool
}

// SampleBase meets the minimum requirements for identifying the data sample.
type SampleBase struct {
	index int
	size  int
}

// Index returns the index of the data sample.
func (sample SampleBase) Index() int {
	return sample.index
}

// Size returns the relative size of the data sample.
func (sample SampleBase) Size() int {
	return sample.size
}

// Less tests whether the current data sample is less than the given argument.
func (sample SampleBase) Less(than Sample) bool {
	return sample.Size() < than.Size()
}
