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

// Package data provides primitives for representing and organizing
// the given dataset.
// In addition to the traditional sharded dataset, it supports
// a partitioned dataset where the data is split into multiple data partitions
// across nodes in the cluster.
package data

// Sample represents a single data sample in the dataset.
type Sample interface {
	// Index provides a primitive to identify a particular data sample
	// in the dataset.
	Index() int

	// Size provides a primitive for the user-defined relative size
	// of the data sample.
	Size() int

	// Less provides a primitive for ordering data samples.
	//
	// This must provide a strict weak ordering; if !a.Less(b) && !b.Less(a),
	// we treat this to mean a == b (i.e., we can only hold one of either a or b
	// in the dataset).
	Less(than Sample) bool
}

// SampleBase meets the minimum requirements for identifying the data sample.
type SampleBase struct {
	index int
	size  int
}

// NewSampleBase creates a new basic sample.
func NewSampleBase(index, size int) *SampleBase {
	return &SampleBase{
		index: index,
		size:  size,
	}
}

// Index returns the index of the data sample.
func (s SampleBase) Index() int {
	return s.index
}

// Size returns the relative size of the data sample.
func (s SampleBase) Size() int {
	return s.size
}

// Less tests whether the current data sample is less than the given argument.
func (s SampleBase) Less(than Sample) bool {
	if s.Size() == than.Size() {
		return s.Index() < than.Index()
	}
	return s.Size() < than.Size()
}
