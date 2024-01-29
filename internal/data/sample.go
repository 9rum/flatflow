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

package data

import "github.com/9rum/chronica/internal/btree"

// Sample represents a single data sample in the data set.
type Sample struct {
	btree.ItemBase
}

// NewSample creates a new data sample with the given arguments.
func NewSample(index, size int) Sample {
	return Sample{
		ItemBase: btree.NewItemBase(index, size),
	}
}

// Less tests whether the current data sample is less than the given argument.
// This allows the underlying container to non-deterministically return items
// for a given key while keeping the sorting order.
func (s Sample) Less(than btree.Item) bool {
	if s.Size() == than.Size() {
		return mapping[s.Index()] < mapping[than.Index()]
	}
	return s.Size() < than.Size()
}
