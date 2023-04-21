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

package btree

import "reflect"

// Item represents a single object in the tree.
type Item interface {
	// Index provides a primitive to identify a particular item in the tree.
	Index() int

	// Size provides a primitive for the user-defined relative size of the item.
	Size() int

	// Less tests whether the current item is less than the given argument.
	//
	// This must provide a strict weak ordering; if !a.Less(b) && !b.Less(a),
	// we treat this to mean a == b (i.e., we can only hold one of either a or b
	// in the tree).
	Less(than Item) bool
}

// NewItem creates a new item with the given constraints.  This uses reflection
// to mimic type switches. If type switches are supported in the future,
// reflection may be removed; there is no language support for type switches on
// the type parameter.
func NewItem[T Item](index, size int) T {
	var (
		zero     T
		zeroBase *ItemBase
	)

	switch reflect.TypeOf(zero) {
	case reflect.TypeOf(zeroBase):
		return Item(NewItemBase(index, size)).(T)
	default:
		panic("invalid type")
	}
}

// ItemBase meets the minimum requirements for identifying the item.
type ItemBase struct {
	index int
	size  int
}

// NewItemBase creates a new base item with the given arguments.
func NewItemBase(index, size int) *ItemBase {
	return &ItemBase{
		index: index,
		size:  size,
	}
}

// Index returns the index of the item.
func (i ItemBase) Index() int {
	return i.index
}

// Size returns the relative size of the item.
func (i ItemBase) Size() int {
	return i.size
}

// Less tests whether the current item is less than the given argument.
func (i ItemBase) Less(than Item) bool {
	if i.Size() == than.Size() {
		return i.Index() < than.Index()
	}
	return i.Size() < than.Size()
}
