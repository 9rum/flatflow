// Copyright 2014 Google Inc.
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

//go:build go1.18
// +build go1.18

// Package btree implements in-memory B-trees of arbitrary degree.
//
// btree implements an in-memory B-tree for use as an ordered data structure.
// It is not meant for persistent storage solutions.
//
// It has a flatter structure than an equivalent red-black or other binary tree,
// which in some cases yields better memory usage and/or performance.
// See some discussion on the matter here:
//
//	http://google-opensource.blogspot.com/2013/01/c-containers-that-save-memory-and-time.html
//
// Note, though, that this project is in no way related to the C++ B-tree
// implementation written about there.
//
// Within this tree, each node contains a slice of samples and a (possibly nil)
// slice of children.  For basic numeric values or raw structs, this can cause
// efficiency differences when compared to equivalent C++ template code that
// stores values in arrays within the node:
//   - Due to the overhead of storing values as interfaces (each
//     value needs to be stored as the value itself, then 2 words for the
//     interface pointing to that value and its type), resulting in higher
//     memory use.
//   - Since interfaces can point to values anywhere in memory, values are
//     most likely not stored in contiguous blocks, resulting in a higher
//     number of cache misses.
//
// These issues don't tend to matter, though, when working with strings or other
// heap-allocated structures, since C++-equivalent structures also must store
// pointers and also distribute their values across the heap.
//
// This implementation is designed to be a drop-in replacement to gollrb.LLRB
// trees, (http://github.com/petar/gollrb), an excellent and probably the most
// widely used ordered tree implementation in the Go ecosystem currently.
// Its functions, therefore, exactly mirror those of
// llrb.LLRB where possible.  Unlike gollrb, though, we currently don't
// support storing multiple equivalent values.
package btree

import (
	"sort"
	"sync"

	"github.com/9rum/chronica/internal/data"
)

const DefaultFreeListSize = 32

// FreeList represents a free list of BTree nodes. By default each
// BTree has its own FreeList, but multiple BTrees can share the same
// FreeList.
// Two BTrees using the same freelist are safe for concurrent write access.
type FreeList[T data.Sample] struct {
	mu       sync.Mutex
	freelist children[T]
}

// NewFreeList creates a new free list.
// size is the maximum size of the returned free list.
func NewFreeList[T data.Sample](size int) *FreeList[T] {
	return &FreeList[T]{freelist: make(children[T], 0, size)}
}

func (f *FreeList[T]) newNode() (n *node[T]) {
	f.mu.Lock()
	index := len(f.freelist) - 1
	if index < 0 {
		f.mu.Unlock()
		return new(node[T])
	}
	n = f.freelist[index]
	f.freelist[index] = nil
	f.freelist = f.freelist[:index]
	f.mu.Unlock()
	return
}

// freeNode adds the given node to the list, returning true if it was added
// and false if it was discarded.
func (f *FreeList[T]) freeNode(n *node[T]) (out bool) {
	f.mu.Lock()
	if len(f.freelist) < cap(f.freelist) {
		f.freelist = append(f.freelist, n)
		out = true
	}
	f.mu.Unlock()
	return
}

// SampleIterator allows callers of {A/De}scend* to iterate in-order over portions of
// the tree.  When this function returns false, iteration will stop and the
// associated {A/De}scend* function will immediately return.
type SampleIterator[T data.Sample] func(T) bool

// New creates a new B-tree with the given degree.
//
// New(2), for example, will create a 2-3-4 tree (each node contains 1-3 samples
// and 2-4 children).
func New[T data.Sample](degree int) *BTree[T] {
	return NewWithFreeList(degree, NewFreeList[T](DefaultFreeListSize))
}

// NewWithFreeList creates a new B-tree that uses the given node free list.
func NewWithFreeList[T data.Sample](degree int, f *FreeList[T]) *BTree[T] {
	if degree <= 1 {
		panic("bad degree")
	}
	return &BTree[T]{
		degree: degree,
		cow:    &copyOnWriteContext[T]{freelist: f},
	}
}

// samples stores samples in a node.
type samples[T data.Sample] []T

// insertAt inserts a value into the given index, pushing all subsequent values
// forward.
func (s *samples[T]) insertAt(index int, sample T) {
	var zero T
	*s = append(*s, zero)
	if index < len(*s) {
		copy((*s)[index+1:], (*s)[index:])
	}
	(*s)[index] = sample
}

// removeAt removes a value at a given index, pulling all subsequent values
// back.
func (s *samples[T]) removeAt(index int) T {
	sample := (*s)[index]
	copy((*s)[index:], (*s)[index+1:])
	var zero T
	(*s)[len(*s)-1] = zero
	*s = (*s)[:len(*s)-1]
	return sample
}

// pop removes and returns the last element in the list.
func (s *samples[T]) pop() (out T) {
	index := len(*s) - 1
	out = (*s)[index]
	var zero T
	(*s)[index] = zero
	*s = (*s)[:index]
	return
}

// truncate truncates this instance at index so that it contains only the
// first index samples. index must be less than or equal to length.
func (s *samples[T]) truncate(index int) {
	var toClear samples[T]
	*s, toClear = (*s)[:index], (*s)[index:]
	var zero T
	for i := 0; i < len(toClear); i++ {
		toClear[i] = zero
	}
}

// find returns the index where the given sample should be inserted into this
// list.  'found' is true if the sample already exists in the list at the given
// index.
func (s samples[T]) find(sample T) (index int, found bool) {
	i := sort.Search(len(s), func(i int) bool {
		return sample.Less(s[i])
	})
	if 0 < i && !s[i-1].Less(sample) {
		return i - 1, true
	}
	return i, false
}

// children stores child nodes in a node.
type children[T data.Sample] []*node[T]

// insertAt inserts a value into the given index, pushing all subsequent values
// forward.
func (c *children[T]) insertAt(index int, n *node[T]) {
	*c = append(*c, nil)
	if index < len(*c) {
		copy((*c)[index+1:], (*c)[index:])
	}
	(*c)[index] = n
}

// removeAt removes a value at a given index, pulling all subsequent values
// back.
func (c *children[T]) removeAt(index int) *node[T] {
	n := (*c)[index]
	copy((*c)[index:], (*c)[index+1:])
	(*c)[len(*c)-1] = nil
	*c = (*c)[:len(*c)-1]
	return n
}

// pop removes and returns the last element in the list.
func (c *children[T]) pop() (out *node[T]) {
	index := len(*c) - 1
	out = (*c)[index]
	(*c)[index] = nil
	*c = (*c)[:index]
	return
}

// truncate truncates this instance at index so that it contains only the
// first index children. index must be less than or equal to length.
func (s *children[T]) truncate(index int) {
	var toClear children[T]
	*s, toClear = (*s)[:index], (*s)[index:]
	for i := 0; i < len(toClear); i++ {
		toClear[i] = nil
	}
}

// node is an internal node in a tree.
//
// It must at all times maintain the invariant that either
//   - len(children) == 0, len(samples) unconstrained
//   - len(children) == len(samples) + 1
type node[T data.Sample] struct {
	samples  samples[T]
	children children[T]
	cow      *copyOnWriteContext[T]
}

func (n *node[T]) mutableFor(cow *copyOnWriteContext[T]) *node[T] {
	if n.cow == cow {
		return n
	}
	out := cow.newNode()
	if len(n.samples) <= cap(out.samples) {
		out.samples = out.samples[:len(n.samples)]
	} else {
		out.samples = make(samples[T], len(n.samples), cap(n.samples))
	}
	copy(out.samples, n.samples)
	// Copy children
	if len(n.children) <= cap(out.children) {
		out.children = out.children[:len(n.children)]
	} else {
		out.children = make(children[T], len(n.children), cap(n.children))
	}
	copy(out.children, n.children)
	return out
}

func (n *node[T]) mutableChild(i int) *node[T] {
	c := n.children[i].mutableFor(n.cow)
	n.children[i] = c
	return c
}

// split splits the given node at the given index.  The current node shrinks,
// and this function returns the sample that existed at that index and a new node
// containing all samples/children after it.
func (n *node[T]) split(i int) (T, *node[T]) {
	sample := n.samples[i]
	next := n.cow.newNode()
	next.samples = append(next.samples, n.samples[i+1:]...)
	n.samples.truncate(i)
	if 0 < len(n.children) {
		next.children = append(next.children, n.children[i+1:]...)
		n.children.truncate(i + 1)
	}
	return sample, next
}

// maybeSplitChild checks if a child should be split, and if so splits it.
// Returns whether or not a split occurred.
func (n *node[T]) maybeSplitChild(i, maxSamples int) bool {
	if len(n.children[i].samples) < maxSamples {
		return false
	}
	first := n.mutableChild(i)
	sample, second := first.split(maxSamples / 2)
	n.samples.insertAt(i, sample)
	n.children.insertAt(i+1, second)
	return true
}

// insert inserts a sample into the subtree rooted at this node, making sure
// no nodes in the subtree exceed maxSamples samples.  Should an equivalent
// sample be found/replaced by insert, it will be returned.
func (n *node[T]) insert(sample T, maxSamples int) (_ T, _ bool) {
	i, found := n.samples.find(sample)
	if found {
		out := n.samples[i]
		n.samples[i] = sample
		return out, true
	}
	if len(n.children) == 0 {
		n.samples.insertAt(i, sample)
		return
	}
	if n.maybeSplitChild(i, maxSamples) {
		inTree := n.samples[i]
		switch {
		case sample.Less(inTree):
			// no change, we want first split node
		case inTree.Less(sample):
			i++ // we want second split node
		default:
			out := n.samples[i]
			n.samples[i] = sample
			return out, true
		}
	}
	return n.mutableChild(i).insert(sample, maxSamples)
}

// get finds the given key in the subtree and returns it.
func (n *node[T]) get(key T) (_ T, _ bool) {
	i, found := n.samples.find(key)
	if found {
		return n.samples[i], true
	} else if 0 < len(n.children) {
		return n.children[i].get(key)
	}
	return
}

// min returns the first sample in the subtree.
func min[T data.Sample](n *node[T]) (_ T, found bool) {
	if n == nil {
		return
	}
	for 0 < len(n.children) {
		n = n.children[0]
	}
	if len(n.samples) == 0 {
		return
	}
	return n.samples[0], true
}

// max returns the last sample in the subtree.
func max[T data.Sample](n *node[T]) (_ T, found bool) {
	if n == nil {
		return
	}
	for 0 < len(n.children) {
		n = n.children[len(n.children)-1]
	}
	if len(n.samples) == 0 {
		return
	}
	return n.samples[len(n.samples)-1], true
}

// toRemove details what sample to remove in a node.remove call.
type toRemove int

const (
	removeSample toRemove = iota // removes the given sample
	removeMin                    // removes smallest sample in the subtree
	removeMax                    // removes largest sample in the subtree
)

// remove removes a sample from the subtree rooted at this node.
func (n *node[T]) remove(sample T, minSamples int, typ toRemove) (_ T, _ bool) {
	var (
		i     int
		found bool
	)
	switch typ {
	case removeMax:
		if len(n.children) == 0 {
			return n.samples.pop(), true
		}
		i = len(n.samples)
	case removeMin:
		if len(n.children) == 0 {
			return n.samples.removeAt(0), true
		}
		i = 0
	case removeSample:
		i, found = n.samples.find(sample)
		if len(n.children) == 0 {
			if found {
				return n.samples.removeAt(i), true
			}
			return
		}
	default:
		panic("invalid type")
	}
	// If we get to here, we have children.
	if len(n.children[i].samples) <= minSamples {
		return n.growChildAndRemove(i, sample, minSamples, typ)
	}
	child := n.mutableChild(i)
	// Either we had enough samples to begin with, or we've done some
	// merging/stealing, because we've got enough now and we're ready to return
	// stuff.
	if found {
		// The sample exists at index 'i', and the child we've selected can give us a
		// predecessor, since if we've gotten here it's got minSamples < samples in it.
		out := n.samples[i]
		// We use our special-case 'remove' call with typ=removeMax to pull the
		// predecessor of sample i (the rightmost leaf of our immediate left child)
		// and set it into where we pulled the sample from.
		var zero T
		n.samples[i], _ = child.remove(zero, minSamples, removeMax)
		return out, true
	}
	// Final recursive call.  Once we're here, we know that the sample isn't in this
	// node and that the child is big enough to remove from.
	return child.remove(sample, minSamples, typ)
}

// growChildAndRemove grows child 'i' to make sure it's possible to remove a
// sample from it while keeping it at minSamples, then calls remove to actually
// remove it.
//
// Most documentation says we have to do two sets of special casing:
//  1. sample is in this node
//  2. sample is in child
//
// In both cases, we need to handle the two subcases:
//
//	A) node has enough values that it can spare one
//	B) node doesn't have enough values
//
// For the latter, we have to check:
//
//	a) left sibling has node to spare
//	b) right sibling has node to spare
//	c) we must merge
//
// To simplify our code here, we handle cases #1 and #2 the same:
// If a node doesn't have enough samples, we make sure it does (using a,b,c).
// We then simply redo our remove call, and the second time (regardless of
// whether we're in case 1 or 2), we'll have enough samples and can guarantee
// that we hit case A.
func (n *node[T]) growChildAndRemove(i int, sample T, minSamples int, typ toRemove) (T, bool) {
	if 0 < i && minSamples < len(n.children[i-1].samples) {
		// Steal from left child
		child := n.mutableChild(i)
		stealFrom := n.mutableChild(i - 1)
		stolenSample := stealFrom.samples.pop()
		child.samples.insertAt(0, n.samples[i-1])
		n.samples[i-1] = stolenSample
		if 0 < len(stealFrom.children) {
			child.children.insertAt(0, stealFrom.children.pop())
		}
	} else if i < len(n.samples) && minSamples < len(n.children[i+1].samples) {
		// steal from right child
		child := n.mutableChild(i)
		stealFrom := n.mutableChild(i + 1)
		stolenSample := stealFrom.samples.removeAt(0)
		child.samples = append(child.samples, n.samples[i])
		n.samples[i] = stolenSample
		if 0 < len(stealFrom.children) {
			child.children = append(child.children, stealFrom.children.removeAt(0))
		}
	} else {
		if len(n.samples) <= i {
			i--
		}
		child := n.mutableChild(i)
		// merge with right child
		mergeSample := n.samples.removeAt(i)
		mergeChild := n.children.removeAt(i + 1)
		child.samples = append(child.samples, mergeSample)
		child.samples = append(child.samples, mergeChild.samples...)
		child.children = append(child.children, mergeChild.children...)
		n.cow.freeNode(mergeChild)
	}
	return n.remove(sample, minSamples, typ)
}

type direction int

const (
	descend = direction(-1)
	ascend  = direction(+1)
)

type optionalSample[T data.Sample] struct {
	sample T
	valid  bool
}

func optional[T data.Sample](sample T) optionalSample[T] {
	return optionalSample[T]{sample: sample, valid: true}
}

func empty[T data.Sample]() optionalSample[T] {
	return optionalSample[T]{}
}

// iterate provides a simple method for iterating over elements in the tree.
//
// When ascending, the 'start' should be less than 'stop' and when descending,
// the 'start' should be greater than 'stop'. Setting 'includeStart' to true
// will force the iterator to include the first sample when it equals 'start',
// thus creating a "greaterOrEqual" or "lessThanEqual" rather than just a
// "greaterThan" or "lessThan" queries.
func (n *node[T]) iterate(dir direction, start, stop optionalSample[T], includeStart bool, hit bool, iter SampleIterator[T]) (bool, bool) {
	var (
		ok, found bool
		index     int
	)
	switch dir {
	case ascend:
		if start.valid {
			index, _ = n.samples.find(start.sample)
		}
		for i := index; i < len(n.samples); i++ {
			if 0 < len(n.children) {
				if hit, ok = n.children[i].iterate(dir, start, stop, includeStart, hit, iter); !ok {
					return hit, false
				}
			}
			if !includeStart && !hit && start.valid && !start.sample.Less(n.samples[i]) {
				hit = true
				continue
			}
			hit = true
			if stop.valid && !n.samples[i].Less(stop.sample) {
				return hit, false
			}
			if !iter(n.samples[i]) {
				return hit, false
			}
		}
		if 0 < len(n.children) {
			if hit, ok = n.children[len(n.children)-1].iterate(dir, start, stop, includeStart, hit, iter); !ok {
				return hit, false
			}
		}
	case descend:
		if start.valid {
			index, found = n.samples.find(start.sample)
			if !found {
				index--
			}
		} else {
			index = len(n.samples) - 1
		}
		for i := index; 0 <= i; i-- {
			if start.valid && !n.samples[i].Less(start.sample) {
				if !includeStart || hit || start.sample.Less(n.samples[i]) {
					continue
				}
			}
			if 0 < len(n.children) {
				if hit, ok = n.children[i+1].iterate(dir, start, stop, includeStart, hit, iter); !ok {
					return hit, false
				}
			}
			if stop.valid && !stop.sample.Less(n.samples[i]) {
				return hit, false //	continue
			}
			hit = true
			if !iter(n.samples[i]) {
				return hit, false
			}
		}
		if 0 < len(n.children) {
			if hit, ok = n.children[0].iterate(dir, start, stop, includeStart, hit, iter); !ok {
				return hit, false
			}
		}
	}
	return hit, true
}

// BTree is a generic implementation of a B-tree.
//
// BTree stores Sample instances in an ordered structure, allowing easy
// insertion, removal, and iteration.
//
// Write operations are not safe for concurrent mutation by multiple
// goroutines, but Read operations are.
type BTree[T data.Sample] struct {
	degree int
	length int
	root   *node[T]
	cow    *copyOnWriteContext[T]
}

// copyOnWriteContext pointers determine node ownership... a tree with a write
// context equivalent to a node's write context is allowed to modify that node.
// A tree whose write context does not match a node's is not allowed to modify
// it, and must create a new, writable copy (IE: it's a Clone).
//
// When doing any write operation, we maintain the invariant that the current
// node's context is equal to the context of the tree that requested the write.
// We do this by, before we descend into any node, creating a copy with the
// correct context if the contexts don't match.
//
// Since the node we're currently visiting on any write has the requesting
// tree's context, that node is modifiable in place.  Children of that node may
// not share context, but before we descend into them, we'll make a mutable
// copy.
type copyOnWriteContext[T data.Sample] struct {
	freelist *FreeList[T]
}

// Clone clones the tree, lazily.  Clone should not be called concurrently,
// but the original tree (t) and the new tree (t2) can be used concurrently
// once the Clone call completes.
//
// The internal tree structure of b is marked read-only and shared between t and
// t2.  Writes to both t and t2 use copy-on-write logic, creating new nodes
// whenever one of b's original nodes would have been modified.  Read operations
// should have no performance degredation.  Write operations for both t and t2
// will initially experience minor slow-downs caused by additional allocs and
// copies due to the aforementioned copy-on-write logic, but should converge to
// the original performance characteristics of the original tree.
func (t *BTree[T]) Clone() (t2 *BTree[T]) {
	// Create two entirely new copy-on-write contexts.
	// This operation effectively creates three trees:
	//   the original, shared nodes (old b.cow)
	//   the new b.cow nodes
	//   the new out.cow nodes
	cow1, cow2 := *t.cow, *t.cow
	out := *t
	t.cow = &cow1
	out.cow = &cow2
	return &out
}

// maxSamples returns the max number of samples to allow per node.
func (t *BTree[T]) maxSamples() int {
	return t.degree*2 - 1
}

// minSamples returns the min number of samples to allow per node
// (ignored for the root node).
func (t *BTree[T]) minSamples() int {
	return t.degree - 1
}

func (c *copyOnWriteContext[T]) newNode() (n *node[T]) {
	n = c.freelist.newNode()
	n.cow = c
	return
}

type freeType int

const (
	ftFreelistFull freeType = iota // node was freed (available for GC, not stored in freelist)
	ftStored                       // node was stored in the freelist for later use
	ftNotOwned                     // node was ignored by COW, since it's owned by another one
)

// freeNode frees a node within a given COW context, if it's owned by that
// context.  It returns what happened to the node (see freeType const
// documentation).
func (c *copyOnWriteContext[T]) freeNode(n *node[T]) freeType {
	if n.cow == c {
		// clear to allow GC
		n.samples.truncate(0)
		n.children.truncate(0)
		n.cow = nil
		if c.freelist.freeNode(n) {
			return ftStored
		} else {
			return ftFreelistFull
		}
	} else {
		return ftNotOwned
	}
}

// ReplaceOrInsert adds the given sample to the tree.  If a sample in the tree
// already equals the given one, it is removed from the tree and returned,
// and the second return value is true.  Otherwise, (zeroValue, false).
func (t *BTree[T]) ReplaceOrInsert(sample T) (_ T, _ bool) {
	if t.root == nil {
		t.root = t.cow.newNode()
		t.root.samples = append(t.root.samples, sample)
		t.length++
		return
	} else {
		t.root = t.root.mutableFor(t.cow)
		if t.maxSamples() <= len(t.root.samples) {
			sample2, second := t.root.split(t.maxSamples() / 2)
			oldroot := t.root
			t.root = t.cow.newNode()
			t.root.samples = append(t.root.samples, sample2)
			t.root.children = append(t.root.children, oldroot, second)
		}
	}
	out, outb := t.root.insert(sample, t.maxSamples())
	if !outb {
		t.length++
	}
	return out, outb
}

// Delete removes a sample equal to the passed in sample from the tree,
// returning it.  If no such sample exists, returns (zeroValue, false).
func (t *BTree[T]) Delete(sample T) (T, bool) {
	return t.delete(sample, removeSample)
}

// DeleteMin removes the smallest sample in the tree and returns it.
// If no such sample exists, returns (zeroValue, false).
func (t *BTree[T]) DeleteMin() (T, bool) {
	var zero T
	return t.delete(zero, removeMin)
}

// DeleteMax removes the largest sample in the tree and returns it.
// If no such sample exists, returns (zeroValue, false).
func (t *BTree[T]) DeleteMax() (T, bool) {
	var zero T
	return t.delete(zero, removeMax)
}

func (t *BTree[T]) delete(sample T, typ toRemove) (_ T, _ bool) {
	if t.root == nil || len(t.root.samples) == 0 {
		return
	}
	t.root = t.root.mutableFor(t.cow)
	out, outb := t.root.remove(sample, t.minSamples(), typ)
	if len(t.root.samples) == 0 && 0 < len(t.root.children) {
		oldroot := t.root
		t.root = t.root.children[0]
		t.cow.freeNode(oldroot)
	}
	if outb {
		t.length--
	}
	return out, outb
}

// AscendRange calls the iterator for every value in the tree within the range
// [greaterOrEqual, lessThan), until iterator returns false.
func (t *BTree[T]) AscendRange(greaterOrEqual, lessThan T, iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(ascend, optional(greaterOrEqual), optional(lessThan), true, false, iterator)
}

// AscendLessThan calls the iterator for every value in the tree within the range
// [first, pivot), until iterator returns false.
func (t *BTree[T]) AscendLessThan(pivot T, iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(ascend, empty[T](), optional(pivot), false, false, iterator)
}

// AscendGreaterOrEqual calls the iterator for every value in the tree within
// the range [pivot, last], until iterator returns false.
func (t *BTree[T]) AscendGreaterOrEqual(pivot T, iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(ascend, optional(pivot), empty[T](), true, false, iterator)
}

// Ascend calls the iterator for every value in the tree within the range
// [first, last], until iterator returns false.
func (t *BTree[T]) Ascend(iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(ascend, empty[T](), empty[T](), false, false, iterator)
}

// DescendRange calls the iterator for every value in the tree within the range
// [lessOrEqual, greaterThan), until iterator returns false.
func (t *BTree[T]) DescendRange(lessOrEqual, greaterThan T, iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(descend, optional(lessOrEqual), optional(greaterThan), true, false, iterator)
}

// DescendLessOrEqual calls the iterator for every value in the tree within the range
// [pivot, first], until iterator returns false.
func (t *BTree[T]) DescendLessOrEqual(pivot T, iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(descend, optional(pivot), empty[T](), true, false, iterator)
}

// DescendGreaterThan calls the iterator for every value in the tree within
// the range [last, pivot), until iterator returns false.
func (t *BTree[T]) DescendGreaterThan(pivot T, iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(descend, empty[T](), optional(pivot), false, false, iterator)
}

// Descend calls the iterator for every value in the tree within the range
// [last, first], until iterator returns false.
func (t *BTree[T]) Descend(iterator SampleIterator[T]) {
	if t.root == nil {
		return
	}
	t.root.iterate(descend, empty[T](), empty[T](), false, false, iterator)
}

// Get looks for the key sample in the tree, returning it.  It returns
// (zeroValue, false) if unable to find that sample.
func (t *BTree[T]) Get(key T) (_ T, _ bool) {
	if t.root == nil {
		return
	}
	return t.root.get(key)
}

// Min returns the smallest sample in the tree, or (zeroValue, false) if the tree is empty.
func (t *BTree[T]) Min() (T, bool) {
	return min(t.root)
}

// Max returns the largest sample in the tree, or (zeroValue, false) if the tree is empty.
func (t *BTree[T]) Max() (T, bool) {
	return max(t.root)
}

// Has returns true if the given key is in the tree.
func (t *BTree[T]) Has(key T) bool {
	_, ok := t.Get(key)
	return ok
}

// Len returns the number of samples currently in the tree.
func (t *BTree[T]) Len() int {
	return t.length
}

// Clear removes all samples from the tree.  If addNodesToFreelist is true,
// t's nodes are added to its freelist as part of this call, until the freelist
// is full.  Otherwise, the root node is simply dereferenced and the subtree
// left to Go's normal GC processes.
//
// This can be much faster
// than calling Delete on all elements, because that requires finding/removing
// each element in the tree and updating the tree accordingly.  It also is
// somewhat faster than creating a new tree to replace the old one, because
// nodes from the old tree are reclaimed into the freelist for use by the new
// one, instead of being lost to the garbage collector.
//
// This call takes:
//
//	O(1): when addNodesToFreelist is false, this is a single operation.
//	O(1): when the freelist is already full, it breaks out immediately
//	O(freelist size):  when the freelist is empty and the nodes are all owned
//	    by this tree, nodes are added to the freelist until full.
//	O(tree size):  when all nodes are owned by another tree, all nodes are
//	    iterated over looking for nodes to add to the freelist, and due to
//	    ownership, none are.
func (t *BTree[T]) Clear(addNodesToFreelist bool) {
	if t.root != nil && addNodesToFreelist {
		t.root.reset(t.cow)
	}
	t.root, t.length = nil, 0
}

// reset returns a subtree to the freelist.  It breaks out immediately if the
// freelist is full, since the only benefit of iterating is to fill that
// freelist up.  Returns true if parent reset call should continue.
func (n *node[T]) reset(c *copyOnWriteContext[T]) bool {
	for _, child := range n.children {
		if !child.reset(c) {
			return false
		}
	}
	return c.freeNode(n) != ftFreelistFull
}
