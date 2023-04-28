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
// the given dataset.  In addition to the traditional sharded dataset,
// it supports a partitioned dataset where the data is split into
// multiple data partitions across nodes in the cluster.
package data

import (
	"math/rand"
	"time"

	"github.com/9rum/chronica/internal/btree"
)

// Dataset represents the given dataset.
type Dataset interface {
	// Getitem provides a mechanism to retrieve a data sample with the given
	// arguments.  This must provide an index identifying the scheduled
	// data sample and its size.
	Getitem(rank, size int) (index, siz int)

	// Len provides a primitive for the number of data samples currently
	// in the dataset.
	Len(rank int) int

	// Rand provides a primitive for selecting an arbitrary data sample
	// from the dataset.
	Rand(rank int) (index, size int)

	// OnEpochEnd provides a mechanism to be called at the end of every epoch.
	OnEpochEnd()

	// OnTrainEnd provides a mechanism to terminate the training environment.
	OnTrainEnd()
}

// ShardedDataset represents a sharded dataset where every node in the cluster
// has a replica of the given dataset; hence it ignores rank when looking for
// the data sample.
type ShardedDataset[T btree.Item] struct {
	items      *btree.BTree[T]
	recycleBin *btree.BTree[T]
}

// NewShardedDataset creates a new sharded dataset with the given arguments.
func NewShardedDataset[T btree.Item](sizes []int) *ShardedDataset[T] {
	// We use the default degree for the nodes to fit on a single memory page.
	dataset := &ShardedDataset[T]{
		items:      btree.New[T](0),
		recycleBin: btree.New[T](0),
	}

	for index, size := range sizes {
		if _, found := dataset.items.ReplaceOrInsert(btree.NewItem[T](index, size)); found {
			panic("insert found item")
		}
	}

	return dataset
}

// Getitem looks for the data sample with the size nearest to the given size.
func (d *ShardedDataset[T]) Getitem(rank, size int) (index, siz int) {
	if item, ok := d.items.DeleteNearest(btree.NewItem[T](size, size)); !ok {
		panic("didn't find item")
	} else {
		index, siz = item.Index(), item.Size()
		if _, found := d.recycleBin.ReplaceOrInsert(item); found {
			panic("insert found item")
		}
	}
	return
}

// Len returns the number of data samples currently in the dataset.
func (d ShardedDataset[T]) Len(rank int) int {
	return d.items.Len()
}

// Rand selects a random data sample from the dataset.
func (d *ShardedDataset[T]) Rand(rank int) (index, size int) {
	rand.Seed(time.Now().Unix())

	item, ok := d.items.Min()
	if !ok {
		panic("didn't find item")
	}
	min := item.Size()

	item, ok = d.items.Max()
	if !ok {
		panic("didn't find item")
	}
	max := item.Size()

	pivot := rand.Intn(max-min+1) + min
	item, ok = d.items.DeleteNearest(btree.NewItem[T](pivot, pivot))
	if !ok {
		panic("didn't find item")
	}
	index, size = item.Index(), item.Size()

	if _, found := d.recycleBin.ReplaceOrInsert(item); found {
		panic("insert found item")
	}

	return
}

// OnEpochEnd resets the data samples.
func (d *ShardedDataset[T]) OnEpochEnd() {
	for item, ok := d.items.DeleteMin(); ok; item, ok = d.items.DeleteMin() {
		if _, found := d.recycleBin.ReplaceOrInsert(item); found {
			panic("insert found item")
		}
	}
	d.items, d.recycleBin = d.recycleBin, d.items
}

// OnTrainEnd terminates the training environment.
func (d *ShardedDataset[T]) OnTrainEnd() {
	d.items.Clear(false)
	d.recycleBin.Clear(false)
}

// PartitionedDataset represents a partitioned dataset where each of the nodes
// in the cluster holds only a portion of the given dataset.
type PartitionedDataset[T btree.Item] struct {
	partitions  []*btree.BTree[T]
	recycleBins []*btree.BTree[T]
}

// Getitem looks for the data sample with the size nearest to the given size
// in the partition with the given rank.
func (d *PartitionedDataset[T]) Getitem(rank, size int) (index, siz int) {
	if item, ok := d.partitions[rank].DeleteNearest(btree.NewItem[T](size, size)); !ok {
		panic("didn't find item")
	} else {
		index, siz = item.Index(), item.Size()
		if _, found := d.recycleBins[rank].ReplaceOrInsert(item); found {
			panic("insert found item")
		}
	}
	return
}

// Len returns the number of data samples currently in the dataset.
func (d PartitionedDataset[T]) Len(rank int) int {
	return d.partitions[rank].Len()
}

// Rand selects a random data sample from the dataset.
func (d *PartitionedDataset[T]) Rand(rank int) (index, size int) {
	rand.Seed(time.Now().Unix())

	item, ok := d.partitions[rank].Min()
	if !ok {
		panic("didn't find item")
	}
	min := item.Size()

	item, ok = d.partitions[rank].Max()
	if !ok {
		panic("didn't find item")
	}
	max := item.Size()

	pivot := rand.Intn(max-min+1) + min
	item, ok = d.partitions[rank].DeleteNearest(btree.NewItem[T](pivot, pivot))
	if !ok {
		panic("didn't find item")
	}
	index, size = item.Index(), item.Size()

	if _, found := d.recycleBins[rank].ReplaceOrInsert(item); found {
		panic("insert found item")
	}

	return
}

// OnTrainEnd terminates the training environment.
func (d *PartitionedDataset[T]) OnTrainEnd() {
	for rank, partition := range d.partitions {
		partition.Clear(false)
		d.recycleBins[rank].Clear(false)
	}
}
