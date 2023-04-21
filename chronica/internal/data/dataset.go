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

import "github.com/9rum/chronica/internal/btree"

// Dataset represents the given dataset.
type Dataset interface {
	// BatchSize provides a primitive for the number of data samples to select
	// at each step.
	BatchSize() int

	// Len provides a primitive for the number of times to schedule.
	Len() int

	// GetItem provides a mechanism to retrieve a data sample with the given
	// arguments.  This must provide an index identifying the scheduled
	// data sample.
	GetItem(rank, size int) (index int)

	// OnBatchEnd provides a mechanism to be called at the end of every step.
	OnBatchEnd()

	// OnEpochEnd provides a mechanism to be called at the end of every epoch.
	OnEpochEnd()

	// OnTrainEnd provides a mechanism to be called at the end of training.
	OnTrainEnd()
}

// ShardedDataset represents a sharded dataset where every node in the cluster
// has a replica of the given dataset; hence it ignores rank when looking for
// the data sample.
type ShardedDataset[T btree.Item] struct {
	length     int
	batchSize  int
	shuffle    bool
	dropLast   bool
	items      *btree.BTree[T]
	recycleBin *btree.BTree[T]
}

// NewShardedDataset creates a new sharded dataset with the given arguments.
func NewShardedDataset[T btree.Item](sizes []int, batchSize int, shuffle, dropLast bool) *ShardedDataset[T] {
	dataset := &ShardedDataset[T]{
		batchSize: batchSize,
		shuffle:   shuffle,
		dropLast:  dropLast,
	}

	if len(sizes)%batchSize == 0 || dropLast {
		dataset.length = len(sizes) / batchSize
	} else {
		dataset.length = len(sizes)/batchSize + 1
	}

	// We use the default degree for the nodes to fit on a single page.
	dataset.items = btree.New[T](0)
	dataset.recycleBin = btree.New[T](0)

	for index, size := range sizes {
		if _, found := dataset.items.ReplaceOrInsert(btree.NewItem[T](index, size)); found {
			panic("insert found item")
		}
	}

	return dataset
}

// BatchSize returns the batch size.
func (d ShardedDataset[T]) BatchSize() int {
	return d.batchSize
}

// Len returns the number of remaining steps currently in the training epoch.
func (d ShardedDataset[T]) Len() int {
	return d.length
}

// GetItem looks for the data sample with the size nearest to the given size.
func (d *ShardedDataset[T]) GetItem(rank, size int) (index int) {
	if item, ok := d.items.DeleteNearest(btree.NewItem[T](size, size)); !ok {
		panic("didn't find item")
	} else {
		index = item.Index()
		if _, found := d.recycleBin.ReplaceOrInsert(item); found {
			panic("insert found item")
		}
	}
	return
}

// PartitionedDataset represents a partitioned dataset where each of the nodes
// in the cluster holds only a portion of the given dataset.
type PartitionedDataset[T btree.Item] struct {
	length      int
	batchSize   int
	shuffle     bool
	dropLast    bool
	partitions  []*btree.BTree[T]
	recycleBins []*btree.BTree[T]
}
