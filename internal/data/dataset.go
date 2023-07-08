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
	"errors"
	"math/rand"

	"github.com/9rum/chronica/internal/btree"
)

// Dataset represents the given dataset.
// All implementations must embed DatasetBase for forward compatibility.
type Dataset interface {
	// Getitem retrieves a data sample with the given arguments.  This must provide
	// an index identifying the scheduled data sample and its size.
	Getitem(rank, size int) (index, siz int)

	// Len returns the number of data samples currently in the dataset.
	Len(rank int) int

	// Rand retrieves an arbitrary data sample from the dataset.
	Rand(rank int) (index, size int)

	// OnBatchEnd is called at the end of a training batch.
	OnBatchEnd(rank int)

	// OnEpochEnd is called at the end of an epoch during training.
	OnEpochEnd(epoch int64)

	// OnTrainEnd terminates the training environment.
	OnTrainEnd()
}

// DatasetBase must be embedded to have forward compatible implementations.
type DatasetBase struct {
}

func (DatasetBase) Getitem(rank, size int) (index, siz int) {
	return
}
func (DatasetBase) Len(rank int) int {
	return 0
}
func (DatasetBase) Rand(rank int) (index, size int) {
	return
}
func (DatasetBase) OnBatchEnd(rank int)    {}
func (DatasetBase) OnEpochEnd(epoch int64) {}
func (DatasetBase) OnTrainEnd()            {}

// ShardedDataset represents a sharded dataset where every node in the cluster
// has a replica of the given dataset; hence it ignores rank when looking for
// the data sample.
type ShardedDataset[T btree.Item] struct {
	DatasetBase
	items      *btree.BTree[T]
	recycleBin *btree.BTree[T]
}

// NewShardedDataset creates a new sharded dataset with the given argument.
func NewShardedDataset[T btree.Item](sizes []int) (Dataset, error) {
	// We use the default degree for the nodes to fit on a single memory page.
	dataset := &ShardedDataset[T]{
		items:      btree.New[T](0),
		recycleBin: btree.New[T](0),
	}

	for index, size := range sizes {
		if _, found := dataset.items.ReplaceOrInsert(btree.NewItem[T](index, size)); found {
			dataset.OnTrainEnd()
			return nil, errors.New("insert found item")
		}
	}

	return dataset, nil
}

// Getitem looks for the data sample with the size nearest to the given size.
func (d ShardedDataset[T]) Getitem(rank, size int) (index, siz int) {
	item, ok := d.items.DeleteNearest(btree.NewItem[T](size, size))
	if !ok {
		return
	}
	index, siz = item.Index(), item.Size()
	d.recycleBin.ReplaceOrInsert(item)
	return
}

// Len returns the number of data samples currently in the dataset.
func (d ShardedDataset[T]) Len(rank int) int {
	return d.items.Len()
}

// Rand selects a random data sample from the dataset.
func (d ShardedDataset[T]) Rand(rank int) (index, size int) {
	item, ok := d.items.Min()
	if !ok {
		return
	}
	min := item.Size()

	item, ok = d.items.Max()
	if !ok {
		return
	}
	max := item.Size()

	pivot := rand.Intn(max-min+1) + min
	item, ok = d.items.DeleteNearest(btree.NewItem[T](pivot, pivot))
	if !ok {
		return
	}
	index, size = item.Index(), item.Size()

	d.recycleBin.ReplaceOrInsert(item)

	return
}

// OnEpochEnd resets the data samples.
func (d *ShardedDataset[T]) OnEpochEnd(epoch int64) {
	rand.Seed(epoch)

	for item, ok := d.items.DeleteMin(); ok; item, ok = d.items.DeleteMin() {
		d.recycleBin.ReplaceOrInsert(item)
	}
	d.items, d.recycleBin = d.recycleBin, d.items
}

// OnTrainEnd terminates the training environment.
func (d *ShardedDataset[T]) OnTrainEnd() {
	d.items.Clear(false)
	d.items = nil
	d.recycleBin.Clear(false)
	d.recycleBin = nil
}

// PartitionedDataset represents a partitioned dataset where each of the nodes
// in the cluster holds only a portion of the given dataset.
type PartitionedDataset[T btree.Item] struct {
	DatasetBase
	groups      []int
	partitions  []*btree.BTree[T]
	recycleBins []*btree.BTree[T]
}

// NewPartitionedDataset creates a new partitioned dataset with the given arguments.
func NewPartitionedDataset[T btree.Item](groups []int, partitions [][]int) (Dataset, error) {
	dataset := &PartitionedDataset[T]{
		groups:      groups,
		partitions:  make([]*btree.BTree[T], 0, len(partitions)),
		recycleBins: make([]*btree.BTree[T], 0, len(partitions)),
	}

	// We assume that the indices are sequentially distributed across workers.
	base := 0

	for rank, partition := range partitions {
		// We use the default degree for the nodes to fit on a single memory page.
		dataset.partitions = append(dataset.partitions, btree.New[T](0))
		dataset.recycleBins = append(dataset.recycleBins, btree.New[T](0))
		for index, size := range partition {
			if _, found := dataset.partitions[rank].ReplaceOrInsert(btree.NewItem[T](base+index, size)); found {
				dataset.OnTrainEnd()
				return nil, errors.New("insert found item")
			}
		}
		base += len(partition)
	}

	return dataset, nil
}

// Getitem looks for the data sample with the size nearest to the given size
// in the partition with the given rank.
func (d PartitionedDataset[T]) Getitem(rank, size int) (index, siz int) {
	item, ok := d.partitions[d.groups[rank]].DeleteNearest(btree.NewItem[T](size, size))
	if !ok {
		return
	}
	index, siz = item.Index(), item.Size()
	d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)
	return
}

// Len returns the number of data samples currently in the dataset.
func (d PartitionedDataset[T]) Len(rank int) int {
	return d.partitions[d.groups[rank]].Len()
}

// Rand selects a random data sample from the dataset.
func (d PartitionedDataset[T]) Rand(rank int) (index, size int) {
	item, ok := d.partitions[d.groups[rank]].Min()
	if !ok {
		return
	}
	min := item.Size()

	item, ok = d.partitions[d.groups[rank]].Max()
	if !ok {
		return
	}
	max := item.Size()

	pivot := rand.Intn(max-min+1) + min
	item, ok = d.partitions[d.groups[rank]].DeleteNearest(btree.NewItem[T](pivot, pivot))
	if !ok {
		return
	}
	index, size = item.Index(), item.Size()

	d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)

	return
}

// OnEpochEnd resets the data partitions.
func (d *PartitionedDataset[T]) OnEpochEnd(epoch int64) {
	rand.Seed(epoch)

	for rank, partition := range d.partitions {
		for item, ok := partition.DeleteMin(); ok; item, ok = partition.DeleteMin() {
			d.recycleBins[rank].ReplaceOrInsert(item)
		}
	}
	d.partitions, d.recycleBins = d.recycleBins, d.partitions
}

// OnTrainEnd terminates the training environment.
func (d *PartitionedDataset[T]) OnTrainEnd() {
	for rank, partition := range d.partitions {
		partition.Clear(false)
		d.partitions[rank] = nil
		d.recycleBins[rank].Clear(false)
		d.recycleBins[rank] = nil
	}
	d.partitions = nil
	d.recycleBins = nil
}
