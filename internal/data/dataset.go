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

// Package data provides primitives for representing and organizing the given
// data sets.  In addition to the traditional sharded data set, it supports a
// partitioned data set where the data is split into multiple data partitions
// across nodes in the cluster.
package data

import (
	"errors"
	"math/rand"

	"github.com/9rum/chronica/internal/btree"
)

var mapping []int

// Dataset represents the given data set.
// All implementations must embed DatasetBase for forward compatibility.
type Dataset interface {
	// Getitem retrieves a data sample with the given arguments.  This must provide
	// an index identifying the scheduled data sample and its size.
	Getitem(rank, size int) (_, _ int)

	// Rand retrieves an arbitrary data sample from the data set.
	Rand(rank int) (_, _ int)

	// Len returns the number of data samples currently in the data set.
	Len(rank int) int

	// OnEpochEnd is called at the end of an epoch during training.
	OnEpochEnd(epoch int64)

	// OnTrainEnd terminates the training environment.
	OnTrainEnd()
}

// DatasetBase must be embedded to have forward compatible implementations.
type DatasetBase struct {
}

func (DatasetBase) Getitem(rank, size int) (_, _ int) {
	return
}
func (DatasetBase) Rand(rank int) (_, _ int) {
	return
}
func (DatasetBase) Len(rank int) (_ int) {
	return
}
func (DatasetBase) OnEpochEnd(epoch int64) {}
func (DatasetBase) OnTrainEnd()            {}

// ShardedDataset represents a sharded data set where every node in the cluster
// has a replica of the given data set; hence it ignores rank when looking for
// the data sample.
type ShardedDataset struct {
	DatasetBase
	items      *btree.BTree[Sample]
	recycleBin *btree.BTree[Sample]
}

// NewShardedDataset creates a new sharded data set with the given argument.
func NewShardedDataset(sizes []int) (*ShardedDataset, error) {
	// We use the default degree for the items to fit on a single memory page.
	dataset := &ShardedDataset{
		items:      btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()),
		recycleBin: btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()),
	}
	mapping = rand.Perm(len(sizes))

	for index, size := range sizes {
		if _, found := dataset.recycleBin.ReplaceOrInsert(NewSample(index, size)); found {
			dataset.OnTrainEnd()
			return nil, errors.New("insert found item")
		}
	}

	return dataset, nil
}

// Getitem looks for the data sample with the size nearest to the given size.
func (d *ShardedDataset) Getitem(rank, size int) (_, _ int) {
	item, ok := d.items.DeleteNearest(NewSample(0, size))
	if !ok {
		return
	}
	defer d.recycleBin.ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// Rand selects a random data sample from the data set.
func (d *ShardedDataset) Rand(rank int) (_, _ int) {
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
	item, ok = d.items.DeleteNearest(NewSample(0, pivot))
	if !ok {
		return
	}
	defer d.recycleBin.ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// Len returns the number of data samples currently in the data set.
func (d *ShardedDataset) Len(rank int) int {
	return d.items.Len()
}

// OnEpochEnd restores the data samples.
func (d *ShardedDataset) OnEpochEnd(epoch int64) {
	rand.Seed(epoch)
	mapping = rand.Perm(len(mapping))

	for item, ok := d.items.DeleteMin(); ok; item, ok = d.items.DeleteMin() {
		d.recycleBin.ReplaceOrInsert(item)
	}
	d.items, d.recycleBin = d.recycleBin, d.items
}

// OnTrainEnd terminates the training environment.
func (d *ShardedDataset) OnTrainEnd() {
	d.items.Clear(false)
	d.items = nil
	d.recycleBin.Clear(false)
	d.recycleBin = nil
}

// PartitionedDataset represents a partitioned data set where each of the nodes
// in the cluster holds only a portion of the given data set.
type PartitionedDataset struct {
	DatasetBase
	groups      []int
	partitions  []*btree.BTree[Sample]
	recycleBins []*btree.BTree[Sample]
}

// NewPartitionedDataset creates a new partitioned data set with the given arguments.
func NewPartitionedDataset(groups []int, partitions [][]int) (*PartitionedDataset, error) {
	dataset := &PartitionedDataset{
		groups:      groups,
		partitions:  make([]*btree.BTree[Sample], 0, len(partitions)),
		recycleBins: make([]*btree.BTree[Sample], 0, len(partitions)),
	}
	mapping = rand.Perm(func() (sum int) {
		for _, partition := range partitions {
			sum += len(partition)
		}
		return
	}())

	// We assume that the indices are sequentially distributed across workers.
	base := 0

	for rank, partition := range partitions {
		// We use the default degree for the items to fit on a single memory page.
		dataset.partitions = append(dataset.partitions, btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()))
		dataset.recycleBins = append(dataset.recycleBins, btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()))
		for index, size := range partition {
			if _, found := dataset.recycleBins[rank].ReplaceOrInsert(NewSample(base+index, size)); found {
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
func (d *PartitionedDataset) Getitem(rank, size int) (_, _ int) {
	item, ok := d.partitions[d.groups[rank]].DeleteNearest(NewSample(0, size))
	if !ok {
		return
	}
	defer d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// Rand selects a random data sample from the data set.
func (d *PartitionedDataset) Rand(rank int) (_, _ int) {
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
	item, ok = d.partitions[d.groups[rank]].DeleteNearest(NewSample(0, pivot))
	if !ok {
		return
	}
	defer d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// Len returns the number of data samples currently in the data set.
func (d *PartitionedDataset) Len(rank int) int {
	return d.partitions[d.groups[rank]].Len()
}

// OnEpochEnd restores the data partitions.
func (d *PartitionedDataset) OnEpochEnd(epoch int64) {
	rand.Seed(epoch)
	mapping = rand.Perm(len(mapping))

	for rank, partition := range d.partitions {
		for item, ok := partition.DeleteMin(); ok; item, ok = partition.DeleteMin() {
			d.recycleBins[rank].ReplaceOrInsert(item)
		}
	}
	d.partitions, d.recycleBins = d.recycleBins, d.partitions
}

// OnTrainEnd terminates the training environment.
func (d *PartitionedDataset) OnTrainEnd() {
	for rank, partition := range d.partitions {
		partition.Clear(false)
		d.partitions[rank] = nil
		d.recycleBins[rank].Clear(false)
		d.recycleBins[rank] = nil
	}
	d.partitions = nil
	d.recycleBins = nil
}
