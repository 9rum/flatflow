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
	"math"
	"math/rand"

	"github.com/9rum/chronica/internal/btree"
)

// Index mapping for shuffling data samples with the same size. This makes
// scheduling non-deterministic and data well shuffled in that most of the data
// sample sizes are similar in data sets with power-law distribution.
//
// NOTE:
//
// If we store the index mapping inside data set, additional interface is
// required to inform the index mapping and a copy of the data set is created
// for each data sample. When storing the index mapping on every data sample,
// memory usage for storing data samples increases from 16 bytes to 40 bytes
// and the B-tree degree is severely reduced from 128 to 51. Even if we store
// the index mapping as a pointer, there are pointer dereference overheads and
// memory usage still increases to 24 bytes and the B-tree degree is reduced to
// 85. Thus we store the index mapping as a package variable.
var mapping []int

// Dataset represents the given data set.
type Dataset interface {
	// Getitem retrieves a data sample with the given arguments.  This must provide
	// an index identifying the scheduled data sample and its size.
	Getitem(rank int, size int64) (int64, int64)

	// Rand retrieves an arbitrary data sample from the data set.
	Rand(rank int) (int64, int64)

	// OnEpochEnd is called at the end of an epoch during training.
	OnEpochEnd(epoch int64)

	// OnTrainEnd terminates the training environment.
	OnTrainEnd()
}

// New creates a new data set with the given arguments.
func New(sizes, groups []int64, seed int64, partition bool) Dataset {
	if partition {
		return NewPartitionedDataset(sizes, groups, seed)
	}
	return NewShardedDataset(sizes, seed)
}

// ShardedDataset represents a sharded data set where every node in the cluster
// has a replica of the given data set; hence it ignores rank when looking for
// the data sample.
type ShardedDataset struct {
	seed       int64
	items      *btree.BTree[Sample]
	recycleBin *btree.BTree[Sample]
}

// NewShardedDataset creates a new sharded data set with the given argument.
func NewShardedDataset(sizes []int64, seed int64) *ShardedDataset {
	// We use the default degree for the items to fit on a single memory page.
	dataset := &ShardedDataset{
		seed:       seed,
		items:      btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()),
		recycleBin: btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()),
	}
	mapping = rand.Perm(len(sizes))

	for index, size := range sizes {
		if _, found := dataset.recycleBin.ReplaceOrInsert(NewSample(int64(index), size)); found {
			panic("insert found item")
		}
	}

	return dataset
}

// Getitem looks for the data sample with the size nearest to the given size.
func (d *ShardedDataset) Getitem(rank int, size int64) (_, _ int64) {
	if size == math.MinInt {
		item, ok := d.items.DeleteMin()
		if !ok {
			return
		}
		defer d.recycleBin.ReplaceOrInsert(item)
		return item.Index(), item.Size()
	}

	if size == math.MaxInt {
		item, ok := d.items.DeleteMax()
		if !ok {
			return
		}
		defer d.recycleBin.ReplaceOrInsert(item)
		return item.Index(), item.Size()
	}

	item, ok := d.items.DeleteNearest(NewSample(0, size))
	if !ok {
		return
	}
	defer d.recycleBin.ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// Rand selects a random data sample from the data set.
func (d *ShardedDataset) Rand(rank int) (_, _ int64) {
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

	pivot := int64(rand.Intn(int(max-min+1))) + min
	item, ok = d.items.DeleteNearest(NewSample(0, pivot))
	if !ok {
		return
	}
	defer d.recycleBin.ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// OnEpochEnd restores the data samples.
func (d *ShardedDataset) OnEpochEnd(epoch int64) {
	rand.Seed(d.seed + epoch)
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
	seed        int64
	groups      []int64
	partitions  []*btree.BTree[Sample]
	recycleBins []*btree.BTree[Sample]
}

// NewPartitionedDataset creates a new partitioned data set with the given arguments.
func NewPartitionedDataset(sizes, groups []int64, seed int64) *PartitionedDataset {
	nodes := func() int64 {
		maxRank := groups[0]
		for _, rank := range groups[1:] {
			maxRank = max(maxRank, rank)
		}
		return maxRank + 1
	}()
	partitionSize := len(sizes) / len(groups)
	partitionSizes := make([]int, nodes)
	for _, rank := range groups {
		partitionSizes[rank] += partitionSize
	}

	dataset := &PartitionedDataset{
		seed:        seed,
		groups:      groups,
		partitions:  make([]*btree.BTree[Sample], 0, len(partitionSizes)),
		recycleBins: make([]*btree.BTree[Sample], 0, len(partitionSizes)),
	}
	mapping = rand.Perm(len(sizes))

	// We assume that the indices are sequentially distributed across workers.
	base := 0

	for rank, partitionSize := range partitionSizes {
		// We use the default degree for the items to fit on a single memory page.
		dataset.partitions = append(dataset.partitions, btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()))
		dataset.recycleBins = append(dataset.recycleBins, btree.New[Sample](btree.DefaultTargetNodeSize[Sample]()))
		for index, size := range sizes[base : base+partitionSize] {
			if _, found := dataset.recycleBins[rank].ReplaceOrInsert(NewSample(int64(base+index), size)); found {
				panic("insert found item")
			}
		}
		base += partitionSize
	}

	return dataset
}

// Getitem looks for the data sample with the size nearest to the given size
// in the partition with the given rank.
func (d *PartitionedDataset) Getitem(rank int, size int64) (_, _ int64) {
	if size == math.MinInt {
		item, ok := d.partitions[d.groups[rank]].DeleteMin()
		if !ok {
			return
		}
		defer d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)
		return item.Index(), item.Size()
	}

	if size == math.MaxInt {
		item, ok := d.partitions[d.groups[rank]].DeleteMax()
		if !ok {
			return
		}
		defer d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)
		return item.Index(), item.Size()
	}

	item, ok := d.partitions[d.groups[rank]].DeleteNearest(NewSample(0, size))
	if !ok {
		return
	}
	defer d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// Rand selects a random data sample from the data set.
func (d *PartitionedDataset) Rand(rank int) (_, _ int64) {
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

	pivot := int64(rand.Intn(int(max-min+1))) + min
	item, ok = d.partitions[d.groups[rank]].DeleteNearest(NewSample(0, pivot))
	if !ok {
		return
	}
	defer d.recycleBins[d.groups[rank]].ReplaceOrInsert(item)
	return item.Index(), item.Size()
}

// OnEpochEnd restores the data partitions.
func (d *PartitionedDataset) OnEpochEnd(epoch int64) {
	rand.Seed(d.seed + epoch)
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
