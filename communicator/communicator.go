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

// Package communicator implements an intermediary to communicate with
// Chronica's scheduler.
package communicator

import (
	"context"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"

	"github.com/9rum/chronica/internal/data"
	"github.com/9rum/chronica/scheduler"
	"github.com/golang/glog"
	"github.com/golang/protobuf/ptypes/empty"
	"golang.org/x/exp/constraints"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// communicatorServer implements the server API for Communicator service.
type communicatorServer struct {
	UnimplementedCommunicatorServer
	scheduler scheduler.Scheduler
	done      chan<- os.Signal
	fanin     chan struct{}
	fanout    []chan []int
	steps     int
}

// NewCommunicatorServer creates a new communicator server.
func NewCommunicatorServer(done chan<- os.Signal) CommunicatorServer {
	return &communicatorServer{
		done:  done,
		fanin: make(chan struct{}),
	}
}

// Init initializes the training environment.
func (c *communicatorServer) Init(ctx context.Context, in *InitRequest) (*empty.Empty, error) {
	worldSize := int(in.GetWorldSize())
	batchSize := int(in.GetBatchSize())

	glog.Infof("Init called with world size: %d batch size: %d type: %s", worldSize, batchSize, in.GetType())

	c.fanout = make([]chan []int, 0, worldSize)
	for len(c.fanout) < cap(c.fanout) {
		c.fanout = append(c.fanout, make(chan []int))
	}

	// initialize a data set with the given sizes
	var (
		dataset data.Dataset
		err     error
	)
	sizes := cast[int64, int](in.GetSizes())
	c.steps = ceil(len(sizes), batchSize)

	if in.GetPartition() {
		groups := cast[int64, int](in.GetGroups())
		partitionSize := len(sizes) / worldSize
		partitionSizes := make([]int, max(groups...)+1)
		for _, rank := range groups {
			partitionSizes[rank] += partitionSize
		}

		// We assume that the indices are sequentially distributed across workers.
		partitions := make([][]int, 0, len(partitionSizes))
		base := 0
		for _, partitionSize = range partitionSizes {
			partitions = append(partitions, sizes[base:base+partitionSize])
			base += partitionSize
		}
		dataset, err = data.NewPartitionedDataset(groups, partitions)
	} else {
		dataset, err = data.NewShardedDataset(sizes)
	}

	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// initialize a scheduler based on the given schedule type
	switch in.GetType() {
	case Schedule_STATIC:
		c.scheduler = scheduler.NewStaticScheduler(dataset, worldSize, batchSize, sizes)
	case Schedule_DYNAMIC:
		c.scheduler = scheduler.NewDynamicScheduler(dataset, worldSize, batchSize)
	default:
		panic("invalid type")
	}

	return new(empty.Empty), nil
}

// cast casts the given slice.
func cast[T, U constraints.Integer](slice []T) []U {
	out := make([]U, len(slice))
	stride := ceil(len(slice), runtime.NumCPU())

	var wg sync.WaitGroup
	for base := 0; base < len(slice); base += stride {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for index := base; index < min(base+stride, len(slice)); index++ {
				out[index] = U(slice[index])
			}
		}(base)
	}
	wg.Wait()

	return out
}

// ceil returns the least integer value greater than or equal to numerator / denominator.
// This is an alternative to the Ceil function in the standard math package.
func ceil(numerator, denominator int) int {
	if numerator%denominator == 0 {
		return numerator / denominator
	}
	return numerator/denominator + 1
}

// min returns the minimum value in the given slice.
func min[T constraints.Ordered](slice ...T) (min T) {
	if len(slice) == 0 {
		return
	}
	min = slice[0]
	for _, v := range slice[1:] {
		if v < min {
			min = v
		}
	}
	return
}

// max returns the maximum value in the given slice.
func max[T constraints.Ordered](slice ...T) (max T) {
	if len(slice) == 0 {
		return
	}
	max = slice[0]
	for _, v := range slice[1:] {
		if max < v {
			max = v
		}
	}
	return
}

// Bcast broadcasts the schedule to all workers. If the scheduler provides a
// feedback-directed optimization, the performance indicators in the given
// feedback are used to estimate the training time.
func (c *communicatorServer) Bcast(ctx context.Context, in *BcastRequest) (*BcastResponse, error) {
	glog.Infof("epoch: %d Bcast called from rank %d", in.GetEpoch(), in.GetRank())

	go func() {
		c.scheduler.OnEpochEnd(in.GetEpoch(), in.GetRank(), in.GetCoefficient(), in.GetIntercept())
		c.fanin <- struct{}{}
	}()

	if in.GetRank() == 0 {
		go func() {
			for range c.fanout {
				<-c.fanin
			}
			for rank, indices := range scheduler.NextN(c.scheduler, c.steps) {
				c.fanout[rank] <- indices
			}
		}()
	}

	indices := <-c.fanout[in.GetRank()]
	return &BcastResponse{Indices: cast[int, int64](indices)}, nil
}

// Finalize terminates the training environment.
func (c *communicatorServer) Finalize(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	defer func() {
		close(c.fanin)
		for _, ch := range c.fanout {
			close(ch)
		}
		signal.Notify(c.done, syscall.SIGTERM)
		close(c.done)
	}()

	glog.Info("Finalize called")
	defer glog.Flush()

	c.scheduler.OnTrainEnd()
	c.scheduler = nil

	return new(empty.Empty), nil
}
