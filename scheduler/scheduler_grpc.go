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

package scheduler

import (
	"context"
	"math"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"

	"github.com/9rum/chronica/internal/btree"
	"github.com/9rum/chronica/internal/data"
	"github.com/golang/glog"
	"github.com/golang/protobuf/ptypes/empty"
	"golang.org/x/exp/constraints"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// schedulerServer implements the server API for Scheduler service.
type schedulerServer struct {
	UnimplementedSchedulerServer
	scheduler Scheduler
	done      chan<- os.Signal
	fanin     chan struct{}
	fanout    []chan []int
}

// NewSchedulerServer creates a new scheduler server.
func NewSchedulerServer(done chan<- os.Signal) SchedulerServer {
	return &schedulerServer{
		done:  done,
		fanin: make(chan struct{}),
	}
}

// Init initializes the training environment.
func (s *schedulerServer) Init(ctx context.Context, in *Arguments) (*empty.Empty, error) {
	glog.Infof("Init called with world size: %d batch size: %d type: %s", in.GetWorldSize(), in.GetBatchSize(), in.GetType())

	s.fanout = make([]chan []int, in.GetWorldSize())
	for rank := range s.fanout {
		s.fanout[rank] = make(chan []int)
	}

	// initialize a dataset with the given sizes
	var (
		dataset data.Dataset
		err     error
	)
	sizes := cast[int64, int](in.GetSizes())

	if in.GetPartition() {
		groups := cast[int64, int](in.GetGroups())
		partitionSize := len(sizes) / int(in.GetWorldSize())
		partitionSizes := make([]int, max(groups)+1)
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
		dataset, err = data.NewPartitionedDataset[*btree.ItemBase](groups, partitions)
	} else {
		dataset, err = data.NewShardedDataset[*btree.ItemBase](sizes)
	}

	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// initialize a scheduler based on the given schedule type
	switch in.GetType() {
	case SCHEDULE_STATIC:
		s.scheduler = NewStaticScheduler(dataset, int(in.GetWorldSize()), int(in.GetBatchSize()),
			binSize(in.GetSizes(), in.GetWorldSize(), in.GetBatchSize()), ceil(len(sizes), int(in.GetBatchSize())))
	case SCHEDULE_DYNAMIC:
		s.scheduler = NewDynamicScheduler(dataset, int(in.GetWorldSize()), int(in.GetBatchSize()))
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
			for index := base; index < base+stride; index++ {
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

// max returns the maximum value in the given slice.
func max[T constraints.Ordered](slice []T) (max T) {
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

// binSize returns the bin size to be used for static scheduling.
func binSize[T constraints.Integer](sizes []T, worldSize, batchSize T) int {
	return int(math.Round(mean(sizes) * float64(batchSize) / float64(worldSize)))
}

// mean averages the given sizes.
func mean[T constraints.Integer](sizes []T) float64 {
	var sum T
	for _, size := range sizes {
		sum += size
	}
	return float64(sum) / float64(len(sizes))
}

// Bcast broadcasts the schedule to all workers. If the scheduler provides a
// feedback-directed optimization, the performance indicators in the given
// feedback are used to estimate the training time.
func (s schedulerServer) Bcast(ctx context.Context, in *Feedback) (*Schedule, error) {
	glog.Infof("Bcast called from rank %d", in.GetRank())

	go func() {
		s.scheduler.OnBatchEnd(int(in.GetRank()), in.GetCoefficient(), in.GetIntercept())
		s.fanin <- struct{}{}
	}()

	if in.GetRank() == 0 {
		go func() {
			for range s.fanout {
				<-s.fanin
			}
			for rank, indices := range s.scheduler.Schedule() {
				s.fanout[rank] <- indices
			}
		}()
	}

	indices := <-s.fanout[in.GetRank()]
	return &Schedule{Indices: cast[int, int64](indices)}, nil
}

// Reset is called at the end of an epoch during training. It resets the
// training environment for scheduling in the next training epoch.
func (s schedulerServer) Reset(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	glog.Info("Reset called")

	s.scheduler.OnEpochEnd()

	return new(empty.Empty), nil
}

// Finalize terminates the training environment.
func (s *schedulerServer) Finalize(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	defer func() {
		close(s.fanin)
		for _, c := range s.fanout {
			close(c)
		}
		signal.Notify(s.done, syscall.SIGTERM)
		close(s.done)
	}()

	glog.Info("Finalize called")
	defer glog.Flush()

	s.scheduler.OnTrainEnd()
	s.scheduler = nil

	return new(empty.Empty), nil
}
