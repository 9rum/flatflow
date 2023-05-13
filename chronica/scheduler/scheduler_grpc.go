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
	"syscall"

	"github.com/9rum/chronica/internal/btree"
	"github.com/9rum/chronica/internal/data"
	"github.com/golang/protobuf/ptypes/empty"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// schedulerServer implements the server API for Scheduler service.
type schedulerServer struct {
	UnimplementedSchedulerServer
	scheduler Scheduler
	done      chan<- os.Signal
}

// NewSchedulerServer creates a new scheduler server.
func NewSchedulerServer(done chan<- os.Signal) SchedulerServer {
	return &schedulerServer{done: done}
}

// Init initializes the training environment.
func (s *schedulerServer) Init(ctx context.Context, in *Arguments) (*empty.Empty, error) {
	// initialize a dataset with the given sizes
	var dataset data.Dataset
	if in.GetPartition() {
		partitions := make([][]int64, 0, in.GetWorldSize())
		for base := 0; base < len(in.GetSizes()); base += int(in.GetPartitionSize()) {
			partitions = append(partitions, in.GetSizes()[base:base+int(in.GetPartitionSize())])
		}
		dataset = data.NewPartitionedDataset[*btree.ItemBase](partitions)
	} else {
		dataset = data.NewShardedDataset[*btree.ItemBase](in.GetSizes())
	}

	// initialize a scheduler based on the given schedule type
	switch in.GetType() {
	case Schedule_STATIC:
		s.scheduler = NewStaticScheduler(dataset, int(in.GetWorldSize()), int(in.GetBatchSize()), binSize(in))
	case Schedule_DYNAMIC:
		fallthrough
	default:
		panic("invalid type")
	}

	if r := recover(); r != nil {
		return nil, status.Errorf(codes.Internal, "%v", r)
	}
	return new(empty.Empty), nil
}

// binSize returns the bin size to be used for static scheduling.
func binSize(in *Arguments) (_ int) {
	if in.GetType() == Schedule_STATIC {
		return int(math.Round(mean(in.GetSizes()) * float64(in.GetBatchSize()) / float64(in.GetWorldSize())))
	}
	return
}

// mean averages the given sizes.
func mean(sizes []int64) float64 {
	sum := int64(0)
	for _, size := range sizes {
		sum += size
	}
	return float64(sum) / float64(len(sizes))
}

// Bcast broadcasts the schedule to all workers. If the scheduler provides a
// feedback-directed optimization, the performance indicators in the given
// feedback are used to estimate the training time.
func (s schedulerServer) Bcast(ctx context.Context, in *Feedback) (*Indices, error)

// Reset is called at the end of an epoch during training. This typically resets
// the training environment for scheduling in the next training epoch.
func (s schedulerServer) Reset(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	s.scheduler.OnEpochEnd()

	if r := recover(); r != nil {
		return nil, status.Errorf(codes.Internal, "%v", r)
	}
	return new(empty.Empty), nil
}

// Finalize terminates the training environment.
func (s *schedulerServer) Finalize(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	defer func() {
		signal.Notify(s.done, syscall.SIGTERM)
		close(s.done)
	}()

	s.scheduler.OnTrainEnd()
	s.scheduler = nil

	if r := recover(); r != nil {
		return nil, status.Errorf(codes.Internal, "%v", r)
	}
	return new(empty.Empty), nil
}
