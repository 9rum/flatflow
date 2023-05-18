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
	"log"
	"math"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"

	"github.com/9rum/chronica/internal/btree"
	"github.com/9rum/chronica/internal/data"
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
	logger    *log.Logger
}

// NewSchedulerServer creates a new scheduler server.
func NewSchedulerServer(done chan<- os.Signal) SchedulerServer {
	return &schedulerServer{
		done:   done,
		fanin:  make(chan struct{}),
		logger: log.New(os.Stderr, "", log.LstdFlags),
	}
}

// Init initializes the training environment.
func (s *schedulerServer) Init(ctx context.Context, in *Arguments) (*empty.Empty, error) {
	s.logger.Printf("Init called with\nworld size: %d\nbatch size: %d\npartition: %t\npartition size: %d\ntype: %s", in.GetWorldSize(), in.GetBatchSize(), in.GetPartition(), in.GetPartitionSize(), in.GetType())

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
		partitions := make([][]int, 0, in.GetWorldSize())
		for base := 0; base < len(sizes); base += int(in.GetPartitionSize()) {
			partitions = append(partitions, sizes[base:base+int(in.GetPartitionSize())])
		}
		dataset, err = data.NewPartitionedDataset[*btree.ItemBase](partitions)
	} else {
		dataset, err = data.NewShardedDataset[*btree.ItemBase](sizes)
	}

	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// initialize a scheduler based on the given schedule type
	switch in.GetType() {
	case SCHEDULE_STATIC:
		s.scheduler = NewStaticScheduler(dataset, int(in.GetWorldSize()), int(in.GetBatchSize()), binSize(in.GetSizes(), in.GetWorldSize(), in.GetBatchSize()))
	case SCHEDULE_DYNAMIC:
		fallthrough
	default:
		panic("invalid type")
	}

	return new(empty.Empty), nil
}

// cast casts the given slice.
func cast[T, U constraints.Integer](slice []T) []U {
	out := make([]U, len(slice))
	stride := int(math.Ceil(float64(len(slice)) / float64(runtime.NumCPU())))

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
func (s schedulerServer) Bcast(ctx context.Context, in *Feedback) (*Indices, error) {
	s.logger.Printf("Bcast called from rank %d", in.GetRank())

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
	return &Indices{Indices: cast[int, int64](indices)}, nil
}

// Reset is called at the end of an epoch during training. It resets the
// training environment for scheduling in the next training epoch.
func (s schedulerServer) Reset(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	s.logger.Print("Reset called")

	s.scheduler.OnEpochEnd()

	return new(empty.Empty), nil
}

// Finalize terminates the training environment.
func (s *schedulerServer) Finalize(ctx context.Context, in *empty.Empty) (*empty.Empty, error) {
	defer func() {
		signal.Notify(s.done, syscall.SIGTERM)
		close(s.done)
	}()

	s.logger.Print("Finalize called")

	s.scheduler.OnTrainEnd()
	s.scheduler = nil

	close(s.fanin)
	for _, ch := range s.fanout {
		close(ch)
	}

	return new(empty.Empty), nil
}
