# Copyright 2024 The FlatFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing as mp
import re
import time
from collections import defaultdict
from typing import Any

import torch.distributed
from megatron.core import parallel_state

__all__ = ["ComputeProfiler", "MemoryProfiler"]


class ComputeProfiler:
    def __init__(self, rank, hook_handles=[]):
        self.start_times = {}
        self.forward_times = []
        self.buffer = defaultdict(float)
        self.rank = rank
        self.world_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.current_batch_id = 0
        self.hook_handles = hook_handles
        self.last_stage_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        self.mp_group = parallel_state.get_model_parallel_group()
        self.mp_src_rank = torch.distributed.get_process_group_ranks(self.mp_group)[0]
        self.pp_group = parallel_state.get_pipeline_model_parallel_group()
        self.event = mp.Event()

    def _generate_batch_key(self, microbatch_id: int) -> str:
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        tensor_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
        return f"dp{data_parallel_rank}_pp{pipeline_parallel_rank}_tp{tensor_parallel_rank}_batch{microbatch_id}"

    def set_microbatch_id(self, microbatch_id: int):
        self.current_batch_id = microbatch_id

    def update_microbatch_id(self):
        self.current_batch_id += 1

    def record_start(self, module: Any, input: Any) -> None:
        batch_key = self._generate_batch_key(self.current_batch_id)
        self.start_times[batch_key] = time.perf_counter()

    def record_end(self, module: Any, input: Any, output: Any) -> None:
        batch_key = self._generate_batch_key(self.current_batch_id)
        if batch_key in self.start_times:
            elapsed_time = (time.perf_counter() - self.start_times[batch_key]) * 1000
            self.buffer[batch_key] += elapsed_time

    def gather_times(self):
        """Synchronize time in pipeline group and model parallel group
        Broadcast recent microbatch id from last stage.
        In model parallel source rank, perpare data list as size of model parallel world size.
        gather self.buffer to accumulate all forward times based on unique key.

            profile_time = [
                dict(dp0_pp0_tp0_batch0: 1.0, dp0_pp0_tp0_batch1: 2.0, ...),
                dict(dp1_pp0_tp0_batch0: 1.0, dp1_pp0_tp0_batch1: 2.0, ...),
                ...
        """
        if parallel_state.get_pipeline_model_parallel_rank() == self.last_stage_rank:
            latest_microbatch_id = [-1]
            for key in self.buffer.keys():
                latest_microbatch_id[0] = max(latest_microbatch_id[0], int(key.split("batch")[1]))
        else:
            latest_microbatch_id = [-1]

        torch.distributed.broadcast_object_list(latest_microbatch_id, src=self.last_stage_rank, group=self.pp_group)

        profile_times = (
            [None]
            * parallel_state.get_pipeline_model_parallel_world_size()
            * parallel_state.get_tensor_model_parallel_world_size()
        )

        torch.distributed.all_gather_object(profile_times, self.buffer, group=self.mp_group)

        if self.rank == self.mp_src_rank:
            tp_max_times = defaultdict(float)
            for data in profile_times:
                if data:
                    for key, value in data.items():
                        base_key = re.sub(r"_tp_\d+", "", key)
                        tp_max_times[base_key] = max(tp_max_times[base_key], value)

            processed_times = defaultdict(float)
            for key, value in tp_max_times.items():
                base_key = re.sub(r"_pp\d+", "", key)
                processed_times[base_key] += value

            sorted_times = [
                value for _, value in sorted(processed_times.items(), key=lambda x: self.get_batch_id(x[0]))
            ]

            self.forward_times.extend(sorted_times)
            self.event.set()

        self.update(latest_microbatch_id[0])

    def get_batch_id(self, key):
        match = re.search(r"batch(\d+)", key)
        return int(match.group(1)) if match else -1

    def set_event(self):
        self.event.set()

    def update(self, latest_microbatch_id: int):
        keys_to_remove = []

        for key in self.buffer.keys():
            batch_id = int(key.split("batch")[1]) if not isinstance(key, int) else int(key)
            if latest_microbatch_id >= batch_id:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.buffer[key]

    def wait(self, timeout=None):
        return self.event.wait(timeout)

    def extract(self):
        times = self.forward_times
        self.forward_times = []
        self.event.clear()
        return times


class MemoryProfiler:
    def __init__(self, rank, hook_handles=[]):
        self.rank = rank
        self.world_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.current_batch_id = 0
        self.hook_handles = hook_handles
        self.last_stage_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        self.mp_group = parallel_state.get_model_parallel_group()
        self.mp_src_rank = torch.distributed.get_process_group_ranks(self.mp_group)[0]
        self.pp_group = parallel_state.get_pipeline_model_parallel_group()
        self.memory_tracker = defaultdict(float)
        self.memory_buffer = {}

    def _generate_batch_key(self, microbatch_id: int) -> str:
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        tensor_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
        return f"dp{data_parallel_rank}_pp{pipeline_parallel_rank}_tp{tensor_parallel_rank}_batch{microbatch_id}"

    def set_microbatch_id(self, microbatch_id: int):
        self.current_batch_id = microbatch_id

    def record_start(self, module: Any, input: Any) -> None:
        batch_key = self._generate_batch_key(self.current_batch_id)
        self.memory_buffer[batch_key] = torch.cuda.memory_allocated() / 1024 / 1024

    def record_end(self, module: Any, input: Any, output: Any) -> None:
        batch_key = self._generate_batch_key(self.current_batch_id)
        if batch_key in self.memory_buffer:
            self.memory_tracker[batch_key] += (
                torch.cuda.memory_allocated() / 1024 / 1024 - self.memory_buffer[batch_key]
            )

    def save_memory_log(self):
        memory_gather_object = (
            [None]
            * parallel_state.get_pipeline_model_parallel_world_size()
            * parallel_state.get_tensor_model_parallel_world_size()
        )

        torch.distributed.all_gather_object(memory_gather_object, self.memory_tracker, group=self.mp_group)

        if self.rank == self.mp_src_rank:
            tp_max_memory = defaultdict(float)
            for data in memory_gather_object:
                if data:
                    for key, value in data.items():
                        base_key = re.sub(r"_tp_\d+", "", key)
                        tp_max_memory[base_key] = max(tp_max_memory[base_key], value)

            processed_memory = defaultdict(float)
            for key, value in tp_max_memory.items():
                base_key = re.sub(r"_pp\d+", "", key)
                processed_memory[base_key] += value

            group_size = (
                parallel_state.get_pipeline_model_parallel_world_size()
                * parallel_state.get_tensor_model_parallel_world_size()
            )
            num_groups = self.world_size // group_size

            memory_object = [None] * num_groups
        else:
            processed_memory = None
            memory_object = None

        if self.rank == 0:
            memory_object = [None] * self.world_size

        torch.distributed.gather_object(obj=processed_memory, object_gather_list=memory_object, dst=0)
        if self.rank == 0:
            assert memory_object is not None
            revised = [memory for memory in memory_object if memory is not None]
            with open("memory_profile.json", "w") as f:
                json.dump(revised, f, indent=4)
