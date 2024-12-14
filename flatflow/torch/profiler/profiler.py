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

import multiprocessing as mp
import re
import time
from collections import defaultdict
from typing import Any

import torch

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

class ComputeProfiler:
    def __init__(self, rank ,hook_handles=[]):
        self.start_times = {}
        self.forward_times = list()
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
        self.event.set()

    def _generate_batch_key(self, microbatch_id: int) -> str:
        return f"dp{parallel_state.get_data_parallel_rank()}_pp{parallel_state.get_pipeline_model_parallel_rank()}_tp{parallel_state.get_tensor_model_parallel_rank()}_batch{microbatch_id}"

    def update_microbatch_id(self):
        self.current_batch_id+=1

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
        if parallel_state.is_pipeline_last_stage():
            latest_microbatch_id = [-1]
            for key in self.buffer.keys():
                latest_microbatch_id[0] = max(latest_microbatch_id[0], int(key.split('batch')[1]))
        else:
            latest_microbatch_id = [-1]

        torch.distributed.broadcast_object_list(
            latest_microbatch_id, src=self.last_stage_rank, group=self.pp_group
        )

        profile_times = [None] * parallel_state.get_pipeline_model_parallel_world_size() * parallel_state.get_tensor_model_parallel_world_size()

        torch.distributed.all_gather_object(
            profile_times, self.buffer, group=self.mp_group
        )

        if self.rank == self.mp_src_rank:
            tp_max_times = defaultdict(float)
            for data in profile_times:
                if data:
                    for key, value in data.items():
                        base_key = re.sub(r'_tp_\d+', '', key)
                        tp_max_times[base_key] = max(tp_max_times[base_key], value)

            processed_times = defaultdict(float)
            for key, value in tp_max_times.items():
                base_key = re.sub(r'_pp\d+', '', key)
                processed_times[base_key] += value

            sorted_times = [value for _, value in sorted(processed_times.items(), key=lambda x: self.get_batch_id(x[0]))]

            self.forward_times = sorted_times
            self.event.set()

        self.update(latest_microbatch_id[0])

    def get_batch_id(self,key):
        match = re.search(r'batch(\d+)', key)
        return int(match.group(1)) if match else -1

    def update(self, latest_microbatch_id: int):
        keys_to_remove = []

        for key in self.buffer.keys():
            batch_id = int(key.split('batch')[1]) if not isinstance(key, int) else int(key)
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
