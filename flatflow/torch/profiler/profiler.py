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

import contextlib
import datetime
import json
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict

import torch
import torch.distributed
from megatron.core import parallel_state

__all__ = ["ComputeProfiler", "MemoryProfiler"]


class ComputeProfiler:
    def __init__(self, rank, layers, layer_events):
        self.timestamp = {}
        self.forward_times = []
        self.buffer = defaultdict(float)
        self.rank = rank
        self.world_size = torch.distributed.get_world_size()
        self.current_batch_id = 0
        self.hook_handles = []
        self.last_stage_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        self.mp_group = parallel_state.get_model_parallel_group()
        self.mp_src_rank = torch.distributed.get_process_group_ranks(self.mp_group)[0]
        self.pp_group = parallel_state.get_pipeline_model_parallel_group()
        self.event = mp.Event()
        self.layers = layers
        self.layer_events = layer_events
        self.times = defaultdict(float)
        for name in self.layers.keys():
            self.register_hooks(name)

    def register_hooks(self, name):
        events = self.layer_events[name]
        layer = self.layers[name]

        def forward_pre_hook(module, _):
            batch_key = "forward_" + self._generate_batch_key(self.current_batch_id)
            torch.cuda.synchronize()
            events["forward_pre"].record()
            if batch_key not in self.timestamp:
                self.timestamp[batch_key] = time.perf_counter()

        def forward_hook(module, _, _1):
            events["forward_post"].record()
            batch_key = "forward_" + self._generate_batch_key(self.current_batch_id)
            torch.cuda.synchronize()
            self.times[batch_key] += events["forward_pre"].elapsed_time(events["forward_post"])

        def backward_pre_hook(module, _):
            batch_key = "backward_" + self._generate_batch_key(self.current_batch_id)
            torch.cuda.synchronize()
            events["backward_pre"].record()
            if batch_key not in self.timestamp:
                self.timestamp[batch_key] = time.perf_counter()

        def backward_hook(module, _, _1):
            events["backward_post"].record()
            batch_key = "backward_" + self._generate_batch_key(self.current_batch_id)
            torch.cuda.synchronize()
            self.times[batch_key] += events["backward_pre"].elapsed_time(events["backward_post"])


        self.hook_handles.append(layer.register_forward_pre_hook(forward_pre_hook))
        self.hook_handles.append(layer.register_forward_hook(forward_hook))

        self.hook_handles.append(layer.register_full_backward_pre_hook(backward_pre_hook))
        self.hook_handles.append(layer.register_full_backward_hook(backward_hook))

    def save_latency_log(self):

        combined_data = {
        'elapsed_time': self.times,
        'timestamps': self.timestamp,
        }

        compute_times = (
            [None]
            * parallel_state.get_pipeline_model_parallel_world_size()
            * parallel_state.get_tensor_model_parallel_world_size()
        )

        torch.distributed.all_gather_object(compute_times, combined_data, group=self.mp_group)
        if self.rank == self.mp_src_rank:
            elapsed = defaultdict(float)
            initial_time = dict()

            for data in compute_times:
                if data:
                    for key, value in data['elapsed_time'].items():
                        base_key = re.sub(r"_tp_\d+", "", key)
                        elapsed[base_key] = max(elapsed[base_key], value)
                    for key, value in data['timestamps'].items():
                        base_key = re.sub(r"_tp_\d+", "", key)
                        if base_key not in initial_time or value < initial_time[base_key]:
                            initial_time[base_key] = value

            processed_data = {
                'timestamp': initial_time,
                'elapsed_time': elapsed,
            }

        else:
            processed_data = None

        data_object = [None] * self.world_size if self.rank == 0 else None

        torch.distributed.gather_object(obj=processed_data, object_gather_list=data_object, dst=0)
        if self.rank == 0:
            assert data_object is not None
            revised = [data for data in data_object if data is not None]
            with open("latency_profile.json", "w") as f:
                json.dump(revised, f, indent=4)

    def _generate_batch_key(self, microbatch_id: int) -> str:
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        tensor_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
        return f"dp{data_parallel_rank}_pp{pipeline_parallel_rank}_tp{tensor_parallel_rank}_batch{microbatch_id}"

    def set_microbatch_id(self, microbatch_id):
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
    _current_step = 0
    _profile_start_step = None
    _profile_end_step = None
    _enabled = True
    _step_interval = 1  # profile N step at a time
    
    @classmethod
    def configure(cls, start_step: int = None, end_step: int = None, 
                  step_interval: int = 1, enabled: bool = True):
        cls._profile_start_step = start_step
        cls._profile_end_step = end_step
        cls._step_interval = step_interval
        cls._enabled = enabled
        
        # Allow configuration via environment variables
        if os.environ.get("MEMORY_PROFILE_START_STEP"):
            cls._profile_start_step = int(os.environ.get("MEMORY_PROFILE_START_STEP"))
        if os.environ.get("MEMORY_PROFILE_END_STEP"):
            cls._profile_end_step = int(os.environ.get("MEMORY_PROFILE_END_STEP"))
        if os.environ.get("MEMORY_PROFILE_INTERVAL"):
            cls._step_interval = int(os.environ.get("MEMORY_PROFILE_INTERVAL"))
        if os.environ.get("MEMORY_PROFILE_ENABLED"):
            cls._enabled = os.environ.get("MEMORY_PROFILE_ENABLED").lower() == "true"
            
        print(f"MemoryProfiler configured: start_step={cls._profile_start_step}, "
              f"end_step={cls._profile_end_step}, interval={cls._step_interval}, enabled={cls._enabled}")
    
    @classmethod
    def set_step(cls, step):
        cls._current_step = int(step) if isinstance(step, str) else step
    
    @classmethod
    def increment_step(cls):
        cls._current_step += 1
    
    @classmethod
    def should_profile(cls) -> bool:
        if not cls._enabled:
            return False
        if cls._profile_start_step is not None and cls._current_step < cls._profile_start_step:
            return False
        if cls._profile_end_step is not None and cls._current_step > cls._profile_end_step:
            return False
        if cls._step_interval > 1 and cls._current_step % cls._step_interval != 0:
            return False
            
        return True
    
    @staticmethod
    def _get_output_filename() -> str:
        dp_rank = parallel_state.get_data_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        rank = os.environ.get("LOCAL_RANK", "0")
        return f"memory_profile_rank{rank}_dp{dp_rank}_pp{pp_rank}_tp{tp_rank}.jsonl"
    
    @classmethod
    @contextlib.contextmanager
    def profile(cls, tag: str = "", **metadata):
        # Return if profiling is not needed
        if not cls.should_profile():
            yield
            return
            
        if not torch.cuda.is_available():
            yield
            return

        torch.cuda.synchronize()
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()

            allocated_diff = (end_allocated - start_allocated) / 1024**2
            reserved_diff = (end_reserved - start_reserved) / 1024**2

            new_data = {
                "step": cls._current_step,
                "allocated_mb": allocated_diff,
                "reserved_mb": reserved_diff,
                "timestamp": datetime.datetime.now().isoformat(),
                "rank": int(os.environ.get("LOCAL_RANK", 0)),
                "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            }

            if metadata:
                for k, v in metadata.items():
                    if k not in new_data:
                        new_data[k] = v

            cls.save_log(tag, new_data, cls._get_output_filename())

    @staticmethod
    def save_log(tag: str, new_data: Dict[str, Any], output_file: str):
        try:
            dir_path = os.path.dirname(output_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            record = {"tag": tag, **new_data}
            metadata = MemoryProfiler._serialize(record)

            try:
                json_str = json.dumps(metadata)
            except (TypeError, ValueError) as json_err:
                print(f"Warning: JSON serialization failed: {json_err}")
                safe_metadata = {k: str(v) for k, v in metadata.items()}
                json_str = json.dumps(safe_metadata)
            
            with open(output_file, 'a') as f:
                f.write(json_str + '\n')

        except (OSError, IOError) as io_err:
            print(f"Warning: File I/O error when saving to {output_file}: {io_err}")
        except Exception as e:
            print(f"Warning: Failed to save memory profile to {output_file}: {e}")

    @staticmethod
    def _serialize(obj):
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.detach().cpu().tolist()
        elif isinstance(obj, (list, tuple)):
            return [MemoryProfiler._serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: MemoryProfiler._serialize(v) for k, v in obj.items()}
        elif hasattr(obj, "item") and callable(getattr(obj, "item")) and not isinstance(obj, torch.Tensor):
            try:
                return obj.item()
            except (ValueError, TypeError):
                return str(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
            try:
                return obj.tolist()
            except (ValueError, TypeError):
                return str(obj)
        else:
            return str(obj)
