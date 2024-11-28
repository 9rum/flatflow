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

import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
import pytest

from flatflow.rpc import CommunicatorClient, run


def lognormal_n(mean: float, sigma: float, n: int, seed: int) -> Sequence[int]:
    random.seed(seed)
    np.random.seed(seed)

    sizes = []
    while len(sizes) < n:
        size = np.random.lognormal(mean, sigma)
        if 0.5 <= size and size < 8192.5:
            sizes.append(round(size))
    return sizes


def launch(
    port: int,
    world_size: int,
    global_batch_size: int,
    micro_batch_size: int,
    order: int,
    rank: int,
    seed: int,
    heterogeneous: bool,
    use_flat_shuffle: bool,
    num_samples: int,
) -> None:
    if rank == 0:
        run(port, world_size)

    sizes = None
    if rank == 0:
        sizes = lognormal_n(5.252, 0.293, num_samples, seed)

    channel = grpc.insecure_channel(f"localhost:{port}")
    client = CommunicatorClient(channel)

    client.Init(global_batch_size, micro_batch_size, order, rank, seed, heterogeneous, use_flat_shuffle, sizes=sizes)

    for epoch in range(4):
        response = client.Broadcast(epoch)
        assert response.Converged()
        assert response.IndicesLength() == num_samples // world_size

    if rank == 0:
        client.Finalize()


@pytest.mark.parametrize("port", [50051])
@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("global_batch_size", [128])
@pytest.mark.parametrize("micro_batch_size", [4])
@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("heterogeneous", [False])
@pytest.mark.parametrize("use_flat_shuffle", [True])
@pytest.mark.parametrize("num_samples", [65536])
def test_communicator(
    port,
    world_size,
    global_batch_size,
    micro_batch_size,
    order,
    seed,
    heterogeneous,
    use_flat_shuffle,
    num_samples,
):
    executor = ThreadPoolExecutor(max_workers=world_size)
    futures = [
        executor.submit(
            launch,
            port,
            world_size,
            global_batch_size,
            micro_batch_size,
            order,
            rank,
            seed,
            heterogeneous,
            use_flat_shuffle,
            num_samples,
        )
        for rank in range(world_size)
    ]
    for future in futures:
        future.result()
