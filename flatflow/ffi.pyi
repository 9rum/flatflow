# ruff: noqa: E501
from typing import Final

import numpy

__version__: Final[str]

def sched(
    indices: numpy.ndarray[tuple[int], numpy.dtype[numpy.uintp]],
    sizes: numpy.ndarray[tuple[int], numpy.dtype[numpy.int64]],
    buf: bytes,
    tensor_parallel_world_size: int,
    context_parallel_world_size: int,
    data_parallel_world_size: int,
    data_parallel_rank: int,
    global_batch_size: int,
    micro_batch_size: int,
    policy: str,
) -> numpy.ndarray[tuple[int], numpy.dtype[numpy.uintp]]:
    """
    Reorders the given computation schedule `indices` for the next training epoch.

    This scheduler is stable; i.e., does not affect the resulting checkpoint by iteratively
    reordering the training sequence at the granularity of mini-batch, which we call *iterative
    reordering*.

    When applicable, unstable scheduling is preferred because stable scheduling may produce somewhat
    suboptimal training performance due to the constraints in optimization scope. See
    [`sched_unstable`].

    `sched` returns an error if the given serialized computational graph `buf` is invalid or if the
    given array `indices` or `sizes` is not contiguous.

    # Scheduling policies

    There are several scheduling policies and one of them can be selected via `policy`. See
    [`Policy`] for the descriptions for each.

    # Panics

    May panic if the given arguments are invalid.

    [`sched_unstable`]: fn@crate::sched::sched_unstable
    [`Policy`]: enum@crate::sched::Policy
    """

def sched_unstable(
    indices: numpy.ndarray[tuple[int], numpy.dtype[numpy.uintp]],
    sizes: numpy.ndarray[tuple[int], numpy.dtype[numpy.int64]],
    buf: bytes,
    tensor_parallel_world_size: int,
    context_parallel_world_size: int,
    data_parallel_world_size: int,
    data_parallel_rank: int,
    global_batch_size: int,
    micro_batch_size: int,
    policy: str,
) -> numpy.ndarray[tuple[int], numpy.dtype[numpy.uintp]]:
    """
    Reorders the given computation schedule `indices` for the next training epoch.

    This scheduler is unstable; i.e., does not preserve the resulting checkpoint. If it is important
    to preserve the resulting checkpoint or if the batch size is sufficiently large, consider using
    the stable counterpart [`sched`].

    `sched_unstable` returns an error if the given serialized computational graph `buf` is invalid
    or if the given array `indices` or `sizes` is not contiguous.

    # Scheduling policies

    There are several scheduling policies and one of them can be selected via `policy`. See
    [`Policy`] for the descriptions for each.

    # Panics

    May panic if the given arguments are invalid.

    [`sched`]: fn@crate::sched::sched
    [`Policy`]: enum@crate::sched::Policy
    """
