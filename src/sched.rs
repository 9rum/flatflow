// SPDX-License-Identifier: Apache-2.0

//! The scheduler is the main component of FlatFlow that decides which data samples will be batched
//! together and the execution order of those micro-batches.

use core::iter::once;
use std::time::Instant;

use flatbuffers::InvalidFlatbuffer;
use log::info;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::{Bound, PyResult, pyfunction};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use scopeguard::defer;

use crate::ops::{root_as_graph, transform};

mod partition;
use partition::{Heuristic, partition};

/// Scheduling policy to select the scheduling objectives. The default policy is `Joint`.
///
/// # Scheduling policies
///
/// There are several scheduling policies, i.e., `Fast`, `Mem` and `Joint`, depending on the
/// objective the scheduler aims to optimize. The descriptions for each are as follows:
///
/// * `Fast` minimizes computation stalls by eliminating pipeline bubbles between micro-batches
///   within a pipeline and reducing synchronization latency across pipelines.
/// * `Mem` prioritizes memory balance by keeping device memory usage as even as possible, helping
///   maximize batch size and promote stable convergence.
/// * `Joint` combines both objectives by first balancing batch-level device memory usage and then
///   reducing pipeline bubbles and synchronization latency to promote both stability and training
///   throughput.
#[derive(Default)]
pub enum Policy {
    Fast,
    Mem,
    #[default]
    Joint,
}

impl From<&str> for Policy {
    #[inline]
    fn from(policy: &str) -> Self {
        match policy {
            "fast" => Self::Fast,
            "mem" => Self::Mem,
            "joint" => Self::Joint,
            _ => unreachable!(),
        }
    }
}

/// Reorders the given computation schedule `indices` for the next training epoch.
///
/// This scheduler is stable; i.e., does not affect the resulting checkpoint by iteratively
/// reordering the training sequence at the granularity of mini-batch, which we call *iterative
/// reordering*.
///
/// When applicable, unstable scheduling is preferred because stable scheduling may produce somewhat
/// suboptimal training performance due to the constraints in optimization scope. See
/// [`sched_unstable`](sched::sched_unstable).
///
/// `sched` returns an error if the given serialized computational graph `buf` is invalid or if the
/// given array `indices` or `sizes` is not contiguous.
///
/// # Scheduling policies
///
/// There are several scheduling policies and one of them can be selected via `policy`. See
/// [`Policy`](sched::Policy) for the descriptions for each.
///
/// # Panics
///
/// May panic if the given arguments are invalid.
#[pyfunction]
pub fn sched<'py>(
    mut indices: PyReadwriteArray1<'py, usize>,
    sizes: PyReadonlyArray1<'py, i64>,
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
    policy: &str,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    assert_ne!(data_parallel_world_size, 0);
    assert_eq!(indices.len() % data_parallel_world_size, 0);
    assert_ne!(micro_batch_size, 0);

    let now = Instant::now();
    let num_samples = indices.len() / data_parallel_world_size;
    defer!(info!(
        "Reordering {} micro-batches took {:?}",
        ((num_samples - 1) / micro_batch_size + 1) * data_parallel_world_size,
        now.elapsed()
    ));

    iterative_reorder_by(
        indices.as_slice_mut()?,
        sizes.as_slice()?,
        buf,
        tensor_parallel_world_size,
        context_parallel_world_size,
        data_parallel_world_size,
        data_parallel_rank,
        global_batch_size,
        micro_batch_size,
        policy.into(),
    )
    .map(|batches| batches.into_pyarray(indices.py()))
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[inline]
fn iterative_reorder_by(
    indices: &mut [usize],
    sizes: &[i64],
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
    policy: Policy,
) -> Result<Vec<usize>, InvalidFlatbuffer> {
    let f = transform(root_as_graph(buf)?, tensor_parallel_world_size, context_parallel_world_size);

    let batches = match policy {
        Policy::Fast => indices
            .par_chunks_mut(global_batch_size)
            .flat_map_iter(|batch| {
                sched_fast(
                    batch,
                    sizes,
                    &f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    micro_batch_size,
                )
            })
            .collect(),
        Policy::Mem => indices
            .par_chunks_mut(global_batch_size)
            .flat_map_iter(|batch| {
                sched_mem(
                    batch,
                    sizes,
                    &f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    micro_batch_size,
                )
            })
            .collect(),
        Policy::Joint => indices
            .par_chunks_mut(global_batch_size)
            .flat_map_iter(|batch| {
                sched_joint(
                    batch,
                    sizes,
                    &f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    micro_batch_size,
                )
            })
            .collect(),
    };

    Ok(batches)
}

#[inline]
fn sched_fast<F>(
    indices: &mut [usize],
    sizes: &[i64],
    f: F,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    micro_batch_size: usize,
) -> impl Iterator<Item = usize>
where
    F: Fn(i64) -> i64,
{
    assert_eq!(indices.len() % data_parallel_world_size, 0);
    let per_replica_batch_size = indices.len() / data_parallel_world_size;
    let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

    indices.sort_unstable_by_key(|&index| f(sizes[index]));
    let batches: Vec<Vec<_>> = partition(
        indices.iter().copied(),
        data_parallel_world_size,
        |&index| f(sizes[index]),
        Some(Heuristic::Meld),
    );

    let mut batch = batches.into_iter().nth(data_parallel_rank).unwrap();
    batch.sort_unstable_by_key(|&index| f(sizes[index]));

    let last_micro_batch_size = per_replica_batch_size % micro_batch_size;
    let last_micro_batch = batch.split_off(per_replica_batch_size - last_micro_batch_size);
    let micro_batches: Vec<Vec<_>> = partition(
        batch,
        gradient_accumulation_steps,
        |&index| f(sizes[index]),
        Some(Heuristic::Meld),
    );

    micro_batches.into_iter().chain(once(last_micro_batch)).flatten()
}

#[inline]
fn sched_mem<F>(
    indices: &mut [usize],
    sizes: &[i64],
    f: F,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    micro_batch_size: usize,
) -> impl Iterator<Item = usize>
where
    F: Fn(i64) -> i64,
{
    assert_eq!(indices.len() % data_parallel_world_size, 0);
    let per_replica_batch_size = indices.len() / data_parallel_world_size;
    let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

    indices.sort_unstable_by_key(|&index| sizes[index]);
    let batches: Vec<Vec<_>> = partition(
        indices.iter().copied(),
        data_parallel_world_size,
        |&index| sizes[index],
        Some(Heuristic::Meld),
    );

    let mut batch = batches.into_iter().nth(data_parallel_rank).unwrap();
    batch.sort_unstable_by_key(|&index| sizes[index]);

    let last_micro_batch_size = per_replica_batch_size % micro_batch_size;
    let last_micro_batch = batch.split_off(per_replica_batch_size - last_micro_batch_size);
    let mut micro_batches: Vec<Vec<_>> =
        partition(batch, gradient_accumulation_steps, |&index| sizes[index], Some(Heuristic::Meld));

    micro_batches.sort_unstable_by_key(|micro_batch| {
        let sum: i64 = micro_batch.iter().map(|&index| f(sizes[index])).sum();
        sum
    });

    micro_batches.into_iter().chain(once(last_micro_batch)).flatten()
}

#[inline]
fn sched_joint<F>(
    indices: &mut [usize],
    sizes: &[i64],
    f: F,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    micro_batch_size: usize,
) -> impl Iterator<Item = usize>
where
    F: Fn(i64) -> i64,
{
    assert_eq!(indices.len() % data_parallel_world_size, 0);
    let per_replica_batch_size = indices.len() / data_parallel_world_size;
    let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

    // Partition the given batch in a memory-balanced manner so that the number of tokens in each
    // replica is as uniform as possible.
    indices.sort_unstable_by_key(|&index| sizes[index]);
    let batches: Vec<Vec<_>> = partition(
        indices.iter().copied(),
        data_parallel_world_size,
        |&index| sizes[index],
        Some(Heuristic::Meld),
    );

    let mut batch = batches.into_iter().nth(data_parallel_rank).unwrap();
    batch.sort_unstable_by_key(|&index| f(sizes[index]));

    // Partition the batch into micro-batches so that any earlier micro-batch takes less execution
    // time than subsequent ones to reduce pipeline bubbles.
    let last_micro_batch_size = per_replica_batch_size % micro_batch_size;
    let last_micro_batch = batch.split_off(per_replica_batch_size - last_micro_batch_size);
    let micro_batches: Vec<Vec<_>> = partition(
        batch,
        gradient_accumulation_steps,
        |&index| f(sizes[index]),
        Some(Heuristic::Meld),
    );

    micro_batches.into_iter().chain(once(last_micro_batch)).flatten()
}

/// Reorders the given computation schedule `indices` for the next training epoch.
///
/// This scheduler is unstable; i.e., does not preserve the resulting checkpoint. If it is important
/// to preserve the resulting checkpoint or if the batch size is sufficiently large, consider using
/// the stable counterpart [`sched`](sched::sched).
///
/// `sched_unstable` returns an error if the given serialized computational graph `buf` is invalid
/// or if the given array `indices` or `sizes` is not contiguous.
///
/// # Scheduling policies
///
/// There are several scheduling policies and one of them can be selected via `policy`. See
/// [`Policy`](sched::Policy) for the descriptions for each.
///
/// # Panics
///
/// May panic if the given arguments are invalid.
#[pyfunction]
pub fn sched_unstable<'py>(
    mut indices: PyReadwriteArray1<'py, usize>,
    sizes: PyReadonlyArray1<'py, i64>,
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
    policy: &str,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    assert_ne!(data_parallel_world_size, 0);
    assert_eq!(indices.len() % data_parallel_world_size, 0);
    assert_ne!(micro_batch_size, 0);

    let now = Instant::now();
    let num_samples = indices.len() / data_parallel_world_size;
    defer!(info!(
        "Reordering {} micro-batches took {:?}",
        ((num_samples - 1) / micro_batch_size + 1) * data_parallel_world_size,
        now.elapsed()
    ));

    reorder_by(
        indices.as_slice_mut()?,
        sizes.as_slice()?,
        buf,
        tensor_parallel_world_size,
        context_parallel_world_size,
        data_parallel_world_size,
        data_parallel_rank,
        global_batch_size,
        micro_batch_size,
        policy.into(),
    )
    .map(|batches| batches.into_pyarray(indices.py()))
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[inline]
fn reorder_by(
    indices: &mut [usize],
    sizes: &[i64],
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
    policy: Policy,
) -> Result<Vec<usize>, InvalidFlatbuffer> {
    assert_ne!(global_batch_size, 0);

    let f = transform(root_as_graph(buf)?, tensor_parallel_world_size, context_parallel_world_size);

    let batches = match policy {
        Policy::Fast => match indices.len() % global_batch_size {
            0 => sched_unstable_fast(
                indices,
                sizes,
                f,
                data_parallel_world_size,
                data_parallel_rank,
                global_batch_size,
                micro_batch_size,
            ),
            last_global_batch_size => {
                let (left, right) = indices.split_at_mut(indices.len() - last_global_batch_size);
                let mut last_batch = sched_unstable_fast(
                    right,
                    sizes,
                    &f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    last_global_batch_size,
                    micro_batch_size,
                );
                let mut batches = sched_unstable_fast(
                    left,
                    sizes,
                    f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    global_batch_size,
                    micro_batch_size,
                );
                batches.append(&mut last_batch);
                batches
            }
        },
        Policy::Mem => match indices.len() % global_batch_size {
            0 => sched_unstable_mem(
                indices,
                sizes,
                f,
                data_parallel_world_size,
                data_parallel_rank,
                global_batch_size,
                micro_batch_size,
            ),
            last_global_batch_size => {
                let (left, right) = indices.split_at_mut(indices.len() - last_global_batch_size);
                let mut last_batch = sched_unstable_mem(
                    right,
                    sizes,
                    &f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    last_global_batch_size,
                    micro_batch_size,
                );
                let mut batches = sched_unstable_mem(
                    left,
                    sizes,
                    f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    global_batch_size,
                    micro_batch_size,
                );
                batches.append(&mut last_batch);
                batches
            }
        },
        Policy::Joint => match indices.len() % global_batch_size {
            0 => sched_unstable_joint(
                indices,
                sizes,
                f,
                data_parallel_world_size,
                data_parallel_rank,
                global_batch_size,
                micro_batch_size,
            ),
            last_global_batch_size => {
                let (left, right) = indices.split_at_mut(indices.len() - last_global_batch_size);
                let mut last_batch = sched_unstable_joint(
                    right,
                    sizes,
                    &f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    last_global_batch_size,
                    micro_batch_size,
                );
                let mut batches = sched_unstable_joint(
                    left,
                    sizes,
                    f,
                    data_parallel_world_size,
                    data_parallel_rank,
                    global_batch_size,
                    micro_batch_size,
                );
                batches.append(&mut last_batch);
                batches
            }
        },
    };

    Ok(batches)
}

fn sched_unstable_fast<F>(
    indices: &mut [usize],
    sizes: &[i64],
    f: F,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
) -> Vec<usize>
where
    F: Fn(i64) -> i64 + Sync,
{
    assert_eq!(global_batch_size % data_parallel_world_size, 0);
    let per_replica_batch_size = global_batch_size / data_parallel_world_size;
    let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

    indices.sort_unstable_by_key(|&index| f(sizes[index]));
    let batches: Vec<Vec<_>> = partition(
        indices.iter().copied(),
        indices.len() / per_replica_batch_size,
        |&index| f(sizes[index]),
        Some(Heuristic::Meld),
    );

    let batches: Vec<_> =
        batches.into_iter().skip(data_parallel_rank).step_by(data_parallel_world_size).collect();

    batches
        .into_par_iter()
        .flat_map_iter(|mut batch| {
            batch.sort_unstable_by_key(|&index| f(sizes[index]));

            let last_micro_batch_size = per_replica_batch_size % micro_batch_size;
            let last_micro_batch = batch.split_off(per_replica_batch_size - last_micro_batch_size);
            let micro_batches: Vec<Vec<_>> = partition(
                batch,
                gradient_accumulation_steps,
                |&index| f(sizes[index]),
                Some(Heuristic::Meld),
            );
            micro_batches.into_iter().chain(once(last_micro_batch)).flatten()
        })
        .collect()
}

fn sched_unstable_mem<F>(
    indices: &mut [usize],
    sizes: &[i64],
    f: F,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
) -> Vec<usize>
where
    F: Fn(i64) -> i64 + Sync,
{
    assert_eq!(global_batch_size % data_parallel_world_size, 0);
    let per_replica_batch_size = global_batch_size / data_parallel_world_size;
    let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

    indices.sort_unstable_by_key(|&index| sizes[index]);
    let batches: Vec<Vec<_>> = partition(
        indices.iter().copied(),
        indices.len() / per_replica_batch_size,
        |&index| sizes[index],
        Some(Heuristic::Meld),
    );

    let batches: Vec<_> =
        batches.into_iter().skip(data_parallel_rank).step_by(data_parallel_world_size).collect();

    batches
        .into_par_iter()
        .flat_map_iter(|mut batch| {
            batch.sort_unstable_by_key(|&index| sizes[index]);

            let last_micro_batch_size = per_replica_batch_size % micro_batch_size;
            let last_micro_batch = batch.split_off(per_replica_batch_size - last_micro_batch_size);
            let mut micro_batches: Vec<Vec<_>> = partition(
                batch,
                gradient_accumulation_steps,
                |&index| sizes[index],
                Some(Heuristic::Meld),
            );

            micro_batches.sort_unstable_by_key(|micro_batch| {
                let sum: i64 = micro_batch.iter().map(|&index| f(sizes[index])).sum();
                sum
            });

            micro_batches.into_iter().chain(once(last_micro_batch)).flatten()
        })
        .collect()
}

fn sched_unstable_joint<F>(
    indices: &mut [usize],
    sizes: &[i64],
    f: F,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
) -> Vec<usize>
where
    F: Fn(i64) -> i64 + Sync,
{
    assert_eq!(global_batch_size % data_parallel_world_size, 0);
    let per_replica_batch_size = global_batch_size / data_parallel_world_size;
    let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

    // Partition the given indices in a memory-balanced manner so that the number of tokens in each
    // batch is as uniform as possible, thereby preventing devices from running out of memory while
    // promoting stable convergence.
    indices.sort_unstable_by_key(|&index| sizes[index]);
    let mut batches: Vec<Vec<_>> = partition(
        indices.iter().copied(),
        indices.len() / per_replica_batch_size,
        |&index| sizes[index],
        Some(Heuristic::Meld),
    );

    // Sort batches in order to reduce synchronization latency across pipelines.
    batches.sort_unstable_by_key(|batch| {
        let sum: i64 = batch.iter().map(|&index| f(sizes[index])).sum();
        sum
    });
    let batches: Vec<_> =
        batches.into_iter().skip(data_parallel_rank).step_by(data_parallel_world_size).collect();

    // Partition each batch into micro-batches so that any earlier micro-batch takes less execution
    // time than subsequent ones to reduce pipeline bubbles.
    batches
        .into_par_iter()
        .flat_map_iter(|mut batch| {
            batch.sort_unstable_by_key(|&index| f(sizes[index]));

            let last_micro_batch_size = per_replica_batch_size % micro_batch_size;
            let last_micro_batch = batch.split_off(per_replica_batch_size - last_micro_batch_size);
            let micro_batches: Vec<Vec<_>> = partition(
                batch,
                gradient_accumulation_steps,
                |&index| f(sizes[index]),
                Some(Heuristic::Meld),
            );
            micro_batches.into_iter().chain(once(last_micro_batch)).flatten()
        })
        .collect()
}
