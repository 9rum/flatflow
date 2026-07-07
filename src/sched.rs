// SPDX-License-Identifier: Apache-2.0

//! The scheduler is the main component of FlatFlow that decides which data samples will be batched
//! together and the execution order of those micro-batches.

use core::iter::once;
use std::time::Instant;

use flatbuffers::InvalidFlatbuffer;
use log::info;
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, pyfunction};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use scopeguard::defer;

use crate::ops::{root_as_graph, transform};

mod partition;
use partition::partition;

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
pub enum Policy {
    Joint,
    Fast,
    Mem,
}

impl From<&str> for Policy {
    #[inline]
    fn from(policy: &str) -> Self {
        match policy {
            "joint" => Self::Joint,
            "fast" => Self::Fast,
            "mem" => Self::Mem,
            _ => unreachable!(),
        }
    }
}

#[pyfunction]
#[inline]
pub fn sched(
    indices: Vec<usize>,
    sizes: Vec<i64>,
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
    policy: &str,
) -> PyResult<Vec<usize>> {
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

    let mut indices = indices;

    match policy.into() {
        Policy::Joint => sched_joint(
            indices.as_mut_slice(),
            sizes.as_slice(),
            buf,
            tensor_parallel_world_size,
            context_parallel_world_size,
            data_parallel_world_size,
            data_parallel_rank,
            global_batch_size,
            micro_batch_size,
        ),
        Policy::Fast => todo!(),
        Policy::Mem => todo!(),
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn sched_joint(
    indices: &mut [usize],
    sizes: &[i64],
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
) -> Result<Vec<usize>, InvalidFlatbuffer> {
    assert_eq!(global_batch_size % data_parallel_world_size, 0);
    let per_replica_batch_size = global_batch_size / data_parallel_world_size;
    let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

    let proj =
        transform(root_as_graph(buf)?, tensor_parallel_world_size, context_parallel_world_size);

    // Partition the given indices in a memory-balanced manner so that the number of tokens in each
    // batch is as uniform as possible.
    indices.sort_unstable_by_key(|&index| sizes[index]);
    let batches: Vec<Vec<_>> =
        partition(indices.iter().copied(), data_parallel_world_size, |&index| sizes[index], None);

    let mut batch = batches.into_iter().nth(data_parallel_rank).unwrap();
    batch.sort_unstable_by_key(|&index| proj(sizes[index]));

    // Partition the batch into micro-batches so that any earlier micro-batch takes less execution
    // time than subsequent ones to reduce pipeline bubbles.
    let last_micro_batch_size = batch.len() % micro_batch_size;
    let last_micro_batch = batch.split_off(batch.len() - last_micro_batch_size);
    let micro_batches: Vec<Vec<_>> =
        partition(batch, gradient_accumulation_steps, |&index| proj(sizes[index]), None);

    Ok(micro_batches.into_iter().chain(once(last_micro_batch)).flatten().collect())
}

#[pyfunction]
#[inline]
pub fn sched_unstable(
    indices: Vec<usize>,
    sizes: Vec<i64>,
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
    policy: &str,
) -> PyResult<Vec<usize>> {
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

    let mut indices = indices;

    match policy.into() {
        Policy::Joint => sched_unstable_joint(
            indices.as_mut_slice(),
            sizes.as_slice(),
            buf,
            tensor_parallel_world_size,
            context_parallel_world_size,
            data_parallel_world_size,
            data_parallel_rank,
            global_batch_size,
            micro_batch_size,
        ),
        Policy::Fast => todo!(),
        Policy::Mem => todo!(),
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn sched_unstable_joint(
    indices: &mut [usize],
    sizes: &[i64],
    buf: &[u8],
    tensor_parallel_world_size: i64,
    context_parallel_world_size: i64,
    data_parallel_world_size: usize,
    data_parallel_rank: usize,
    global_batch_size: usize,
    micro_batch_size: usize,
) -> Result<Vec<usize>, InvalidFlatbuffer> {
    assert_ne!(global_batch_size, 0);

    match indices.len() % global_batch_size {
        0 => {
            assert_eq!(global_batch_size % data_parallel_world_size, 0);
            let per_replica_batch_size = global_batch_size / data_parallel_world_size;
            let gradient_accumulation_steps = per_replica_batch_size / micro_batch_size;

            let proj = transform(
                root_as_graph(buf)?,
                tensor_parallel_world_size,
                context_parallel_world_size,
            );

            // Partition the given indices in a memory-balanced manner so that the number of tokens
            // in each batch is as uniform as possible, thereby preventing devices from running out
            // of memory while promoting stable convergence.
            indices.sort_unstable_by_key(|&index| sizes[index]);
            let mut batches: Vec<Vec<_>> = partition(
                indices.iter().copied(),
                indices.len() / per_replica_batch_size,
                |&index| sizes[index],
                None,
            );

            // Sort batches in order to reduce synchronization latency across pipelines.
            batches.sort_unstable_by_key(|batch| {
                let sum: i64 = batch.iter().map(|&index| proj(sizes[index])).sum();
                sum
            });
            let batches: Vec<_> = batches
                .into_iter()
                .skip(data_parallel_rank)
                .step_by(data_parallel_world_size)
                .collect();

            // Partition each batch into micro-batches so that any earlier micro-batch takes less
            // execution time than subsequent ones to reduce pipeline bubbles.
            let indices = batches
                .into_par_iter()
                .flat_map_iter(|mut batch| {
                    batch.sort_unstable_by_key(|&index| proj(sizes[index]));

                    let last_micro_batch_size = per_replica_batch_size % micro_batch_size;
                    let last_micro_batch = batch.split_off(batch.len() - last_micro_batch_size);
                    let micro_batches: Vec<Vec<_>> = partition(
                        batch,
                        gradient_accumulation_steps,
                        |&index| proj(sizes[index]),
                        None,
                    );
                    micro_batches.into_iter().chain(once(last_micro_batch)).flatten()
                })
                .collect();

            Ok(indices)
        }
        last_global_batch_size => {
            let (left, right) = indices.split_at_mut(indices.len() - last_global_batch_size);
            let mut batches = sched_unstable_joint(
                left,
                sizes,
                buf,
                tensor_parallel_world_size,
                context_parallel_world_size,
                data_parallel_world_size,
                data_parallel_rank,
                global_batch_size,
                micro_batch_size,
            )?;
            let mut last_batch = sched_unstable_joint(
                right,
                sizes,
                buf,
                tensor_parallel_world_size,
                context_parallel_world_size,
                data_parallel_world_size,
                data_parallel_rank,
                last_global_batch_size,
                micro_batch_size,
            )?;
            batches.append(&mut last_batch);
            Ok(batches)
        }
    }
}
