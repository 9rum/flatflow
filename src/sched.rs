// SPDX-License-Identifier: Apache-2.0

//! The scheduler is the main component of FlatFlow that decides which data samples will be batched
//! together and the execution order of those micro-batches.

use flatbuffers::InvalidFlatbuffer;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
            assert_ne!(data_parallel_world_size, 0);
            assert_eq!(global_batch_size % data_parallel_world_size, 0);
            let per_replica_batch_size = global_batch_size / data_parallel_world_size;
            assert_ne!(micro_batch_size, 0);
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
            let n_batches = indices.len() / per_replica_batch_size;
            let mut batches: Vec<Vec<_>> =
                partition(indices.iter().copied(), n_batches, |&index| sizes[index], None);

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
                    match per_replica_batch_size % micro_batch_size {
                        0 => {
                            let micro_batches: Vec<Vec<_>> = partition(
                                batch,
                                gradient_accumulation_steps,
                                |&index| proj(sizes[index]),
                                None,
                            );
                            micro_batches.into_iter().flatten()
                        }
                        last_micro_batch_size => {
                            let last_micro_batch =
                                batch.split_off(per_replica_batch_size - last_micro_batch_size);
                            let mut micro_batches: Vec<Vec<_>> = partition(
                                batch,
                                gradient_accumulation_steps,
                                |&index| proj(sizes[index]),
                                None,
                            );
                            micro_batches.push(last_micro_batch);
                            micro_batches.into_iter().flatten()
                        }
                    }
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
