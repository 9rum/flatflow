import numpy as np
import torch.distributed
from megatron.core import parallel_state


def get_model_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group.
    """
    world_size = torch.distributed.get_world_size()
    total_rank = np.arange(world_size)
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
    total_rank = total_rank.reshape(pipeline_parallel_size, -1, tensor_parallel_size)
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    return total_rank[:, data_parallel_rank, :].min()
