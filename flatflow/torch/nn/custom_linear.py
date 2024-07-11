import torch
from torch import nn


class Linear(nn.Module):
    """Custom Linear layer for flatflow.
    Offsets are required to handle variable length inputs.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )

    def forward(self, x, offsets):
        """
        Args:
            x : torch.Tensor
            offsets : Dict[str, Union[List[Any], int]]
                Offset values of cu_seqlens and max_seqlen
                There are four keys : cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        """
        return self.proj(x), offsets
