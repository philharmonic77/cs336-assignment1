import numpy as np
import torch

from torch import Tensor
from jaxtyping import Float, Int
import numpy.typing as npt


def get_batch(x: npt.NDArray,
                 batch_size: int,
                 context_length: int,
                 device: str) -> tuple[Int[Tensor, "B S"], Int[Tensor, "B S"]]:
    
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")

    n = x.shape[0]
    m = context_length
    if n < m + 1:
        raise ValueError("Need at least context_length + 1 tokens to form (inputs, targets).")

    # start positions i must satisfy: i + m < n  (so i <= n - m - 1)
    max_start = n - m - 1
    starts: Int[npt.NDArray, "B,"] = np.random.randint(0, max_start + 1, size=(batch_size,), dtype=np.int64)

    offsets = np.arange(m, dtype=np.int64)[None, :]   # (1, m)          
    idx: Int[npt.NDArray, "B, m"] = starts[:, None] + offsets   # broadcast: (B, 1) + (1, m) -> (B, m)  

    inputs_np = x[idx]                                   
    targets_np = x[idx + 1]                             

    inputs = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
    return inputs, targets
