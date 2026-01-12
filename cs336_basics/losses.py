from jaxtyping import Float, Int
import torch
from torch import Tensor, nn

def cross_entropy(
        logits: Float[Tensor, "... seq vocab_size"],
        targets: Int[Tensor, "... seq"]
) -> Float[Tensor, ""]:
    """
    Computes the cross-entropy loss for next-token prediction.

    The per-position loss is:

        ℓ_i = −log softmax(o_i)[y]
            = log(∑_j exp(o_{i,j})) − o_{i,y}

    For numerical stability, the log-sum-exp term is computed as:

        log_Z = log(∑_j exp(o_{i,j}))
        = m_i + log(∑_j exp(o_{i,j} − m_i)),
          where m_i = max_j o_{i,j}.

    The loss is averaged over all batch-like dimensions.
    """
    assert targets.dtype == torch.long
    o_max: Float[Tensor, "... seq 1"] = logits.max(dim=-1, keepdim=True).values
    log_Z: Float[Tensor, "... seq 1"] = o_max + torch.log(
        torch.sum(torch.exp(logits - o_max), dim=-1, keepdim=True)
        )
    
    # torch.gather(..., dim=-1, ...)对每个 (..., seq) 位置，在 vocab 维度上取 index = y_i 的那个 logit
    o_y: Float[Tensor, "... seq 1"] = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    loss: Float[Tensor, ""] = (log_Z - o_y).mean()

    return loss
