from collections.abc import Iterable
import torch


def gradient_clipping(params: Iterable[torch.nn.Parameter], 
                      max_l2_norm: float,
                      eps: float = 1e-6) -> float:
    grads = [p.grad for  p in params if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    
    l2_norm = torch.sqrt(
        sum((torch.sum(g * g) for g in grads),
                start=torch.zeros((), device=grads[0].device, dtype=grads[0].dtype))
        )

    if l2_norm > max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + eps)
        for g in grads:
            g.mul_(clip_coef)
    return float(l2_norm)

def compute_grad_l2_norm(params: Iterable[torch.nn.Parameter]) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    
    l2_norm = torch.sqrt(
        sum((torch.sum(g * g) for g in grads),
                start=torch.zeros((), device=grads[0].device, dtype=grads[0].dtype))
        )
    return float(l2_norm)
        
        

