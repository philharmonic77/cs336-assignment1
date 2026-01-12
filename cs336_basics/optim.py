from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9,0.999),
            eps=1e-8,
            weight_decay=0.0
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue 
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                grad = p.grad.data 
                # Update the first moment estimate
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1-beta1)
                # Update the second moment estimat
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                # Bias correction: Compute adjusted α for iteration t
                adjusted_lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t) 
                with torch.no_grad(): # Using with torch.no_grad() makes parameter updates explicit, safe, and compatible with PyTorch’s autograd system, avoiding graph pollution and replacing the unsafe use of p.data.
                    # Update the parameters in place
                    p.addcdiv_(state["exp_avg"], state["exp_avg_sq"].sqrt().add_(eps), value=-adjusted_lr)
                    # Apply weight decay
                    p.mul_(1 - lr * weight_decay)

        return loss
    

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int):
    assert t >= 0
    assert alpha_max >= alpha_min
    assert T_c > T_w >= 0

    # Warm-up
    if T_w > 0 and t < T_w:
        return t / T_w * alpha_max
    # Cosine annealin
    elif t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    # Post-annealing
    else:
        return alpha_min
    
    



