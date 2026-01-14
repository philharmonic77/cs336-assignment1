import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional
from dataclasses import dataclass
from cs336_basics.nn.attention import softmax

@dataclass(frozen=True)
class SamplingConfig:
    """
    Generation / sampling configuration.

    Notes:
    - max_new_tokens: number of tokens to generate (not including prompt length).
    - temperature:
        * 1.0 = no change
        * <1.0 sharper (more greedy)
        * >1.0 flatter (more random)
        * 0.0 is often treated as greedy (we'll handle as a special case)
    - top_p:
        * 1.0 = no nucleus filtering
        * smaller = keep only smallest set whose cumulative prob >= top_p
    - eos_token_id: stop when generated token == eos_token_id (if provided)
    """
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    eos_token_id: Optional[int] = None

@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: Int[Tensor, "B S"],
    cfg: SamplingConfig,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:

    if prompt_ids.ndim != 2:
        raise ValueError(f"prompt_ids must be (B, S), got {tuple(prompt_ids.shape)}")
    if cfg.max_new_tokens <= 0:
        return prompt_ids
    
    ctx = getattr(model, "context_length", None)
    if not isinstance(ctx, int):
        raise ValueError("model must have int attribute `context_length`")

    was_training = model.training 

    model.eval()    
    out = prompt_ids.clone()
    
    for _ in range(cfg.max_new_tokens):
        if out.size(1) > ctx: 
            input_ids = out[:, -ctx:]
        else:
            input_ids = out
        logits: Float[Tensor, "B S V"] = model(input_ids)
        next_logit: Float[Tensor, "B V"] = logits[:, -1, :]

        next_token = sample_next_token(
            next_logit,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            generator=generator)

        out = torch.concat([out, next_token], dim=1)

        if cfg.eos_token_id is not None:
            if (next_token == cfg.eos_token_id).all():
                break

    if was_training:
        model.train()
    return out

def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, V], got {tuple(logits.shape)}")

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
    
    logits = logits / temperature

    masked_logits = logits
    if top_p < 1.0:
        masked_logits = top_p_filtering(logits, top_p)
    
    probs = softmax(masked_logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_token


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:

    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        # argmax
        out = torch.full_like(logits, float("-inf"))
        idx = torch.argmax(logits, dim=-1, keepdim=True)
        out.scatter_(dim=-1, index=idx, src=torch.gather(logits, -1, idx))
        return out

    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = softmax(sorted_logits, dim=-1)
    cumprobs: Float[Tensor, "B V"] = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumprobs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1]
    sorted_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # unsort back
    masked_logits = torch.empty_like(logits)
    masked_logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return masked_logits
    
