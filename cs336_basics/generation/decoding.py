import torch
import json
from pathlib import Path
from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional
from dataclasses import dataclass
from cs336_basics.nn.attention import softmax
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.training.serialization import load_checkpoint
from cs336_basics.text.tokenizer import Tokenizer

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
        * 0.0 treated as greedy 
    - top_p:
        * 1.0 = no nucleus filtering
        * smaller = keep only smallest set whose cumulative prob >= top_p
    - eos_token_id: stop when generated token == eos_token_id
    """
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    eos_token_id: Optional[int] = 256

def generate_sample(
    tokenizer: Tokenizer,
    prompt: str,
    runs_path: Path,
    run_name: str,
    ckpt_name: str = "latest.pt",
    sample_cfg: SamplingConfig | None= None,
    device: torch.device | str = torch.device("cuda:0"),
    random_seed: int = 1234
):
    # 1) prepare model

    with open(runs_path / run_name / "config.json", "r") as f:
        cfg = json.load(f)
    mcfg = cfg["model"]
    S = int(cfg["data"]["context_length"]) 

    model = TransformerLM(
        vocab_size=mcfg["vocab_size"],
        context_length=S,               
        num_layers=mcfg["num_layers"],
        d_model=mcfg["d_model"],
        num_heads=mcfg["num_heads"],
        d_ff=mcfg["d_ff"],
        rope_theta=mcfg["rope"]["theta"]
    )

    # 2) load state
    device = torch.device(device)
    ckpt_path = runs_path / run_name / "ckpt" / ckpt_name
    load_checkpoint(ckpt_path, model, optimizer=None, device="cpu")
    model.to(device)

    # 3) encode -> generate -> decode
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

    gen = torch.Generator(device=device).manual_seed(random_seed)

    if sample_cfg is None:
        sample_cfg = SamplingConfig()
    new_tokens = generate(model, prompt_ids, sample_cfg, gen).detach().cpu().tolist()

    out = tokenizer.decode(new_tokens)
    return out

@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: Int[Tensor, "(S,)"],
    cfg: SamplingConfig,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError(f"prompt_ids must be (S,), got {tuple(prompt_ids.shape)}")
    if cfg.max_new_tokens <= 0:
        return prompt_ids

    ctx = getattr(model, "context_length", None)
    if not isinstance(ctx, int):
        raise ValueError("model must have int attribute `context_length`")

    was_training = model.training
    model.eval()

    out = prompt_ids.clone()

    for _ in range(cfg.max_new_tokens):
        input_ids = out[-ctx:] if out.size(0) > ctx else out
        input_ids_2d = input_ids.unsqueeze(0)          # [1, S]

        logits = model(input_ids_2d)                   # [1, S, V]
        next_logit = logits[0, -1, :]                  # [V]

        next_token = sample_next_token(
            next_logit,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            generator=generator,
        )                                              # [1]

        out = torch.cat([out, next_token], dim=0)

        if cfg.eos_token_id is not None and int(next_token.item()) == int(cfg.eos_token_id):
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
    
    if logits.ndim != 1:
        raise ValueError(f"logits must be (V, ), got {tuple(logits.shape)}")

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True).view(1)   # shape (1,) 
    
    logits = logits / temperature

    masked_logits = logits
    if top_p < 1.0:
        masked_logits = top_p_filtering(logits, top_p)
    
    probs = softmax(masked_logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_token.view(1)  # ensure shape (1,)


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
    cumprobs: Float[Tensor, "V"] = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumprobs > top_p
    sorted_mask[1:] = sorted_mask[ :-1]
    sorted_mask[0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # unsort back
    masked_logits = torch.empty_like(logits)
    masked_logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return masked_logits
    

if __name__ == '__main__':

    repo_root = Path(__file__).resolve().parents[2]

    TS_vocab_path = repo_root / "artifacts" / "bpe" / "tinystories_vocab.json"
    TS_merges_path = repo_root / "artifacts" / "bpe" / "tinystories_merges.txt"

    tinystories_tokenizer = Tokenizer.from_file(str(TS_vocab_path),
                                                str(TS_merges_path),
                                                special_tokens=["<|endoftext|>"])

    eos_ids = tinystories_tokenizer.encode("<|endoftext|>")
    if len(eos_ids) != 1:
        raise ValueError(f"Expected <|endoftext|> to be 1 token, got {eos_ids}")
    eos_id = eos_ids[0]

    sample_cfg = SamplingConfig(eos_token_id=eos_id)

    out = generate_sample(tinystories_tokenizer,
                           prompt="Let's play games!",
                           runs_path=repo_root / "runs",
                           run_name="exp_lr3e-4_bs128",
                           sample_cfg=sample_cfg,
                           device="cpu",
                           random_seed=2345
                           )
    print(out)