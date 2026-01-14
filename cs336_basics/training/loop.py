import torch
from torch import nn
import numpy as np
import math
from pathlib import Path
import shutil
import random
import argparse
import importlib.util
from typing import Any, Optional
from cs336_basics.data import get_batch
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.training.grad_utils import gradient_clipping
from cs336_basics.training.scheduler import learning_rate_schedule
from cs336_basics.training.serialization import save_checkpoint, load_checkpoint
from cs336_basics.training import init_wandb, apply_overrides, save_run_config

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_py_config(path: str) -> dict[str, Any]:
    p = Path(path)
    spec = importlib.util.spec_from_file_location("user_cfg", str(p))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "cfg", None)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must define a dict variable named `cfg`")
    return cfg

def train(cfg: dict[str, Any]) -> None:

    # 0) parse cfg

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    ocfg = cfg["optim"]
    tcfg = cfg["training"]
    icfg = cfg["io"]
    scfg = tcfg["schedule"]

    # 1.1) key params

    seed = int(tcfg.get("seed", 0))
    if seed:
        set_seed(seed)

    device_str: str = tcfg["device"]
    dtype_str: str = tcfg["dtype"]

    device = torch.device(device_str)
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Must be one of {list(_DTYPE_MAP)}")
    param_dtype = _DTYPE_MAP[dtype_str]

    train_token_path: str = dcfg["train_token_npy_path"]
    valid_token_path: str = dcfg["valid_token_npy_path"]
    B: int = int(dcfg["batch_size"])
    S: int = int(dcfg["context_length"]) 

    # 1.2) build model/optim

    model = TransformerLM(
        vocab_size=mcfg["vocab_size"],
        context_length=S,               
        num_layers=mcfg["num_layers"],
        d_model=mcfg["d_model"],
        num_heads=mcfg["num_heads"],
        d_ff=mcfg["d_ff"],
        rope_theta=mcfg["rope"]["theta"]
    )

    optimizer = AdamW(
        model.parameters(),
        lr=ocfg["lr"],
        betas=ocfg["betas"],
        eps=ocfg["eps"],
        weight_decay=ocfg["weight_decay"],
    ) 

    # 1.3) other params  

    max_norm = tcfg["grad_clip"]["max_norm"]   

    resume_from: Optional[str] = tcfg.get("resume_from")
    max_iters: int = int(tcfg["max_iters"])

    log_interval: int = int(icfg["log_interval"])
    save_interval: int = int(icfg["save_interval"])

    eval_interval = int(tcfg.get("eval_interval", 100))
    eval_batches = int(tcfg.get("eval_batches", 10))

    # 2) io + wandb

    run_name: str = icfg["run_name"]
    if not run_name:
        raise ValueError("io.run_name must be a non-empty string")
    
    out_dir_str = icfg["out_dir"]
    if not out_dir_str:
        raise ValueError("io.out_dir must be a non-empty path")
    out_dir = Path(out_dir_str)

    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run = init_wandb(cfg, out_dir=out_dir, run_name=run_name)

    # 3) load data   
     
    train_tokens = np.load(train_token_path, mmap_mode="r")
    valid_tokens = np.load(valid_token_path, mmap_mode="r")
    if train_tokens.ndim != 1:
        raise ValueError(f"train token array must be 1D, got shape {train_tokens.shape}!")
    if valid_tokens.ndim != 1:
        raise ValueError(f"valid token array must be 1D, got shape {valid_tokens.shape}!")

    model = model.to(device=device, dtype=param_dtype)
    model.train()
    start_iter = 0

    # 4) resume

    if resume_from is not None:
        start_iter = load_checkpoint(resume_from, model, optimizer)
        model.train()

    # 5) loop

    for it in range(start_iter, max_iters):

        if scfg["type"] == "cosine":
            lr_t = learning_rate_schedule(
                t=it,
                alpha_max=ocfg["lr"],
                alpha_min=scfg["alpha_min"],
                T_w=scfg["T_w"],
                T_c=scfg["T_c"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr_t

        x, y = get_batch(train_tokens, B, S, device) 
            
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        if max_norm is not None:
            gradient_clipping(model.parameters(), max_norm)

        optimizer.step()

        # 6) eval、log、save ckpt

        do_log = ((it + 1) % log_interval == 0)
        do_eval = (eval_interval > 0) and ((it + 1) % eval_interval == 0)
        if do_log or do_eval:
            lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "iter": it + 1,
                "train/loss": float(loss.item()),
                "train/lr": float(lr),
            }

            if do_eval:
                valid_loss, valid_ppl = evaluate(
                    model=model,
                    tokens=valid_tokens,
                    batch_size=B,
                    context_length=S,
                    device=device,
                    num_batches=eval_batches,
                )
                metrics.update({
                    "valid/loss": valid_loss,
                    "valid/ppl": valid_ppl,
                })

            if do_log:
                msg = (
                    f"[{run_name}] iter={it+1} "
                    f"train_loss={metrics['train/loss']:.6f} "
                    f"lr={metrics['train/lr']:.3e}"
                )
                if do_eval:
                    msg += (
                        f" | valid_loss={metrics['valid/loss']:.6f} "
                        f"valid_ppl={metrics['valid/ppl']:.3f}"
                    )
                print(msg)                

            elif do_eval:
                print(
                    f"[{run_name}] iter={it+1} "
                    f"valid_loss={metrics['valid/loss']:.6f} "
                    f"valid_ppl={metrics['valid/ppl']:.3f}"
                )

            if run is not None:
                run.log(metrics, step=it + 1)

        if (it + 1) % save_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_iter{it+1:06d}.pt"
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            latest_path = ckpt_dir / "latest.pt"
            shutil.copyfile(ckpt_path, latest_path)

    # 7) finalize

    if run is not None:
        run.finish()

@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int
):
    model.eval()
    
    losses: list[float] = []
    for _ in range(num_batches):
        x, y = get_batch(tokens, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())

    mean_loss = float(sum(losses) / len(losses))
    ppl = float(math.exp(mean_loss))

    model.train()

    return mean_loss, ppl
    



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])  # 允许传 0~多个 override
    args = ap.parse_args()

    cfg = load_py_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    out_dir = Path(cfg["io"]["out_dir"])
    save_run_config(out_dir, cfg, overrides=args.override)

    train(cfg)

if __name__ == "__main__":
    main()
