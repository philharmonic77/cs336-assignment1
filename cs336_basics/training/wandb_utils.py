from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import wandb


def init_wandb(cfg: dict[str, Any], out_dir: Path, run_name: str) -> Optional[Any]:
    """
    Initialize a Weights & Biases run if enabled in cfg["io"]["wandb"].

    Conventions:
    - If enabled=False: return None.
    - mode defaults to "offline".
    - project is required when enabled=True.
    - dir defaults to out_dir/"wandb" (keeps run artifacts co-located with checkpoints/logs).
    - Only pass non-empty optional fields (entity/tags/notes).
    - Only set resume when mode == "online".
    """
    wcfg = cfg.get("io", {}).get("wandb", {})
    if not wcfg.get("enabled", False):
        return None

    project = wcfg.get("project")
    if not project:
        raise ValueError("io.wandb.project must be set when wandb is enabled")

    mode = str(wcfg.get("mode", "offline"))

    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "config": cfg,
        "mode": mode,
        "dir": str(wcfg.get("dir") or (out_dir / "wandb")),
    }

    entity = wcfg.get("entity")
    if entity:
        init_kwargs["entity"] = entity

    tags = wcfg.get("tags")
    if tags:
        init_kwargs["tags"] = tags

    notes = wcfg.get("notes")
    if notes:
        init_kwargs["notes"] = notes

    if mode == "online":
        init_kwargs["resume"] = wcfg.get("resume", "allow")

    return wandb.init(**init_kwargs)