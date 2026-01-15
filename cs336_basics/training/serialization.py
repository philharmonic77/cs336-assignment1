import torch
import os 
from typing import BinaryIO, IO
from torch import nn


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device | str = "cpu"):
    
    checkpoint = torch.load(src, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
