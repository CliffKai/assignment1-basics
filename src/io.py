# src/io.py
from __future__ import annotations
from typing import BinaryIO, IO
import os
import torch

def _is_pathlike(x) -> bool:
    return isinstance(x, (str, bytes, os.PathLike))

def save_checkpoint(
    *, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    序列化 {model_state, optimizer_state, iteration} 到 out（路径或二进制 file-like）。
    """
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    if _is_pathlike(out):
        with open(out, "wb") as f:
            torch.save(payload, f)
    else:
        # 假设是二进制 file-like（具有 .write）
        torch.save(payload, out)

def load_checkpoint(
    *, 
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    从 src（路径或二进制 file-like）加载 checkpoint，恢复 model/optimizer，并返回 iteration。
    """
    if _is_pathlike(src):
        with open(src, "rb") as f:
            payload = torch.load(f, map_location="cpu")
    else:
        payload = torch.load(src, map_location="cpu")

    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["iteration"])
