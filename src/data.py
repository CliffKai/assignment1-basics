# src/data.py
from __future__ import annotations          
import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    *, dataset_np, batch_size: int, context_length: int,
    device: str, pin_memory: bool = False
):
    is_cuda = str(device).startswith("cuda")
    if pin_memory and not is_cuda:
        raise ValueError("pin_memory=True only makes sense when device is CUDA")

    if pin_memory:
        # CPU + pinned 路径
        toks = torch.as_tensor(dataset_np, dtype=torch.long, device="cpu").pin_memory()
        idx_device = "cpu"
    else:
        # 直接放目标设备
        toks = torch.as_tensor(dataset_np, dtype=torch.long, device=device)
        idx_device = device  # 索引必须与 toks 在同设备

    n = toks.numel()
    if n < context_length + 1:
        raise ValueError(f"need at least {context_length+1}, got {n}")

    max_start = n - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), device=idx_device)
    ar = torch.arange(context_length, device=idx_device)
    idx = starts[:, None] + ar[None, :]
    X = toks[idx]
    Y = toks[idx + 1]

    if pin_memory:
        # 仅在 pinned 路径下做异步 H2D
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

    return X, Y


