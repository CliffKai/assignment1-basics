# src/data.py
from __future__ import annotations          
import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    *,
    dataset_np,
    batch_size: int,
    context_length: int,
    device: str,
    pin_memory: bool = False   # 默认 False
):
    is_cuda = str(device).startswith("cuda")

    if pin_memory and not is_cuda:
        raise ValueError("pin_memory=True only makes sense when device is CUDA")

    if not pin_memory:
        toks = torch.as_tensor(dataset_np, dtype=torch.long, device=device)
        n = toks.numel()
        if n < context_length + 1:
            raise ValueError(f"need at least {context_length+1}, got {n}")

        max_start = n - context_length - 1
        starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
        ar = torch.arange(context_length, device=device)
        idx = starts[:, None] + ar[None, :]

        X = toks[idx]
        Y = toks[idx + 1]
        return X, Y

    else:
        toks_cpu = torch.as_tensor(dataset_np, dtype=torch.long, device="cpu")
        if pin_memory:
            toks_cpu = toks_cpu.pin_memory()
        n = toks_cpu.numel()
        if n < context_length + 1:
            raise ValueError(f"need at least {context_length+1}, got {n}")

        max_start = n - context_length - 1
        starts = torch.randint(0, max_start + 1, (batch_size,), device="cpu")
        ar = torch.arange(context_length, device="cpu")
        idx = starts[:, None] + ar[None, :]

        X_cpu = toks_cpu[idx]
        Y_cpu = toks_cpu[idx + 1]

        X = X_cpu.to(device, non_blocking=True)
        Y = Y_cpu.to(device, non_blocking=True)
        return X, Y
