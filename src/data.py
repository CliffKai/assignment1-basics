# src/data.py
from __future__ import annotations          
import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    *,
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    pin_memory: bool = False   # 默认 False
):
    """
    规则：
    1. pin_memory == False 且 device 是 GPU -> 所有操作在 GPU 上完成
    2. pin_memory == False 且 device 是 CPU -> 所有操作在 CPU 上完成
    3. pin_memory == True 且 device 是 GPU -> 在 CPU 上索引，然后 pinned memory -> GPU
    4. pin_memory == True 且 device 是 CPU -> 报错
    """

    is_cuda = str(device).startswith("cuda")

    # case 4: pin_memory=True 且 device=cpu -> 报错
    if pin_memory and not is_cuda:
        raise ValueError("pin_memory=True only makes sense when device is CUDA")

    # case 1 & 2: 直接在目标 device 上完成所有操作
    if not pin_memory:
        toks = torch.as_tensor(dataset, dtype=torch.long, device=device)
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

    # case 3: pin_memory=True 且 device 是 GPU -> CPU 上索引, 再搬到 GPU
    else:
        toks_cpu = torch.as_tensor(dataset, dtype=torch.long, device="cpu")
        toks_cpu = toks_cpu.pin_memory()  # <-- 正确做法
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

