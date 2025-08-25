# src/data.py
from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    *, dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维 token 序列里随机采样 (batch_size, context_length) 的输入与右移标签。
    返回的张量 dtype 为 torch.long，放置到给定 device。
    """
    n = int(dataset.shape[0])
    if n < context_length + 1:
        raise ValueError(
            f"dataset too short: need at least context_length+1={context_length+1}, got {n}"
        )

    # 起点 i 的可选范围：[0, n - context_length - 1]（含端点）
    max_start = n - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=(batch_size,))  # [low, high)

    # 逐个切片再堆叠成 (B, T)
    X = np.stack([dataset[i : i + context_length] for i in starts], axis=0)
    Y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts], axis=0)

    # 转为 torch.long 并放置到 device
    X_t = torch.as_tensor(X, dtype=torch.long, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.long, device=device)
    return X_t, Y_t
