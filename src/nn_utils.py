# src/nn_utils.py
from __future__ import annotations
from typing import Iterable
import torch
from torch import Tensor
from jaxtyping import Float

def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=dim, keepdim=True)

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # log_softmax(x) = x - logsumexp(x)
    logsumexp = torch.logsumexp(inputs, dim=1, keepdim=True)   # (B, 1)
    log_probs = inputs - logsumexp                             # (B, V)
    gathered = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
    return -gathered.mean()                                    # 标准 mean reduction


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    
    # 直接在原始梯度上计算总范数
    with torch.no_grad():
        total_norm = torch.norm(torch.stack([torch.norm(p.grad, 2) for p in params]), 2)
    
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        for p in params:
            # 直接在原始梯度上进行原地缩放
            p.grad.mul_(clip_coef)

def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    公式: SiLU(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)