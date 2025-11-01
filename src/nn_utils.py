# src/nn_utils.py
from __future__ import annotations
# from jaxtyping import Iterable
from collections.abc import Iterable
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
    device = params[0].grad.device
    grads_norms = torch.stack([p.grad.detach().norm(2) for p in params]).to(device)
    total_norm = grads_norms.norm(2)
    clip_coef = max_l2_norm / (total_norm + 1e-6)  # +epsilon 防 0
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))  # 原地缩放

def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    公式: SiLU(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)