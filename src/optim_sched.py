# src/optim_sched.py
from __future__ import annotations
import math
from typing import Iterable, Optional
import torch
from torch.optim import Optimizer


class AdamWCustom(Optimizer):
    r"""
    纯手写 AdamW，实现与 PyTorch AdamW 等价的更新语义（Decoupled Weight Decay + 偏置校正）。

    参数（与 torch.optim.AdamW 的关键子集对齐）:
      - params: 可训练参数（iterable of Tensors 或 param groups）
      - lr: 学习率 (默认 1e-3)
      - betas: 一阶/二阶动量系数 (默认 (0.9, 0.999))
      - eps: 数值稳定项 (默认 1e-8)
      - weight_decay: decoupled 权重衰减系数 (默认 0.01)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """执行一次参数更新；返回 closure() 的 loss（若提供）。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # AdamW 不支持稀疏梯度（与官方实现一致）
                if grad.is_sparse:
                    raise RuntimeError("AdamWCustom does not support sparse gradients")

                state = self.state[p]
                # 状态初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step: int = state["step"]

                # —— Decoupled Weight Decay（与 Adam 的 L2 正则不同，直接对参数做衰减）——
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                # —— 一阶/二阶动量 —— 
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # —— 偏置校正（和 PyTorch AdamW 语义对齐）——
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # denom = sqrt(v_hat) + eps，其中 v_hat = exp_avg_sq / (1 - beta2^step)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 参数更新
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_adamw_cls():
    """
    返回“我们自己实现的AdamW 类”，语义上是对其 torch.optim.AdamW 的。
    """
    return AdamWCustom


def get_lr_cosine_schedule(
    *,
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    线性 warmup + 余弦退火到 min_lr：
      - 0 <= it <= warmup_iters: 线性插值 0 → max_lr（包含端点）
      - warmup_iters < it < cosine_cycle_iters: 余弦从 max_lr → min_lr
      - it >= cosine_cycle_iters: 固定 min_lr

    注意：cosine_cycle_iters 表示“余弦段结束时刻”的迭代编号，而不是“长度”。
    """
    # 线性 warmup（含端点）
    if it <= warmup_iters:
        if warmup_iters <= 0:
            return float(max_learning_rate)
        return float(max_learning_rate) * (it / float(warmup_iters))

    # 余弦段
    span = max(1, cosine_cycle_iters - warmup_iters)  # 防止除零
    progress = (it - warmup_iters) / float(span)
    if progress >= 1.0:
        return float(min_learning_rate)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine

    # 数值钳制，避免浮点误差越界
    if lr < min_learning_rate:
        lr = min_learning_rate
    if lr > max_learning_rate:
        lr = max_learning_rate
    return float(lr)
