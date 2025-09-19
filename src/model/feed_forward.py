# src/model/feed_forward.py
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.model.linear import Linear
from src.nn_utils import silu

class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络的实现。
    遵循 cs336_spring2025_assignment1_basics.pdf 3.5.2 节的规范。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化 SwiGLU 模块。

        Args:
            d_model: 模型的隐藏维度。
            d_ff: 内部前馈层的维度。如果为 None，则根据作业中规范计算。
            device: 参数所在的设备。
            dtype: 参数的数据类型。
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if d_ff is None:
            # 根据作业要求中规范: d_ff 约等于 8/3 * d_model，且是 64 的倍数
            d_ff = int((8 / 3) * d_model)
            d_ff = (d_ff + 63) // 64 * 64
        
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        对输入张量应用 SwiGLU 前馈网络。

        Args:
            x: 输入张量，形状为 (..., d_model)。

        Returns:
            输出张量，形状为 (..., d_model)。
        """
        # FFN(x) = W2(SiLU(W1x) ⊙ W3x)
        gate = self.w3(x)
        hidden = silu(self.w1(x))
        return self.w2(hidden * gate)