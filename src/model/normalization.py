# src/model/normalization.py
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

class RMSNorm(nn.Module):
    """
    均方根层归一化 (RMSNorm) 的实现。
    遵循 cs336_spring2025_assignment1_basics.pdf 3.5.1 节的规范。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化 RMSNorm 模块。

        Args:
            d_model: 模型的隐藏维度。
            eps: 为防止除以零而添加的小值。
            device: 参数所在的设备。
            dtype: 参数的数据类型。
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        对输入张量应用 RMSNorm。

        Args:
            x: 输入张量，形状为 (..., d_model)。

        Returns:
            归一化后的张量，形状与输入相同。
        """
        in_dtype = x.dtype
        # 为保证数值稳定性，上浮到 float32
        x = x.to(torch.float32)

        # 计算 RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 归一化并应用可学习的 gain 参数
        normalized_x = x * rms * self.weight

        # 转换回原始数据类型
        return normalized_x.to(in_dtype)