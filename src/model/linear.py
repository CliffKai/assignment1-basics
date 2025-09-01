# src/model/linear.py
import math
import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor

class Linear(nn.Module):
    """
    自定义的线性变换模块，不含偏置项。
    遵循 cs336_spring2025_assignment1_basics.pdf 3.4.2 节的规范。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化线性层。

        Args:
            in_features: int, 输入特征的维度。
            out_features: int, 输出特征的维度。
            device: torch.device | None, 参数所在的设备。
            dtype: torch.dtype | None, 参数的数据类型。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 创建权重参数
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化权重。
        遵循 cs336_spring2025_assignment1_basics.pdf 3.4.1 节的初始化策略。
        使用截断正态分布初始化。
        """
        # 计算标准差
        std = math.sqrt(2 / (self.in_features + self.out_features))
        # 使用截断正态分布进行初始化
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        对输入应用线性变换。

        Args:
            x: 输入张量，形状为 (..., in_features)。

        Returns:
            输出张量，形状为 (..., out_features)。
        """
        # 使用 einsum 实现矩阵乘法，可读性强且能处理任意批处理维度
        return einsum(
            x, self.weight, "... d_in, d_out d_in -> ... d_out"
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"