# src/model/attention.py
import math
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor

from src.model.embedding import RotaryPositionalEmbedding
from src.model.linear import Linear
from src.nn_utils import softmax

def scaled_dot_product_attention(
    q: Float[Tensor, "... queries d_k"],
    k: Float[Tensor, "... keys d_k"],
    v: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    缩放点积注意力的实现。
    遵循 cs336_spring2025_assignment1_basics.pdf 3.5.4 节的规范。
    """
    d_k = q.size(-1)
    
    # 计算注意力分数: (Q * K^T) / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    # 计算注意力权重
    attn_weights = softmax(scores, dim=-1)

    # 应用注意力权重到 V
    output = torch.matmul(attn_weights, v)
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力机制的实现。
    遵循 cs336_spring2025_assignment1_basics.pdf 3.5.5 节的规范。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化多头自注意力模块。

        Args:
            d_model: 模型的隐藏维度。
            num_heads: 注意力头的数量。
            rope: 旋转位置嵌入模块。
            device: 参数所在的设备。
            dtype: 参数的数据类型。
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope
        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

        self.register_buffer("causal_mask", None, persistent=False)

    def get_causal_mask(self, seq_len: int, device: torch.device) -> Bool[Tensor, "seq_len seq_len"]:
        """创建或获取因果掩码"""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self.causal_mask = ~mask
        return self.causal_mask[:seq_len, :seq_len]

    def forward(
        self, 
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        多头自注意力的前向传播。

        Args:
            x: 输入张量。
            token_positions: token 在序列中的位置。

        Returns:
            输出张量。
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分为多头
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # 3. 应用 RoPE
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # 4. 创建因果掩码
        causal_mask = self.get_causal_mask(seq_len, x.device)

        # 5. 缩放点积注意力
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # 6. 合并多头
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

        # 7. 输出投影
        return self.output_proj(attn_output)