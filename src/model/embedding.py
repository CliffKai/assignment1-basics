# src/model/embedding.py
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange

class Embedding(nn.Module):
    """
    自定义的词嵌入模块。
    遵循 cs336_spring2025_assignment1_basics.pdf 3.4.3 节的规范。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化词嵌入层。

        Args:
            num_embeddings: int, 词汇表大小。
            embedding_dim: int, 嵌入向量的维度 (d_model)。
            device: torch.device | None, 参数所在的设备。
            dtype: torch.dtype | None, 参数的数据类型。
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化嵌入矩阵。
        遵循 cs336_spring2025_assignment1_basics.pdf 3.4.1 节的初始化策略。
        """
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        """
        根据 token_ids 查找嵌入向量。

        Args:
            token_ids: 输入的 token ID 张量。

        Returns:
            对应的嵌入向量张量。
        """
        return self.weight[token_ids]


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置嵌入 (RoPE) 的实现。
    遵循 cs336_spring2025_assignment1_basics.pdf 3.5.3 节的规范。
    """

    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
    ):
        """
        初始化 RoPE 模块并预计算 sin/cos 值。

        Args:
            d_k: 查询和键向量的维度。
            max_seq_len: 支持的最大序列长度。
            theta: RoPE 的 theta 参数。
            device: 缓冲区所在的设备。
        """
        super().__init__()
        # 确保 d_k 是偶数
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for Rotary Positional Embedding.")
        
        # 计算频率
        # freqs_cis 的形状将是 (max_seq_len, d_k / 2)
        # freqs 的形状是 (d_k / 2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        # t 的形状是 (max_seq_len)
        t = torch.arange(max_seq_len, device=device)
        
        # freqs 的形状是 (max_seq_len, d_k / 2)
        freqs = torch.outer(t, freqs)
        
        # freqs_cis 的形状是 (max_seq_len, d_k / 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # 注册为缓冲区，不参与模型训练
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_k"], 
        token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        对输入张量应用 RoPE。

        Args:
            x: 输入张量 (查询或键)，形状为 (..., seq_len, d_k)。
            token_positions: token 在序列中的位置，形状为 (..., seq_len)。

        Returns:
            应用 RoPE 后的张量，形状与输入相同。
        """
        # 将 x 视为复数 (..., seq_len, d_k/2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # 根据 token_positions 获取对应的频率
        # freqs_cis 的形状是 (..., seq_len, d_k/2)
        freqs_cis = self.freqs_cis[token_positions]
        
        # 扩展维度以匹配 x_complex 的批处理维度
        if x_complex.dim() == 4: # 典型的 (batch, head, seq, dim)
            freqs_cis = freqs_cis.unsqueeze(1)

        # 应用旋转: (a + ib) * (cos + isin) = (a*cos - b*sin) + i*(a*sin + b*cos)
        x_rotated = x_complex * freqs_cis
        
        # 将复数转换回实数张量
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)

        return x_out.type_as(x)