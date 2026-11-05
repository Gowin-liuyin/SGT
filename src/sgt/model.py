"""SGT 模型 M3：序列依赖的均值/方差估计器 µθ / Σθ（SPEC §6, A1）。

论文 §3 只写 "The sequence dependent mean and variance estimator functions are
modeled as transformers"，未给出层数、宽度、位置编码等细节。本实现按 SPEC A1 的
决策与 M3 任务单：

    h      = W_in x + p_t                                 # 线性投影 + 可学习绝对位置编码
    h      = TransformerEncoder(h, key_padding_mask)      # n_layers 层，双向（非因果），pre-LN
    (mu, log_sigma2) = split(W_out h + b_out, d)          # 各 [B, T, d]

Eq (1) 的采样 `x̃ = x + µ + σ ⊙ u` 由 :meth:`SGTModel.sample` 完成；`sigma` 是标准差
（见 :func:`sgt.sampling.sample_sgt`）。`log_sigma2` 为逐维对数方差，`sample` 只在下游
施加 `sigma_min` 下限，`forward` 返回未截断的原始值（训练损失需要原始量）。

约定：
- 输入 `x` 为 `[B, T, d]`，`attention_mask` 为 `[B, T]`，True = 有效 token；掩码实现为
  注意力的 key padding mask，因此有效位置的输出不依赖 padding 位置的内容。
- 参数 dtype 默认 float32；模型与输入必须同 dtype（float64 输入请先 `model.double()`，
  或先 `torch.set_default_dtype(torch.float64)` 再构造），forward 不做隐式 dtype 转换。
- forward 不修改输入张量，也不含随机性（随机采样只在 :meth:`sample` 内发生）。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .sampling import sample_sgt

__all__ = ["SGTModel"]


class SGTModel(nn.Module):
    """序列依赖的 µθ / log Σθ 估计器：双向 pre-LN Transformer + 零初始化输出头。

    Args:
        d: embedding 维度。
        hidden: Transformer 隐藏维度（`d -> hidden` 线性投影）。
        n_layers: 双向 TransformerEncoderLayer 层数。
        n_heads: 多头注意力头数（`hidden` 必须整除）。
        ffn_mult: FFN 隐层倍数，`dim_feedforward = ffn_mult * hidden`。
        max_len: 支持的最大序列长度（可学习绝对位置编码行数）。
        sigma_init: 初始化时 `sigma ≡ sigma_init`（`log_sigma2` 头偏置 = log(sigma_init²)）。
        sigma_min: `sample` 中 sigma 的下限。
        dropout: Transformer 内 dropout 概率（默认 0，即确定性）。

    Note:
        输出头（`head`）在初始化时权重为零：`mu ≡ 0`、`sigma ≡ sigma_init`。因此**初始**
        前向对输入不敏感；依赖性的验证需在头权重非零后（训练后或测试中注入）进行。
    """

    def __init__(
        self,
        d: int,
        *,
        hidden: int = 512,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_mult: int = 4,
        max_len: int = 512,
        sigma_init: float = 1.0,
        sigma_min: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = self._positive_int("d", d)
        hidden = self._positive_int("hidden", hidden)
        n_layers = self._positive_int("n_layers", n_layers)
        n_heads = self._positive_int("n_heads", n_heads)
        ffn_mult = self._positive_int("ffn_mult", ffn_mult)
        max_len = self._positive_int("max_len", max_len)
        if hidden % n_heads != 0:
            raise ValueError(f"hidden={hidden} 必须能被 n_heads={n_heads} 整除")
        if not (math.isfinite(sigma_init) and sigma_init > 0.0):
            raise ValueError(f"sigma_init 必须为正的有限值，实际 {sigma_init!r}")
        if not (math.isfinite(sigma_min) and sigma_min > 0.0):
            raise ValueError(f"sigma_min 必须为正的有限值，实际 {sigma_min!r}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout 必须在 [0, 1) 内，实际 {dropout!r}")

        self.d = d
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.max_len = max_len
        self.sigma_init = float(sigma_init)
        self.sigma_min = float(sigma_min)
        self.log_sigma_min2 = math.log(self.sigma_min**2)

        self.input_proj = nn.Linear(d, hidden)
        # 可学习绝对位置编码（论文未提，SPEC A1 备注）；默认初始化同 nn.Embedding。
        self.pos_emb = nn.Embedding(max_len, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=ffn_mult * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        # enable_nested_tensor=False：无 mask / 有 mask 两条路径使用同一实现，避免
        # 因 fast path 造成的掩码前后数值差异（padding 不变性测试要求逐位可比）。
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(hidden, 2 * d)
        self._reset_output_head()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    @staticmethod
    def _positive_int(name: str, value) -> int:
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} 必须为正整数，实际 {value!r}")
        return int(value)

    @torch.no_grad()
    def _reset_output_head(self) -> None:
        """输出头零初始化：mu 头（权重与偏置）为零，log_sigma2 头权重为零、偏置为常数。"""
        self.head.weight.zero_()
        self.head.bias.zero_()
        self.head.bias[self.d :].fill_(math.log(self.sigma_init**2))

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 `(mu, log_sigma2)`，均为 `[B, T, d]`。

        Args:
            x: `[B, T, d]` 输入 embedding 序列。
            attention_mask: `[B, T]`，True = 有效 token；None 表示全部有效。

        Raises:
            ValueError: 形状不合法、`T > max_len`、mask 每行无有效 token、dtype 与参数不一致。
            TypeError: 输入不是浮点张量。
        """
        x = torch.as_tensor(x)
        if x.dim() != 3:
            raise ValueError(f"x 必须为 [B, T, d]，实际 {tuple(x.shape)}")
        batch, seq_len, d = x.shape
        if d != self.d:
            raise ValueError(f"x 的最后一维必须为 d={self.d}，实际 {d}")
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 T={seq_len} 超过 max_len={self.max_len}")
        if not x.is_floating_point():
            raise TypeError(f"x 必须为浮点张量，实际 dtype={x.dtype}")
        param_dtype = self.head.weight.dtype
        if x.dtype != param_dtype:
            raise ValueError(
                f"输入 dtype {x.dtype} 与模型参数 dtype {param_dtype} 不一致；"
                "float64 输入请先调用 model.double()（forward 不隐式转换以避免降精度）"
            )

        key_padding_mask = None
        if attention_mask is not None:
            mask = torch.as_tensor(attention_mask).to(device=x.device, dtype=torch.bool)
            if tuple(mask.shape) != (batch, seq_len):
                raise ValueError(
                    f"attention_mask 形状必须为 {(batch, seq_len)}，实际 {tuple(mask.shape)}"
                )
            if not bool(mask.any(dim=1).all()):
                raise ValueError("attention_mask 每行至少需要一个 True（有效 token）")
            key_padding_mask = ~mask

        # 投影 + 位置编码（[1, T, hidden] 广播；位置编码对所有位置施加，padding 位置
        # 只能通过 key padding mask 被屏蔽，不能污染有效位置）。
        h = self.input_proj(x) + self.pos_emb.weight[:seq_len]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        out = self.head(h)
        return out[..., : self.d], out[..., self.d :]

    # ------------------------------------------------------------------ #
    # 采样（Eq (1)）
    # ------------------------------------------------------------------ #
    def sample(
        self,
        x: torch.Tensor,
        *,
        u: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """按 Eq (1) 采样：`x̃ = x + mu + exp(0.5·clamp(log_sigma2, min=log(sigma_min²))) ⊙ u`。"""
        mu, log_sigma2 = self.forward(x)
        sigma = torch.exp(0.5 * log_sigma2.clamp(min=self.log_sigma_min2))
        return sample_sgt(x, mu, sigma, u=u, generator=generator)
