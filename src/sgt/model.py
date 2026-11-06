r"""SGT 模型 M3：序列依赖的均值/方差估计器 µθ / Σθ（SPEC §6, A1）。

论文 §3 只写 "The sequence dependent mean and variance estimator functions are
modeled as transformers"，未给出层数、宽度、位置编码等细节。本实现按 SPEC A1 的
决策与 M3 任务单：

    h      = W_in x + p_t                                 # p_t 由 pos_encoding 决定（A1）
    h      = TransformerEncoder(h, key_padding_mask)      # n_layers 层，双向（非因果），pre-LN
    h      = LayerNorm(h)                                 # final_norm=True；A1：论文未给归一化位置
    (mu, log_sigma2) = split(W_out h + b_out, d)          # 各 [B, T, d]（可再按 A21 夹紧）

Eq (1) 的采样 `x̃ = x + µ + σ ⊙ u` 由 :meth:`SGTModel.sample` 完成；`sigma` 是标准差
（见 :func:`sgt.sampling.sample_sgt`）。`log_sigma2` 为逐维对数方差；`forward` 默认返回
未截断的原始值（训练损失需要原始量），`sample` 只在下游施加 `sigma_min` 下限；
给定 `sigma_max` / `sigma_floor` 时 `forward` 额外施加 `log(sigma_max²)` 上界 /
`log(sigma_floor²)` 下界；给定 `mu_scale` 时 µ 走 `M·tanh(raw_mu / M)`（均见下）。

声明性口径（论文未给 / 审计要求）：

- **A1（宽度）**：论文 §3 只说 "modeled as transformers"，未给宽度。默认 `hidden=512`，
  但本类**可配置任意宽度**；审计要求可探测 2048（`n_heads` 必须整除 `hidden`，由构造时
  校验）。宽度 / 层数 / 位置编码都只影响「用哪个 Transformer」，不改变论文公式。
- **A1（位置编码初始化）**：论文未给任何初始化细节。可学习绝对位置编码默认按标准
  Transformer 的嵌入初始化取 `N(0, 0.02²)`（`pos_emb_init_std=0.02`）。`nn.Embedding`
  的原生初始化是 `N(0, 1)`，比 `input_proj` 的输出（量级 ~0.02，见 Llama 的嵌入 std）
  大约 50 倍，会让位置项在 pre-LN 前主导残差流、抑制内容信号；`1.0` 可复现该旧行为用于
  对照。这是**实现选择**（SPEC A1），不改变论文公式。
- **A1（输出头前的最终归一化）**：论文只写 µθ/Σθ "modeled as transformers"，未给归一化
  位置。默认（`final_norm=True`）在编码器输出后、输出头前加一层 `LayerNorm(hidden)`，
  把每个 token 的 hidden 压到单位尺度；否则编码器输出幅值未受约束（实测 |h_i| 可达 ~19），
  输出头的单步更新量 `ΔW·h` 被同比例放大。`final_norm=False` 复现「头直接接编码器输出」
  的旧路径用于对照。这是**实现选择**（SPEC A1），不改变论文公式。
- **A21（Σθ 的有界参数化，对应审计 GAP-3）**：Eq (12) 的 Comp.Entr. 损失
  $-\log|\Sigma_\theta|$ 按字面**无下界**（F31：训练会把 σ 推向 0 而发散），要复现
  Table 1 的该行必须限制参数化范围。构造时传 `sigma_max` 会让 `forward` 把
  `log_sigma2` 夹到 `log(sigma_max²)` 上界，且**返回夹过的值**，因此 `sample()`、训练
  损失、评测看到同一个 σ。这是为了让 Eq (12) 可训练的**声明性实现选择**，不改变论文
  公式；默认 `sigma_max=None` 严格保持 Def 3.1 的无界定义（σ>0 即可）。
- **A21（Σθ 的下界，`sigma_floor`）**：Eq (12) 的 $-\log|\Sigma_\theta|$ 与 Eq (9) 的
  `|cos|` 都偏好 σ→0，实测 `log_sigma2` 会塌到 −3000（评测时 MI 无定义）。构造时传
  `sigma_floor=f>0` 会让 `forward` 把 `log_sigma2` 夹到 `log(f²)` 下限（与 `sigma_max`
  上界对称），且**返回夹过的值**，因此 `sample()`、训练损失、评测看到同一个 σ。这是
  数值稳定性所需的**声明性实现选择**；默认 `sigma_floor=None` 只在 `sample` 内施加
  `sigma_min`，`forward` 返回原始值。
- **A21（µθ 的上界，`mu_scale`）**：Eq (9) 的 `|cos(µ_x, µ_x̃)|` 对 µ 的尺度不变
  （µ → cµ 时 cos 不变），该目标对 µ 幅度无回归力，无界头在 Adam 下会发散（实测
  hidden=2048 时单步 µ 可增 ~1.6、30 步到 1e5）。构造时传 `mu_scale=M>0` 会让 `forward`
  返回 `mu = M·tanh(raw_mu / M)`（逐元素，数学上严格落在 (−M, M)），把 µ 限制在 ±M；
  训练与评测看到同一个 µ。这是**数值稳定性的声明性实现选择**：Eq (1) 的形式
  `x̃ = x + µ + σ⊙u` 不变，只限制 µθ 的参数化范围。默认 `mu_scale=None` 严格保持
  Def 3.1 的无界定义。注意 float32 下 `tanh` 在 `|raw_mu|/M ≳ 9` 时饱和到 ±1，此时
  返回值可能恰为 ±M——那是浮点极限，不是额外的 clamp。
- **A1（位置编码与长度泛化，`pos_encoding`）**：论文未规定位置编码。`"learned"`（默认）
  是可学习绝对位置编码（`max_len` 行，`T > max_len` 报错）；`"sinusoidal"` 是 Vaswani 式
  固定正弦编码（**无参数**，按序列长度动态生成，任意 T，不受 `max_len` 限制）；
  `"none"` 无位置编码。后两种是 A1 的**长度泛化选择**：论文评测 HellaSwag 10-shot 的
  prompt 长度 p50≈810、max≈971 token，而可学习绝对位置编码只能覆盖训练长度（如 256）。
  `max_len` 的 T 上限校验只对 `"learned"` 生效。

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
        max_len: 支持的最大序列长度（`"learned"` 位置编码的行数；只在
            `pos_encoding="learned"` 时作为 T 的上限校验，见 `pos_encoding`）。
        pos_encoding: 位置编码类型（A1；论文未规定）：
            `"learned"`（默认）——可学习绝对位置编码，`max_len` 行，`T > max_len` 报
            `ValueError`；`"sinusoidal"`——Vaswani 式固定正弦编码，**无参数**、按序列长度
            动态生成，任意 T（不受 `max_len` 限制），`hidden` 为奇数时最后一维置 0；
            `"none"`——无位置编码，同样不受 `max_len` 限制。后两种是长度泛化选择（如
            评测 prompt 长度超过训练长度）。不同 `pos_encoding` 的 `state_dict` 键不同
            （`"learned"` 含 `pos_emb.weight`），互不通用。
        sigma_init: 初始化时 `sigma ≡ sigma_init`（`log_sigma2` 头偏置 = log(sigma_init²)）。
        sigma_min: `sample` 中 sigma 的下限。
        sigma_max: Σθ 的上界（`None` = 论文 Def 3.1 的无界定义）。给定后 `forward` 返回的
            `log_sigma2` 被夹到 `log(sigma_max²)`，因此 `sample`/损失/评测共用同一 σ。
            这是 A21 的声明性实现选择（Eq (12) 可训练所必需，见模块 docstring）；
            必须满足 `sigma_max > sigma_min`。
        sigma_floor: Σθ 的下界（`None` = 不施加，只在 `sample` 内用 `sigma_min`）。给定后
            `forward` 返回的 `log_sigma2` 被夹到 `log(sigma_floor²)`（与 `sigma_max` 上界
            对称），因此 `sample`/损失/评测共用同一 σ。这是 A21 的声明性实现选择
            （Eq (12) 的 `-log|Σ|` 与 Eq (9) 的 `|cos|` 都会把 σ 推向 0，见模块
            docstring）；必须满足 `sigma_min <= sigma_floor`，`sigma_max` 给定时还要满足
            `sigma_floor <= sigma_max`。
        mu_scale: µθ 的有界参数化尺度（`None` = 论文 Def 3.1 的无界定义）。给定正数 M 时
            `forward` 返回 `mu = M·tanh(raw_mu / M)`，把 µ 限制在 ±M；因为 Eq (9) 的
            `|cos|` 对 µ 尺度不变，无界头在 Adam 下会发散（A21）。不改变 Eq (1) 的形式。
            必须为正的有限值。
        dropout: Transformer 内 dropout 概率（默认 0，即确定性）。
        pos_emb_init_std: 可学习绝对位置编码的初始化标准差，默认 `0.02`（标准
            Transformer 嵌入初始化；SPEC A1 的实现选择——论文未给初始化）。传 `1.0`
            得到 `nn.Embedding` 的原生 `N(0, 1)` 分布，用于对照（**分布**一致；因重新
            初始化会额外消耗一次 RNG，同 seed 下与旧代码的权重不逐位相同）。
        final_norm: 是否在输出头前加 `LayerNorm(hidden)`（标准初始化 weight=1/bias=0），
            默认 True。论文只写 µθ/Σθ "modeled as transformers"、未给归一化位置，这是
            SPEC A1 的架构细节：无归一化时编码器输出幅值不受约束（实测 |h_i| 可达 ~19），
            输出头的单步更新 `ΔW·h` 被同比例放大；最终 LN 把 |h_i| 压到 ~1，单步 Δµ
            缩小约 25×。False 复现旧路径（`h = encoder_out`）用于对照。

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
        pos_encoding: str = "learned",
        sigma_init: float = 1.0,
        sigma_min: float = 1e-6,
        sigma_max: float | None = None,
        sigma_floor: float | None = None,
        mu_scale: float | None = None,
        dropout: float = 0.0,
        pos_emb_init_std: float = 0.02,
        final_norm: bool = True,
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
        if sigma_max is not None and not (
            math.isfinite(sigma_max) and sigma_max > sigma_min
        ):
            raise ValueError(
                f"sigma_max 必须为满足 sigma_max > sigma_min={sigma_min} 的有限值，"
                f"实际 {sigma_max!r}；None 表示无界（论文 Def 3.1）"
            )
        if sigma_floor is not None:
            if not (math.isfinite(sigma_floor) and sigma_floor > 0.0):
                raise ValueError(
                    f"sigma_floor 必须为正的有限值，实际 {sigma_floor!r}；"
                    "None 表示不施加下界"
                )
            if sigma_floor < sigma_min:
                raise ValueError(
                    f"sigma_floor={sigma_floor} 必须 >= sigma_min={sigma_min}"
                )
            if sigma_max is not None and sigma_floor > sigma_max:
                raise ValueError(
                    f"sigma_floor={sigma_floor} 必须 <= sigma_max={sigma_max}"
                )
        if mu_scale is not None and not (
            math.isfinite(mu_scale) and mu_scale > 0.0
        ):
            raise ValueError(
                f"mu_scale 必须为正的有限值，实际 {mu_scale!r}；"
                "None 表示无界（论文 Def 3.1）"
            )
        if pos_encoding not in ("learned", "sinusoidal", "none"):
            raise ValueError(
                f"pos_encoding 必须为 'learned' / 'sinusoidal' / 'none' 之一，"
                f"实际 {pos_encoding!r}"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout 必须在 [0, 1) 内，实际 {dropout!r}")
        if not (math.isfinite(pos_emb_init_std) and pos_emb_init_std >= 0.0):
            raise ValueError(
                f"pos_emb_init_std 必须为非负有限值，实际 {pos_emb_init_std!r}"
            )

        self.d = d
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.max_len = max_len
        self.pos_encoding = pos_encoding
        self.sigma_init = float(sigma_init)
        self.sigma_min = float(sigma_min)
        self.log_sigma_min2 = math.log(self.sigma_min**2)
        # A21：Σθ 的上界（None = 无界）。log 形式供 forward 的 clamp 使用。
        self.sigma_max = None if sigma_max is None else float(sigma_max)
        self.log_sigma_max2 = (
            None if sigma_max is None else math.log(self.sigma_max**2)
        )
        # A21：Σθ 的下界（None = 不施加），与上界对称，log 形式供 forward 的 clamp 使用。
        self.sigma_floor = None if sigma_floor is None else float(sigma_floor)
        self.log_sigma_floor2 = (
            None if sigma_floor is None else math.log(self.sigma_floor**2)
        )
        # A21：µθ 的有界尺度（None = 无界）；给定后 mu = M·tanh(raw_mu / M)。
        self.mu_scale = None if mu_scale is None else float(mu_scale)

        self.input_proj = nn.Linear(d, hidden)
        # 位置编码（论文未提，SPEC A1）：learned = 可学习绝对位置编码；sinusoidal / none
        # 无参数（sinusoidal 在 forward 内按 T 动态生成，因此不受 max_len 限制）。
        self.pos_emb_init_std = float(pos_emb_init_std)
        if pos_encoding == "learned":
            # 默认按标准 Transformer 嵌入初始化取 N(0, 0.02²)，避免 nn.Embedding 原生
            # N(0, 1) 相对 input_proj 输出（量级 ~0.02）过大、在 pre-LN 前主导残差流。
            # pos_emb_init_std=1.0 复现旧行为。
            self.pos_emb = nn.Embedding(max_len, hidden)
            nn.init.normal_(self.pos_emb.weight, std=self.pos_emb_init_std)
        else:
            self.pos_emb = None
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
        # A1（论文未给归一化位置）：输出头前的最终 LayerNorm，标准初始化 weight=1/bias=0。
        self.final_norm = nn.LayerNorm(hidden) if final_norm else None
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
    # 位置编码（A1：论文未规定）
    # ------------------------------------------------------------------ #
    def _sinusoidal_pos_encoding(
        self,
        seq_len: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        r"""Vaswani 式固定正弦位置编码，`[T, hidden]`，无参数、按 `seq_len` 动态生成。

        .. math::

            pe[p, 2i]   = \sin(p / 10000^{2i/hidden}) \\
            pe[p, 2i+1] = \cos(p / 10000^{2i/hidden})

        其中 `i = 0..hidden//2-1`。`hidden` 为奇数时最后一维无法配对，**置 0**
        （见类 docstring：不报错，保持任意 `hidden` 可用）。因此该模式下 `T` 不受
        `max_len` 限制（A1 的长度泛化选择）。
        """
        if dtype is None:
            dtype = self.head.weight.dtype
        if device is None:
            device = self.head.weight.device
        pe = torch.zeros(seq_len, self.hidden, device=device, dtype=dtype)
        n_pairs = self.hidden // 2
        if n_pairs > 0:
            pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
            # 1 / 10000^(2i/hidden) = exp(-log(10000) · 2i / hidden)
            div = torch.exp(
                torch.arange(n_pairs, device=device, dtype=dtype)
                * (-2.0 * math.log(10000.0) / self.hidden)
            )
            pe[:, 0 : 2 * n_pairs : 2] = torch.sin(pos * div)
            pe[:, 1 : 2 * n_pairs : 2] = torch.cos(pos * div)
        return pe

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 `(mu, log_sigma2)`，均为 `[B, T, d]`。

        给定 `sigma_max` / `sigma_floor` 时 `log_sigma2` 已被夹到 `[log(f²), log(M²)]`
        （A21），返回的就是夹过的值，因此 `sample()`、训练损失、评测看到同一个 σ；两者
        均为 `None` 时返回原始值（论文 Def 3.1 的无界定义）。给定 `mu_scale` 时 `mu` 为
        `M·tanh(raw_mu / M)`（A21），不会超出 ±M。

        Args:
            x: `[B, T, d]` 输入 embedding 序列。
            attention_mask: `[B, T]`，True = 有效 token；None 表示全部有效。

        Raises:
            ValueError: 形状不合法、`pos_encoding="learned"` 时 `T > max_len`、mask 每行
                无有效 token、dtype 与参数不一致。
            TypeError: 输入不是浮点张量。
        """
        x = torch.as_tensor(x)
        if x.dim() != 3:
            raise ValueError(f"x 必须为 [B, T, d]，实际 {tuple(x.shape)}")
        batch, seq_len, d = x.shape
        if d != self.d:
            raise ValueError(f"x 的最后一维必须为 d={self.d}，实际 {d}")
        if self.pos_encoding == "learned" and seq_len > self.max_len:
            # A1：只有可学习绝对位置编码受 max_len 限制；sinusoidal / none 可长度泛化。
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
        h = self.input_proj(x)
        if self.pos_encoding == "learned":
            h = h + self.pos_emb.weight[:seq_len]
        elif self.pos_encoding == "sinusoidal":
            # A1：无参数、按 T 动态生成，因此 T > max_len 也可前向。
            h = h + self._sinusoidal_pos_encoding(
                seq_len, device=x.device, dtype=h.dtype
            )
        # pos_encoding == "none"：不施加任何位置信息（A1 的消融 / 长度泛化选择）。
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        if self.final_norm is not None:
            # A1：输出头前的最终归一化（final_norm=False 时保持旧路径 h = encoder_out）。
            h = self.final_norm(h)
        out = self.head(h)
        mu = out[..., : self.d]
        log_sigma2 = out[..., self.d :]
        if self.log_sigma_max2 is not None:
            # A21：有界参数化。clamp 的返回值用于一切下游（sample/损失/评测），
            # 保证它们看到同一个 σ；不改变论文公式，只限制参数化范围。
            log_sigma2 = log_sigma2.clamp(max=self.log_sigma_max2)
        if self.log_sigma_floor2 is not None:
            # A21：Σθ 的下界（与 sigma_max 上界对称；sigma_floor 已校验 <= sigma_max）。
            log_sigma2 = log_sigma2.clamp(min=self.log_sigma_floor2)
        if self.mu_scale is not None:
            # A21：µθ 的有界参数化。Eq (9) 的 |cos| 对 µ 尺度不变，无界头在 Adam 下
            # 会发散；M·tanh(raw/M) 数学上严格落在 (−M, M)，不改变 Eq (1) 的形式。
            mu = self.mu_scale * torch.tanh(mu / self.mu_scale)
        return mu, log_sigma2

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
        """按 Eq (1) 采样：`x̃ = x + mu + exp(0.5·clamp(log_sigma2, min=log(sigma_min²))) ⊙ u`。

        `log_sigma2` 来自 :meth:`forward`，因此给定 `sigma_max` / `sigma_floor` 时它已含
        `log(sigma_max²)` 上界 / `log(sigma_floor²)` 下界——`sample` 与损失、评测使用的是
        同一个（夹过的）σ（A21）。`mu_scale` 同理：采样用的 µ 就是 `forward` 返回的有界值。
        """
        mu, log_sigma2 = self.forward(x)
        sigma = torch.exp(0.5 * log_sigma2.clamp(min=self.log_sigma_min2))
        return sample_sgt(x, mu, sigma, u=u, generator=generator)
