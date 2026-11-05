"""SGT 采样器（Eq (1)）。

    x̃ = x + μ + σ ⊙ u,   u ~ N(0, I)（逐元素 iid）

`sigma` 是标准差（不是方差）。u 可由调用方显式给出，用于门禁逐元素复算。
"""

from __future__ import annotations

import torch


def sample_sgt(x, mu, sigma, *, u=None, generator=None) -> torch.Tensor:
    """按 Eq (1) 采样 x̃，输入形状均为 [B, T, d]（或任意同形状张量）。"""
    x = torch.as_tensor(x)
    mu = torch.as_tensor(mu)
    sigma = torch.as_tensor(sigma)
    if x.shape != mu.shape or x.shape != sigma.shape:
        raise ValueError(
            f"x/mu/sigma 形状必须一致，实际 {tuple(x.shape)}/{tuple(mu.shape)}/{tuple(sigma.shape)}"
        )
    if u is None:
        u = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    else:
        u = torch.as_tensor(u)
        if u.shape != x.shape:
            raise ValueError(f"u 形状必须为 {tuple(x.shape)}，实际 {tuple(u.shape)}")
    return x + mu + sigma * u
