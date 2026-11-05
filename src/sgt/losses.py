"""SGT 损失函数：Eq (8)–(13) 与效用损失 L_U（SPEC §3, A2/A3/A4/A5/A6/A16）。

约定：序列张量 `[B, T, d]`；`log_sigma2` 为逐维对数方差（`sigma2 = exp(log_sigma2)`）；
`mask` 为 `[B, T]` bool，True = 有效 token。所有函数不修改输入，返回 dtype 跟随
输入浮点提升。
"""

from __future__ import annotations

import torch

__all__ = [
    "mi_loss",
    "abs_cos_loss",
    "cos_sim_loss",
    "median_norm_loss",
    "comp_entropy_loss",
    "obfuscation_loss",
    "utility_loss",
]

# full 模式单块 [N, C, d] 的元素预算（与 theory.gmm_log_prob 的 2e6 同量级），
# 用于把“按 x_prime token 分块累加”的临时张量内存限制在约 48 MB（float64）。
_FULL_CHUNK_BUDGET = 2_000_000
# 余弦分母下限：只在 ‖x‖·‖x̃‖ → 0 时生效，避免 0/0 产生 NaN。
_COS_DENOM_EPS = 1e-12


def _promoted_float_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """输入浮点 dtype 的提升；全非浮点时退回默认 dtype。"""
    dtype = None
    for t in tensors:
        if t.is_floating_point():
            dtype = t.dtype if dtype is None else torch.promote_types(dtype, t.dtype)
    return torch.get_default_dtype() if dtype is None else dtype


def _masked_mean(values: torch.Tensor, mask) -> torch.Tensor:
    """对有效位置取均值；mask=None 表示全有效，空选择抛错而不是静默返回 NaN。"""
    if mask is None:
        if values.numel() == 0:
            raise ValueError("输入为空，无法归约")
        return values.mean()
    m = torch.as_tensor(mask).to(device=values.device).bool()
    if tuple(m.shape) != tuple(values.shape):
        raise ValueError(f"mask 形状必须为 {tuple(values.shape)}，实际 {tuple(m.shape)}")
    valid = values[m]
    if valid.numel() == 0:
        raise ValueError("mask 未选中任何有效 token")
    return valid.mean()


def _flat_mask(mask, prefix_shape, *, name: str, device: torch.device) -> torch.Tensor:
    """把 [B, T] 掩码校验后展平成 [B*T]。"""
    m = torch.as_tensor(mask)
    if tuple(m.shape) != tuple(prefix_shape):
        raise ValueError(f"{name} 形状必须为 {tuple(prefix_shape)}，实际 {tuple(m.shape)}")
    return m.to(device=device).bool().reshape(-1)


def _validate_mask_shape(mask, shape) -> None:
    m = torch.as_tensor(mask)
    if tuple(m.shape) != tuple(shape):
        raise ValueError(f"mask 形状必须为 {tuple(shape)}，实际 {tuple(m.shape)}")


def mi_loss(
    x_tilde,
    x_prime,
    mu_prime,
    log_sigma2_prime,
    log_sigma2,
    *,
    mask=None,
    mask_prime=None,
    mode="pair",
    half_factor=False,
) -> torch.Tensor:
    """Eq (8)：Σ_j[logσ²_{ℓ',j}−logσ²_{i,j}] + Σ_j(x̃_{i,j}−x'_{ℓ',j}−µ'_{ℓ',j})²/σ²_{ℓ',j} 的均值。

    mode="pair"：同下标配对（Alg.1 L6），归一化 1/(B·T)，单一 mask 同时作用于两侧；
    mode="full"：全交叉 (B·T)×(B'·T')，归一化 1/(B·T·B'·T')，两侧各自掩码。
    half_factor=True 整体乘 0.5（Eq (7) 展开的 1/2 因子，SPEC A3）。
    """
    if mode not in ("pair", "full"):
        raise ValueError(f"mode 只能是 'pair' 或 'full'，实际 {mode!r}")

    x_tilde = torch.as_tensor(x_tilde)
    x_prime = torch.as_tensor(x_prime)
    mu_prime = torch.as_tensor(mu_prime)
    log_sigma2_prime = torch.as_tensor(log_sigma2_prime)
    log_sigma2 = torch.as_tensor(log_sigma2)

    if x_tilde.ndim < 1:
        raise ValueError("x_tilde 至少 1 维（最后一维为特征维 d）")
    if x_tilde.shape != log_sigma2.shape:
        raise ValueError(
            f"x_tilde/log_sigma2 形状必须一致：{tuple(x_tilde.shape)} vs {tuple(log_sigma2.shape)}"
        )
    if not (x_prime.shape == mu_prime.shape == log_sigma2_prime.shape):
        raise ValueError(
            "x_prime/mu_prime/log_sigma2_prime 形状必须一致："
            f"{tuple(x_prime.shape)}/{tuple(mu_prime.shape)}/{tuple(log_sigma2_prime.shape)}"
        )
    if x_prime.shape[-1] != x_tilde.shape[-1]:
        raise ValueError(
            f"两侧特征维 d 必须一致：{x_tilde.shape[-1]} vs {x_prime.shape[-1]}"
        )

    dtype = _promoted_float_dtype(
        x_tilde, x_prime, mu_prime, log_sigma2_prime, log_sigma2
    )
    device = x_tilde.device
    x_tilde = x_tilde.to(device=device, dtype=dtype)
    x_prime = x_prime.to(device=device, dtype=dtype)
    mu_prime = mu_prime.to(device=device, dtype=dtype)
    log_sigma2 = log_sigma2.to(device=device, dtype=dtype)
    log_sigma2_prime = log_sigma2_prime.to(device=device, dtype=dtype)

    d = x_tilde.shape[-1]
    scale = 0.5 if half_factor else 1.0

    if mode == "pair":
        if x_prime.shape != x_tilde.shape:
            raise ValueError(
                "mode='pair' 要求 x_tilde 与 x_prime 同形，实际 "
                f"{tuple(x_tilde.shape)} vs {tuple(x_prime.shape)}"
            )
        if mask_prime is not None:
            raise ValueError(
                "mode='pair' 只接受单一 mask（同下标配对时两侧共享掩码），"
                "请改用 mode='full' 或删除 mask_prime"
            )
        flat_tilde = x_tilde.reshape(-1, d)
        flat_prime = x_prime.reshape(-1, d)
        flat_mu = mu_prime.reshape(-1, d)
        flat_lv = log_sigma2.reshape(-1, d)
        flat_lvp = log_sigma2_prime.reshape(-1, d)

        logdet = (flat_lvp - flat_lv).sum(dim=-1)
        residuals = flat_tilde - flat_prime - flat_mu
        mahalanobis = (residuals.square() * torch.exp(-flat_lvp)).sum(dim=-1)
        per_pair = logdet + mahalanobis

        if mask is not None:
            keep = _flat_mask(
                mask, x_tilde.shape[:-1], name="mask", device=device
            )
            per_pair = per_pair[keep]
        if per_pair.numel() == 0:
            raise ValueError("没有有效配对参与 MI 计算")
        return scale * per_pair.mean()

    # mode == "full"
    flat_tilde = x_tilde.reshape(-1, d)
    flat_lv = log_sigma2.reshape(-1, d)
    if mask is not None:
        keep = _flat_mask(mask, x_tilde.shape[:-1], name="mask", device=device)
        flat_tilde = flat_tilde[keep]
        flat_lv = flat_lv[keep]

    flat_prime = (x_prime + mu_prime).reshape(-1, d)
    flat_lvp = log_sigma2_prime.reshape(-1, d)
    if mask_prime is not None:
        keep_prime = _flat_mask(
            mask_prime, x_prime.shape[:-1], name="mask_prime", device=device
        )
        flat_prime = flat_prime[keep_prime]
        flat_lvp = flat_lvp[keep_prime]

    n_base = flat_tilde.shape[0]
    n_prime = flat_prime.shape[0]
    if n_base == 0 or n_prime == 0:
        raise ValueError("没有有效配对参与 MI 计算")

    base_logdet = flat_lv.sum(dim=-1)
    prime_logdet = flat_lvp.sum(dim=-1)
    prime_inv_var = torch.exp(-flat_lvp)

    # 按 x_prime 的 token 分块：每块临时 [n_base, C, d]，内存 O(n_base·C·d)。
    chunk = max(1, _FULL_CHUNK_BUDGET // max(1, n_base * d))
    total = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    for start in range(0, n_prime, chunk):
        stop = min(start + chunk, n_prime)
        block = flat_prime[start:stop]
        diff = flat_tilde.unsqueeze(1) - block.unsqueeze(0)
        mahalanobis = (
            diff.square() * prime_inv_var[start:stop].unsqueeze(0)
        ).sum(dim=-1)
        logdet = prime_logdet[start:stop].unsqueeze(0) - base_logdet.unsqueeze(1)
        total = total + (logdet + mahalanobis).sum().to(torch.float64)
        count += n_base * (stop - start)
    return scale * (total / count).to(dtype=dtype)


def _cosine_terms(x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
    """逐 token 有符号余弦，返回 [B, T]（或输入去掉最后一维的形状）。"""
    x = torch.as_tensor(x)
    x_tilde = torch.as_tensor(x_tilde)
    if x.shape != x_tilde.shape:
        raise ValueError(f"x/x_tilde 形状必须一致：{tuple(x.shape)} vs {tuple(x_tilde.shape)}")
    if x.ndim < 1:
        raise ValueError("x 至少 1 维（最后一维为特征维 d）")
    dtype = _promoted_float_dtype(x, x_tilde)
    device = x.device
    x = x.to(device=device, dtype=dtype)
    x_tilde = x_tilde.to(device=device, dtype=dtype)
    numer = (x * x_tilde).sum(dim=-1)
    denom = (x.norm(dim=-1) * x_tilde.norm(dim=-1)).clamp_min(_COS_DENOM_EPS)
    return numer / denom


def abs_cos_loss(x, x_tilde, *, mask=None, reduce=True) -> torch.Tensor:
    """Eq (9)：逐 token |cos(x_t, x̃_t)|；reduce=True 取有效 token 均值，否则返回 [B,T]。"""
    per_token = _cosine_terms(x, x_tilde).abs()
    if not reduce:
        if mask is not None:
            _validate_mask_shape(mask, per_token.shape)
        return per_token
    return _masked_mean(per_token, mask)


def cos_sim_loss(x, x_tilde, *, mask=None, reduce=True) -> torch.Tensor:
    """Eq (13)：逐 token 有符号余弦；reduce 语义同 abs_cos_loss。"""
    per_token = _cosine_terms(x, x_tilde)
    if not reduce:
        if mask is not None:
            _validate_mask_shape(mask, per_token.shape)
        return per_token
    return _masked_mean(per_token, mask)


def median_norm_loss(x_plus_mu, target_norm, *, mask=None) -> torch.Tensor:
    """Eq (10)：逐 token | ‖(x+µ)_t‖₂ − T | 的有效均值（T 为标量中位数范数，SPEC A6）。"""
    x_plus_mu = torch.as_tensor(x_plus_mu)
    if x_plus_mu.ndim < 1:
        raise ValueError("x_plus_mu 至少 1 维（最后一维为特征维 d）")
    target = torch.as_tensor(target_norm)
    if target.numel() != 1:
        raise ValueError(f"target_norm 必须是标量，实际形状 {tuple(target.shape)}")
    dtype = _promoted_float_dtype(x_plus_mu, target)
    device = x_plus_mu.device
    x_plus_mu = x_plus_mu.to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype).reshape(())
    per_token = (x_plus_mu.norm(dim=-1) - target).abs()
    return _masked_mean(per_token, mask)


def comp_entropy_loss(log_sigma2, *, mask=None) -> torch.Tensor:
    """Eq (12)（SPEC A5 修正）：逐 token −Σ_j logσ²_{t,j} 的有效均值。"""
    log_sigma2 = torch.as_tensor(log_sigma2)
    if log_sigma2.ndim < 1:
        raise ValueError("log_sigma2 至少 1 维（最后一维为特征维 d）")
    dtype = _promoted_float_dtype(log_sigma2)
    log_sigma2 = log_sigma2.to(dtype=dtype)
    per_token = -log_sigma2.sum(dim=-1)
    return _masked_mean(per_token, mask)


def obfuscation_loss(*, mi=None, acs=None, mnp=None, alphas=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """Eq (11)：α₁·L^MI + α₂·L^ACS + α₃·L^MNP；None 项按 0 处理。"""
    if len(alphas) != 3:
        raise ValueError(f"alphas 必须是长度 3 的序列，实际 {len(alphas)}")
    terms = (mi, acs, mnp)
    tensors = [torch.as_tensor(t) for t in terms if t is not None]
    if not tensors:
        return torch.zeros((), dtype=torch.get_default_dtype())
    dtype = _promoted_float_dtype(*tensors)
    device = tensors[0].device
    total = torch.zeros((), dtype=dtype, device=device)
    for alpha, term in zip(alphas, terms):
        if term is None:
            continue
        value = torch.as_tensor(term)
        if value.numel() != 1:
            raise ValueError(f"损失项必须是标量，实际形状 {tuple(value.shape)}")
        total = total + float(alpha) * value.to(device=device, dtype=dtype).reshape(())
    return total


def utility_loss(logits_clean, logits_sgt, *, attention_mask=None, temperature=1.0) -> torch.Tensor:
    """L_U（SPEC §3, A16）：mean_{有效 token}[ −Σ_v softmax(z_clean/τ)_v · log_softmax(z_sgt/τ)_v ]。

    输入是原始 logits [B, T, V]；temperature 缩放 logits 后做 softmax/log_softmax。
    """
    logits_clean = torch.as_tensor(logits_clean)
    logits_sgt = torch.as_tensor(logits_sgt)
    if logits_clean.shape != logits_sgt.shape:
        raise ValueError(
            f"logits_clean/logits_sgt 形状必须一致："
            f"{tuple(logits_clean.shape)} vs {tuple(logits_sgt.shape)}"
        )
    if logits_clean.ndim < 2:
        raise ValueError("logits 至少 2 维 [..., T, V]")
    tau = float(temperature)
    if not tau > 0.0:
        raise ValueError(f"temperature 必须为正，实际 {temperature}")

    dtype = _promoted_float_dtype(logits_clean, logits_sgt)
    device = logits_clean.device
    z_clean = logits_clean.to(device=device, dtype=dtype) / tau
    z_sgt = logits_sgt.to(device=device, dtype=dtype) / tau
    log_p_sgt = torch.log_softmax(z_sgt, dim=-1)
    p_clean = torch.softmax(z_clean, dim=-1)
    per_token = -(p_clean * log_p_sgt).sum(dim=-1)
    return _masked_mean(per_token, attention_mask)
