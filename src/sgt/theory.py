"""SGT 理论核心：PAC-Adv 反解与 GMM 熵/MI（Eq (4)–(7)）。

约定：`log_vars` 是逐维对数方差；标量估计内部统一用 float64。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import quad

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2PIE = math.log(2.0 * math.pi * math.e)


def delta_kl(delta, delta0) -> float:
    """Eq (4) 左端：Δ_KL(δ) = δ log(δ/δ0) + (1−δ) log((1−δ)/(1−δ0))，nats。"""
    delta = float(delta)
    delta0 = float(delta0)
    # 端点按 0·log 0 = 0 处理，使 Δ_KL(1) = log(1/δ0) 可表示。
    first = 0.0 if delta == 0.0 else delta * math.log(delta / delta0)
    second = (
        0.0
        if delta == 1.0
        else (1.0 - delta) * math.log((1.0 - delta) / (1.0 - delta0))
    )
    return first + second


def invert_pac_adv(mi, n, tol=1e-10) -> float:
    """反解 Eq (4)：返回满足 Δ_KL(δ) ≥ mi 的最小 δ，δ0 = 1/n。"""
    mi = float(mi)
    delta0 = 1.0 / float(n)
    if mi <= 0.0:
        return delta0
    if delta_kl(1.0, delta0) < mi:
        return 1.0
    lo, hi = delta0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if delta_kl(mid, delta0) >= mi:
            hi = mid
        else:
            lo = mid
    return hi


def _validate_log_vars(means: torch.Tensor, log_vars: torch.Tensor) -> None:
    if means.ndim != 2 or log_vars.ndim != 2:
        raise ValueError(f"means/log_vars 必须为 [N, d]，实际 {means.ndim}/{log_vars.ndim} 维")
    if means.shape != log_vars.shape:
        raise ValueError(f"means/log_vars 形状不一致：{tuple(means.shape)} vs {tuple(log_vars.shape)}")


def gmm_log_prob(z, means, log_vars) -> torch.Tensor:
    """逐分量 log 密度：[..., d] × [N, d] -> [..., N]。"""
    z = torch.as_tensor(z)
    means = torch.as_tensor(means)
    log_vars = torch.as_tensor(log_vars)
    _validate_log_vars(means, log_vars)
    if z.ndim < 1 or z.shape[-1] != means.shape[-1]:
        raise ValueError(f"z 最后一维应为 {means.shape[-1]}，实际 {tuple(z.shape)}")
    dtype = torch.promote_types(z.dtype, torch.promote_types(means.dtype, log_vars.dtype))
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    z = z.to(dtype=dtype)
    means = means.to(device=z.device, dtype=dtype)
    log_vars = log_vars.to(device=z.device, dtype=dtype)
    n, d = means.shape
    flat = z.reshape(-1, d)
    if flat.shape[0] == 0:
        return z.new_empty(z.shape[:-1] + (n,))
    # 分块限制 [block, N, d] 临时张量，避免大网格/大分量数 OOM。
    block = max(1, 2_000_000 // max(1, n * d))
    inv_var = torch.exp(-log_vars)
    parts = []
    for start in range(0, flat.shape[0], block):
        sq = (flat[start : start + block].unsqueeze(-2) - means).square()
        parts.append(-0.5 * (sq * inv_var + log_vars + _LOG_2PI).sum(dim=-1))
    log_prob = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return log_prob.reshape(z.shape[:-1] + (n,))


def _log_weights(weights, n: int, *, dtype, device) -> torch.Tensor:
    if weights is None:
        return torch.full((n,), -math.log(n), dtype=dtype, device=device)
    w = torch.as_tensor(weights, dtype=dtype, device=device)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"weights 必须为长度 {n} 的向量，实际 {tuple(w.shape)}")
    return torch.log(w)


def gmm_pdf(z, means, log_vars, weights=None) -> torch.Tensor:
    """混合密度 p(z) = Σ_i w_i N(z; m_i, diag(v_i))；weights 默认均匀。"""
    log_prob = gmm_log_prob(z, means, log_vars)
    log_w = _log_weights(
        weights, log_prob.shape[-1], dtype=log_prob.dtype, device=log_prob.device
    )
    return torch.exp(torch.logsumexp(log_prob + log_w, dim=-1))


def gmm_entropy_upper_bound(weights, log_vars) -> float:
    """Eq (6) 右端：Σ_i w_i · ½ Σ_j log(2πe σ²_ij)。"""
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1).cpu()
    lv = torch.as_tensor(log_vars, dtype=torch.float64).cpu()
    if lv.ndim == 1:
        lv = lv.unsqueeze(-1)
    if lv.ndim != 2:
        raise ValueError(f"log_vars 必须为 [N, d] 或 [N]，实际 {tuple(lv.shape)}")
    if w.numel() != lv.shape[0]:
        raise ValueError(f"weights 长度 {w.numel()} 与分量数 {lv.shape[0]} 不一致")
    return float((w * 0.5 * (lv + _LOG_2PIE).sum(dim=1)).sum())


def cond_entropy(weights, log_vars) -> float:
    """Eq (5) 的 E_X[h(N)]，与 Eq (6) 右端同式：Σ_i w_i · ½ Σ_j log(2πe σ²_ij)。"""
    return gmm_entropy_upper_bound(weights, log_vars)


def _gmm_f64(weights, means, log_vars):
    """校验并转成同设备 float64 的 (w, means, log_vars)。"""
    w = torch.as_tensor(weights)
    means = torch.as_tensor(means)
    log_vars = torch.as_tensor(log_vars)
    _validate_log_vars(means, log_vars)
    w = w.to(device=means.device, dtype=torch.float64)
    if w.ndim != 1 or w.shape[0] != means.shape[0]:
        raise ValueError(f"weights 必须为长度 {means.shape[0]} 的向量，实际 {tuple(w.shape)}")
    return w, means.to(dtype=torch.float64), log_vars.to(dtype=torch.float64)


def _gmm_sample(weights, means, log_vars, n_samples, *, generator=None) -> torch.Tensor:
    """从 GMM 采样（gmm_entropy_mc 与测试共用的采样流程）。"""
    idx = torch.multinomial(weights, n_samples, replacement=True, generator=generator)
    eps = torch.randn(
        n_samples,
        means.shape[1],
        generator=generator,
        dtype=means.dtype,
        device=means.device,
    )
    return means[idx] + torch.exp(0.5 * log_vars[idx]) * eps


def gmm_entropy_mc(weights, means, log_vars, n_samples, *, generator=None, chunk=4096) -> float:
    """MC 估计微分熵 h(X̃) = −E[log p(z)]，分块计算 z 与逐分量 log 密度。"""
    w, means, log_vars = _gmm_f64(weights, means, log_vars)
    n = int(n_samples)
    chunk = int(chunk)
    if n <= 0:
        raise ValueError(f"n_samples 必须为正，实际 {n_samples}")
    if chunk <= 0:
        raise ValueError(f"chunk 必须为正，实际 {chunk}")
    log_w = torch.log(w)
    total = 0.0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        z = _gmm_sample(w, means, log_vars, m, generator=generator)
        log_p = torch.logsumexp(gmm_log_prob(z, means, log_vars) + log_w, dim=-1)
        total += float((-log_p).sum())
        done += m
    return total / n


def mutual_information_mc(weights, means, log_vars, n_samples, *, generator=None) -> float:
    """Eq (5)/(7) 的 GMM 特例：MI = h(X̃) − E_X[h(N)]。"""
    return gmm_entropy_mc(
        weights, means, log_vars, n_samples, generator=generator
    ) - cond_entropy(weights, log_vars)


def gmm_entropy_quad_1d(weights, means, log_vars, *, lo=None, hi=None) -> float:
    """一维 GMM 熵的数值真值：∫ −p(z) log p(z) dz，默认区间覆盖各分量 ±12σ。"""
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1).cpu().numpy()
    m = torch.as_tensor(means, dtype=torch.float64).reshape(-1).cpu().numpy()
    lv = torch.as_tensor(log_vars, dtype=torch.float64).reshape(-1).cpu().numpy()
    if w.size != m.size or w.size != lv.size:
        raise ValueError(f"weights/means/log_vars 长度不一致：{w.size}/{m.size}/{lv.size}")
    var = np.exp(lv)
    if lo is None:
        lo = float(np.min(m - 12.0 * np.sqrt(var)))
    if hi is None:
        hi = float(np.max(m + 12.0 * np.sqrt(var)))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        raise ValueError(f"积分区间无效：[{lo}, {hi}]")
    log_w = np.log(w)

    def integrand(t: float) -> float:
        lp = -0.5 * ((t - m) ** 2 / var + lv + _LOG_2PI) + log_w
        peak = float(lp.max())
        log_p = peak + math.log(float(np.exp(lp - peak).sum()))
        return -math.exp(log_p) * log_p

    points = m[(m > lo) & (m < hi)]
    kwargs = {"epsabs": 1e-12, "epsrel": 1e-12, "limit": 400}
    if points.size:
        kwargs["points"] = np.sort(points)
    value, _ = quad(integrand, lo, hi, **kwargs)
    return float(value)
