"""M4 隐私/变换指标：秩、重建、TTR、秩直方图熵、PAC-Adv 与 epsilon-DP。

定义见 SPEC §4 与论文 §6（p.7–8）：

- 秩 ``R(x̃)``：干净 token embedding 在 ``{‖x̃ − X_k‖₂}`` 中由小到大的位置（1-based）；
  距离并列时按 token id 升序稳定排序。
- NN 重建 = 秩 1；MRP 重建 = 秩 ``r`` 的 token，``r`` 为秩直方图的众数（1-based，并列取小）。
- TTR-k = 100·|{k < R}|/|D|；SymTTR-k = 100·|{k < R < V − k}|/|D|。
- Hist. Entropy = H[R]/log(V)（自然对数，比值与底无关）。
- PAC-Adv = 100·δ，δ 由 Eq (4) 反解（复用 :func:`sgt.theory.invert_pac_adv`）。
- epsilon-DP（常数高斯机制）= Δ·sqrt(2 ln(1.25/δ))/σ。

距离一律在分块内用 ``‖x‖² + ‖e‖² − 2x·e`` 展开计算，临时张量为 ``O(chunk × V)``。
"""

from __future__ import annotations

import math

import torch

from . import theory

__all__ = [
    "ranks",
    "ttr",
    "symttr",
    "hist_entropy",
    "rank_histogram",
    "failure_rate",
    "pac_adv",
    "eps_dp",
    "nn_reconstruct",
    "mrp_rank",
    "mrp_reconstruct",
]


# --------------------------------------------------------------------------- #
# 校验与分块距离
# --------------------------------------------------------------------------- #
def _as_float_tensor(value, name: str) -> torch.Tensor:
    t = torch.as_tensor(value)
    if not t.is_floating_point():
        t = t.to(torch.get_default_dtype())
    return t


def _validate_table(embedding_table) -> torch.Tensor:
    table = _as_float_tensor(embedding_table, "embedding_table")
    if table.ndim != 2:
        raise ValueError(f"embedding_table 必须为 [V, d]，实际 {tuple(table.shape)}")
    if table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError(f"embedding_table 不能为空，实际 {tuple(table.shape)}")
    return table


def _validate_points(x_tilde, table: torch.Tensor) -> torch.Tensor:
    x = _as_float_tensor(x_tilde, "x_tilde")
    if x.ndim != 2:
        raise ValueError(f"x_tilde 必须为 [N, d]，实际 {tuple(x.shape)}")
    if x.shape[1] != table.shape[1]:
        raise ValueError(
            f"x_tilde 维度 {x.shape[1]} 与 embedding_table 维度 {table.shape[1]} 不一致"
        )
    return x


def _validate_chunk(chunk) -> int:
    chunk = int(chunk)
    if chunk <= 0:
        raise ValueError(f"chunk 必须为正，实际 {chunk}")
    return chunk


def _prepare(x_tilde, embedding_table, chunk):
    table = _validate_table(embedding_table)
    x = _validate_points(x_tilde, table)
    chunk = _validate_chunk(chunk)
    # 统一 dtype/device；float64 输入保持 float64（并列判定更可靠）。
    dtype = torch.promote_types(x.dtype, table.dtype)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        dtype = torch.get_default_dtype()
    x = x.to(dtype=dtype)
    table = table.to(device=x.device, dtype=dtype)
    return x, table, chunk


def _sqdist_chunk(x_chunk: torch.Tensor, table: torch.Tensor, table_sq: torch.Tensor):
    """‖x − e‖² 的展开式；clamp 消除浮点负零/微小负值。"""
    x_sq = x_chunk.square().sum(dim=1, keepdim=True)
    dist = x_sq - 2.0 * (x_chunk @ table.T) + table_sq.unsqueeze(0)
    return dist.clamp_min_(0.0)


# --------------------------------------------------------------------------- #
# 秩与重建
# --------------------------------------------------------------------------- #
def ranks(x_tilde, target_ids, embedding_table, *, chunk: int = 4096) -> torch.Tensor:
    """返回每个 ``x̃`` 中真实干净 token 的 1-based 秩，形状 ``[N]``（int64）。

    并列距离按 token id 升序稳定排序：秩 = 1 + #{d_k < d_target}
    + #{d_k = d_target 且 k < target}。
    """
    x, table, chunk = _prepare(x_tilde, embedding_table, chunk)
    target = torch.as_tensor(target_ids, device=x.device).reshape(-1).long()
    n, vocab = x.shape[0], table.shape[0]
    if target.numel() != n:
        raise ValueError(f"target_ids 长度必须为 {n}，实际 {target.numel()}")
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)
    if int(target.min()) < 0 or int(target.max()) >= vocab:
        raise ValueError(f"target_ids 必须在 [0, {vocab - 1}]，实际范围 "
                         f"[{int(target.min())}, {int(target.max())}]")

    table_sq = table.square().sum(dim=1)
    token_ids = torch.arange(vocab, device=x.device)
    out = torch.empty(n, dtype=torch.long, device=x.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        xc = x[start:stop]
        tc = target[start:stop]
        dist = _sqdist_chunk(xc, table, table_sq)
        d_target = dist.gather(1, tc.unsqueeze(1))
        closer = (dist < d_target).sum(dim=1)
        tied_before = ((dist == d_target) & token_ids.unsqueeze(0).lt(tc.unsqueeze(1))).sum(
            dim=1
        )
        out[start:stop] = closer + tied_before + 1
    return out


def nn_reconstruct(x_tilde, embedding_table, *, chunk: int = 4096) -> torch.Tensor:
    """NN 重建：每个 ``x̃`` 的最近 token id（距离并列取较小 token id）。"""
    x, table, chunk = _prepare(x_tilde, embedding_table, chunk)
    table_sq = table.square().sum(dim=1)
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        out[start:stop] = _sqdist_chunk(x[start:stop], table, table_sq).argmin(dim=1)
    return out


def mrp_rank(rank_hist) -> int:
    """MRP 的 r*：秩直方图众数，1-based；并列取较小 r。"""
    hist = torch.as_tensor(rank_hist).reshape(-1)
    if hist.numel() == 0:
        raise ValueError("rank_hist 不能为空")
    if not torch.is_floating_point(hist):
        hist = hist.to(torch.float64)
    return int(torch.argmax(hist)) + 1


def mrp_reconstruct(
    x_tilde, embedding_table, r, *, chunk: int = 4096
) -> torch.Tensor:
    """MRP 重建：每个 ``x̃`` 的第 ``r`` 近邻 token id（1-based，并列按 id 升序）。"""
    x, table, chunk = _prepare(x_tilde, embedding_table, chunk)
    vocab = table.shape[0]
    r = int(r)
    if not 1 <= r <= vocab:
        raise ValueError(f"r 必须在 [1, {vocab}]，实际 {r}")
    table_sq = table.square().sum(dim=1)
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        dist = _sqdist_chunk(x[start:stop], table, table_sq)
        # stable=True：并列距离保持 token id 升序，与 ranks 的约定一致。
        order = torch.argsort(dist, dim=1, stable=True)
        out[start:stop] = order[:, r - 1]
    return out


# --------------------------------------------------------------------------- #
# 变换率与直方图
# --------------------------------------------------------------------------- #
def _validate_ranks(ranks, vocab_size) -> torch.Tensor:
    r = torch.as_tensor(ranks).reshape(-1).long()
    vocab_size = int(vocab_size)
    if vocab_size < 1:
        raise ValueError(f"vocab_size 必须为正，实际 {vocab_size}")
    if r.numel() == 0:
        raise ValueError("ranks 不能为空")
    if int(r.min()) < 1 or int(r.max()) > vocab_size:
        raise ValueError(f"ranks 必须在 [1, {vocab_size}]，实际范围 "
                         f"[{int(r.min())}, {int(r.max())}]")
    return r


def ttr(ranks, k, vocab_size) -> float:
    """TTR-k（百分比）：100·|{k < R}|/|D|。"""
    r = _validate_ranks(ranks, vocab_size)
    k = int(k)
    if k < 0:
        raise ValueError(f"k 必须非负，实际 {k}")
    return float(100.0 * (r > k).to(torch.float64).mean())


def symttr(ranks, k, vocab_size) -> float:
    """SymTTR-k（百分比）：100·|{k < R < V − k}|/|D|。"""
    r = _validate_ranks(ranks, vocab_size)
    vocab_size = int(vocab_size)
    k = int(k)
    if k < 0:
        raise ValueError(f"k 必须非负，实际 {k}")
    return float(100.0 * ((r > k) & (r < vocab_size - k)).to(torch.float64).mean())


def rank_histogram(ranks, vocab_size) -> torch.Tensor:
    """秩直方图计数，形状 ``[V]``（int64）。"""
    r = _validate_ranks(ranks, vocab_size)
    return torch.bincount(r, minlength=int(vocab_size))[: int(vocab_size)].long()


def hist_entropy(ranks, vocab_size) -> float:
    """Hist. Entropy = H[R]/log(V)；均匀秩直方图为 1，单点分布为 0。"""
    r = _validate_ranks(ranks, vocab_size)
    vocab_size = int(vocab_size)
    if vocab_size == 1:
        # H[R] = log(V) = 0，约定返回 0（无不确定性）。
        return 0.0
    counts = torch.bincount(r, minlength=vocab_size).to(torch.float64)
    p = counts / counts.sum()
    nz = p > 0
    entropy = float(-(p[nz] * torch.log(p[nz])).sum())
    return entropy / math.log(vocab_size)


def failure_rate(pred_ids, target_ids) -> float:
    """重建失败率（百分比）：pred ≠ target 的位置占比；支持任意相同形状。"""
    pred = torch.as_tensor(pred_ids).reshape(-1).long()
    target = torch.as_tensor(target_ids).reshape(-1).long()
    if pred.numel() != target.numel():
        raise ValueError(f"pred/target 元素数不一致：{pred.numel()} vs {target.numel()}")
    if pred.numel() == 0:
        raise ValueError("pred/target 不能为空")
    return float(100.0 * (pred != target).to(torch.float64).mean())


# --------------------------------------------------------------------------- #
# 隐私上界与 DP
# --------------------------------------------------------------------------- #
def pac_adv(mi, n) -> float:
    """PAC-Adv（百分比）：由 Eq (4) 反解 δ，返回 100·δ，δ0 = 1/n。"""
    n = int(n)
    if n < 1:
        raise ValueError(f"n 必须为正，实际 {n}")
    delta = theory.invert_pac_adv(float(mi), n)
    return float(min(100.0, max(0.0, 100.0 * delta)))


def eps_dp(sigma, *, sensitivity, delta) -> float:
    """常数高斯机制的 epsilon：Δ·sqrt(2 ln(1.25/δ))/σ。"""
    sigma = float(sigma)
    sensitivity = float(sensitivity)
    delta = float(delta)
    if sigma <= 0.0:
        raise ValueError(f"sigma 必须为正，实际 {sigma}")
    if sensitivity < 0.0:
        raise ValueError(f"sensitivity 必须非负，实际 {sensitivity}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta 必须在 (0, 1) 内，实际 {delta}")
    if sensitivity == 0.0:
        return 0.0
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
