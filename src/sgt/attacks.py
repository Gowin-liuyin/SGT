"""M4 重建攻击：NN、MRP 与最小忠实版 BeamClean（SPEC §4 / A9）。

BeamClean 只按论文 §6.2 的公开描述实现（EM/矩估计噪声尺度 + LM 先验 beam search，
非官方代码）：

1. 噪声模型 ``x̃ = x + n``，``n ~ N(0, sigma² I)``。``sigma=None`` 时用 top-k
   候选上的 EM 估计（初值取 NN 残差矩估计），迭代 ``n_iter`` 次后返回估计值。
2. 每个位置的似然为 ``exp(−‖x_t − x̃_t‖²/(2σ²))``，保留 top-``topk`` 候选 token。
3. 固定长度 T 的前缀 beam search，分数 = Σ_t log 似然 + LM 整段 log-prob。
   ``lm_score_fn`` 需要对长度 ``t ≤ T`` 的 token 前缀 ``[M, t]`` 返回 ``[M]`` 的
   整段 log-prob（最终解码长度恒为 T）。
4. σ → 0 时后验退化为 NN（实现里显式短路），重建结果与 :func:`nn_attack` 相同。
"""

from __future__ import annotations

import math

import torch

from . import metrics

__all__ = ["nn_attack", "mrp_attack", "beamclean"]

# 内部 top-k 距离计算的元素预算（约 2M 个数，float64 下 ~16 MB）。
_TOPK_BLOCK_BUDGET = 2_000_000
# sigma 低于此值时似然为点质量，直接返回 NN（避免 1/sigma² 溢出/无意义的 beam 展开）。
_NN_SIGMA = 1e-9
_SIGMA2_FLOOR = 1e-12


def nn_attack(x_tilde, embedding_table, *, chunk: int = 4096) -> torch.Tensor:
    """NN 攻击：逐位置最近邻 token id，等价于秩 1 重建。"""
    return metrics.nn_reconstruct(x_tilde, embedding_table, chunk=chunk)


def mrp_attack(x_tilde, embedding_table, r, *, chunk: int = 4096) -> torch.Tensor:
    """MRP 攻击：逐位置第 ``r`` 近邻 token id（1-based）。"""
    return metrics.mrp_reconstruct(x_tilde, embedding_table, r, chunk=chunk)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _validate_sequence_inputs(x_tilde, embedding_table):
    """返回 ``(x[N,T,d], table, was_flat)``；2D 输入按逐位置长度 1 处理。"""
    table = metrics._validate_table(embedding_table)
    x = metrics._as_float_tensor(x_tilde)
    was_flat = x.ndim == 2
    if was_flat:
        x = x.unsqueeze(1)
    elif x.ndim != 3:
        raise ValueError(f"x_tilde 必须为 [N, T, d] 或 [N, d]，实际 {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"x_tilde 的 N/T 必须为正，实际 {tuple(x.shape)}")
    if x.shape[2] != table.shape[1]:
        raise ValueError(
            f"x_tilde 维度 {x.shape[2]} 与 embedding_table 维度 {table.shape[1]} 不一致"
        )
    dtype = torch.promote_types(x.dtype, table.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    elif dtype not in (torch.float32, torch.float64):
        dtype = torch.get_default_dtype()
    return x.to(dtype=dtype), table.to(device=x.device, dtype=dtype), was_flat


def _topk_sqdist(flat_x, table, k, *, chunk=None):
    """每个点的前 k 小平方距离与 token id：``([M, k], [M, k])``。"""
    m, d = flat_x.shape
    vocab = table.shape[0]
    k = min(int(k), vocab)
    if chunk is None:
        chunk = max(1, min(m, _TOPK_BLOCK_BUDGET // max(1, vocab)))
    chunk = max(1, int(chunk))
    table_sq = table.square().sum(dim=1)
    out_d = torch.empty(m, k, dtype=torch.float64, device=flat_x.device)
    out_i = torch.empty(m, k, dtype=torch.long, device=flat_x.device)
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        dist = metrics._sqdist_chunk(flat_x[start:stop], table, table_sq)
        # stable=True 保证距离并列时取较小 token id。
        vals, idx = torch.sort(dist, dim=1, stable=True)
        out_d[start:stop] = vals[:, :k].to(torch.float64)
        out_i[start:stop] = idx[:, :k]
    return out_d, out_i


def _estimate_sigma(x_tilde, table, *, topk: int, n_iter: int) -> float:
    """top-k 责任度上的 EM 估计 sigma；初值为 NN 残差矩估计。"""
    flat = x_tilde.reshape(-1, x_tilde.shape[-1])
    d = table.shape[1]
    top_d, _ = _topk_sqdist(flat, table, topk)
    # E‖n‖² = d·σ²；NN 残差偏小，作为 EM 初值。
    sigma2 = max(float(top_d[:, 0].mean()) / d, _SIGMA2_FLOOR)
    for _ in range(max(0, int(n_iter))):
        # 均匀 token 先验下的高斯混合责任度（截断到 top-k）。
        logits = -top_d / (2.0 * sigma2)
        gamma = torch.softmax(logits, dim=1)
        sigma2 = max(float((gamma * top_d).sum()) / (flat.shape[0] * d), _SIGMA2_FLOOR)
    return math.sqrt(sigma2)


def _lm_scores(lm_score_fn, prefixes, t: int, device) -> torch.Tensor:
    """调用 LM 打分并对返回形状做校验。"""
    tokens = torch.tensor(prefixes, dtype=torch.long, device=device)
    if tuple(tokens.shape) != (len(prefixes), t):
        raise RuntimeError(f"内部前缀形状错误：{tuple(tokens.shape)}")
    scores = lm_score_fn(tokens)
    scores = torch.as_tensor(scores, dtype=torch.float64, device=device).reshape(-1)
    if scores.numel() != len(prefixes):
        raise ValueError(
            f"lm_score_fn 必须返回 {len(prefixes)} 个分数，实际 {scores.numel()}"
        )
    return scores


def _beam_decode_one(top_ids, top_loglik, lm_score_fn, beam: int, device) -> list[int]:
    """单条序列的前缀 beam search；分数 = Σ log 似然 + LM 前缀 log-prob。"""
    length, k = top_ids.shape
    # 每个 beam：(token 前缀 tuple, Σ log 似然)；LM 分数在每次展开时按整段前缀重算。
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for t in range(length):
        cand_ids = top_ids[t].tolist()
        cand_ll = top_loglik[t].tolist()
        prefixes: list[tuple[int, ...]] = []
        loglik_sums: list[float] = []
        for seq, ll_sum in beams:
            for j in range(k):
                prefixes.append(seq + (int(cand_ids[j]),))
                loglik_sums.append(ll_sum + float(cand_ll[j]))
        lm = _lm_scores(lm_score_fn, prefixes, t + 1, device)
        total = torch.tensor(loglik_sums, dtype=torch.float64, device=device) + lm
        keep = torch.argsort(total, descending=True, stable=True)[:beam]
        beams = [(prefixes[int(i)], loglik_sums[int(i)]) for i in keep]
    # 最后一步的 LM 分数已进入 total；重新打分选择最终序列（beam 内的前缀长度均为 T）。
    final_prefixes = [seq for seq, _ in beams]
    lm = _lm_scores(lm_score_fn, final_prefixes, length, device)
    final = torch.tensor([ll for _, ll in beams], dtype=torch.float64, device=device) + lm
    best = int(torch.argmax(final))
    return list(final_prefixes[best])


# --------------------------------------------------------------------------- #
# BeamClean
# --------------------------------------------------------------------------- #
def beamclean(
    x_tilde,
    embedding_table,
    lm_score_fn,
    *,
    sigma=None,
    beam: int = 8,
    topk: int = 32,
    n_iter: int = 3,
) -> tuple[torch.Tensor, float]:
    """BeamClean 最小实现：返回 ``(重建 token ids [N, T] 或 [N], 估计/给定的 sigma)``。

    2D 输入 ``[N, d]`` 按 ``T=1`` 的独立位置处理并返回 ``[N]``；3D 输入返回 ``[N, T]``。
    """
    x, table, was_flat = _validate_sequence_inputs(x_tilde, embedding_table)
    beam = int(beam)
    topk = int(topk)
    n_iter = int(n_iter)
    if beam < 1:
        raise ValueError(f"beam 必须为正，实际 {beam}")
    if topk < 1:
        raise ValueError(f"topk 必须为正，实际 {topk}")
    if n_iter < 0:
        raise ValueError(f"n_iter 必须非负，实际 {n_iter}")

    n, length, d = x.shape
    vocab = table.shape[0]
    if sigma is None:
        sigma = _estimate_sigma(x, table, topk=topk, n_iter=n_iter)
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma 必须为有限非负数，实际 {sigma}")

    k = min(topk, vocab)
    top_d, top_i = _topk_sqdist(x.reshape(-1, d), table, k)
    top_d = top_d.reshape(n, length, k)
    top_i = top_i.reshape(n, length, k)

    # σ=0 / 数值意义上的 0：只有 NN 候选有非零后验。
    if sigma <= _NN_SIGMA:
        ids = top_i[:, :, 0].clone()
        return (ids.reshape(-1) if was_flat else ids), sigma

    loglik = -top_d / (2.0 * sigma * sigma)
    out = torch.empty(n, length, dtype=torch.long, device=x.device)
    for i in range(n):
        out[i] = torch.tensor(
            _beam_decode_one(top_i[i], loglik[i], lm_score_fn, beam, x.device),
            dtype=torch.long,
            device=x.device,
        )
    return (out.reshape(-1) if was_flat else out), sigma
