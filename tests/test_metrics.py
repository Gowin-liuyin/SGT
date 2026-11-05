"""M4 指标与攻击的验证：秩/重建/TTR/熵/PAC/DP 与 BeamClean。

对拍真值尽量用独立的朴素循环或 `argsort` 参考实现，不复用 `sgt.metrics` 的向量化路径
（BeamClean 的 LM 先验除外，它只能由构造的合成 LM 提供）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgt import attacks as A  # noqa: E402
from sgt import metrics as M  # noqa: E402

F64 = torch.float64


# --------------------------------------------------------------------------- #
# 独立参考实现
# --------------------------------------------------------------------------- #
def _brute_ranks_chunked(x, target, table, chunk=64):
    """逐行直接算 [N, V, d] 距离，再用稳定 argsort 取目标位置（1-based）。"""
    out = torch.empty(x.shape[0], dtype=torch.long)
    table = table.to(dtype=torch.float64)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        xc = x[start:stop].to(dtype=torch.float64)
        dist = (xc[:, None, :] - table[None, :, :]).square().sum(dim=-1)
        order = torch.argsort(dist, dim=1, stable=True)
        for i in range(stop - start):
            pos = (order[i] == int(target[start + i])).nonzero(as_tuple=True)[0]
            out[start + i] = int(pos[0]) + 1
    return out


def _synthetic_lm(phrases, vocab):
    """由 10 条“短语”构造的 bigram LM；训练/评估数据均为合成，不读取真实语料。"""
    logp = torch.full((vocab, vocab), -math.log(vocab), dtype=F64)
    for p in phrases:
        for t in range(p.shape[0] - 1):
            logp[p[t], p[t + 1]] += 4.0
    logp = torch.log_softmax(logp, dim=1)
    start = torch.full((vocab,), -math.log(vocab), dtype=F64)

    def lm_score_fn(tokens):
        scores = start[tokens[:, 0]].clone()
        for t in range(1, tokens.shape[1]):
            scores = scores + logp[tokens[:, t - 1], tokens[:, t]]
        return scores

    return lm_score_fn


def _beam_data(*, vocab=50, d=8, length=6, n=150, scale=0.3, seed=20241105):
    """固定种子合成 embedding 表、短语、真实序列与共享噪声基底。"""
    g = torch.Generator().manual_seed(seed)
    table = scale * torch.randn(vocab, d, generator=g, dtype=F64)
    phrases = torch.randint(0, vocab, (10, length), generator=g, dtype=torch.long)
    truth = phrases[torch.randint(0, 10, (n,), generator=g)]
    base_noise = torch.randn(
        n, length, d, generator=torch.Generator().manual_seed(seed + 1), dtype=F64
    )
    return table, phrases, truth, base_noise


# --------------------------------------------------------------------------- #
# 1. 秩：对拍 + 并列稳定性
# --------------------------------------------------------------------------- #
def test_ranks_matches_bruteforce():
    g = torch.Generator().manual_seed(0)
    vocab, d, n = 10, 3, 8
    table = torch.randn(vocab, d, generator=g, dtype=F64)
    target = torch.randint(0, vocab, (n,), generator=g)
    x = table[target] + 0.4 * torch.randn(n, d, generator=g, dtype=F64)

    got = M.ranks(x, target, table)
    expected = _brute_ranks_chunked(x, target, table)
    assert torch.equal(got, expected), f"got={got.tolist()} exp={expected.tolist()}"
    assert got.dtype == torch.long


def test_ranks_ties_follow_token_id_order():
    table = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=F64,
    )
    x = torch.zeros(4, 3, dtype=F64)  # 到 4 个 token 的距离全部为 1
    target = torch.arange(4)
    assert torch.equal(M.ranks(x, target, table), torch.tensor([1, 2, 3, 4]))

    # 再插入一个更近的 token，验证并列组整体后移。
    table2 = torch.cat([table, torch.tensor([[0.5, 0.0, 0.0]], dtype=F64)], dim=0)
    got = M.ranks(torch.zeros(1, 3, dtype=F64), torch.tensor([3]), table2)
    assert int(got[0]) == 5  # 近点(0.25)排第 1，4 个等距 token 中 id=3 排第 5


# --------------------------------------------------------------------------- #
# 2. TTR / SymTTR
# --------------------------------------------------------------------------- #
def test_ttr_symttr_hand_cases():
    ranks = [1, 5, 10]
    vocab = 10
    assert M.ttr(ranks, 1, vocab) == pytest.approx(200.0 / 3.0, abs=1e-12)
    assert M.symttr(ranks, 1, vocab) == pytest.approx(100.0 / 3.0, abs=1e-12)
    # 定义 k < R < V-k：rank 5 严格落在 (2, 8) 内，故 SymTTR-2 = 33.33。
    assert M.symttr(ranks, 2, vocab) == pytest.approx(100.0 / 3.0, abs=1e-12)
    # 任务单示例中的 SymTTR-2=0 对应 ranks=[1,2,10]（rank 2 不满足 R>2）。
    assert M.symttr([1, 2, 10], 2, vocab) == 0.0
    assert M.symttr(ranks, 5, vocab) == 0.0
    # SymTTR 不可能超过 TTR（同一阈值下集合包含关系）。
    for k in range(1, vocab):
        assert M.symttr(ranks, k, vocab) <= M.ttr(ranks, k, vocab) + 1e-12


# --------------------------------------------------------------------------- #
# 3. 直方图与熵
# --------------------------------------------------------------------------- #
def test_rank_histogram_and_entropy():
    vocab = 10
    uniform = torch.arange(1, vocab + 1)
    assert torch.equal(M.rank_histogram(uniform, vocab), torch.ones(vocab, dtype=torch.long))
    assert M.hist_entropy(uniform, vocab) == pytest.approx(1.0, abs=1e-12)

    single = torch.ones(10, dtype=torch.long)
    assert M.hist_entropy(single, vocab) == 0.0
    assert torch.equal(M.rank_histogram(single, vocab), torch.tensor([10] + [0] * 9))


# --------------------------------------------------------------------------- #
# 4. 失败率
# --------------------------------------------------------------------------- #
def test_failure_rate_all_wrong_and_all_right():
    target = torch.tensor([3, 1, 4, 1, 5])
    assert M.failure_rate(torch.tensor([0, 2, 0, 2, 0]), target) == 100.0
    assert M.failure_rate(target, target) == 0.0
    assert M.failure_rate(target[:-1], target[:-1]) == 0.0
    # 形状不同的同元素数输入也可按位置比较。
    assert M.failure_rate(torch.tensor([[0, 2], [0, 2]]), torch.tensor([[1, 1], [1, 1]])) == 100.0


# --------------------------------------------------------------------------- #
# 5. NN / MRP 重建
# --------------------------------------------------------------------------- #
def test_nn_and_mrp_reconstruct():
    table = torch.tensor([[9.0], [1.0], [7.0], [3.0], [5.0],
                          [2.0], [8.0], [4.0], [6.0], [0.5]], dtype=F64)
    x = torch.zeros(1, 1, dtype=F64)
    expected_order = [9, 1, 5, 3, 7, 4, 8, 2, 6, 0]

    nn = M.nn_reconstruct(x, table)
    assert int(nn[0]) == 9 == expected_order[0]
    assert torch.equal(nn, A.nn_attack(x, table))
    assert int(nn[0]) == int((table - x[0]).square().sum(dim=1).argmin())

    for r in (1, 3, 10):
        want = expected_order[r - 1]
        assert int(M.mrp_reconstruct(x, table, r)[0]) == want
        assert int(A.mrp_attack(x, table, r)[0]) == want

    hist = torch.tensor([0, 2, 5, 5, 1])
    assert M.mrp_rank(hist) == 3  # 众数并列时取较小 r（index 2）
    assert M.mrp_rank(torch.tensor([0, 0, 0, 7])) == 4


# --------------------------------------------------------------------------- #
# 6. PAC-Adv / 7. epsilon-DP
# --------------------------------------------------------------------------- #
def test_pac_adv_table_value():
    got = M.pac_adv(1.11, 128_256)
    assert abs(got - 12.67) <= 0.15, f"PAC-Adv={got}, want 12.67±0.15"
    assert M.pac_adv(22.19, 128_256) == 100.0
    assert M.pac_adv(0.0, 128_256) == pytest.approx(100.0 / 128_256, rel=1e-12)


def test_eps_dp_formula_and_monotonicity():
    delta = 1e-5
    sensitivity = 1.5
    sigmas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    values = [M.eps_dp(s, sensitivity=sensitivity, delta=delta) for s in sigmas]
    assert all(a >= b for a, b in zip(values, values[1:])), values
    for sigma, got in zip(sigmas, values):
        want = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
        assert got == pytest.approx(want, rel=1e-14)
    assert M.eps_dp(0.1, sensitivity=0.0, delta=delta) == 0.0


# --------------------------------------------------------------------------- #
# 8. BeamClean
# --------------------------------------------------------------------------- #
def test_beamclean_noise_free_degenerates_to_nn():
    table, phrases, truth, base = _beam_data()
    lm = _synthetic_lm(phrases, table.shape[0])
    xt = table[truth] + 1e-3 * base

    pred, sigma_est = A.beamclean(xt, table, lm, sigma=1e-3, beam=8, topk=32)
    nn = A.nn_attack(xt.reshape(-1, table.shape[1]), table).reshape(truth.shape)
    assert sigma_est == pytest.approx(1e-3, rel=1e-15)
    assert pred.shape == truth.shape and pred.dtype == torch.long
    assert torch.equal(pred, nn), "sigma→0 时 BeamClean 必须逐位置退化为 NN"

    # 2D 输入按 T=1 处理（M4 门禁的调用方式），返回 [N]。
    pred_flat, _ = A.beamclean(
        xt.reshape(-1, table.shape[1]), table, lm, sigma=1e-3, beam=8, topk=32
    )
    assert pred_flat.shape == (truth.numel(),)
    assert torch.equal(pred_flat, nn.reshape(-1))


def test_beamclean_failure_rate_monotone_in_sigma():
    table, phrases, truth, base = _beam_data()
    lm = _synthetic_lm(phrases, table.shape[0])
    sigmas = [0.01, 0.1, 0.5]
    beam_fr, nn_fr = [], []
    for sigma in sigmas:
        xt = table[truth] + sigma * base
        pred, _ = A.beamclean(xt, table, lm, sigma=sigma, beam=8, topk=32)
        nn = A.nn_attack(xt.reshape(-1, table.shape[1]), table).reshape(truth.shape)
        beam_fr.append(M.failure_rate(pred, truth))
        nn_fr.append(M.failure_rate(nn, truth))
        assert beam_fr[-1] <= nn_fr[-1] + 1e-12, (sigma, beam_fr[-1], nn_fr[-1])

    assert beam_fr[0] <= beam_fr[1] <= beam_fr[2], f"BeamClean FR 非单调：{beam_fr}"
    assert nn_fr[0] <= nn_fr[1] <= nn_fr[2], f"NN FR 非单调：{nn_fr}"


def test_beamclean_estimates_sigma_from_em():
    table, phrases, truth, base = _beam_data()
    lm = _synthetic_lm(phrases, table.shape[0])
    xt = table[truth] + 0.1 * base
    pred, sigma_est = A.beamclean(xt, table, lm, sigma=None, beam=8, topk=32, n_iter=3)
    assert 0.08 <= sigma_est <= 0.12, f"EM sigma={sigma_est}"
    assert pred.shape == truth.shape


# --------------------------------------------------------------------------- #
# 9. 分块一致性（大词表）
# --------------------------------------------------------------------------- #
def test_ranks_chunked_matches_bruteforce_large_vocab():
    g = torch.Generator().manual_seed(7)
    vocab, d, n = 5000, 8, 500
    table = torch.randn(vocab, d, generator=g)
    target = torch.randint(0, vocab, (n,), generator=g)
    x = table[target] + 0.2 * torch.randn(n, d, generator=g)

    got = M.ranks(x, target, table, chunk=64)  # N=500 → 8 个分块
    expected = _brute_ranks_chunked(x, target, table, chunk=64)
    assert torch.equal(got, expected)
