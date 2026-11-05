"""M7 实验脚本（``experiments/run_ablation.py``、``run_constant_noise.py``）的验证（任务单 C）。

全部用**小合成数据**：不加载 1B 模型、不联网、不读 data_cache。被测函数：

- :func:`run_ablation.rank_metrics`：恒等变换 / 最远映射的退化行为；
- :func:`run_ablation.feature_mi`：d=1 常数 σ GMM 下 Eq (8) 逐特征均值的解析对照；
- :func:`run_constant_noise.nn_fr_for_sigma`：σ=0 与 σ 单调性；
- :func:`run_ablation.write_json` / ``json_safe``：严格 JSON、NaN 记为 null。

关于任务单 C3（``feature_mi`` 与 ``sgt.theory.mutual_information_mc`` ±10%）：
本文件用 ``xfail(strict=True)`` 显式记录该断言**在数学上不成立**（推导见该测试的
``reason``），并分别用两个可成立的测试验证两个估计量的闭式解；详细报告见交付说明。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_ablation as A  # noqa: E402
import run_constant_noise as C  # noqa: E402
from sgt import theory  # noqa: E402

# d=1 常数 σ GMM 的固定参数（测试 3）。
SIGMA2 = 2.0
GMM_N = 4000
GMM_SEED = 20241105
PAIR_SEED = 7


# --------------------------------------------------------------------------- #
# 1. 秩指标：恒等变换
# --------------------------------------------------------------------------- #
def test_rank_metrics_identity_transform():
    """x̃ = x（恒等）：秩全为 1 ⇒ NN-FR/TTR-10/SymTTR-10/Hist.Entropy 全为 0。"""
    generator = torch.Generator().manual_seed(3)
    vocab, d, n = 50, 6, 40
    table = torch.randn(vocab, d, generator=generator, dtype=torch.float64)
    target_ids = torch.randint(0, vocab, (n,), generator=generator)
    result = A.rank_metrics(table[target_ids], target_ids, table, chunk=1024)

    assert result["n_tokens"] == n
    assert result["nn_fr"] == 0.0
    assert result["ttr_10"] == 0.0
    assert result["symttr_10"] == 0.0
    assert result["hist_entropy"] == 0.0
    assert result["mrp_r_star"] == 1
    assert result["mrp_fr"] == 0.0


# --------------------------------------------------------------------------- #
# 2. 秩指标：所有 token 都映射到最远的 embedding
# --------------------------------------------------------------------------- #
def test_rank_metrics_farthest_embedding():
    """目标行放在远处、x̃ 落在其余全同的零行上：NN 必错、秩=V。"""
    vocab, d, n = 32, 4, 16
    table = torch.zeros(vocab, d, dtype=torch.float64)
    table[1] = 10.0  # token 1 是“最远”的 embedding
    target_ids = torch.ones(n, dtype=torch.long)
    x_tilde = torch.zeros(n, d, dtype=torch.float64)  # 最近邻是 token 0（id 最小的零行）

    result = A.rank_metrics(x_tilde, target_ids, table, chunk=1024)
    assert result["nn_fr"] == 100.0
    assert result["ttr_10"] == 100.0
    assert result["mrp_r_star"] == vocab  # 所有秩 = V，直方图众数在 V
    assert result["mrp_fr"] == 0.0  # 第 V 近邻恰好是目标行（稳定的退化反例）
    assert result["hist_entropy"] == 0.0  # 秩分布单点 ⇒ 熵 0


# --------------------------------------------------------------------------- #
# 3. 特征级 MI：d=1 常数 σ GMM
# --------------------------------------------------------------------------- #
def _constant_sigma_gmm():
    """x ~ N(0,1)、µ=0、σ² 固定；x̃ = x + σu；prime 是独立同分布样本。"""
    sigma = math.sqrt(SIGMA2)
    generator = torch.Generator().manual_seed(GMM_SEED)
    x = torch.randn(GMM_N, 1, generator=generator, dtype=torch.float64)
    x_prime = torch.randn(GMM_N, 1, generator=generator, dtype=torch.float64)
    noise = torch.randn(GMM_N, 1, generator=generator, dtype=torch.float64)
    x_tilde = x + sigma * noise
    log_var = torch.full_like(x, math.log(SIGMA2))
    return x, x_prime, x_tilde, log_var


def test_feature_mi_matches_eq8_expectation():
    """Eq (8) 逐特征项在该 GMM 下的解析期望：E[(x̃−x′−µ′)²]/σ² = 1 + 2/σ²。

    推导：Var(x̃)=1+σ²、Var(x′)=1、二者独立 ⇒ E[(x̃−x′)²]=2+σ²；
    常数 σ 使 log σ²′−log σ²=0。MC（20000 对）与有限样本允许 ±10%。
    """
    x, x_prime, x_tilde, log_var = _constant_sigma_gmm()
    result = A.feature_mi(
        x_tilde,
        x_prime,
        torch.zeros_like(x_prime),
        log_var,
        log_var,
        n_pairs=20000,
        seed=PAIR_SEED,
    )
    expected = 1.0 + 2.0 / SIGMA2
    assert result["n_pairs"] == 20000
    assert result["n_tokens"] == GMM_N
    assert abs(result["mi_feature_mean"] - expected) / expected < 0.10
    assert result["mi_sum"] == pytest.approx(result["mi_feature_mean"] * 1.0)
    assert result["mi_feature_median"] == pytest.approx(result["mi_feature_mean"])
    assert math.isfinite(result["mi_feature_std"])


def test_theory_mutual_information_mc_matches_closed_form():
    """同一 GMM 的真实 MI：0.5·log(1 + 1/σ²)（`theory` 的 MC 估计，±10%）。"""
    x, _, _, log_var = _constant_sigma_gmm()
    weights = torch.full((GMM_N,), 1.0 / GMM_N, dtype=torch.float64)
    estimate = theory.mutual_information_mc(
        weights, x, log_var, 20000, generator=torch.Generator().manual_seed(8)
    )
    closed_form = 0.5 * math.log1p(1.0 / SIGMA2)
    assert abs(estimate - closed_form) / closed_form < 0.10


@pytest.mark.xfail(
    strict=True,
    reason=(
        "任务单 C3 要求 feature_mi 与 theory.mutual_information_mc 在 ±10% 内，"
        "但 A7 的 Eq(8) 逐特征均值与真实 MI 不是同一个统计量：本构造下 "
        "feature_mi → 1+2/σ² = 2.0，而 theory MI → 0.5·log(1+1/σ²) = 0.203，"
        "相差约 9.9 倍（任何 σ 下都不可能相等：左边 ≥1，右边随 σ 单调且最大 0.347）。"
        "两个估计量各自的闭式解已由上两个测试分别验证；需主控确认 C3 的口径。"
    ),
)
def test_feature_mi_matches_theory_mi_as_requested_by_c3():
    x, x_prime, x_tilde, log_var = _constant_sigma_gmm()
    feature = A.feature_mi(
        x_tilde, x_prime, torch.zeros_like(x_prime), log_var, log_var,
        n_pairs=20000, seed=PAIR_SEED,
    )["mi_feature_mean"]
    weights = torch.full((GMM_N,), 1.0 / GMM_N, dtype=torch.float64)
    reference = theory.mutual_information_mc(
        weights, x, log_var, 20000, generator=torch.Generator().manual_seed(8)
    )
    assert abs(feature - reference) <= 0.10 * abs(reference)


# --------------------------------------------------------------------------- #
# 4. 常数噪声：σ=0 与单调性
# --------------------------------------------------------------------------- #
def test_constant_noise_nn_fr_sigma_zero_and_monotone():
    """σ=0 时 nn_fr==0；σ 增大时 nn_fr 单调不降（3 个 σ、200 token、固定 seed）。"""
    generator = torch.Generator().manual_seed(11)
    vocab, d, n = 100, 8, 200
    table = torch.randn(vocab, d, generator=generator)
    target_ids = torch.randint(0, vocab, (n,), generator=generator)
    x_clean = table[target_ids]

    rates = [
        C.nn_fr_for_sigma(x_clean, target_ids, table, sigma, seed=17)
        for sigma in (0.0, 0.5, 1.5)
    ]
    assert rates[0] == 0.0
    assert rates[0] <= rates[1] <= rates[2]
    assert all(0.0 <= rate <= 100.0 for rate in rates)


def test_constant_noise_literal_formula():
    """σ=0 时 constant_noise 逐元素等于输入；σ>0 时均值/方差符合 x+σu。"""
    generator = torch.Generator().manual_seed(5)
    d = 16
    x = torch.randn(64, d, generator=generator)
    zero = C.constant_noise(x, 0.0, generator=torch.Generator().manual_seed(0))
    assert torch.equal(zero, x)
    noisy = C.constant_noise(x, 3.0, generator=torch.Generator().manual_seed(1))
    delta = (noisy - x).to(torch.float64)
    assert abs(float(delta.mean())) < 0.2
    assert abs(float(delta.square().mean()) - 9.0) < 1.5


# --------------------------------------------------------------------------- #
# 5. JSON：严格可序列化、无 NaN
# --------------------------------------------------------------------------- #
def test_json_writer_is_strict_and_nan_free(tmp_path):
    payload = {
        "variant": "synthetic",
        "metrics": {"nn_fr": float("nan"), "eps": float("inf"), "ok": 1.5},
        "nested": [float("-inf"), {"x": 2}],
    }
    path = A.write_json(tmp_path / "ablation.json", payload)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text

    loaded = json.loads(text)
    assert loaded["metrics"]["nn_fr"] is None
    assert loaded["metrics"]["eps"] is None
    assert loaded["nested"][0] is None
    assert loaded["metrics"]["ok"] == 1.5
    json.dumps(loaded, allow_nan=False)  # 严格序列化不抛错
