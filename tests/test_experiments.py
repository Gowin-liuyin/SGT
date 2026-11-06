"""M7 实验脚本（``experiments/run_ablation.py``、``run_constant_noise.py``）的验证（任务单 C）。

全部用**小合成数据**：不加载 1B 模型、不联网、不读 data_cache。被测函数：

- :func:`run_ablation.rank_metrics`：恒等变换 / 最远映射的退化行为；
- :func:`run_ablation.feature_mi`：d=1 常数 σ GMM 下 Eq (8) 逐特征均值的解析对照；
- :func:`run_constant_noise.nn_fr_for_sigma`：σ=0 与 σ 单调性；
- :func:`run_ablation.write_json` / ``json_safe``：严格 JSON、NaN 记为 null。

审计 GAP 的额外覆盖（同样全合成）：

- GAP-7（§7 按 SymTTR 选 checkpoint）：:func:`run_ablation.plan_selection_steps` /
  :func:`run_ablation.build_selection` / :func:`run_ablation.selected_step`，用合成
  「checkpoint 序列」验证最佳步、trace 长度、``--no-select-by-symttr`` 取最后；
- GAP-4（SPEC A8）：:func:`run_ablation.mrp_calibration_split` 与 ``rank_metrics`` 的
  calibration/test 切分——r* 只来自 calibration、test 指标不随 calibration 数据变化、
  校准集过小必须抛错；
- GAP-5：``configs/ablation.yaml`` 的规模字段与 ``train.lm_dtype`` 解析/回退。

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
from sgt import metrics, theory  # noqa: E402

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


# --------------------------------------------------------------------------- #
# 6. GAP-7：按 SymTTR 选 checkpoint（合成 checkpoint 序列，不跑训练）
# --------------------------------------------------------------------------- #
def test_selection_plan_contains_cadence_and_final_step():
    """trace 的步数 = every 的整数倍 + 末步（末步不是整数倍时补一个点）。"""
    assert A.plan_selection_steps(10, 5) == [5, 10]
    assert A.plan_selection_steps(12, 5) == [5, 10, 12]
    assert A.plan_selection_steps(3, 5) == [3]  # every > steps：只评测末步
    assert A.plan_selection_steps(0, 5) == []
    with pytest.raises(ValueError):
        A.plan_selection_steps(10, 0)


def test_selection_picks_max_symttr_and_records_trace():
    values = {5: 40.0, 10: 70.0}
    block = A.build_selection(lambda step: values[step], steps=10, every=5)

    assert block["enabled"] is True
    assert block["metric"] == "SymTTR-100"
    assert block["every"] == 5
    assert [entry["step"] for entry in block["trace"]] == [5, 10]
    assert len(block["trace"]) == 2
    assert block["trace"][1]["symttr_100"] == 70.0
    assert block["best_step"] == 10
    assert block["best_symttr_100"] == 70.0
    assert A.selected_step(block, steps=10) == 10

    # 末步不是 every 的整数倍时 trace 补末步，且末步可以胜出
    values2 = {5: 10.0, 10: 20.0, 12: 90.0}
    block2 = A.build_selection(lambda step: values2[step], steps=12, every=5)
    assert [entry["step"] for entry in block2["trace"]] == [5, 10, 12]
    assert block2["best_step"] == 12

    # 非有限值不参与选择（JSON 里经 json_safe 记为 null）
    block3 = A.build_selection(
        lambda step: {5: float("nan"), 10: 33.0}[step], steps=10, every=5
    )
    assert block3["best_step"] == 10
    assert block3["best_symttr_100"] == 33.0
    with pytest.raises(ValueError, match="有限值"):
        A.build_selection(lambda step: float("nan"), steps=5, every=5)

    # 并列（SymTTR 全 0 的退化情形）取较晚的步：不把训练步数截断到第一个采样点。
    block_tie = A.build_selection(lambda step: 0.0, steps=10, every=5)
    assert block_tie["best_step"] == 10
    assert [entry["symttr_100"] for entry in block_tie["trace"]] == [0.0, 0.0]


def test_selection_disabled_uses_last_checkpoint():
    called = []

    def _unexpected(step):  # 选择关闭时不得评测任何 checkpoint
        called.append(step)
        raise AssertionError("选择关闭时不应调用评测")

    block = A.build_selection(_unexpected, steps=10, every=5, enabled=False)
    assert block == {"enabled": False, "checkpoint": "last"}
    assert called == []
    assert A.selected_step(block, steps=10) == 10  # 最后一个 checkpoint


def test_ablation_cli_selection_defaults_and_no_flag():
    args = A.parse_args(["--variant", "abscos"])
    assert args.select_by_symttr is True
    assert args.select_every is None  # 未给时取 yaml，再缺省 50
    assert args.calibration_examples is None
    assert args.calibration_fraction is None
    assert args.mrp_r is None
    assert args.lm_dtype == "auto"

    args = A.parse_args(
        [
            "--variant",
            "abscos",
            "--no-select-by-symttr",
            "--select-every",
            "7",
            "--calibration-examples",
            "3",
            "--calibration-fraction",
            "0.5",
            "--mrp-r",
            "9",
            "--lm-dtype",
            "float32",
        ]
    )
    assert args.select_by_symttr is False
    assert args.select_every == 7
    assert args.calibration_examples == 3
    assert args.calibration_fraction == 0.5
    assert args.mrp_r == 9
    assert args.lm_dtype == "float32"


def test_selection_and_eval_option_resolution():
    raw = {
        "selection": {"enabled": True, "every": 25, "calibration_examples": 10},
        "eval": {"calibration_fraction": 0.2},
    }
    block = A.resolve_selection(raw, enabled=True)
    assert block["enabled"] is True
    assert block["every"] == 25
    assert block["calibration_examples"] == 10

    # CLI 关闭或 yaml 关闭都能整体关闭
    assert A.resolve_selection(raw, enabled=False)["enabled"] is False
    assert A.resolve_selection({"selection": {"enabled": False}})["enabled"] is False

    default = A.resolve_selection({}, enabled=True)
    assert default["every"] == A.DEFAULT_SELECTION_EVERY == 50
    assert default["calibration_examples"] == A.DEFAULT_CALIBRATION_EXAMPLES == 50

    assert A.resolve_eval_options(raw)["calibration_fraction"] == 0.2
    assert A.resolve_eval_options({})["calibration_fraction"] == 0.25
    assert A.resolve_eval_options({})["min_calibration_tokens"] == 100
    assert A.resolve_eval_options({}, mrp_r=7)["mrp_r"] == 7
    with pytest.raises(ValueError):
        A.resolve_eval_options({}, calibration_fraction=1.0)
    with pytest.raises(ValueError):
        A.resolve_eval_options({}, mrp_r=0)


def test_symttr_on_sequences_matches_metrics_symttr():
    """σ≈0 的确定性 SGT 下，SymTTR 与 metrics.symttr(ranks) 一致（同一 k 口径）。"""

    class ShiftSGT(torch.nn.Module):
        def forward(self, x, attention_mask=None):  # noqa: D401 - 测试桩
            return torch.full_like(x, 7.0), torch.full_like(x, -1e4)  # σ = exp(-5000) == 0

    vocab = 64
    table = torch.arange(1, vocab + 1, dtype=torch.float64).reshape(-1, 1)
    sequences = [[3, 10, 20, 30]]
    result = A.symttr_on_sequences(
        ShiftSGT(),
        table,
        sequences,
        pad_id=0,
        batch_seqs=1,
        seed=0,
        rank_chunk=16,
        k=10,
    )
    ids = torch.tensor(sequences[0])
    x_tilde = table[ids] + 7.0
    ranks = metrics.ranks(x_tilde, ids, table, chunk=16)
    expected = metrics.symttr(ranks, 10, vocab)
    assert result["symttr"] == pytest.approx(expected)
    assert result["n_tokens"] == 4
    assert result["metric"] == "SymTTR-10"
    assert 0.0 <= result["symttr"] <= 100.0


def test_lm_dtype_probe_and_explicit_fallback():
    plain = A.resolve_lm_dtype("float32", probe=None)
    assert plain["effective"] == "float32" and plain["mixed_precision"] is False

    ok = A.resolve_lm_dtype("bfloat16", probe=lambda: (True, "pong"))
    assert ok["effective"] == "bfloat16"
    assert ok["mixed_precision"] is True
    assert ok["reason"] == "pong"

    fallback = A.resolve_lm_dtype(
        "bfloat16", probe=lambda: (False, "MPS bfloat16 不可用")
    )
    assert fallback["effective"] == "float32"
    assert fallback["requested"] == "bfloat16"
    assert fallback["mixed_precision"] is False
    assert "MPS bfloat16 不可用" in fallback["reason"]

    with pytest.raises(ValueError):
        A.resolve_lm_dtype("float64", probe=None)
    with pytest.raises(ValueError):
        A.resolve_lm_dtype("bfloat16", probe=None)


def test_build_train_config_carries_lm_dtype_and_selection_cadence():
    raw = {"train": {"steps": 7, "lr": 1.0e-3, "log_every": 3}}
    cfg = A.build_train_config(
        raw,
        loss_terms=("acs",),
        weights={"acs": 1.0},
        max_steps=None,
        device="cpu",
        seed=0,
        lm_dtype="bfloat16",
        checkpoint_dir="/tmp/sgt_select_test",
        log_every=5,
    )
    assert cfg.steps == 7
    assert cfg.dtype == "bfloat16"  # 记录的是 LM 侧精度（SGT 恒 fp32）
    assert cfg.log_every == 5
    assert cfg.checkpoint_dir == "/tmp/sgt_select_test"

    default = A.build_train_config(
        raw, loss_terms=("acs",), weights={}, max_steps=None, device="cpu", seed=0
    )
    assert default.log_every == 3
    assert default.checkpoint_dir is None
    assert default.dtype == "float32"


# --------------------------------------------------------------------------- #
# 7. GAP-4：MRP 的 r* 用独立 calibration 子集
# --------------------------------------------------------------------------- #
def _rank_points(
    vocab: int, ranks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 d=1 的 embedding 表与 ``x̃``，使目标 token 的秩**精确等于** ``ranks``。

    表值：token 0 = 0（目标），token k = 10·k（等距、无并列）；``x̃ = 5·r − 2.5`` 时
    ``|10k − x̃| < |0 − x̃|`` 当且仅当 ``k < r``，所以比目标更近的恰有 r−1 个 token，
    即秩 = r。
    """
    vocab = int(vocab)
    values = 10.0 * torch.arange(0, vocab, dtype=torch.float64)
    table = values.reshape(-1, 1)
    x_tilde = (5.0 * ranks.to(torch.float64) - 2.5).reshape(-1, 1)
    target_ids = torch.zeros(int(ranks.numel()), dtype=torch.long)
    return x_tilde, target_ids, table


def test_mrp_r_star_comes_from_calibration_not_test():
    vocab, n = 64, 400
    ranks = torch.cat(
        [
            torch.full((100,), 5, dtype=torch.long),  # calibration：众数 5
            torch.full((300,), 20, dtype=torch.long),  # test：众数 20
        ]
    )
    x_tilde, target_ids, table = _rank_points(vocab, ranks)
    assert torch.equal(metrics.ranks(x_tilde, target_ids, table, chunk=1024), ranks)

    result = A.rank_metrics(
        x_tilde, target_ids, table, chunk=1024, aux_chunk=32, calibration_fraction=0.25
    )
    assert result["mrp_r_star_source"] == "calibration"
    assert result["n_calibration_tokens"] == 100
    assert result["n_test_tokens"] == 300
    assert result["n_tokens"] == 300
    assert result["mrp_r_star"] == 5  # 只来自 calibration 的秩直方图
    # 若按旧口径用全体评测集的众数会得到 20 —— 证明确实没有用 test 求 r*。
    assert metrics.mrp_rank(metrics.rank_histogram(ranks, vocab)) == 20


def test_mrp_test_metrics_do_not_depend_on_calibration_data():
    vocab, n = 64, 400
    test_ranks = torch.full((300,), 12, dtype=torch.long)
    results = []
    for calib_rank in (5, 30):
        ranks = torch.cat([torch.full((100,), calib_rank, dtype=torch.long), test_ranks])
        x_tilde, target_ids, table = _rank_points(vocab, ranks)
        results.append(
            A.rank_metrics(
                x_tilde,
                target_ids,
                table,
                chunk=1024,
                aux_chunk=32,
                calibration_fraction=0.25,
            )
        )
    first, second = results
    assert first["mrp_r_star"] == 5
    assert second["mrp_r_star"] == 30
    # r* 之外的 test 指标逐位相同（test token 完全没变）。
    for key in (
        "n_test_tokens",
        "nn_fr",
        "ttr_10",
        "ttr_50",
        "symttr_10",
        "symttr_100",
        "hist_entropy",
        "rank_mean",
        "rank_median",
    ):
        assert first[key] == second[key], key
    # 显式固定 r 时，MRP-FR 也不受 calibration 影响。
    fixed = []
    for calib_rank in (5, 30):
        ranks = torch.cat([torch.full((100,), calib_rank, dtype=torch.long), test_ranks])
        x_tilde, target_ids, table = _rank_points(vocab, ranks)
        fixed.append(
            A.rank_metrics(
                x_tilde,
                target_ids,
                table,
                chunk=1024,
                aux_chunk=32,
                calibration_fraction=0.25,
                mrp_r=12,
            )
        )
    assert fixed[0]["mrp_fr"] == fixed[1]["mrp_fr"]
    assert fixed[0]["mrp_r_star_source"] == "override"


def test_mrp_calibration_too_small_raises():
    with pytest.raises(ValueError, match="校准集太小"):
        A.mrp_calibration_split(300, 0.25)  # 75 < 100
    with pytest.raises(ValueError, match="校准集太小"):
        A.rank_metrics(
            torch.zeros(300, 4, dtype=torch.float64),
            torch.zeros(300, dtype=torch.long),
            torch.eye(4, dtype=torch.float64),
            chunk=64,
            aux_chunk=8,
            calibration_fraction=0.25,
        )
    # 边界：n=400 × 0.25 = 100 恰好达到下限
    split = A.mrp_calibration_split(400, 0.25)
    assert split == {"n_calibration": 100, "n_test": 300, "start_test": 100, "n_total": 400}
    # 显式 --mrp-r 时不需要从 calibration 求 r*（切分仍需成立）
    assert A.mrp_calibration_split(300, 0.25, min_calibration_tokens=1)["n_calibration"] == 75
    with pytest.raises(ValueError):
        A.mrp_calibration_split(100, 1.0)


def test_mrp_legacy_no_split_keeps_old_behavior():
    """calibration_fraction=0（库级缺省）保留旧口径：r* 与指标都取自全体评测集。"""
    vocab, n = 64, 400
    ranks = torch.cat(
        [torch.full((100,), 5, dtype=torch.long), torch.full((300,), 20, dtype=torch.long)]
    )
    x_tilde, target_ids, table = _rank_points(vocab, ranks)
    legacy = A.rank_metrics(x_tilde, target_ids, table, chunk=1024, aux_chunk=32)
    assert legacy["n_tokens"] == n
    assert legacy["n_calibration_tokens"] == 0
    assert legacy["mrp_r_star"] == 20
    assert legacy["mrp_r_star_source"] == "evaluation_set"


# --------------------------------------------------------------------------- #
# 8. GAP-5：训练规模配置值
# --------------------------------------------------------------------------- #
def test_ablation_config_gap5_values():
    import yaml

    config_path = REPO_ROOT / "configs" / "ablation.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["data"]["max_examples"] == 20000
    assert raw["data"]["frac"] == 0.15  # 论文协议（只作记录）
    assert raw["train"]["steps"] == 800
    assert raw["train"]["batch_size"] == 1
    assert raw["train"]["max_len"] == 256
    assert raw["train"]["lr"] == pytest.approx(1.0e-3)
    assert raw["train"]["lm_dtype"] == "bfloat16"
    assert raw["sgt"]["sigma_max"] == pytest.approx(0.05)
    assert raw["selection"] == {"enabled": True, "every": 50, "calibration_examples": 50}
    assert raw["eval"] == {"calibration_fraction": 0.25}
    assert raw["sgt"]["max_len"] >= raw["train"]["max_len"]



def test_selection_checkpoint_state_reads_step_files(tmp_path):
    """GAP-7：每个选择点必须读**该步**的 checkpoint；整数倍步缺文件时不得冒充。"""
    final_state = {"input_proj.weight": torch.zeros(2, 2)}
    for saved_step in (4, 9):
        torch.save(
            {"sgt": {"input_proj.weight": torch.full((2, 2), float(saved_step))}},
            tmp_path / f"step_{saved_step}.pt",
        )
    state_5 = A._selection_checkpoint_state(tmp_path, 5, 5, final_state)
    assert torch.equal(state_5["input_proj.weight"], torch.full((2, 2), 4.0))
    state_10 = A._selection_checkpoint_state(tmp_path, 10, 5, final_state)
    assert torch.equal(state_10["input_proj.weight"], torch.full((2, 2), 9.0))
    # 末步不是 every 的整数倍：用内存里的最终模型（与末步 checkpoint 逐位相同）
    state_12 = A._selection_checkpoint_state(tmp_path, 12, 5, final_state)
    assert torch.equal(state_12["input_proj.weight"], final_state["input_proj.weight"])
    # 整数倍步缺 checkpoint -> 报错（静默用最终模型会让选择轨迹失真）
    (tmp_path / "step_4.pt").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        A._selection_checkpoint_state(tmp_path, 5, 5, final_state)


def test_best_checkpoint_path_naming():
    assert A.best_checkpoint_path("results/ckpt_abscos.pt") == Path(
        "results/ckpt_abscos_best.pt"
    )


# --------------------------------------------------------------------------- #
# 9. 训练探针 --skip-eval / MI 健壮化 / 位置编码初始化
# --------------------------------------------------------------------------- #
def _skip_eval_fixture(monkeypatch):
    """替换 ``main`` 的 IO 与重部件：tiny LM + 2 个合成 batch（不读 parquet/Alpaca/1B 模型）。

    同时把评测入口全部替换成「一调用就失败」，用来证明 ``--skip-eval`` 确实跳过了
    评测数据加载、MI/PAC 计算与 SymTTR 选择，而不是静默跑完却写 null。
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tiny = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
        )
    ).eval()
    for parameter in tiny.parameters():
        parameter.requires_grad_(False)

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    generator = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(2):
        ids = torch.randint(0, 32, (2, 8), generator=generator)
        batches.append(
            {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
        )

    raw = {
        "model": "tiny",
        "d": 8,
        "sgt": {
            "hidden": 8,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_mult": 2,
            "max_len": 8,
            "sigma_init": 0.5,
            "sigma_max": 0.05,
        },
        "train": {
            "steps": 2,
            "lr": 1.0e-3,
            "batch_size": 2,
            "max_len": 8,
            "seed": 0,
            "lm_dtype": "float32",
            "grad_clip": 1.0,
        },
        "data": {"max_examples": 4},
        "alphas": {"acs": 1.0},
        "variants": {"abscos": {"loss_terms": ["acs"]}},
        "selection": {"enabled": True, "every": 50, "calibration_examples": 50},
        "eval": {"calibration_fraction": 0.25},
    }
    monkeypatch.setattr(A, "read_yaml", lambda path: raw)
    monkeypatch.setattr(
        A,
        "load_train_examples",
        lambda **kwargs: (
            [{"question": "q", "response": "a"}] * 4,
            {"data_source": "synthetic", "n_examples": 4},
        ),
    )
    monkeypatch.setattr(A, "load_lm", lambda path, *, device, dtype: (_Tokenizer(), tiny))
    monkeypatch.setattr(
        A,
        "build_train_batches",
        lambda examples, tokenizer, **kwargs: (
            batches,
            {
                "n_batches": 2,
                "batch_size": 2,
                "max_len": 8,
                "padded_width": 8,
                "n_examples": 4,
                "tokens_per_epoch": 16,
                "dropped_last": False,
                "seed": 0,
            },
        ),
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("--skip-eval 不得进入评测/MI/PAC 或 SymTTR 选择")

    monkeypatch.setattr(A, "load_eval_examples", _unexpected)
    monkeypatch.setattr(A, "evaluate_sgt", _unexpected)
    monkeypatch.setattr(A, "symttr_on_sequences", _unexpected)
    return raw


def test_skip_eval_trains_two_steps_and_writes_skipped_json(monkeypatch, tmp_path):
    """--skip-eval：2 步训练 + checkpoint 后写完整 JSON，不加载评测集、不算 MI/PAC。"""
    _skip_eval_fixture(monkeypatch)
    out = tmp_path / "probe.json"
    ckpt = tmp_path / "ckpt.pt"
    code = A.main(
        [
            "--config",
            str(tmp_path / "fake.yaml"),
            "--variant",
            "abscos",
            "--device",
            "cpu",
            "--max-steps",
            "2",
            "--max-examples",
            "4",
            "--skip-eval",
            "--ckpt",
            str(ckpt),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["eval"] == {"status": "skipped"}
    assert payload["skip_eval"] is True
    assert payload["steps"] == 2
    assert payload["selected_step"] == 2
    assert len(payload["train_history"]) == 2
    assert {"sigma_mean", "cos_mean", "loss_u"} <= set(payload["train_history"][0])
    # selection 必须显式标注「被跳过」，而不是静默缺失（yaml 里 selection.enabled=true）
    assert payload["selection"]["enabled"] is False
    assert payload["selection"]["skip_eval"] is True
    assert payload["selection"]["selection_configured_enabled"] is True
    assert "reason" in payload["selection"]
    assert ckpt.is_file()
    # 训练期临时 checkpoint 目录只服务 SymTTR 选择，探针模式不再需要
    select_dir = payload["config_echo"]["selection"]["checkpoint_dir"]
    assert select_dir is not None and not Path(select_dir).exists()
    for key in ("data_s", "lm_load_s", "sgt_init_s", "train_s", "ckpt_s", "total_s"):
        assert key in payload["timings"]
    json.dumps(payload, allow_nan=False)  # 严格可序列化


def test_feature_mi_undefined_sigma_too_small_does_not_crash():
    """log σ² = −800（σ→0）：MI 记 null + mi_status，不抛错；正常输入仍为 ok。"""
    torch.manual_seed(1)
    n, d = 32, 4
    x = torch.randn(n, d)
    zeros = torch.zeros_like(x)
    log_var_ok = torch.full((n, d), math.log(0.01))

    ok = A._mi_block(x, x, zeros, log_var_ok, log_var_ok, n_pairs=256, seed=3, vocab_size=64)
    assert ok["mi_status"] == "ok"
    assert ok["mi_trigger"] == "ok"
    assert math.isfinite(ok["mi_feature_mean"])
    assert ok["pac_adv"] == pytest.approx(metrics.pac_adv(ok["mi_feature_mean"], 64))

    log_var_bad = torch.full((n, d), -800.0)
    bad = A._mi_block(x, x, zeros, log_var_bad, log_var_bad, n_pairs=256, seed=3, vocab_size=64)
    assert bad["mi_status"] == "undefined_sigma_too_small"
    assert bad["mi_trigger"] == "sigma_below_floor"
    for key in ("mi_feature_mean", "mi_sum", "mi_feature_median", "pac_adv"):
        assert bad[key] is None, key
    assert "undefined" in bad["mi_note"]
    assert bad["pac_adv_status"] == "skipped_mi_undefined"

    # 直接调用 feature_mi 也不抛错（任务单口径：返回值里 pac_adv 同样为 None）
    direct = A.feature_mi(x, x, zeros, log_var_bad, log_var_bad, n_pairs=32, seed=0)
    assert direct["mi_status"] == "undefined_sigma_too_small"
    assert direct["pac_adv"] is None
    json.dumps(A.json_safe(bad), allow_nan=False)


def test_feature_mi_nonfinite_term_is_undefined_not_crash():
    """σ² 未过预检门但 exp(-log σ²)·residual² 在 float32 溢出：同样返回占位块。"""
    n, d = 8, 4
    huge = torch.full((n, d), 1e20, dtype=torch.float32)
    zeros = torch.zeros_like(huge)
    log_var_edge = torch.full((n, d), A.MI_LOG_SIGMA2_FLOOR + 1.0, dtype=torch.float32)
    block = A.feature_mi(huge, zeros, zeros, log_var_edge, log_var_edge, n_pairs=8, seed=0)
    assert block["mi_status"] == "undefined_sigma_too_small"
    assert block["mi_trigger"] == "nonfinite_term"
    assert block["mi_feature_mean"] is None


def test_feature_mi_still_raises_on_real_bugs():
    """形状/NaN 输入是调用方 bug，不能被 σ 门吞掉：照旧抛 ValueError。"""
    n, d = 8, 4
    x = torch.randn(n, d)
    zeros = torch.zeros_like(x)
    log_var = torch.zeros(n, d)
    with pytest.raises(ValueError, match="特征维"):
        A.feature_mi(x[:, :2], x, zeros, log_var, log_var, n_pairs=8, seed=0)
    nan = x.clone()
    nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match="非有限"):
        A.feature_mi(nan, x, zeros, log_var, log_var, n_pairs=8, seed=0)
    with pytest.raises(ValueError):
        A.feature_mi(x, x, zeros, log_var, log_var, n_pairs=0, seed=0)


def test_evaluate_sgt_marks_mi_gate_when_sigma_too_small():
    """evaluate_sgt 端到端：σ→0 时 MI/PAC 为 null，mi_gate 显式判为不可用；秩指标不受影响。"""

    class _UndefinedSigmaSGT(torch.nn.Module):
        def forward(self, x, attention_mask=None):
            return torch.zeros_like(x), torch.full_like(x, -800.0)

    vocab = 64
    table = torch.arange(1, vocab + 1, dtype=torch.float64).reshape(-1, 1)
    result = A.evaluate_sgt(
        _UndefinedSigmaSGT(),
        table,
        [[3, 10, 20, 30]],
        pad_id=0,
        eval_batch=1,
        seed=0,
        rank_chunk=16,
        n_pairs=64,
    )
    assert result["feature_mi"]["mi_status"] == "undefined_sigma_too_small"
    assert result["metrics"]["mi_status"] == "undefined_sigma_too_small"
    for key in ("mi_feature_mean", "mi_sum", "mi_feature_median", "pac_adv"):
        assert result["metrics"][key] is None, key
    gate = result["mi_gate"]
    assert gate["mi_status"] == "undefined_sigma_too_small"
    assert gate["mi_and_pac_conclusive"] is False
    assert "undefined" in gate["note"]
    # σ 小不等于「评测失败」：秩指标仍是有限数值
    assert 0.0 <= result["metrics"]["nn_fr"] <= 100.0
    assert math.isfinite(result["metrics"]["symttr_100"])
    json.dumps(A.json_safe(result), allow_nan=False)


def test_sgt_pos_emb_init_std_default_and_legacy_control():
    """SPEC A1 实现选择：pos_emb 默认 N(0,0.02²)；std=1.0 复现旧 N(0,1)（对照）。"""
    from sgt.model import SGTModel

    torch.manual_seed(0)
    default = SGTModel(8, hidden=512, n_layers=1, n_heads=4, max_len=512, sigma_init=0.5)
    std_default = float(default.pos_emb.weight.detach().std())
    assert abs(std_default - 0.02) <= 0.3 * 0.02  # n = 512×512 样本，允许 30% 误差

    torch.manual_seed(0)
    legacy = SGTModel(
        8, hidden=512, n_layers=1, n_heads=4, max_len=512, sigma_init=0.5,
        pos_emb_init_std=1.0,
    )
    std_legacy = float(legacy.pos_emb.weight.detach().std())
    assert abs(std_legacy - 1.0) <= 0.3 * 1.0

    # 只改位置编码初始化：零初始化输出头与 sigma_init 语义原样保留（A21 不触碰）。
    for model in (default, legacy):
        assert float(model.head.weight.detach().abs().max()) == 0.0
        assert torch.equal(model.head.bias[:8], torch.zeros(8))
        assert torch.equal(model.head.bias[8:], torch.full((8,), math.log(0.5**2)))

    with pytest.raises(ValueError, match="pos_emb_init_std"):
        SGTModel(8, hidden=16, n_heads=4, pos_emb_init_std=-0.1)


def test_symttr_on_sequences_is_reproducible_for_same_seed():
    """GAP-7：选择指标用固定 seed 重新采样，同一 seed 的两次评测逐位一致。"""

    class UnitNoiseSGT(torch.nn.Module):
        def forward(self, x, attention_mask=None):  # noqa: D401 - 测试桩
            return torch.zeros_like(x), torch.zeros_like(x)  # µ=0, σ=1

    vocab = 64
    table = torch.arange(1, vocab + 1, dtype=torch.float64).reshape(-1, 1)
    sequences = [[3, 10, 20, 30], [5, 6, 7]]
    kwargs = dict(pad_id=0, batch_seqs=1, seed=123, rank_chunk=16, k=10)
    first = A.symttr_on_sequences(UnitNoiseSGT(), table, sequences, **kwargs)
    second = A.symttr_on_sequences(UnitNoiseSGT(), table, sequences, **kwargs)
    assert first["symttr"] == second["symttr"]
    assert first["n_tokens"] == second["n_tokens"] == 7


# --------------------------------------------------------------------------- #
# 10. A1 / A21：sgt: 段的 4 个结构超参接入两个实验脚本
# --------------------------------------------------------------------------- #
def test_sgt_structural_options_wired_from_yaml(monkeypatch):
    """``sgt:`` 段的 final_norm/mu_scale/sigma_floor/pos_encoding 必须原样传给 SGTModel。

    两个实验脚本各有一条构造路径（``run_ablation.build_sgt_model`` /
    ``train_sgt._build_sgt_model``）：monkeypatch 掉模块级 ``SGTModel`` 捕获 kwargs，
    不真的建模型/加载 LM。同时验证缺省回落到 ``True/None/None/"learned"``。
    """
    import yaml

    import train_sgt as T

    captured = []

    class _Recorder:
        def __init__(self, *args, **kwargs):
            captured.append({"args": args, "kwargs": kwargs})

    sgt_raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "ablation.yaml").read_text(encoding="utf-8")
    )["sgt"]
    expected_from_yaml = {
        "final_norm": True,
        "mu_scale": pytest.approx(0.1),
        "sigma_floor": pytest.approx(0.005),
        "pos_encoding": "sinusoidal",
    }
    expected_defaults = {
        "final_norm": True,
        "mu_scale": None,
        "sigma_floor": None,
        "pos_encoding": "learned",
    }
    assert A.resolve_sgt_options(sgt_raw) == expected_from_yaml
    assert T._resolve_sgt_options(sgt_raw) == expected_from_yaml
    assert A.resolve_sgt_options({}) == expected_defaults
    assert T._resolve_sgt_options({}) == expected_defaults

    monkeypatch.setattr(A, "SGTModel", _Recorder)
    monkeypatch.setattr(T, "SGTModel", _Recorder)
    builders = (
        lambda raw: A.build_sgt_model(8, raw, sigma_min=1e-6),
        lambda raw: T._build_sgt_model(8, raw, sigma_min=1e-6),
    )
    for builder in builders:
        for raw, expected in ((sgt_raw, expected_from_yaml), ({}, expected_defaults)):
            captured.clear()
            builder(raw)
            assert len(captured) == 1
            kwargs = captured[0]["kwargs"]
            for key, value in expected.items():
                assert kwargs[key] == value, f"{key}: {kwargs[key]!r} != {value!r}"
