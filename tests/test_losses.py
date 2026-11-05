"""M2 损失函数的数值验证：Alg.1 与 Eq (8)–(13)、效用损失 L_U。

对拍真值全部由本文件内的朴素逐元素循环独立计算，不复用 `sgt.losses` 的向量化路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgt import losses as L  # noqa: E402

F64 = torch.float64


# --------------------------------------------------------------------------- #
# 数据与朴素参考实现
# --------------------------------------------------------------------------- #
def _mi_data(*, B=2, T=3, d=4, Bp=None, Tp=None, seed=7, dtype=F64):
    """构造两侧独立 batch；pair 模式要求两侧同形（Bp=B, Tp=T）。"""
    Bp = B if Bp is None else Bp
    Tp = T if Tp is None else Tp
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=dtype)

    x = rand(B, T, d)
    mu = 0.3 * rand(B, T, d)
    x_tilde = x + mu + 0.5 * rand(B, T, d)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -1] = False
    mask_prime = torch.ones(Bp, Tp, dtype=torch.bool)
    if Bp * Tp > 1:
        mask_prime[-1, 0] = False
    return {
        "x": x,
        "mu": mu,
        "x_tilde": x_tilde,
        "x_prime": rand(Bp, Tp, d),
        "mu_prime": 0.3 * rand(Bp, Tp, d),
        "lvp": torch.log(torch.rand(Bp, Tp, d, generator=g, dtype=dtype) + 0.2),
        "lv": torch.log(torch.rand(B, T, d, generator=g, dtype=dtype) + 0.2),
        "mask": mask,
        "mask_prime": mask_prime,
    }


def _assert_close_rel(got, exp, tol=1e-9):
    got, exp = float(got), float(exp)
    diff = abs(got - exp)
    assert diff <= tol * max(1.0, abs(exp)), f"got={got!r} exp={exp!r} |diff|={diff:.3e}"


def _naive_mi_pair(xt, xp, mup, lvp, lv, mask=None, half_factor=False):
    """Eq (8) 直译：同下标配对，逐 pair 逐 j 三重循环。"""
    B, T, d = xt.shape
    vals = []
    for b in range(B):
        for t in range(T):
            if mask is not None and not bool(mask[b, t]):
                continue
            total = 0.0
            for j in range(d):
                total += float(lvp[b, t, j] - lv[b, t, j])
                diff = float(xt[b, t, j] - xp[b, t, j] - mup[b, t, j])
                total += diff**2 / float(torch.exp(lvp[b, t, j]))
            vals.append(total)
    value = sum(vals) / len(vals)
    return 0.5 * value if half_factor else value


def _naive_mi_full(xt, xp, mup, lvp, lv, mask=None, mask_prime=None):
    """Eq (8) 直译：全交叉 (B·T)×(B'·T')，逐元素循环。"""
    B, T, d = xt.shape
    Bp, Tp, _ = xp.shape
    vals = []
    for b in range(B):
        for t in range(T):
            if mask is not None and not bool(mask[b, t]):
                continue
            for b2 in range(Bp):
                for t2 in range(Tp):
                    if mask_prime is not None and not bool(mask_prime[b2, t2]):
                        continue
                    total = 0.0
                    for j in range(d):
                        total += float(lvp[b2, t2, j] - lv[b, t, j])
                        diff = float(xt[b, t, j] - xp[b2, t2, j] - mup[b2, t2, j])
                        total += diff**2 / float(torch.exp(lvp[b2, t2, j]))
                    vals.append(total)
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------- #
# 1. mi_loss 对拍
# --------------------------------------------------------------------------- #
def test_mi_pair_matches_naive_loop():
    data = _mi_data()
    got = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"]
    )
    exp = _naive_mi_pair(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"]
    )
    _assert_close_rel(got, exp)

    # 覆盖 mask：同一掩码作用于两侧
    got_m = L.mi_loss(
        data["x_tilde"],
        data["x_prime"],
        data["mu_prime"],
        data["lvp"],
        data["lv"],
        mask=data["mask"],
    )
    exp_m = _naive_mi_pair(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"],
    )
    _assert_close_rel(got_m, exp_m)
    assert abs(float(got_m) - float(got)) > 1e-6, "mask 未改变结果，掩码可能未生效"


def test_mi_pair_matches_naive_on_other_shapes():
    data = _mi_data(B=3, T=2, d=5, seed=101)
    got = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"],
    )
    exp = _naive_mi_pair(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"],
    )
    _assert_close_rel(got, exp)


def test_mi_full_matches_naive_loop():
    data = _mi_data(B=2, T=2, d=3, Bp=2, Tp=2, seed=11)
    got = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"], mask_prime=data["mask_prime"], mode="full",
    )
    exp = _naive_mi_full(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"], mask_prime=data["mask_prime"],
    )
    _assert_close_rel(got, exp)

    # mask_prime=None → x' 侧全有效
    got_np = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"], mode="full",
    )
    exp_np = _naive_mi_full(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"],
    )
    _assert_close_rel(got_np, exp_np)


def test_mi_full_matches_naive_with_ragged_prime_batch():
    """x' 的 batch/长度与 x 不同，仍按全交叉计算。"""
    data = _mi_data(B=2, T=3, d=4, Bp=2, Tp=2, seed=17)
    got = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"], mask_prime=data["mask_prime"], mode="full",
    )
    exp = _naive_mi_full(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"], mask_prime=data["mask_prime"],
    )
    _assert_close_rel(got, exp)


def test_mi_full_reduces_to_pair_for_single_prime_token():
    data = _mi_data(B=2, T=3, d=4, Bp=1, Tp=1, seed=13)
    B, T, d = data["x_tilde"].shape
    got_full = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"], mode="full",
    )
    exp_pair = L.mi_loss(
        data["x_tilde"],
        data["x_prime"].expand(B, T, d),
        data["mu_prime"].expand(B, T, d),
        data["lvp"].expand(B, T, d),
        data["lv"],
        mask=data["mask"],
    )
    _assert_close_rel(got_full, float(exp_pair), tol=1e-12)


def test_mi_full_chunked_accumulation_matches_naive(monkeypatch):
    data = _mi_data(B=2, T=2, d=3, Bp=2, Tp=3, seed=23)
    args = (
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
    )
    exp = _naive_mi_full(*args, mask=data["mask"], mask_prime=data["mask_prime"])
    got = L.mi_loss(
        *args, mask=data["mask"], mask_prime=data["mask_prime"], mode="full"
    )
    _assert_close_rel(got, exp)

    # 预算置 1 → 每个块只含 1 个 x' token，强制走多块累加路径
    monkeypatch.setattr(L, "_FULL_CHUNK_BUDGET", 1)
    got_chunked = L.mi_loss(
        *args, mask=data["mask"], mask_prime=data["mask_prime"], mode="full"
    )
    assert float(got_chunked) == pytest.approx(float(exp), abs=1e-12)


def test_mi_half_factor_is_exactly_half():
    data = _mi_data()
    full = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"],
    )
    half = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mask=data["mask"], half_factor=True,
    )
    assert float(half) == pytest.approx(0.5 * float(full), abs=1e-12)
    fullf = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mode="full",
    )
    halff = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
        mode="full", half_factor=True,
    )
    assert float(halff) == pytest.approx(0.5 * float(fullf), abs=1e-12)


def test_mi_loss_gradients_reach_log_sigma2():
    data = _mi_data()
    lv = data["lv"].clone().requires_grad_(True)
    lvp = data["lvp"].clone().requires_grad_(True)
    loss = L.mi_loss(
        data["x_tilde"], data["x_prime"], data["mu_prime"], lvp, lv, mask=data["mask"]
    )
    loss.backward()
    for grad in (lv.grad, lvp.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# 2. Eq (9) / Eq (13) 余弦
# --------------------------------------------------------------------------- #
def test_abs_cos_and_cos_sim_hand_cases():
    x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]], dtype=F64)
    x_tilde = torch.tensor(
        [[[0.0, 2.0, 0.0], [0.0, -3.0, 0.0], [2.0, 4.0, 0.0]]], dtype=F64
    )
    # 第 1 对正交 → 0；第 2 对反平行 → |·|=1 / 有符号 −1；第 3 对同向 → 1 / +1
    assert float(L.abs_cos_loss(x, x_tilde)) == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert float(L.cos_sim_loss(x, x_tilde)) == pytest.approx(0.0, abs=1e-12)

    mask = torch.tensor([[True, False, True]])
    assert float(L.abs_cos_loss(x, x_tilde, mask=mask)) == pytest.approx(0.5, abs=1e-12)
    assert float(L.cos_sim_loss(x, x_tilde, mask=mask)) == pytest.approx(0.5, abs=1e-12)


def test_cosines_match_torch_cosine_similarity_elementwise():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(2, 5, 7, generator=g, dtype=F64)
    b = torch.randn(2, 5, 7, generator=g, dtype=F64)
    ref = F.cosine_similarity(a, b, dim=-1)

    per_cos = L.cos_sim_loss(a, b, reduce=False)
    per_abs = L.abs_cos_loss(a, b, reduce=False)
    assert per_cos.shape == (2, 5) and per_abs.shape == (2, 5)
    assert torch.allclose(per_cos, ref, atol=1e-12, rtol=0)
    assert torch.allclose(per_abs, ref.abs(), atol=1e-12, rtol=0)

    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 1] = False
    mask[1, 4] = False
    assert float(L.cos_sim_loss(a, b, mask=mask)) == pytest.approx(
        float(ref[mask].mean()), abs=1e-12
    )
    assert float(L.abs_cos_loss(a, b, mask=mask)) == pytest.approx(
        float(ref.abs()[mask].mean()), abs=1e-12
    )


def test_cosine_zero_vector_is_finite():
    x = torch.zeros(1, 2, 4, dtype=F64)
    x[0, 0, 0] = 1.0
    assert torch.isfinite(L.abs_cos_loss(x, x)).all()
    assert torch.isfinite(L.cos_sim_loss(x, x)).all()
    assert torch.isfinite(L.abs_cos_loss(x, x, reduce=False)).all()


# --------------------------------------------------------------------------- #
# 3. Eq (10) 中位数范数惩罚
# --------------------------------------------------------------------------- #
def test_median_norm_hand_case():
    v = torch.tensor([[[0.0, 0.5], [0.0, 1.0], [0.0, 2.0]]], dtype=F64)  # 范数 0.5/1/2
    assert float(L.median_norm_loss(v, 1.0)) == pytest.approx(0.5, abs=1e-12)
    assert float(L.median_norm_loss(v, torch.tensor(1.0, dtype=F64))) == pytest.approx(
        0.5, abs=1e-12
    )

    mask = torch.tensor([[True, False, True]])
    assert float(L.median_norm_loss(v, 1.0, mask=mask)) == pytest.approx(
        (0.5 + 1.0) / 2, abs=1e-12
    )

    # 非 1 的 T 与一般范数：手算 |3−2|+|4−2| 平均
    w = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]], dtype=F64)
    assert float(L.median_norm_loss(w, 2.0)) == pytest.approx(1.5, abs=1e-12)


# --------------------------------------------------------------------------- #
# 4. Eq (12) 分量熵
# --------------------------------------------------------------------------- #
def test_comp_entropy_matches_minus_log_sigma2_sum():
    lv = torch.log(
        torch.tensor([[[0.5, 2.0], [1.0, 4.0], [4.0, 8.0]]], dtype=F64)
    )
    per_token = -lv.sum(dim=-1)
    assert float(L.comp_entropy_loss(lv)) == pytest.approx(
        float(per_token.mean()), abs=1e-12
    )

    mask = torch.tensor([[True, False, True]])
    assert float(L.comp_entropy_loss(lv, mask=mask)) == pytest.approx(
        float(per_token[mask].mean()), abs=1e-12
    )

    # 逐维平移 c：−Σ(logσ²+c) 应恰好减少 d·c
    c = 0.7
    d = lv.shape[-1]
    assert float(L.comp_entropy_loss(lv + c)) == pytest.approx(
        float(L.comp_entropy_loss(lv)) - d * c, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# 5. Eq (11) 混淆损失加权和
# --------------------------------------------------------------------------- #
def test_obfuscation_weighted_sum_and_linearity():
    mi = torch.tensor(1.0, dtype=F64)
    acs = torch.tensor(2.0, dtype=F64)
    mnp = torch.tensor(3.0, dtype=F64)

    total = L.obfuscation_loss(mi=mi, acs=acs, mnp=mnp, alphas=(2.0, 0.5, 3.0))
    assert float(total) == pytest.approx(2.0 * 1.0 + 0.5 * 2.0 + 3.0 * 3.0, abs=1e-12)

    # 接受 Python float
    assert float(L.obfuscation_loss(mi=1.0, acs=2.0, mnp=3.0)) == pytest.approx(
        6.0, abs=1e-12
    )
    # None 项按 0 处理
    assert float(L.obfuscation_loss(mi=1.5, alphas=(2.0, 5.0, 5.0))) == pytest.approx(
        3.0, abs=1e-12
    )
    assert float(L.obfuscation_loss()) == 0.0

    # 线性：α 整体翻倍 → 值翻倍；只改 α1 → Δ = (α1'−α1)·MI
    base = float(L.obfuscation_loss(mi=mi, acs=acs, mnp=mnp))
    doubled = float(L.obfuscation_loss(mi=mi, acs=acs, mnp=mnp, alphas=(2.0, 2.0, 2.0)))
    assert doubled == pytest.approx(2.0 * base, abs=1e-12)
    bumped = float(L.obfuscation_loss(mi=mi, acs=acs, mnp=mnp, alphas=(4.0, 1.0, 1.0)))
    assert bumped - base == pytest.approx(3.0 * 1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 6. 效用损失 L_U
# --------------------------------------------------------------------------- #
def _utility_data(seed=5):
    g = torch.Generator().manual_seed(seed)
    zc = torch.randn(2, 3, 5, generator=g, dtype=F64)
    zs = torch.randn(2, 3, 5, generator=g, dtype=F64)
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask[1, 2] = False
    return zc, zs, mask


def test_utility_loss_elementwise_matches_hand_formula():
    zc, zs, mask = _utility_data()
    p = torch.softmax(zc, dim=-1)
    per_token = -(p * torch.log_softmax(zs, dim=-1)).sum(dim=-1)  # [B, T]

    assert float(L.utility_loss(zc, zs)) == pytest.approx(
        float(per_token.mean()), abs=1e-12
    )
    assert float(L.utility_loss(zc, zs, attention_mask=mask)) == pytest.approx(
        float(per_token[mask].mean()), abs=1e-12
    )
    # 用 one-hot mask 通过公开 API 取出单个 token，验证“逐元素一致”
    for b in range(2):
        for t in range(3):
            one = torch.zeros(2, 3, dtype=torch.bool)
            one[b, t] = True
            assert float(L.utility_loss(zc, zs, attention_mask=one)) == pytest.approx(
                float(per_token[b, t]), abs=1e-12
            )


def test_utility_loss_mask_positions_do_not_matter():
    zc, zs, mask = _utility_data()
    value = float(L.utility_loss(zc, zs, attention_mask=mask))

    g = torch.Generator().manual_seed(99)
    n_masked = int((~mask).sum())
    zc2, zs2 = zc.clone(), zs.clone()
    zc2[~mask] = 10.0 * torch.randn(n_masked, 5, generator=g, dtype=F64)
    zs2[~mask] = 10.0 * torch.randn(n_masked, 5, generator=g, dtype=F64)
    assert float(L.utility_loss(zc2, zs2, attention_mask=mask)) == pytest.approx(
        value, abs=1e-12
    )


def test_utility_loss_temperature_scales_logits():
    zc, zs, _ = _utility_data()
    tau = 2.5
    p = torch.softmax(zc / tau, dim=-1)
    ref = float(-(p * torch.log_softmax(zs / tau, dim=-1)).sum(-1).mean())
    assert float(L.utility_loss(zc, zs, temperature=tau)) == pytest.approx(
        ref, abs=1e-12
    )
    # τ 只改变温度，不改变“交叉熵”的形式：τ=1 即原式
    assert float(L.utility_loss(zc, zs, temperature=1.0)) == pytest.approx(
        float(L.utility_loss(zc, zs)), abs=1e-15
    )


def test_utility_loss_identical_logits_equal_entropy():
    zc, _, _ = _utility_data()
    p = torch.softmax(zc, dim=-1)
    entropy = float(-(p * torch.log_softmax(zc, dim=-1)).sum(-1).mean())
    got = float(L.utility_loss(zc, zc))
    assert got == pytest.approx(entropy, abs=1e-12)
    assert got > 0.5, f"随机 logits 的分布熵应远离 0，实际 {got}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "任务单第 6 条「同分布时 L_U=0」与 L_U=CE(p_clean,p_sgt) 矛盾："
        "z_sgt=z_clean 时 L_U=H(p_clean)≠0（M2 门禁 c-utility-identity 已改为期望 H(p)），"
        "仅 one-hot 极限为 0"
    ),
)
def test_utility_loss_identical_logits_zero_claim_is_false():
    zc, _, _ = _utility_data()
    assert abs(float(L.utility_loss(zc, zc))) < 1e-6


def test_utility_loss_one_hot_limit_is_zero():
    zc = torch.tensor([[[40.0, 0.0, 0.0], [0.0, 40.0, 0.0]]], dtype=F64)
    assert abs(float(L.utility_loss(zc, zc))) < 1e-6


# --------------------------------------------------------------------------- #
# 7. 有限性 / dtype / 不修改输入
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_losses_finite_dtype_and_non_destructive(dtype):
    data = _mi_data(dtype=dtype)
    g = torch.Generator().manual_seed(31)
    zc = torch.randn(2, 3, 5, generator=g, dtype=dtype)
    zs = torch.randn(2, 3, 5, generator=g, dtype=dtype)
    one = torch.tensor(1.0, dtype=dtype)
    inputs = dict(data, zc=zc, zs=zs, one=one)
    snapshot = {k: v.clone() for k, v in inputs.items()}

    results = {
        "mi-pair": L.mi_loss(
            data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
            mask=data["mask"],
        ),
        "mi-pair-nomask": L.mi_loss(
            data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"]
        ),
        "mi-pair-half": L.mi_loss(
            data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
            mask=data["mask"], half_factor=True,
        ),
        "mi-full": L.mi_loss(
            data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
            mask=data["mask"], mask_prime=data["mask_prime"], mode="full",
        ),
        "abs-cos": L.abs_cos_loss(data["x"], data["x_tilde"], mask=data["mask"]),
        "abs-cos-per": L.abs_cos_loss(data["x"], data["x_tilde"], reduce=False),
        "cos-sim": L.cos_sim_loss(data["x"], data["x_tilde"], mask=data["mask"]),
        "cos-sim-per": L.cos_sim_loss(data["x"], data["x_tilde"], reduce=False),
        "mnp": L.median_norm_loss(data["x"], 1.0, mask=data["mask"]),
        "cge": L.comp_entropy_loss(data["lv"], mask=data["mask"]),
        "obf": L.obfuscation_loss(mi=one, acs=2.0 * one, mnp=3.0 * one),
        "util": L.utility_loss(zc, zs, attention_mask=data["mask"]),
    }
    for name, value in results.items():
        assert torch.isfinite(value).all(), f"{name} 出现非有限值"
        if name.endswith("-per"):
            assert value.dtype == dtype, f"{name}: {value.dtype} != {dtype}"
            assert value.shape == data["x"].shape[:-1], name
        else:
            assert value.ndim == 0, f"{name} 应为 0 维"
            assert value.dtype == dtype, f"{name}: {value.dtype} != {dtype}"

    for key, before in snapshot.items():
        assert torch.equal(inputs[key], before), f"输入 {key} 被原地修改"


def test_dtype_promotion_follows_inputs():
    d32 = _mi_data(dtype=torch.float32)
    v64 = L.mi_loss(
        d32["x_tilde"], d32["x_prime"], d32["mu_prime"], d32["lvp"].double(), d32["lv"]
    )
    assert v64.dtype == torch.float64
    v32 = L.mi_loss(
        d32["x_tilde"], d32["x_prime"], d32["mu_prime"], d32["lvp"], d32["lv"]
    )
    assert v32.dtype == torch.float32


# --------------------------------------------------------------------------- #
# 边界与错误路径
# --------------------------------------------------------------------------- #
def test_mi_error_paths():
    data = _mi_data()
    with pytest.raises(ValueError):
        L.mi_loss(
            data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
            mask_prime=data["mask_prime"],
        )  # pair 模式不接受 mask_prime
    with pytest.raises(ValueError):
        L.mi_loss(
            data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
            mode="diag",
        )
    with pytest.raises(ValueError):
        L.mi_loss(
            data["x_tilde"], data["x_prime"][:, :2], data["mu_prime"][:, :2],
            data["lvp"][:, :2], data["lv"],
        )  # pair 模式形状必须相同

    no_valid = torch.zeros(data["mask"].shape, dtype=torch.bool)
    with pytest.raises(ValueError):
        L.mi_loss(
            data["x_tilde"], data["x_prime"], data["mu_prime"], data["lvp"], data["lv"],
            mask=no_valid,
        )


def test_misc_error_paths():
    with pytest.raises(ValueError):
        L.utility_loss(torch.randn(2, 3, 5), torch.randn(2, 3, 4))
    with pytest.raises(ValueError):
        L.utility_loss(torch.randn(2, 3, 5), torch.randn(2, 3, 5), temperature=0.0)
    with pytest.raises(ValueError):
        L.utility_loss(
            torch.randn(2, 3, 5), torch.randn(2, 3, 5),
            attention_mask=torch.ones(3, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        L.comp_entropy_loss(
            torch.randn(2, 3, 4), mask=torch.zeros(2, 3, dtype=torch.bool)
        )
    with pytest.raises(ValueError):
        L.obfuscation_loss(mi=torch.tensor(1.0), alphas=(1.0, 1.0))
    with pytest.raises(ValueError):
        L.median_norm_loss(torch.randn(2, 3, 4), torch.ones(2))
