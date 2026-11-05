"""M1 理论核心的数值验证：Eq (1)/(4)/(5)/(6)/(7)、Prop 3.3、Cor 3.4。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgt import theory as T  # noqa: E402
from sgt.sampling import sample_sgt  # noqa: E402


def test_eq1_exact_with_given_u():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 6, 4, generator=g, dtype=torch.float64)
    mu = torch.randn(3, 6, 4, generator=g, dtype=torch.float64)
    sigma = torch.rand(3, 6, 4, generator=g, dtype=torch.float64) + 0.1
    u = torch.randn(3, 6, 4, generator=g, dtype=torch.float64)
    x0, mu0, sigma0 = x.clone(), mu.clone(), sigma.clone()
    out = sample_sgt(x, mu, sigma, u=u)
    assert torch.equal(out, x + mu + sigma * u)
    assert torch.equal(x, x0) and torch.equal(mu, mu0) and torch.equal(sigma, sigma0)


def test_independent_sampling_and_determinism():
    n, d = 200_000, 3
    mu_v = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    sg_v = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    x = torch.zeros(n, d, dtype=torch.float64)
    mu = mu_v.expand(n, d).clone()
    sigma = sg_v.expand(n, d).clone()
    out = sample_sgt(x, mu, sigma, generator=torch.Generator().manual_seed(1))
    resid = out - (x + mu)
    cov = torch.cov(resid.T)
    target = torch.diag(sg_v**2)
    rel_err = float((cov - target).abs().max() / target.abs().max())
    assert rel_err < 0.02, f"协方差相对误差 {rel_err:.4%}"

    a = sample_sgt(x, mu, sigma, generator=torch.Generator().manual_seed(7))
    b = sample_sgt(x, mu, sigma, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_prop33_normal_marginals():
    n, d = 20_000, 3
    mu_v = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    sg_v = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    x = torch.zeros(n, d, dtype=torch.float64)
    out = sample_sgt(
        x,
        mu_v.expand(n, d).clone(),
        sg_v.expand(n, d).clone(),
        generator=torch.Generator().manual_seed(20250611),
    )
    for j in range(d):
        cdf = lambda t, m=float(mu_v[j]), s=float(sg_v[j]): stats.norm.cdf(t, m, s)
        p = float(stats.kstest(out[:, j].numpy(), cdf).pvalue)
        assert p > 0.01, f"dim {j} KS p={p}"


def test_cor34_gmm_sample():
    n_comp, n_samples, d = 50, 20_000, 2
    g = torch.Generator().manual_seed(123)
    weights = torch.softmax(torch.randn(n_comp, generator=g, dtype=torch.float64), dim=0)
    means = torch.randn(n_comp, d, generator=g, dtype=torch.float64) + torch.tensor(
        [3.0, 2.0], dtype=torch.float64
    )
    log_vars = torch.log(
        torch.rand(n_comp, d, generator=g, dtype=torch.float64) + 0.2
    )
    samples = T._gmm_sample(weights, means, log_vars, n_samples, generator=g)

    w = weights.numpy()
    m = means.numpy()
    lv = log_vars.numpy()
    for j in range(d):
        sd = np.exp(0.5 * lv[:, j])

        def cdf(t, j=j, sd=sd):
            return np.sum(
                w[:, None] * stats.norm.cdf(t[None, :], m[:, j][:, None], sd[:, None]),
                axis=0,
            )

        p = float(stats.kstest(samples[:, j].numpy(), cdf).pvalue)
        assert p > 0.01, f"dim {j} KS p={p}"

    theo_mean = (weights[:, None] * means).sum(dim=0)
    rel_err = ((samples.mean(dim=0) - theo_mean).abs() / theo_mean.abs()).max()
    assert float(rel_err) < 0.01, f"均值相对误差 {float(rel_err):.4%}"


def test_gmm_log_prob_and_pdf_match_scipy():
    g = torch.Generator().manual_seed(42)
    means = torch.randn(5, 3, generator=g, dtype=torch.float64) * 1.5
    log_vars = torch.log(torch.rand(5, 3, generator=g, dtype=torch.float64) + 0.2)
    z = torch.randn(4, 3, generator=g, dtype=torch.float64)
    log_prob = T.gmm_log_prob(z, means, log_vars)
    ref_log_prob = np.stack(
        [
            stats.multivariate_normal(
                means[i].numpy(), np.diag(np.exp(log_vars[i].numpy()))
            ).logpdf(z.numpy())
            for i in range(5)
        ],
        axis=-1,
    )
    assert np.allclose(log_prob.numpy(), ref_log_prob, atol=1e-12)

    weights = torch.softmax(
        torch.randn(5, generator=g, dtype=torch.float64), dim=0
    )
    ref_pdf = np.sum(
        [
            float(weights[i])
            * stats.multivariate_normal(
                means[i].numpy(), np.diag(np.exp(log_vars[i].numpy()))
            ).pdf(z.numpy())
            for i in range(5)
        ],
        axis=0,
    )
    pdf = T.gmm_pdf(z, means, log_vars, weights)
    assert np.allclose(pdf.numpy(), ref_pdf, atol=1e-12)


def test_eq5_eq7_mi_vs_quadrature():
    weights = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    means = torch.tensor([[0.0], [2.0], [-1.5]], dtype=torch.float64)
    log_vars = torch.log(torch.tensor([[0.3], [0.8], [0.2]], dtype=torch.float64))
    mi_mc = T.mutual_information_mc(
        weights, means, log_vars, 200_000,
        generator=torch.Generator().manual_seed(20250611),
    )
    mi_true = T.gmm_entropy_quad_1d(weights, means, log_vars) - T.cond_entropy(
        weights, log_vars
    )
    rel_err = abs(mi_mc - mi_true) / abs(mi_true)
    assert rel_err < 0.02, f"MI 相对误差 {rel_err:.4%}（MC={mi_mc}, 真值={mi_true}）"


@pytest.mark.xfail(
    strict=True,
    reason="F29：Eq (6) 的 '≤ 分量熵加权和' 不成立（正确上界为 Σw_i h_i + H(w)）",
)
def test_eq6_claimed_upper_bound_is_false():
    # 保留任务单原始断言作为反例记录；实际判据见下面两个测试。
    g = torch.Generator().manual_seed(20250611)
    gaps = []
    for _ in range(20):
        weights = torch.softmax(torch.randn(4, generator=g, dtype=torch.float64), dim=0)
        means = torch.randn(4, 1, generator=g, dtype=torch.float64) * 2
        log_vars = torch.log(
            torch.rand(4, 1, generator=g, dtype=torch.float64) + 0.1
        )
        gaps.append(
            T.gmm_entropy_quad_1d(weights, means, log_vars)
            - T.gmm_entropy_upper_bound(weights, log_vars)
        )
    assert max(gaps) <= 1e-9, f"h_quad > upper_bound，最大差 {max(gaps):.6e} nats"


def test_eq6_true_upper_bound():
    g = torch.Generator().manual_seed(20250611)
    for _ in range(20):
        weights = torch.softmax(torch.randn(4, generator=g, dtype=torch.float64), dim=0)
        means = torch.randn(4, 1, generator=g, dtype=torch.float64) * 2
        log_vars = torch.log(
            torch.rand(4, 1, generator=g, dtype=torch.float64) + 0.1
        )
        h_mix = T.gmm_entropy_quad_1d(weights, means, log_vars)
        h_w = float(-(weights * torch.log(weights)).sum())
        true_bound = T.cond_entropy(weights, log_vars) + h_w
        assert h_mix <= true_bound + 1e-9, f"真上界违反：{h_mix} > {true_bound}"


def test_eq6_separated_components_counterexample():
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    means = torch.tensor([[-5.0], [5.0]], dtype=torch.float64)
    log_vars = torch.full((2, 1), math.log(0.1**2), dtype=torch.float64)
    h_mix = T.gmm_entropy_quad_1d(weights, means, log_vars)
    h_cond = T.cond_entropy(weights, log_vars)
    assert h_mix - h_cond > 0.1
    assert abs((h_mix - h_cond) - math.log(2.0)) < 1e-9


def test_pac_adv_inversion():
    n = 128_256
    delta0 = 1.0 / n
    expected = [
        (1.11, 0.1269, 0.0005),
        (1.48, 0.1639, 0.0005),
        (1.91, 0.2056, 0.0005),
        (0.26, 0.0354, 0.0015),
        (0.28, 0.0374, 0.0015),
        (0.23, 0.0303, 0.002),
    ]
    for mi, want, tol in expected:
        got = T.invert_pac_adv(mi, n)
        assert abs(got - want) <= tol, f"MI={mi}: got {got}, want {want}±{tol}"
        residual = abs(T.delta_kl(got, delta0) - mi)
        assert residual < 1e-8, f"MI={mi}: |Δ_KL(δ)-MI|={residual:.3e}"
    assert T.invert_pac_adv(22.19, n) == 1.0

    grid = torch.linspace(delta0, 1.0, 4001)
    vals = [T.delta_kl(float(t), delta0) for t in grid]
    assert all(b >= a for a, b in zip(vals, vals[1:])), "Δ_KL 在 [δ0,1] 上不单调"
