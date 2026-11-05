"""M1 理论核心门禁：Eq (1)/(2)/(4)/(5)/(6)/(7)、Prop 3.3、Cor 3.4 的数值验证。

判据来自 docs/PAPER_ANALYSIS.md（P6–P12、P16–P20）与 docs/MODULE_PLAN.md M1 行。
门禁独立于 worker 的 tests：这里重新算真值（含解析/数值积分），不复用其断言。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402

torch.set_default_dtype(torch.float64)


def projected_cdf(grid: torch.Tensor, w: torch.Tensor, weights, means, log_vars) -> torch.Tensor:
    """1D 投影 w·z 的混合 CDF（每分量仍是高斯）。"""
    m = means @ w                          # [N]
    var = (torch.exp(log_vars) * w**2).sum(dim=1)   # [N]
    sd = var.sqrt()
    cdfs = 0.5 * (1 + torch.erf((grid[None, :] - m[:, None]) / (sd[:, None] * 2**0.5)))
    return (weights[:, None] * cdfs).sum(dim=0)


def main() -> int:
    g = Gate("m1", "theory", module_entry="Eq.(1),(2),(4),(5),(6),(7); Prop 3.3; Cor 3.4",
             torch=torch.__version__)

    try:
        import sgt.sampling as S
        import sgt.theory as T
        import_ok = True
        import_err = ""
    except Exception as exc:  # pragma: no cover
        import_ok = False
        import_err = f"{type(exc).__name__}: {exc}"

    g.check("c-import", expected=True, actual=import_ok, detail=import_err)
    if not import_ok:
        return 0 if g.finish() else 1

    gen = torch.Generator().manual_seed(20250611)

    # --- Eq (1)：给定 u 的精确复算 -----------------------------------------
    x = torch.randn(4, 7, 5, generator=gen)
    mu = torch.randn(4, 7, 5, generator=gen)
    sigma = torch.rand(4, 7, 5, generator=gen) + 0.1
    u = torch.randn(4, 7, 5, generator=gen)
    xt = S.sample_sgt(x, mu, sigma, u=u)
    exact = torch.equal(xt, x + mu + sigma * u)
    g.check("c-eq1", expected="bitwise equal", actual=f"max|diff|={float((xt-(x+mu+sigma*u)).abs().max()):.3e}",
            passed=exact)

    # --- Eq (1)：逐点独立、残差协方差 = diag(sigma^2) ------------------------
    n, d = 200_000, 3
    x0 = torch.zeros(n, 1, d)
    mu0 = torch.tensor([0.5, -1.0, 2.0]).view(1, 1, d).expand(n, 1, d)
    sg = torch.tensor([1.0, 2.0, 0.5]).view(1, 1, d).expand(n, 1, d)
    z = S.sample_sgt(x0, mu0, sg, generator=torch.Generator().manual_seed(1)).reshape(n, d)
    resid = z - (x0 + mu0).reshape(n, d)
    cov = torch.cov(resid.T)
    target = torch.diag(sg[0, 0] ** 2)
    rel_err = float(((cov - target).abs().max() / target.abs().max()))
    g.check("c-indep-cov", expected="rel_err < 2%", actual=f"{rel_err*100:.3f}%", passed=rel_err < 0.02)

    # --- 确定性：同一 generator seed 两次调用结果一致 ------------------------
    a = S.sample_sgt(x0, mu0, sg, generator=torch.Generator().manual_seed(7))
    b = S.sample_sgt(x0, mu0, sg, generator=torch.Generator().manual_seed(7))
    g.check("c-determinism", expected="bitwise equal", actual=torch.equal(a, b), passed=torch.equal(a, b))

    # --- Prop 3.3：经验 CDF vs 解析 N(x+mu, sigma^2)（每维 KS） ---------------
    samples = S.sample_sgt(x0[:20_000], mu0[:20_000], sg[:20_000],
                           generator=torch.Generator().manual_seed(2)).reshape(20_000, d)
    ps = []
    for j in range(d):
        p = stats.kstest(samples[:, j].numpy(),
                         lambda t, m=float((x0 + mu0)[0, 0, j]), s=float(sg[0, 0, j]): stats.norm.cdf(t, m, s)).pvalue
        ps.append(float(p))
    g.check("c-prop33", expected="min KS p > 0.01", actual=f"min p={min(ps):.4f}",
            passed=min(ps) > 0.01, detail=f"p_values={[round(p,4) for p in ps]}")

    # --- Cor 3.4 / Eq (2)：GMM 的投影分布为解析 1D 混合 ---------------------
    N = 50
    weights = torch.softmax(torch.randn(N, generator=gen), dim=0)
    means = torch.randn(N, 2, generator=gen) * 2
    log_vars = torch.log(torch.rand(N, 2, generator=gen) + 0.2)
    # 参考采样（独立于被测实现）：先选分量再加对角高斯
    idx = torch.multinomial(weights, 20_000, replacement=True, generator=gen)
    zs = means[idx] + torch.exp(0.5 * log_vars[idx]) * torch.randn(20_000, 2, generator=gen)
    dirs = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.6, 0.8])]
    proj_ps = []
    for w in dirs:
        proj = zs @ w
        grid = torch.sort(proj).values
        cdf = projected_cdf(grid, w, weights, means, log_vars)
        # 单样本 KS：经验 CDF 与给定 CDF 的最大差
        emp = torch.arange(1, len(grid) + 1) / len(grid)
        D = float((emp - cdf).abs().max())
        crit = 1.63 / len(grid) ** 0.5  # alpha=0.01
        proj_ps.append(D / (1.0 / len(grid) ** 0.5))  # 归一化统计量
        proj_ps[-1] = float(crit - D)  # >0 表示未拒绝
    g.check("c-cor34", expected="3 个投影均未拒绝（crit-D > 0）",
            actual=f"crit-D={[round(v,5) for v in proj_ps]}", passed=all(v > 0 for v in proj_ps))

    # --- gmm_pdf 归一化（2D 数值积分） -------------------------------------
    lo = means.min(dim=0).values - 6 * torch.exp(0.5 * log_vars).max(dim=0).values
    hi = means.max(dim=0).values + 6 * torch.exp(0.5 * log_vars).max(dim=0).values
    gx = torch.linspace(float(lo[0]), float(hi[0]), 401)
    gy = torch.linspace(float(lo[1]), float(hi[1]), 401)
    xx, yy = torch.meshgrid(gx, gy, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    dens = T.gmm_pdf(pts, means, log_vars, weights).reshape(401, 401)
    integral = float(torch.trapezoid(torch.trapezoid(dens, gy, dim=1), gx, dim=0))
    g.check("c-gmm-pdf-normalized", expected="1 ± 1e-3", actual=f"{integral:.6f}",
            passed=abs(integral - 1.0) < 1e-3)

    # --- Eq (5)/(7)：MI MC vs (数值积分熵 − 条件熵) ------------------------
    w1 = torch.tensor([0.2, 0.5, 0.3])
    m1 = torch.tensor([[0.0], [2.0], [-1.5]])
    lv1 = torch.log(torch.tensor([[0.3], [0.8], [0.2]]))
    mi_mc = float(T.mutual_information_mc(w1, m1, lv1, 200_000,
                                          generator=torch.Generator().manual_seed(3)))
    h_quad = float(T.gmm_entropy_quad_1d(w1, m1, lv1))
    h_cond = float(T.cond_entropy(w1, lv1))
    mi_true = h_quad - h_cond
    rel = abs(mi_mc - mi_true) / max(abs(mi_true), 1e-12)
    g.check("c-eq5-eq7", expected="rel_err < 2%", actual=f"{rel*100:.3f}%",
            passed=rel < 0.02, detail=f"mi_mc={mi_mc:.6f} quad-cond={mi_true:.6f}")

    # --- Eq (6)：论文把它写成“分量熵加权和”的上界，但该式**不成立** --------
    # 正确上界：h(mix) <= sum_i w_i h_i + H(w)。见 docs/FINDINGS.md F29。
    true_gap, paper_gap = [], []
    for i in range(20):
        w = torch.softmax(torch.randn(4, generator=gen), dim=0)
        m = torch.randn(4, 1, generator=gen) * 2
        lv = torch.log(torch.rand(4, 1, generator=gen) + 0.1)
        hq = float(T.gmm_entropy_quad_1d(w, m, lv))
        ub = float(T.gmm_entropy_upper_bound(w, lv))
        Hw = float(-(w * torch.log(w)).sum())
        true_gap.append(hq - ub - Hw)
        paper_gap.append(hq - ub)
    g.check("c-bound6-true", expected="h_quad <= sum_i w_i h_i + H(w) (all 20)",
            actual=f"max_gap={max(true_gap):.3e}", passed=max(true_gap) <= 1e-9)
    # 反例：分离分量混合的熵超过论文式 (6) 的右端
    w_ex = torch.tensor([0.5, 0.5])
    m_ex = torch.tensor([[-5.0], [5.0]])
    lv_ex = torch.log(torch.tensor([[0.1], [0.1]]))
    h_ex = float(T.gmm_entropy_quad_1d(w_ex, m_ex, lv_ex))
    ub_ex = float(T.gmm_entropy_upper_bound(w_ex, lv_ex))
    g.check("c-bound6-counterexample", expected="h_quad - upper_bound > 0.1",
            actual=f"{h_ex - ub_ex:.4f} nats (h={h_ex:.4f}, ub={ub_ex:.4f})",
            passed=(h_ex - ub_ex) > 0.1,
            detail="F29：Eq (6) 字面形式不是上界；正确上界需 +H(w)")

    # --- Eq (4) 单调性 + 反解 -------------------------------------------------
    n_vocab = 128_256
    d0 = 1.0 / n_vocab
    grid = torch.linspace(d0, 1.0, 4001)
    vals = torch.tensor([T.delta_kl(float(t), d0) for t in grid])
    mono = bool((vals[1:] >= vals[:-1] - 1e-12).all())
    g.check("c-pac-monotone", expected=True, actual=mono, passed=mono,
            detail=f"delta_kl(1)={float(vals[-1]):.6f}, log(1/delta0)={float(torch.log(torch.tensor(1/d0))):.6f}")

    table = [(1.11, 12.69), (1.48, 16.39), (1.91, 20.56), (22.19, 100.0),
             (0.26, 3.54), (0.28, 3.74), (0.23, 3.03)]
    rows, worst = [], 0.0
    for mi, expect in table:
        got = 100.0 * float(T.invert_pac_adv(mi, n_vocab))
        diff = abs(got - expect)
        worst = max(worst, diff)
        rows.append(f"MI={mi}->{got:.2f}% (论文 {expect}%, Δ={diff:.2f}pp)")
    g.check("c-pac-table", expected="每一行 |Δ| <= 0.15pp", actual=f"worst={worst:.3f}pp",
            passed=worst <= 0.15, detail="; ".join(rows))

    # 反解一致性
    # 仅对存在 δ<1 解的行检查往返（MI >= log(1/delta0) 时按定义饱和到 δ=1）
    mi_solvable = [mi for mi, _ in table if mi < float(torch.log(torch.tensor(1.0 / d0))) - 1e-9]
    cons = max(abs(float(T.delta_kl(float(T.invert_pac_adv(mi, n_vocab)), d0)) - mi) for mi in mi_solvable)
    g.check("c-pac-invert-consistency", expected="< 1e-8", actual=f"{cons:.3e}",
            passed=cons < 1e-8, detail=f"checked MI={mi_solvable}")

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
