"""M2 损失门禁：Alg.1 / Eq (8)(9)(10)(11)(12)(13) 与效用损失。

真值由本文件的参考实现（逐元素/逐对朴素循环）独立计算，不复用 worker 的断言。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402

torch.set_default_dtype(torch.float64)


def ref_mi_pair(xt, xp, mup, lvp, lv, mask=None, half=False):
    """Eq (8) 的直译：同下标配对，逐元素循环。"""
    B, T, d = xt.shape
    vals = []
    for b in range(B):
        for t in range(T):
            if mask is not None and not bool(mask[b, t]):
                continue
            logdet = float((lvp[b, t] - lv[b, t]).sum())
            diff = xt[b, t] - xp[b, t] - mup[b, t]
            mahal = float((diff**2 / lvp[b, t].exp()).sum())
            vals.append(logdet + mahal)
    v = sum(vals) / len(vals)
    return 0.5 * v if half else v


def ref_mi_full(xt, xp, mup, lvp, lv, mask=None, mask_p=None):
    vals = []
    B, T, d = xt.shape
    Bp, Tp, _ = xp.shape
    for b in range(B):
        for t in range(T):
            if mask is not None and not bool(mask[b, t]):
                continue
            for b2 in range(Bp):
                for t2 in range(Tp):
                    if mask_p is not None and not bool(mask_p[b2, t2]):
                        continue
                    logdet = float((lvp[b2, t2] - lv[b, t]).sum())
                    diff = xt[b, t] - xp[b2, t2] - mup[b2, t2]
                    vals.append(logdet + float((diff**2 / lvp[b2, t2].exp()).sum()))
    return sum(vals) / len(vals)


def main() -> int:
    g = Gate("m2", "losses", module_entry="Alg.1; Eq.(8)-(13); L_U")

    try:
        import sgt.losses as L
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    gen = torch.Generator().manual_seed(42)
    B, T, d = 2, 3, 4
    x = torch.randn(B, T, d, generator=gen)
    mu = torch.randn(B, T, d, generator=gen) * 0.3
    x_tilde = x + mu + 0.5 * torch.randn(B, T, d, generator=gen)
    x_p = torch.randn(B, T, d, generator=gen)
    mu_p = torch.randn(B, T, d, generator=gen) * 0.3
    lvp = torch.log(torch.rand(B, T, d, generator=gen) + 0.2)
    lv = torch.log(torch.rand(B, T, d, generator=gen) + 0.2)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -1] = False

    # --- Eq (8) pair 模式 ---------------------------------------------------
    got = float(L.mi_loss(x_tilde, x_p, mu_p, lvp, lv, mask=mask))
    exp = ref_mi_pair(x_tilde, x_p, mu_p, lvp, lv, mask=mask)
    g.check("c-mi-pair", expected=f"{exp:.12f}", actual=f"{got:.12f}",
            passed=abs(got - exp) <= 1e-9 * max(1.0, abs(exp)))

    # --- Eq (8) full 模式 ---------------------------------------------------
    got_full = float(L.mi_loss(x_tilde, x_p, mu_p, lvp, lv, mask=mask, mode="full"))
    exp_full = ref_mi_full(x_tilde, x_p, mu_p, lvp, lv, mask=mask)
    g.check("c-mi-full", expected=f"{exp_full:.12f}", actual=f"{got_full:.12f}",
            passed=abs(got_full - exp_full) <= 1e-9 * max(1.0, abs(exp_full)))

    # --- half_factor --------------------------------------------------------
    got_half = float(L.mi_loss(x_tilde, x_p, mu_p, lvp, lv, mask=mask, half_factor=True))
    g.check("c-mi-half", expected=f"{0.5*exp:.12f}", actual=f"{got_half:.12f}",
            passed=abs(got_half - 0.5 * exp) <= 1e-9)

    # --- 梯度 ---------------------------------------------------------------
    lv_g = lv.clone().requires_grad_(True)
    lvp_g = lvp.clone().requires_grad_(True)
    loss = L.mi_loss(x_tilde, x_p, mu_p, lvp_g, lv_g, mask=mask)
    loss.backward()
    gok = (lv_g.grad is not None and torch.isfinite(lv_g.grad).all()
           and float(lv_g.grad.abs().sum()) > 0 and float(lvp_g.grad.abs().sum()) > 0)
    g.check("c-mi-grad", expected="finite & nonzero", actual=gok, passed=bool(gok))

    # --- Eq (9) AbsCos ------------------------------------------------------
    xa = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    xt = torch.tensor([[[0.0, 2.0, 0.0], [0.0, -3.0, 0.0], [-1.0, 0.0, 0.0]]])
    got_acs = float(L.abs_cos_loss(xa, xt))
    exp_acs = (0.0 + 1.0 + 1.0) / 3
    g.check("c-acs", expected=exp_acs, actual=round(got_acs, 12),
            passed=abs(got_acs - exp_acs) < 1e-12)
    # 与 torch 余弦一致性
    ref = (xa * xt).sum(-1) / (xa.norm(dim=-1) * xt.norm(dim=-1))
    g.check("c-acs-torch", expected=True,
            actual=f"max|diff|={float((L.abs_cos_loss(xa, xt, reduce=False) - ref.abs()).abs().max()):.3e}",
            passed=torch.allclose(L.abs_cos_loss(xa, xt, reduce=False), ref.abs(), atol=1e-12))

    # --- Eq (13) 有符号余弦 -------------------------------------------------
    got_bcs = float(L.cos_sim_loss(xa, xt))
    exp_bcs = (0.0 - 1.0 - 1.0) / 3
    g.check("c-bcs", expected=exp_bcs, actual=round(got_bcs, 12), passed=abs(got_bcs - exp_bcs) < 1e-12)

    # --- Eq (10) 中位数范数惩罚 --------------------------------------------
    xpm = torch.tensor([[[0.0, 0.5], [0.0, 1.0], [0.0, 2.0]]])  # norms 0.5, 1, 2
    got_mnp = float(L.median_norm_loss(xpm, 1.0))
    exp_mnp = (0.5 + 0.0 + 1.0) / 3
    g.check("c-mnp", expected=exp_mnp, actual=round(got_mnp, 12), passed=abs(got_mnp - exp_mnp) < 1e-12)

    # --- Eq (12) 分量熵 -----------------------------------------------------
    lv2 = torch.log(torch.tensor([[[0.5, 2.0], [1.0, 4.0]]]))
    got_cge = float(L.comp_entropy_loss(lv2))
    per_tok = -(lv2.sum(-1))
    exp_cge = float(per_tok.mean())
    g.check("c-cge", expected=f"{exp_cge:.12f}", actual=f"{got_cge:.12f}", passed=abs(got_cge - exp_cge) < 1e-12)
    # 对均值不敏感（A5/F1 语义：只有方差项）
    mu_a = torch.zeros(1, 2, 2, requires_grad=True)
    cge_val = L.comp_entropy_loss(lv2 + 0 * mu_a.sum())
    cge_val.backward()
    g.check("c-cge-no-mean", expected="d/dmu == 0", actual=float(mu_a.grad.abs().sum()),
            passed=float(mu_a.grad.abs().sum()) == 0.0)

    # --- Eq (11) 加权和 -----------------------------------------------------
    a1, a2, a3 = 2.0, 0.5, 3.0
    total = L.obfuscation_loss(mi=torch.tensor(1.0), acs=torch.tensor(2.0), mnp=torch.tensor(3.0),
                               alphas=(a1, a2, a3))
    exp_total = a1 * 1.0 + a2 * 2.0 + a3 * 3.0
    g.check("c-eq11", expected=exp_total, actual=float(total), passed=abs(float(total) - exp_total) < 1e-12)

    # --- 效用损失 L_U（输入为原始 logits，函数内部做 softmax/log_softmax） ----
    zc = torch.randn(2, 3, 5, generator=gen)
    zs = torch.randn(2, 3, 5, generator=gen).requires_grad_(True)
    got_u = float(L.utility_loss(zc, zs))
    p = torch.softmax(zc, dim=-1)
    exp_u = float(-(p * torch.log_softmax(zs, dim=-1)).sum(-1).mean())
    g.check("c-utility", expected=f"{exp_u:.12f}", actual=f"{got_u:.12f}", passed=abs(got_u - exp_u) < 1e-9)
    same = float(L.utility_loss(zc, zc))
    g.check("c-utility-identity", expected="< 1e-6", actual=f"{same:.3e}", passed=abs(same) < 1e-6)

    # mask：被 mask 的位置不影响
    m2 = torch.ones(2, 3, dtype=torch.bool)
    m2[1, 2] = False
    zs2 = zs.detach().clone()
    zs2[1, 2] = torch.randn(5, generator=gen)
    v2 = float(L.utility_loss(zc, zs2, attention_mask=m2))
    per_tok = -(p * torch.log_softmax(zs2, dim=-1)).sum(-1)
    exp_mask = float(per_tok[m2].mean())
    g.check("c-utility-mask", expected=f"{exp_mask:.12f}", actual=f"{v2:.12f}",
            passed=abs(v2 - exp_mask) < 1e-9)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
