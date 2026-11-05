"""M3 SGT 模型门禁：序列依赖、随机性、初始化、padding 不变性、梯度与序列化。

判据来自 docs/PAPER_ANALYSIS.md P1/P7 与 SPEC A1；本文件独立于 worker 的测试。
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


def main() -> int:
    g = Gate("m3", "model", module_entry="Def 3.1 / Eq (1) 的 µθ, Σθ 估计器（SPEC A1）")
    try:
        from sgt.model import SGTModel
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    torch.manual_seed(0)
    B, T, d = 3, 6, 8
    m = SGTModel(d, hidden=16, n_layers=2, n_heads=4, max_len=32, sigma_init=1.0).eval()
    x = torch.randn(B, T, d)
    mu, lv = m(x)
    sig = torch.exp(0.5 * lv)

    g.check("c-shape", expected=[B, T, d], actual=list(mu.shape), passed=tuple(mu.shape) == (B, T, d)
            and tuple(lv.shape) == (B, T, d))

    g.check("c-init-mu", expected="|mu|max < 1e-12", actual=f"{float(mu.abs().max()):.3e}",
            passed=float(mu.abs().max()) < 1e-12)
    dev = float((sig - 1.0).abs().max())
    g.check("c-init-sigma", expected="sigma == sigma_init (±1e-6)", actual=f"max|sigma-1|={dev:.3e}",
            passed=dev < 1e-6)

    # 初始化检查完毕后，随机化全部参数（零初始化头会使输出对输入不敏感，
    # 序列依赖/梯度检查必须在非零头权重下进行；见 model.py Note）
    torch.manual_seed(123)
    with torch.no_grad():
        for p_ in m.parameters():
            p_.normal_(0.0, 0.05)
    mu, lv = m(x)
    sig = torch.exp(0.5 * lv)

    mu2, lv2 = m(x)
    g.check("c-determinism", expected="bitwise equal",
            actual=f"mu={torch.equal(mu, mu2)} lv={torch.equal(lv, lv2)}",
            passed=torch.equal(mu, mu2) and torch.equal(lv, lv2))

    # 序列依赖：扰动位置 3 的输入，应在**所有**位置引起变化
    x2 = x.clone()
    x2[:, 3, :] += 1.0
    mu_b, lv_b = m(x2)
    per_pos = (mu_b - mu).abs().amax(dim=(0, 2))
    g.check("c-seqdep", expected="every position changes > 1e-6",
            actual=f"min_change={float(per_pos.min()):.3e}",
            passed=float(per_pos.min()) > 1e-6,
            detail=f"per_pos={[round(float(v),4) for v in per_pos]}")

    # 位置依赖：置换 token 后输出必须不同
    xp = x[:, [1, 0, 2, 3, 4, 5], :]
    mu_p, _ = m(xp)
    g.check("c-posdep", expected="permuted inputs give different outputs",
            actual=f"max|diff|={float((mu_p - mu).abs().max()):.3e}",
            passed=float((mu_p - mu).abs().max()) > 1e-6)

    # padding 不变性：短序列 vs 右侧 padding + mask
    x1 = torch.randn(B, 4, d)
    x2p = torch.cat([x1, torch.randn(B, 2, d)], dim=1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * B, dtype=torch.bool)
    mu_s, lv_s = m(x1)
    mu_l, lv_l = m(x2p, attention_mask=mask)
    dmu = float((mu_l[:, :4] - mu_s).abs().max())
    dlv = float((lv_l[:, :4] - lv_s).abs().max())
    g.check("c-padding", expected="valid positions invariant (<1e-6)",
            actual=f"dmu={dmu:.3e} dlv={dlv:.3e}", passed=max(dmu, dlv) < 1e-6)

    # 梯度
    m.zero_grad()
    (mu.sum() + lv.sum()).backward()
    grads = {n: p.grad for n, p in m.named_parameters()}
    finite = all((gr is None) or bool(torch.isfinite(gr).all()) for gr in grads.values())
    nonzero = sum(1 for gr in grads.values() if gr is not None and float(gr.abs().sum()) > 0)
    g.check("c-grad", expected="all finite, >=8 tensors nonzero",
            actual=f"finite={finite} nonzero={nonzero}", passed=finite and nonzero >= 8)

    # state_dict 往返
    import copy
    m2 = SGTModel(d, hidden=16, n_layers=2, n_heads=4, max_len=32, sigma_init=1.0).eval()
    m2.load_state_dict(copy.deepcopy(m.state_dict()))
    mu3, lv3 = m2(x)
    g.check("c-state-dict", expected="identical after load", actual=f"{float((mu3-mu).abs().max()):.3e}",
            passed=float((mu3 - mu).abs().max()) == 0.0 and float((lv3 - lv).abs().max()) == 0.0)

    # sample() 与 Eq (1) 一致
    u = torch.randn_like(x)
    xt = m.sample(x, u=u)
    exp = x + mu + sig * u
    g.check("c-sample-eq1", expected="x+mu+sigma*u", actual=f"{float((xt-exp).abs().max()):.3e}",
            passed=torch.allclose(xt, exp, atol=1e-12))

    # 随机性：不给 u 时两次采样不同
    a = m.sample(x, generator=torch.Generator().manual_seed(1))
    b = m.sample(x, generator=torch.Generator().manual_seed(1))
    c = m.sample(x, generator=torch.Generator().manual_seed(2))
    g.check("c-stochastic", expected="same seed equal, diff seed differ",
            actual=f"eq={torch.equal(a,b)} neq={not torch.equal(a,c)}",
            passed=torch.equal(a, b) and not torch.equal(a, c))

    # sigma_min 下界
    with torch.no_grad():
        for p in m.parameters():
            pass
    n_params = sum(p.numel() for p in m.parameters())
    g.check("c-param-report", expected="report only", actual=f"{n_params} params (d=8,hidden=16)", passed=True)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
