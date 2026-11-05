"""M4 指标与攻击门禁：秩/NN/MRP/TTR/SymTTR/直方图熵/PAC/ε-DP/BeamClean。

真值由本文件的暴力实现独立计算。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402

torch.set_default_dtype(torch.float64)


def brute_ranks(xt, tgt, table):
    """O(N·V) 暴力秩（1-based，并列按 id 升序）。"""
    N, V = xt.shape[0], table.shape[0]
    out = []
    for i in range(N):
        d = torch.cdist(xt[i : i + 1], table).reshape(-1)
        order = sorted(range(V), key=lambda k: (float(d[k]), k))
        out.append(order.index(int(tgt[i])) + 1)
    return torch.tensor(out)


def main() -> int:
    g = Gate("m4", "metrics", module_entry="§6 p.7-8：rank/NN/MRP/TTR/SymTTR/HistEntr/PAC/BeamClean")
    try:
        import sgt.metrics as M
        import sgt.attacks as A
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    gen = torch.Generator().manual_seed(7)
    V, d, N = 60, 6, 40
    table = torch.randn(V, d, generator=gen)
    tgt = torch.randint(0, V, (N,), generator=gen)
    xt = table[tgt] + 0.3 * torch.randn(N, d, generator=gen)

    # 秩与暴力一致
    r = M.ranks(xt, tgt, table)
    rb = brute_ranks(xt, tgt, table)
    g.check("c-ranks", expected="== brute force", actual=f"neq={int((r != rb).sum())}",
            passed=torch.equal(r, rb))

    # 并列稳定：重复 embedding（两个 token 距离相同）应取 id 小的
    tab2 = table.clone()
    tab2[5] = tab2[3]
    x1 = tab2[3].reshape(1, -1)
    r_tie = M.ranks(x1, torch.tensor([5]), tab2)
    g.check("c-ties", expected="tie -> smaller id wins (rank 1 for id 3)",
            actual=f"rank_of_id5={int(r_tie[0])}", passed=int(r_tie[0]) == 2)

    # TTR / SymTTR 手算
    rr = torch.tensor([1, 5, 10])
    got_ttr1 = M.ttr(rr, 1, 10)
    got_st1 = M.symttr(rr, 1, 10)
    got_st5 = M.symttr(rr, 5, 10)
    g.check("c-ttr", expected=200.0 / 3.0, actual=round(float(got_ttr1), 6),
            passed=abs(float(got_ttr1) - 200.0 / 3.0) < 1e-6)
    g.check("c-symttr", expected=(100.0 / 3.0, 0.0), actual=(round(float(got_st1), 6), round(float(got_st5), 6)),
            passed=abs(float(got_st1) - 100.0 / 3.0) < 1e-6 and float(got_st5) == 0.0)

    # 直方图熵
    uni = torch.arange(1, 11).repeat(3)
    g.check("c-histent-uniform", expected=1.0, actual=f"{float(M.hist_entropy(uni, 10)):.6f}",
            passed=abs(float(M.hist_entropy(uni, 10)) - 1.0) < 1e-9)
    single = torch.ones(5, dtype=torch.long)
    g.check("c-histent-single", expected=0.0, actual=f"{float(M.hist_entropy(single, 10)):.6f}",
            passed=abs(float(M.hist_entropy(single, 10))) < 1e-9)

    # 失败率
    g.check("c-fr", expected=(100.0, 0.0), actual=(M.failure_rate(torch.tensor([1, 2]), torch.tensor([2, 1])),
                                                   M.failure_rate(torch.tensor([1, 2]), torch.tensor([1, 2]))),
            passed=M.failure_rate(torch.tensor([1, 2]), torch.tensor([2, 1])) == 100.0
            and M.failure_rate(torch.tensor([1, 2]), torch.tensor([1, 2])) == 0.0)

    # NN / MRP
    nn = M.nn_reconstruct(xt, table)
    nn_ref = torch.argmin(torch.cdist(xt, table), dim=1)
    g.check("c-nn", expected="argmin", actual=f"neq={int((nn != nn_ref).sum())}", passed=torch.equal(nn, nn_ref))

    # 构造已知直方图：众数 r=3
    hist = torch.tensor([1, 2, 8, 3, 2])
    g.check("c-mrp-rank", expected=3, actual=M.mrp_rank(hist), passed=M.mrp_rank(hist) == 3)

    mrp = M.mrp_reconstruct(xt, table, 3)
    dist = torch.cdist(xt, table)
    mrp_ref = torch.argsort(dist, dim=1)[:, 2]  # 第 3 近邻（0-based index 2）
    g.check("c-mrp", expected="3rd nearest", actual=f"neq={int((mrp != mrp_ref).sum())}",
            passed=torch.equal(mrp, mrp_ref))

    # PAC-Adv 复用与饱和
    p1 = M.pac_adv(1.11, 128_256)
    p2 = M.pac_adv(22.19, 128_256)
    g.check("c-pac", expected="1.11->12.69±0.15, 22.19->100", actual=f"{p1:.2f}, {p2:.2f}",
            passed=abs(p1 - 12.69) <= 0.15 and abs(p2 - 100.0) < 1e-9)

    # ε-DP
    e1 = M.eps_dp(1.0, sensitivity=1.0, delta=1e-5)
    e2 = M.eps_dp(2.0, sensitivity=1.0, delta=1e-5)
    e_ref = 1.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
    g.check("c-epsdp", expected=f"{e_ref:.4f}", actual=f"{e1:.4f}", passed=abs(e1 - e_ref) < 1e-6)
    g.check("c-epsdp-monotone", expected="eps(sigma=2) < eps(sigma=1)", actual=f"{e2:.4f} < {e1:.4f}",
            passed=e2 < e1)

    # BeamClean：无噪声 → 等于 NN；噪声增大 → 失败率不降
    Vb, db, T = 50, 8, 4
    gb = torch.Generator().manual_seed(11)
    table_b = torch.randn(Vb, db, generator=gb)
    true_ids = torch.randint(0, Vb, (200, T), generator=gb)
    x_clean = table_b[true_ids]                      # [N,T,d]
    u = torch.randn(200, T, db, generator=gb)
    def lm0(ids):
        return torch.zeros(ids.shape[0], dtype=torch.float64)
    frs = []
    for sigma in (1e-3, 0.2, 1.0):
        xt_b = x_clean + sigma * u
        rec, sig_hat = A.beamclean(xt_b, table_b, lm0, sigma=sigma, beam=4, topk=8)
        frs.append(float((rec != true_ids).float().mean() * 100))
    nn_b = torch.argmin(torch.cdist(x_clean.reshape(-1, db), table_b), dim=1).reshape(200, T)
    fr_nn = float((nn_b != true_ids).float().mean() * 100)
    g.check("c-beamclean-nonoise", expected="sigma=1e-3 FR == NN FR",
            actual=f"{frs[0]:.2f}% vs NN {fr_nn:.2f}%", passed=abs(frs[0] - fr_nn) <= 1e-9)
    g.check("c-beamclean-monotone", expected="FR non-decreasing with sigma",
            actual=f"{[round(f,2) for f in frs]}", passed=frs[1] >= frs[0] and frs[2] >= frs[1])

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
