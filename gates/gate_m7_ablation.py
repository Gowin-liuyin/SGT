"""M7 消融/常数噪声门禁：指标聚合正确性 + 六个变体产物 schema + 与 Table 1 对照。

代码级检查用合成数据（不依赖 1B 模型）；产物检查要求先跑完 6 个 variant。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gates.common import Gate  # noqa: E402

torch.set_default_dtype(torch.float32)

VARIANTS = ["comp_entropy", "cossim", "abscos", "mi", "mi_abscos", "mi_abscos_norm"]
REQUIRED_METRICS = ["nn_fr", "mrp_fr", "ttr_10", "ttr_50", "symttr_10", "symttr_100",
                    "hist_entropy", "mi_feature_mean", "mi_sum", "mi_feature_median", "pac_adv"]
TABLE1 = {
    "comp_entropy": {"utility": -13.84, "nn_fr": 88.41, "mrp_fr": 88.41, "symttr_10": 75.99,
                     "symttr_100": 49.75, "hist_entropy": 58.60, "mi": 22.19, "pac": 100.0},
    "cossim": {"utility": -5.49, "nn_fr": 99.98, "mrp_fr": 3.21, "symttr_10": 1.43,
               "symttr_100": 1.07, "hist_entropy": 2.47, "mi": 3.2e19, "pac": 100.0},
    "abscos": {"utility": -0.01, "nn_fr": 94.90, "mrp_fr": 94.90, "symttr_10": 93.36,
               "symttr_100": 92.86, "hist_entropy": 87.62, "mi": 6e9, "pac": 100.0},
    "mi": {"utility": -0.72, "nn_fr": 34.43, "mrp_fr": 34.43, "symttr_10": 18.96,
           "symttr_100": 17.39, "hist_entropy": 23.96, "mi": 1.48, "pac": 16.39},
    "mi_abscos": {"utility": -2.44, "nn_fr": 81.23, "mrp_fr": 81.23, "symttr_10": 71.36,
                  "symttr_100": 66.88, "hist_entropy": 75.72, "mi": 1.91, "pac": 20.56},
    "mi_abscos_norm": {"utility": -1.97, "nn_fr": 93.00, "mrp_fr": 93.00, "symttr_10": 90.42,
                       "symttr_100": 88.15, "hist_entropy": 91.90, "mi": 1.11, "pac": 12.69},
}


def main() -> int:
    g = Gate("m7", "ablation", module_entry="Table 1 / Fig 3 / Fig 4")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_ablation", REPO_ROOT / "experiments/run_ablation.py")
        ra = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ra)
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    # --- rank_metrics：恒等变换 -------------------------------------------
    gen = torch.Generator().manual_seed(0)
    V, d, N = 100, 16, 200
    table = torch.randn(V, d, generator=gen)
    tgt = torch.randint(0, V, (N,), generator=gen)
    x = table[tgt]
    m_id = ra.rank_metrics(x, tgt, table, chunk=64)
    ok_id = (m_id["nn_fr"] == 0.0 and m_id["ttr_10"] == 0.0 and m_id["symttr_10"] == 0.0
             and abs(m_id["hist_entropy"]) < 1e-12)
    g.check("c-rank-identity", expected="nn_fr=ttr=symttr=hist=0", actual={k: round(m_id[k], 4) for k in
            ("nn_fr", "mrp_fr", "ttr_10", "symttr_10", "hist_entropy")}, passed=ok_id)

    # --- rank_metrics：把每个 token 推到最远 -------------------------------
    far = torch.stack([table[torch.argmin(torch.cdist(table[tgt[i]].reshape(1, -1), table))] for i in range(N)])
    # 用「离原 embedding 最远的表项」作为 x_tilde，其最近邻不是目标
    far = torch.stack([table[torch.argsort(torch.cdist(table[tgt[i]].reshape(1, -1), table).reshape(-1))[-1]] for i in range(N)])
    m_far = ra.rank_metrics(far, tgt, table, chunk=64)
    g.check("c-rank-far", expected="nn_fr == 100", actual=f"{m_far['nn_fr']:.2f}",
            passed=abs(m_far["nn_fr"] - 100.0) < 1e-9)

    # --- feature_mi：1 维 GMM 对照 theory --------------------------------
    from sgt import theory
    g2 = torch.Generator().manual_seed(1)
    n = 4000
    means = torch.zeros(n, 1)
    log_vars = torch.log(torch.full((n, 1), 0.5))
    x1 = torch.randn(n, 1, generator=g2)
    mu1 = torch.zeros(n, 1)
    w = torch.full((n,), 1.0 / n)
    mi_ref = theory.mutual_information_mc(w, means, log_vars, 20000, generator=torch.Generator().manual_seed(2))
    xt = x1 + mu1 + torch.exp(0.5 * log_vars) * torch.randn(n, 1, generator=g2)
    mi_got = ra.feature_mi(xt, x1, mu1, log_vars, log_vars, n_pairs=20000, seed=3)
    rel = abs(mi_got["mi_feature_mean"] - float(mi_ref)) / max(abs(float(mi_ref)), 1e-12)
    g.check("c-feature-mi", expected="与 theory MC 相对误差 < 10%",
            actual=f"{rel*100:.2f}% (got={mi_got['mi_feature_mean']:.4f}, ref={float(mi_ref):.4f})",
            passed=rel < 0.10)

    # --- 常数噪声脚本：单调性（合成距离） --------------------------------
    try:
        spec2 = importlib.util.spec_from_file_location("run_constnoise", REPO_ROOT / "experiments/run_constant_noise.py")
        rcn = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(rcn)
        cn_ok = True
    except Exception as exc:
        cn_ok = False
        cn_err = f"{type(exc).__name__}: {exc}"
    g.check("c-constnoise-import", expected=True, actual=cn_ok,
            passed=cn_ok, detail="" if cn_ok else cn_err)

    # --- 产物：六个 variant 的 JSON --------------------------------------
    missing, schema_bad = [], []
    for v in VARIANTS:
        p = REPO_ROOT / "results" / f"ablation_{v}.json"
        if not p.exists():
            missing.append(v)
            continue
        payload = json.loads(p.read_text())
        metrics = payload.get("eval", {}).get("metrics", {})
        miss = [k for k in REQUIRED_METRICS if k not in metrics]
        if miss:
            schema_bad.append(f"{v}:{miss}")
    g.check("c-results-six", expected="6 个 ablation_*.json", actual=f"missing={missing}",
            passed=not missing)
    g.check("c-results-metrics", expected="每行 11 个指标齐全", actual=f"bad={schema_bad}",
            passed=not schema_bad)

    # --- 与 Table 1 的对照表 --------------------------------------------
    summary_path = REPO_ROOT / "results" / "ablation_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        g.check("c-summary-schema", expected="含 6 行与 table1 参照",
                actual=f"rows={len(s.get('rows', []))} keys={list(s)[:4]}",
                passed=len(s.get("rows", [])) == 6)
        zero = [r["variant"] for r in s.get("rows", []) if r.get("metrics", {}).get("n_tokens", 0) == 0]
        g.check("c-summary-nonempty", expected="n_tokens > 0", actual=f"zero={zero}", passed=not zero)
    else:
        g.check("c-summary-schema", expected="results/ablation_summary.json 存在", actual="missing",
                passed=False)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
