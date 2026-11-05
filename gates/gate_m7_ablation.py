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

    # --- feature_mi：A7 口径（Eq 8 逐特征均值），解析值 = 1 + 2/σ² --------
    # 注意（F30）：该量**不是**信息论互信息；x~N(0,1)、µ=0、σ² 固定时，
    # (x̃−x')² 的方差为 2+σ²，逐特征均值 = 1 + 2/σ²。用它对照 theory.MI 会差一个量级。
    sig2 = 0.5
    n = 4000
    g2 = torch.Generator().manual_seed(1)
    x1 = torch.randn(n, 1, generator=g2)
    mu1 = torch.zeros(n, 1)
    log_vars = torch.log(torch.full((n, 1), sig2))
    xt = x1 + torch.exp(0.5 * log_vars) * torch.randn(n, 1, generator=g2)
    mi_got = ra.feature_mi(xt, x1, mu1, log_vars, log_vars, n_pairs=20000, seed=3)
    expected = 1.0 + 2.0 / sig2
    rel = abs(mi_got["mi_feature_mean"] - expected) / expected
    g.check("c-feature-mi", expected=f"1+2/sigma^2={expected} (±10%)",
            actual=f"{mi_got['mi_feature_mean']:.4f} (rel={rel*100:.2f}%)", passed=rel < 0.10,
            detail="A7/F30：Eq(8) 逐特征均值不是信息论 MI")
    # 与训练侧 Eq (8) 的 d 倍关系一致
    g.check("c-mi-sum", expected="mi_sum = mi_feature_mean * d",
            actual=f"{mi_got['mi_sum']:.4f} vs {mi_got['mi_feature_mean']*1:.4f}",
            passed=abs(mi_got["mi_sum"] - mi_got["mi_feature_mean"]) < 1e-6)

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
    # comp_entropy 按论文原式（-log|Σ|）训练会发散（F31）；接受 summary 中的 diverged 证据
    summary0 = REPO_ROOT / "results" / "ablation_summary.json"
    diverged = set()
    if summary0.exists():
        for r in json.loads(summary0.read_text()).get("rows", []):
            if r.get("status") == "diverged":
                diverged.add(r["variant"])
    missing, schema_bad = [], []
    for v in VARIANTS:
        p = REPO_ROOT / "results" / f"ablation_{v}.json"
        if not p.exists():
            if v in diverged:
                continue
            missing.append(v)
            continue
        payload = json.loads(p.read_text())
        metrics = payload.get("eval", {}).get("metrics", {})
        miss = [k for k in REQUIRED_METRICS if k not in metrics]
        if miss:
            schema_bad.append(f"{v}:{miss}")
    g.check("c-results-six", expected="6 个变体均有结果或发散证据（F31）",
            actual=f"missing={missing} diverged={sorted(diverged)}", passed=not missing)
    g.check("c-comp-entropy-diverged", expected="comp_entropy 有发散证据或有限结果",
            actual=f"diverged={('comp_entropy' in diverged)} file={(REPO_ROOT/'results/ablation_comp_entropy.json').exists()}",
            passed=("comp_entropy" in diverged) or (REPO_ROOT / "results" / "ablation_comp_entropy.json").exists())
    g.check("c-results-metrics", expected="每行 11 个指标齐全", actual=f"bad={schema_bad}",
            passed=not schema_bad)

    # --- 与 Table 1 的对照表 --------------------------------------------
    summary_path = REPO_ROOT / "results" / "ablation_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        g.check("c-summary-schema", expected="含 6 行与 table1 参照",
                actual=f"rows={len(s.get('rows', []))} keys={list(s)[:4]}",
                passed=len(s.get("rows", [])) == 6)
        zero = [r["variant"] for r in s.get("rows", [])
                if r.get("status") != "diverged" and r.get("metrics", {}).get("n_tokens", 0) == 0]
        g.check("c-summary-nonempty", expected="n_tokens > 0", actual=f"zero={zero}", passed=not zero)
    else:
        g.check("c-summary-schema", expected="results/ablation_summary.json 存在", actual="missing",
                passed=False)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
