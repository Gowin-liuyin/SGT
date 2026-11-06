"""M10 报告与回归门禁：claim 全部有结论、证据齐备、报告无占位符、上游门禁无回退。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gates.common import Gate  # noqa: E402

EXPECTED_GATES = {
    "m0_skeleton": "m0", "m1_theory": "m1", "m2_losses": "m2", "m3_model": "m3",
    "m4_metrics": "m4", "m5_data": "m5", "m6_train": "m6", "m7_ablation": "m7",
    "m8a_utility": "m8a", "m8b_pii": "m8b", "m9_large": "m9", "s1_security": "s1",
    "c1_conformance": "c1",
}


def main() -> int:
    g = Gate("m10", "report", module_entry="Phase R：声明核对表 + 报告 + 回归")

    # 1) claim 表：无 UNKNOWN，每行有证据
    claims = (REPO_ROOT / "docs/CLAIMS_CHECKLIST.md").read_text(encoding="utf-8")
    rows = [ln for ln in claims.splitlines() if re.match(r"^\|\s*C\d+\s*\|", ln)]
    unknown = [ln.split("|")[1].strip() for ln in rows if "| UNKNOWN |" in ln]
    no_evidence = []
    for ln in rows:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) >= 6 and cells[5] in ("—", "", "-"):
            no_evidence.append(cells[0])
    g.check("c-claims-no-unknown", expected="0 行 UNKNOWN", actual=f"{len(unknown)} {unknown}",
            passed=not unknown)
    g.check("c-claims-evidence", expected="每行有证据路径/说明", actual=f"缺失 {no_evidence}",
            passed=not no_evidence)
    verdicts = {}
    for ln in rows:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        v = cells[4].replace("*", "")
        key = next((k for k in ("VERIFIED", "FALSIFIED", "BLOCKED") if v.startswith(k)), "OTHER")
        verdicts[key] = verdicts.get(key, 0) + 1
    g.check("c-claims-verdicts", expected="VERIFIED+FALSIFIED+BLOCKED == 全部行",
            actual=verdicts, passed=verdicts.get("OTHER", 0) == 0)

    # 2) 报告：必需章节 + 无占位符
    report = (REPO_ROOT / "REPRODUCTION_REPORT.md").read_text(encoding="utf-8")
    sections = ["## 0. 结论摘要", "## 1. 复现范围", "## 2. 环境与硬件差异", "## 3. 方法复现",
                "## 4. 实验数值对照", "## 5. 论文问题清单", "## 6. 歧义与实例化决策",
                "## 7. 复现结论统计", "## 8. 复跑方式", "## 9. 局限"]
    missing_sections = [s for s in sections if s not in report]
    placeholders = [p for p in ("待回填", "待填", "（待") if p in report]
    g.check("c-report-sections", expected="10 节齐全", actual=f"缺 {missing_sections}",
            passed=not missing_sections)
    g.check("c-report-no-placeholder", expected="无占位符", actual=f"{placeholders}",
            passed=not placeholders)

    # 3) 上游门禁 JSON 全部 PASS
    bad = []
    for name, mid in EXPECTED_GATES.items():
        p = REPO_ROOT / "results" / f"gate_{name}.json"
        if not p.exists():
            bad.append(f"{name}:missing")
            continue
        d = json.loads(p.read_text())
        if not d.get("pass"):
            bad.append(f"{name}:{d.get('status')}")
    g.check("c-upstream-gates", expected="13 个上游门禁 PASS", actual=f"{bad}", passed=not bad)

    # 4) 关键结果文件
    need = ["results/ablation_summary.json", "results/fig3_ranksh.json",
            "results/constnoise_low.json", "results/pupa_baseline.json",
            "results/utility_baseline_l50.json"]
    missing = [f for f in need if not (REPO_ROOT / f).exists()]
    g.check("c-results-exist", expected="关键结果齐备", actual=f"缺 {missing}", passed=not missing)

    # 5) 状态表：M0..M10 全 PASS
    state = (REPO_ROOT / "REPRO_STATE.md").read_text(encoding="utf-8")
    status = dict(re.findall(r"^\|\s*([MSC]\d+[ab]?)[^|]*\|[^|]*\|\s*(\w[\w ]*?)\s*\|", state, flags=re.M))
    not_pass = {k: v for k, v in status.items() if v.strip() != "PASS"}
    g.check("c-state-all-pass", expected="M0..M10 + S1（含 M8a/M8b）全 PASS", actual=not_pass or status,
            passed=not not_pass and len(status) == 14)

    # 6) 回归：fail-fast 自检仍通过（不重跑全部门禁，避免递归）
    proc = subprocess.run([sys.executable, "-m", "gates.run_gates", "--self-test"],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    g.check("c-runner-selftest", expected="exit 0", actual=f"exit {proc.returncode}",
            passed=proc.returncode == 0)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
