"""M0 骨架门禁：目录/文档/运行器/状态表就位，且运行器可执行、失败即停、结果可落盘。

对应 docs/MODULE_PLAN.md M0 行。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gates.common import Gate  # noqa: E402

REQUIRED_FILES = [
    "SPEC.md",
    "docs/PAPER_ANALYSIS.md",
    "docs/MODULE_PLAN.md",
    "docs/CLAIMS_CHECKLIST.md",
    "docs/FINDINGS.md",
    "gates/common.py",
    "gates/run_gates.py",
    "REPRO_STATE.md",
    "REPRODUCTION_REPORT.md",
    "README.md",
    "repro.sh",
]

ALLOWED_VERDICTS = {"VERIFIED", "FALSIFIED", "BLOCKED", "UNKNOWN"}


def main() -> int:
    g = Gate("m0", "skeleton", module_entry="Phase 0 骨架")

    # 1) 必需文件存在且非空
    missing = [f for f in REQUIRED_FILES if not (REPO_ROOT / f).is_file()]
    empty = [
        f for f in REQUIRED_FILES
        if (REPO_ROOT / f).is_file() and (REPO_ROOT / f).stat().st_size == 0
    ]
    g.check("c-files", expected="all present & non-empty", actual=f"missing={missing} empty={empty}",
            passed=not missing and not empty)

    # 2) MODULE_PLAN 覆盖 M0..M10 且每行有判据
    plan = (REPO_ROOT / "docs/MODULE_PLAN.md").read_text(encoding="utf-8")
    modules = re.findall(r"\|\s*\*\*(M\d+)\*\*\s*\|", plan)
    expected = [f"M{i}" for i in range(11)]
    g.check("c-modplan", expected=expected, actual=modules, passed=modules == expected)

    # 3) SPEC 的歧义条目 A1..A17 齐全
    spec = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    ambiguities = sorted({int(x) for x in re.findall(r"### A(\d+)[\s:：．.]", spec)})
    contiguous = ambiguities == list(range(1, len(ambiguities) + 1))
    g.check("c-spec-ambiguities", expected="A1..An 连续且 n>=17",
            actual=f"A1..A{max(ambiguities) if ambiguities else 0}",
            passed=contiguous and len(ambiguities) >= 17)

    # 4) claim 表：C1..C34 齐全，判定值合法
    claims = (REPO_ROOT / "docs/CLAIMS_CHECKLIST.md").read_text(encoding="utf-8")
    claim_ids = sorted({int(x) for x in re.findall(r"^\|\s*C(\d+)\s*\|", claims, flags=re.M)})
    verdicts = set(re.findall(r"\|\s*(VERIFIED|FALSIFIED|BLOCKED|UNKNOWN)[^|]*\|", claims))
    g.check("c-claims-ids", expected="C1..Cn 连续且 n>=34", actual=f"C1..C{max(claim_ids) if claim_ids else 0}",
            passed=claim_ids == list(range(1, len(claim_ids) + 1)) and len(claim_ids) >= 34)
    g.check("c-claims-verdicts", expected=sorted(ALLOWED_VERDICTS), actual=sorted(verdicts),
            passed=bool(verdicts) and verdicts <= ALLOWED_VERDICTS)

    # 5) 运行器自检（全通过路径 + fail-fast + JSON pass=false）
    proc = subprocess.run(
        [sys.executable, "-m", "gates.run_gates", "--self-test"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    g.check("c-runner-selftest", expected="exit 0", actual=f"exit {proc.returncode}",
            passed=proc.returncode == 0,
            detail=(proc.stdout + proc.stderr).strip().splitlines()[-1] if (proc.stdout or proc.stderr) else "")

    # 6) results/ 可写
    probe = REPO_ROOT / "results" / "_m0_write_probe.json"
    try:
        probe.write_text(json.dumps({"ok": True}), encoding="utf-8")
        writable = probe.exists()
        probe.unlink()
    except Exception:
        writable = False
    g.check("c-results-writable", expected=True, actual=writable, passed=writable)

    # 7) 状态表包含 M0..M10 且初始状态合法
    state = (REPO_ROOT / "REPRO_STATE.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*(M\d+[ab]?)[^|]*\|[^|]*\|\s*(\w[\w ]*?)\s*\|", state, flags=re.M)
    state_modules = [m for m, _ in rows]
    statuses = {s for _, s in rows}
    expected_state = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8a", "M8b", "M9", "M10"]
    g.check("c-state-modules", expected=expected_state, actual=state_modules,
            passed=sorted(state_modules) == sorted(expected_state))
    g.check("c-state-status", expected="⊆ {PENDING, IN_PROGRESS, PASS, FAIL, BLOCKED}",
            actual=sorted(statuses),
            passed=statuses <= {"PENDING", "IN_PROGRESS", "PASS", "FAIL", "BLOCKED"})

    # 8) git 仓库已初始化
    g.check("c-git", expected=True, actual=(REPO_ROOT / ".git").exists(),
            passed=(REPO_ROOT / ".git").exists())

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
