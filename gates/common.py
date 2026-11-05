"""门禁基类模板（复制到 gates/common.py 后按需修改）。

约定：
- 每个门禁脚本实例化 Gate(gate_id, name)，逐条 check(...)，最后 finish()。
- finish() 把结果写到 results/gate_<id>_<name>.json，并返回 pass。
- 结果 JSON 是复现证据的一部分，必须入库。
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# REPRO_RESULTS_DIR 允许自检把结果写到临时目录，避免污染真实门禁证据。
RESULTS_DIR = Path(os.environ.get("REPRO_RESULTS_DIR", REPO_ROOT / "results"))
if not RESULTS_DIR.is_absolute():
    RESULTS_DIR = (REPO_ROOT / RESULTS_DIR).resolve()


class Gate:
    def __init__(self, gate_id: str, name: str, module_entry: str = "", **env: Any) -> None:
        self.gate_id = gate_id
        self.name = name
        self.module_entry = module_entry
        self.checks: list[dict[str, Any]] = []
        self.env: dict[str, Any] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **env,
        }

    def check(
        self,
        check_id: str,
        *,
        expected: Any,
        actual: Any,
        passed: bool | None = None,
        detail: str = "",
    ) -> bool:
        """记录一条检查。passed 省略时由 expected == actual 推断；复杂判据请显式传 passed。"""
        ok = bool(expected == actual) if passed is None else bool(passed)
        self.checks.append(
            {
                "id": check_id,
                "expected": _render(expected),
                "actual": _render(actual),
                "pass": ok,
                "detail": detail,
            }
        )
        return ok

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["pass"] for c in self.checks)

    def finish(self) -> bool:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "gate": self.gate_id,
            "name": self.name,
            "module_entry": self.module_entry,
            "checks": self.checks,
            "pass": self.passed,
            "status": "PASS" if self.passed else "FAIL",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "env": self.env,
        }
        out = RESULTS_DIR / f"gate_{self.gate_id}_{self.name}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        failed = [c["id"] for c in self.checks if not c["pass"]]
        verdict = "PASS" if self.passed else f"FAIL ({len(failed)} check(s): {', '.join(failed)})"
        print(f"[gate {self.gate_id}/{self.name}] {verdict} -> {out}")
        return self.passed


def _render(value: Any) -> Any:
    """把 numpy 标量/数组转成可 JSON 序列化的表示，保持可读。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
    except ImportError:  # pragma: no cover
        pass
    return repr(value)


def main_exit(pass_: bool) -> None:
    """门禁脚本入口统一用这个退出码，保证 run_gates.py fail-fast。"""
    sys.exit(0 if pass_ else 1)
