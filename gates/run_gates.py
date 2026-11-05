"""顺序门禁运行器：按 M0→M10 串行执行，第一个失败立即停止（fail-fast）。

用法：
  .venv/bin/python -m gates.run_gates --all
  .venv/bin/python -m gates.run_gates --from m1 --to m4
  .venv/bin/python -m gates.run_gates --self-test     # 验证 fail-fast 与结果落盘

退出码：0 = 选中区间全部 PASS；1 = 有门禁 FAIL；2 = 有模块未实现 / 参数错误。

纪律：不允许跳过失败门禁；BLOCKED 必须写在 claim 表里，而不是让门禁"变绿"。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
REPO_ROOT = GATES_DIR.parent

# 模块顺序即依赖顺序。新增模块只能追加，不得跳号或改已有 id。
MODULES: list[tuple[str, str]] = [
    ("m0", "skeleton"),
    ("m1", "theory"),
    ("m2", "losses"),
    ("m3", "model"),
    ("m4", "metrics"),
    ("m5", "data"),
    ("m6", "train"),
    ("m7", "ablation"),
    ("m8a", "utility"),
    ("m8b", "pii"),
    ("m9", "large"),
    ("s1", "security"),
    ("m10", "report"),
]


def gate_script(module: str, name: str, gates_dir: Path = GATES_DIR) -> Path:
    return gates_dir / f"gate_{module}_{name}.py"


def discover_modules(gates_dir: Path) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for path in sorted(gates_dir.glob("gate_*.py")):
        m = re.match(r"gate_(m\d+)_(.+)\.py$", path.name)
        if m:
            found[m.group(1)] = m.group(2)
    def _key(kv):
        digits = "".join(ch for ch in kv[0][1:] if ch.isdigit())
        suffix = "".join(ch for ch in kv[0][1:] if ch.isalpha())
        return (int(digits), suffix)
    return sorted(found.items(), key=_key)


def run_one(module: str, name: str, gates_dir: Path, results_dir: Path) -> int:
    script = gate_script(module, name, gates_dir)
    if not script.exists():
        print(f"[{module}] NOT_IMPLEMENTED ({script.name} missing)")
        return 2
    print(f"\n=== {module} {name} ===", flush=True)
    env = dict(os.environ)
    env["REPRO_RESULTS_DIR"] = str(results_dir)
    proc = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, env=env)
    out = results_dir / f"gate_{module}_{name}.json"
    status = "?"
    if out.exists():
        try:
            status = "PASS" if json.loads(out.read_text())["pass"] else "FAIL"
        except Exception:
            status = "UNREADABLE"
    if proc.returncode == 0:
        print(f"[{module}] PASS ({out.name}: {status})")
    else:
        print(f"[{module}] FAIL (exit {proc.returncode}, {out.name}: {status})")
    return proc.returncode


def select(args: argparse.Namespace, modules: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not args.from_ and not args.to:
        return modules
    ids = [m for m, _ in modules]
    start = ids.index(args.from_)
    end = ids.index(args.to) + 1
    return modules[start:end]


def self_test() -> int:
    """端到端验证：(a) 全通过返回 0；(b) 失败即停且 JSON 落盘 pass=false。"""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gates_dir = tmp_path / "gates"
        results_dir = tmp_path / "results"
        gates_dir.mkdir()
        results_dir.mkdir()
        (gates_dir / "common.py").write_text(
            (GATES_DIR / "common.py").read_text(encoding="utf-8"), encoding="utf-8"
        )

        def write_gate(name: str, pass_: bool, sentinel: Path | None = None) -> None:
            sentinel_line = (
                f"Path({str(sentinel)!r}).write_text('ran')\n" if sentinel else ""
            )
            mod_id, mod_name = name[len("gate_"):-len(".py")].split("_", 1)
            (gates_dir / name).write_text(
                "from pathlib import Path\n"
                "from common import Gate\n"
                f"g = Gate({mod_id!r}, {mod_name!r})\n"
                f"{sentinel_line}"
                f"g.check('c', expected=True, actual={pass_})\n"
                "ok = g.finish()\n"
                "raise SystemExit(0 if ok else 1)\n",
                encoding="utf-8",
            )

        # 场景 1：全通过 → 退出码 0，结果 JSON pass=true
        write_gate("gate_m0_ok.py", True)
        proc = subprocess.run(
            [sys.executable, str(GATES_DIR / "run_gates.py"), "--gates-dir", str(gates_dir),
             "--results-dir", str(results_dir), "--all", "--discover"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        ok_path = results_dir / "gate_m0_ok.json"
        ok_json = json.loads(ok_path.read_text()) if ok_path.exists() else {}
        if proc.returncode != 0 or ok_json.get("pass") is not True:
            failures.append(f"all-pass path broken (rc={proc.returncode}, json={ok_json})")

        # 场景 2：第二个门禁失败 → 退出码 1，JSON pass=false，第三个不得执行
        for f in results_dir.iterdir():
            f.unlink()
        sentinel = tmp_path / "m2_ran.txt"
        write_gate("gate_m0_ok.py", True)
        write_gate("gate_m1_bad.py", False)
        write_gate("gate_m2_never.py", True, sentinel)
        proc = subprocess.run(
            [sys.executable, str(GATES_DIR / "run_gates.py"), "--gates-dir", str(gates_dir),
             "--results-dir", str(results_dir), "--all", "--discover"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        bad_path = results_dir / "gate_m1_bad.json"
        bad_json = json.loads(bad_path.read_text()) if bad_path.exists() else {}
        if proc.returncode != 1:
            failures.append(f"fail-fast exit code wrong: {proc.returncode}")
        if bad_json.get("pass") is not False:
            failures.append(f"failing gate JSON not pass=false: {bad_json}")
        if sentinel.exists():
            failures.append("runner continued past failing gate (sentinel written)")
        if (results_dir / "gate_m2_never.json").exists():
            failures.append("downstream gate result written after failure")

    for f in failures:
        print(f"[self-test] FAIL: {f}")
    if not failures:
        print("[self-test] ok: all-pass=0, fail-fast=1, JSON pass=false persisted")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--from", dest="from_", default=None)
    parser.add_argument("--to", dest="to", default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gates-dir", default=str(GATES_DIR))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--discover", action="store_true", help="按目录扫描模块顺序（自检用）")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    gates_dir = Path(args.gates_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    modules = discover_modules(gates_dir) if args.discover else MODULES
    selected = select(args, modules)
    if not selected:
        print("no module selected (use --all / --from / --to)", file=sys.stderr)
        return 2

    for module, name in selected:
        code = run_one(module, name, gates_dir, results_dir)
        if code != 0:
            print(f"\nSTOP at {module}: upstream gate not PASS; downstream must not start.")
            return 1 if code == 1 else 2
    print(f"\nAll {len(selected)} selected gate(s) PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
