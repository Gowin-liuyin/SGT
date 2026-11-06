"""C1 逐行一致性门禁：CONFORMANCE 审计表 + GAP 关闭状态 + 对应代码能力存在。"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402


def main() -> int:
    g = Gate("c1", "conformance", module_entry="docs/CONFORMANCE.md：论文逐行一致性")
    doc_path = REPO_ROOT / "docs" / "CONFORMANCE.md"
    g.check("c-doc-exists", expected=True, actual=doc_path.exists(), passed=doc_path.exists())
    if not doc_path.exists():
        return 0 if g.finish() else 1
    doc = doc_path.read_text(encoding="utf-8")

    # 1) CF 条目齐全（§3–§7 覆盖）
    cf = sorted({int(x) for x in re.findall(r"\| CF-(\d+) \|", doc)})
    g.check("c-cf-items", expected="CF-01..CF-36 连续", actual=f"CF-01..CF-{max(cf) if cf else 0} ({len(cf)})",
            passed=cf == list(range(1, 37)))

    # 2) GAP 全部 CLOSED
    gap_rows = re.findall(r"^\|\s*GAP-(\d+)\s*\|(.+)$", doc, flags=re.M)
    unclosed = [n for n, rest in gap_rows if "CLOSED" not in rest]
    g.check("c-gaps-closed", expected="GAP-1..GAP-10 全部 CLOSED",
            actual=f"found={sorted(int(n) for n,_ in gap_rows)} unclosed={unclosed}",
            passed=len(gap_rows) >= 10 and not unclosed)

    # 3) 结果段落存在
    g.check("c-results-section", expected="含 GAP 关闭后的实验结果表",
            actual="6.1 修完 GAP 后的实验结果" in doc,
            passed="6.1 修完 GAP 后的实验结果" in doc)

    # 4) 代码能力存在（静态检查）
    from sgt.model import SGTModel
    sig = inspect.signature(SGTModel.__init__)
    needed = {"final_norm", "mu_scale", "sigma_floor", "pos_encoding", "sigma_max"}
    g.check("c-model-options", expected=sorted(needed),
            actual=sorted(set(sig.parameters) & needed), passed=needed <= set(sig.parameters))

    import importlib.util

    def load(name):
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    ra = load("run_ablation")
    for fn, name in (("build_sgt_model", "c-runablation-build"), ("resolve_selection", "c-select"),
                     ("resolve_eval_options", "c-mrp-split"), ("resolve_lm_dtype", "c-lm-dtype")):
        ok = hasattr(ra, fn)
        g.check(name, expected=f"{fn} 存在", actual=ok, passed=ok)
    g.check("c-skip-eval", expected="--skip-eval 在 CLI", actual="--skip-eval" in open(REPO_ROOT / "experiments/run_ablation.py").read(),
            passed="--skip-eval" in (REPO_ROOT / "experiments/run_ablation.py").read_text(encoding="utf-8"))

    ru = load("run_utility")
    for fn, name in (("parse_task_fewshot", "c-fewshot"), ("merge_eval_results", "c-merge"),
                     ("group_by_fewshot", "c-shot-group")):
        ok = hasattr(ru, fn)
        g.check(name, expected=f"{fn} 存在", actual=ok, passed=ok)
    util_src = (REPO_ROOT / "experiments/run_utility.py").read_text(encoding="utf-8")
    g.check("c-max-len-fallback", expected="--max-len-fallback 在 CLI",
            actual="--max-len-fallback" in util_src, passed="--max-len-fallback" in util_src)

    # 5) 加载器读 config.sgt
    from sgt import utility as U
    g.check("c-loader-config", expected="build_sgt_from_state 存在",
            actual=hasattr(U, "build_sgt_from_state"), passed=hasattr(U, "build_sgt_from_state"))
    src = (REPO_ROOT / "src/sgt/utility.py").read_text(encoding="utf-8")
    g.check("c-loader-reads-config", expected="读取 config['sgt']",
            actual="sgt_config_from_payload" in src, passed="sgt_config_from_payload" in src)

    # 6) 配置里有 8 任务与 per-task fewshot
    cfg = (REPO_ROOT / "configs/utility.yaml").read_text(encoding="utf-8")
    tasks = ["arc_challenge", "arc_easy", "openbookqa", "piqa", "truthfulqa_mc2", "ifeval", "mmlu", "hellaswag"]
    missing = [t for t in tasks if t not in cfg]
    g.check("c-utility-8tasks", expected=tasks, actual=f"missing={missing}", passed=not missing)
    import yaml
    ucfg = yaml.safe_load(cfg)
    shot = (ucfg.get("tasks") or {}).get("hellaswag", {}).get("num_fewshot") if isinstance(ucfg.get("tasks"), dict) else None
    g.check("c-hellaswag-10shot", expected="hellaswag num_fewshot=10", actual=shot, passed=shot == 10)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
