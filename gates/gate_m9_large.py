"""M9 大模型启动脚本门禁：配置正确 + 启动命令生成 + 本机 BLOCKED + tiny dry-run。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gates.common import Gate  # noqa: E402


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True)


def main() -> int:
    g = Gate("m9", "large", module_entry="Table 2：70B/32B 大模型训练（本机不可执行）")
    cfg = REPO_ROOT / "configs/large_models.yaml"
    script = REPO_ROOT / "experiments/run_large_models.py"
    g.check("c-files", expected=True, actual=cfg.exists() and script.exists(),
            passed=cfg.exists() and script.exists())
    if not (cfg.exists() and script.exists()):
        return 0 if g.finish() else 1

    import yaml
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    models = raw.get("models", {})
    expect = {
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen3-32b": "Qwen/Qwen3-32B",
        "deepseek-r1-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    }
    got = {k: v.get("repo_id") for k, v in models.items()}
    g.check("c-models", expected=expect, actual=got, passed=got == expect)
    g.check("c-paper-protocol", expected="epochs=2 & loss=mi,acs,mnp & fsdp2",
            actual=str({k: raw.get("train", {}).get(k) for k in ("epochs",)})
                   + f" loss={raw.get('train', {}).get('loss_terms')} par={raw.get('parallel')}",
            passed=raw.get("train", {}).get("epochs") == 2
                   and raw.get("train", {}).get("loss_terms") == ["mi", "acs", "mnp"]
                   and raw.get("parallel", {}).get("fsdp2") is True)

    # 非 dry-run：无集群 → exit 3 + blocked，且不加载 70B
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "blocked.json"
        p = run(["experiments/run_large_models.py", "--config", str(cfg),
                 "--model", "llama-3.3-70b", "--out", str(out)])
        payload = json.loads(out.read_text()) if out.exists() else {}
        g.check("c-blocked", expected="exit 3 + status=blocked + llm_loaded=false",
                actual=f"exit={p.returncode} status={payload.get('status')} llm_loaded={payload.get('llm_loaded')}",
                passed=p.returncode == 3 and payload.get("status") == "blocked"
                and payload.get("llm_loaded") is False)
        cmd = payload.get("launch_command", "")
        g.check("c-launch-command", expected="含 torchrun/nnodes/nproc-per-node/fsdp2",
                actual=cmd[:120],
                passed=all(t in cmd for t in ("torchrun", "--nnodes", "--nproc-per-node", "--fsdp2")))

        # dry-run：tiny 模型 1 步
        tiny = REPO_ROOT / ".models" / "tiny-llama"
        if not tiny.exists():
            g.check("c-dryrun", expected="tiny 模型存在", actual="missing .models/tiny-llama", passed=False)
        else:
            out2 = Path(tmp) / "dry.json"
            p2 = run(["experiments/run_large_models.py", "--config", str(cfg),
                      "--model", "llama-3.3-70b", "--dry-run", "--dry-run-model", str(tiny),
                      "--out", str(out2)])
            d2 = json.loads(out2.read_text()) if out2.exists() else {}
            hist = d2.get("history", [])
            g.check("c-dryrun", expected="exit 0, mode=dry_run, 1 step finite",
                    actual=f"exit={p2.returncode} mode={d2.get('mode')} steps={len(hist)}",
                    passed=p2.returncode == 0 and d2.get("mode") == "dry_run" and len(hist) == 1
                    and abs(float(hist[0]["loss"])) < 1e9)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
