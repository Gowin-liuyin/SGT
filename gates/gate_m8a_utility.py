"""M8a Utility 门禁：SGT embedding 注入 + lm-eval 端到端 + Utility 聚合。

lm-eval 只在 `.venv-lm` 里；门禁通过子进程调用它。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402

LM_VENV = REPO_ROOT / ".venv-lm" / "bin" / "python"
TINY = REPO_ROOT / ".models" / "tiny-llama"


def main() -> int:
    g = Gate("m8a", "utility", module_entry="§7 p.9：8 benchmark 平均差 = Utility")
    try:
        from sgt.model import SGTModel
        from sgt.utility import SGTEmbedding, install_sgt_embedding, utility_scores
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    torch.manual_seed(0)
    base = torch.nn.Embedding(64, 16)
    sgt = SGTModel(16, hidden=8, n_layers=1, n_heads=2, max_len=16, sigma_init=0.5)
    with torch.no_grad():
        for p in sgt.parameters():
            p.normal_(0, 0.1)
    wrap = SGTEmbedding(base, sgt, seed=0)

    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    out_wrap = wrap(ids)
    out_base = base(ids)
    changed = float((out_wrap - out_base).abs().max())
    g.check("c-embedding-tgt1", expected="changed when T>1", actual=f"max|diff|={changed:.3e}",
            passed=changed > 1e-6)

    single = wrap(ids[:, :1])
    g.check("c-embedding-t1-passthrough", expected="T==1 原样",
            actual=f"max|diff|={float((single - base(ids[:, :1])).abs().max()):.3e}",
            passed=torch.equal(single, base(ids[:, :1])))

    a = wrap(ids)
    b = SGTEmbedding(base, sgt, seed=0)(ids)
    c = SGTEmbedding(base, sgt, seed=1)(ids)
    g.check("c-embedding-determinism", expected="same seed equal, diff seed differ",
            actual=f"eq={torch.equal(a, b)} neq={not torch.equal(a, c)}",
            passed=torch.equal(a, b) and not torch.equal(a, c))

    # install 到小 LM
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    lm = LlamaForCausalLM(cfg).eval()
    wrapped = install_sgt_embedding(lm, sgt, seed=0)
    out = lm(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    g.check("c-install", expected="logits finite [2,4,64]",
            actual=f"shape={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}",
            passed=tuple(out.shape) == (2, 4, 64) and bool(torch.isfinite(out).all()))

    # utility 聚合
    base_res = {"results": {"arc_easy": {"acc": 0.50}, "piqa": {"acc": 0.70}, "mmlu": {"acc": 0.25}}}
    sgt_res = {"results": {"arc_easy": {"acc": 0.52}, "piqa": {"acc": 0.65}, "mmlu": {"acc": 0.26}}}
    u = utility_scores(base_res, sgt_res, ["arc_easy", "piqa", "mmlu", "hellaswag"])
    exp = ((0.52 - 0.50) + (0.65 - 0.70) + (0.26 - 0.25)) / 3 * 100
    g.check("c-utility-scores", expected=f"{exp:.6f} pp", actual=f"{u['utility_pp']:.6f} pp",
            passed=abs(u["utility_pp"] - exp) < 1e-9 and u["tasks_missing"] == ["hellaswag"])

    # lm-eval 端到端（子进程）
    if not LM_VENV.exists() or not TINY.exists():
        g.check("c-lm-eval-smoke", expected="run", actual="lm venv 或 tiny 模型缺失", passed=False,
                detail="install .venv-lm / .models/tiny-llama")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "sgt_tiny.pt"
            sgt_tiny = SGTModel(64, hidden=16, n_layers=1, n_heads=2, max_len=64)
            torch.save({"sgt": sgt_tiny.state_dict()}, ck)
            base_out = Path(tmp) / "base.json"
            sgt_out = Path(tmp) / "sgt.json"
            cmds = [
                [str(LM_VENV), "experiments/run_utility.py", "--model", str(TINY),
                 "--tasks", "arc_easy", "--limit", "2", "--batch-size", "1", "--device", "cpu",
                 "--out", str(base_out)],
                [str(LM_VENV), "experiments/run_utility.py", "--model", str(TINY),
                 "--tasks", "arc_easy", "--limit", "2", "--batch-size", "1", "--device", "cpu",
                 "--sgt-ckpt", str(ck), "--out", str(sgt_out)],
            ]
            codes = []
            for cmd in cmds:
                p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                   env={**__import__("os").environ, "HF_HOME": str(Path.home() / ".cache/huggingface")})
                codes.append(p.returncode)
                if p.returncode != 0:
                    err_tail = (p.stdout + p.stderr)[-400:]
                    g.check("c-lm-eval-smoke", expected="exit 0", actual=f"exit {p.returncode}",
                            passed=False, detail=err_tail)
            if all(c == 0 for c in codes):
                b = json.loads(base_out.read_text())
                s = json.loads(sgt_out.read_text())
                ok_json = ("results" in b and "results" in s
                           and "arc_easy" in b["results"] and "arc_easy" in s["results"])
                g.check("c-lm-eval-smoke", expected="两份结果均含 arc_easy",
                        actual=f"base_keys={list(b.get('results', {}))} sgt_keys={list(s.get('results', {}))}",
                        passed=ok_json)
                uu = utility_scores(b, s, ["arc_easy"])
                g.check("c-utility-real", expected="utility_pp 为有限数",
                        actual=f"{uu['utility_pp']}", passed=uu["utility_pp"] is not None)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
