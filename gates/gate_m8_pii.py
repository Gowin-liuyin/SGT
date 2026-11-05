"""M8b PUPA/PII 门禁：PII-Ratio 定义、PUPA 字段、NN 评测流程与 schema。"""

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


def main() -> int:
    g = Gate("m8b", "pii", module_entry="Table 3：PUPA NN/BeamClean FR + PII-Ratio")
    try:
        from sgt.pii import pii_fraction, pii_ratio, pii_units_recovered, pii_mask
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    text = "John lives in Paris."
    units = ["John", "Paris"]
    got = pii_fraction(text, units)
    g.check("c-pii-fraction", expected=9 / len(text), actual=round(got, 6),
            passed=abs(got - 9 / len(text)) < 1e-9)

    r_same = pii_ratio("a John b", "a John b", ["John"])
    r_red = pii_ratio("a John b", "a [REDACTED] b", ["John"])
    r_none = pii_ratio("no pii here", "no pii here", ["John"])
    g.check("c-pii-ratio-identity", expected=1.0, actual=r_same["ratio"],
            passed=r_same["ratio"] is not None and abs(r_same["ratio"] - 1.0) < 1e-12)
    g.check("c-pii-ratio-redacted", expected=0.0, actual=r_red["ratio"],
            passed=r_red["ratio"] is not None and abs(r_red["ratio"]) < 1e-12)
    g.check("c-pii-ratio-undefined", expected="defined=False", actual=r_none["defined"],
            passed=r_none["defined"] is False and r_none["ratio"] is None)

    rec = pii_units_recovered("John and Paris", "John and X", ["John", "Paris"])
    g.check("c-pii-units", expected=0.5, actual=rec["recovered_frac"],
            passed=abs(rec["recovered_frac"] - 0.5) < 1e-12)

    # PUPA 加载与字段
    import sgt.data as D
    rows = D.load_pupa(config="pupa_tnb", max_examples=5)
    ex0 = rows[0] if not hasattr(rows, "column_names") else {k: rows[k][0] for k in rows.column_names}
    g.check("c-pupa-fields", expected="user_query 存在且非空",
            actual=f"keys={sorted(ex0.keys())}",
            passed=isinstance(ex0.get("user_query"), str) and len(ex0["user_query"]) > 0)

    # run_pupa_eval：合成表 + 无损字符 tokenizer，无 LM、跳过 BeamClean
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_pupa", REPO_ROOT / "experiments/run_pupa.py")
        rp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rp)

        class CharTok:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, texts, add_special_tokens=True, padding=False, truncation=False):
                ids = [[ord(c) % 4096 + 2 for c in t] for t in texts]
                return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}

            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr((i - 2) % 4096) if i >= 2 else "" for i in ids)

        torch.manual_seed(0)
        table = torch.randn(4096, 8)
        examples = [{"user_query": "Alice visits Paris", "pii_units": "Alice||Paris"},
                    {"user_query": "Bob calls Carol", "pii_units": "Bob||Carol"}]
        out = rp.run_pupa_eval(examples, tokenizer=CharTok(), table=table, max_len=16,
                               batch_size=2, seed=0, skip_beamclean=True, progress=False)
        ok_json = json.dumps(out, allow_nan=False)
        fr = out["nn"]["fr"]
        g.check("c-pupa-eval-synthetic", expected="nn.fr in [0,100] 且 JSON 合法",
                actual=f"fr={fr}", passed=0.0 <= fr <= 100.0 and out["beamclean"].get("status") == "skipped")
        g.check("c-pupa-no-target-response", expected="rows 只引用 user_query",
                actual=f"row_keys={sorted(out['rows'][0].keys())[:6]}",
                passed=all("target_response" not in json.dumps(r) for r in out["rows"]))
    except Exception as exc:
        g.check("c-pupa-eval-synthetic", expected="run", actual=f"{type(exc).__name__}: {exc}",
                passed=False)

    # 正式产物（若已跑）
    pupa_results = sorted((REPO_ROOT / "results").glob("pupa_*.json"))
    if pupa_results:
        bad = []
        for p in pupa_results:
            d = json.loads(p.read_text())
            if "nn" not in d or "beamclean" not in d:
                bad.append(p.name)
        g.check("c-pupa-results-schema", expected="所有 pupa_*.json 含 nn/beamclean",
                actual=f"{len(pupa_results)} 个文件, bad={bad}", passed=not bad)
    else:
        g.check("c-pupa-results-schema", expected="至少一个 results/pupa_*.json（基线）",
                actual="none", passed=False)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
