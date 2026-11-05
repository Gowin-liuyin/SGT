"""M5 数据管线门禁：OpenOrca 15% 协议、Alpaca、PUPA 字段、tokenize/batch 契约。

网络仅在缓存缺失时触发；已缓存时应在秒级完成。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402


def main() -> int:
    g = Gate("m5", "data", module_entry="§7 p.9：OpenOrca 15%、Alpaca 评估、PUPA")
    try:
        import sgt.data as D
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    # 1) 15% 协议
    counts = D.openorca_counts()
    ok_counts = counts["n_total"] > 0 and counts["n_selected"] == math.ceil(0.15 * counts["n_total"])
    g.check("c-openorca-counts", expected="n_selected == ceil(0.15*n_total)",
            actual=f"n_total={counts['n_total']} n_selected={counts['n_selected']}",
            passed=ok_counts)

    # 2) OpenOrca 加载确定性
    o1 = D.load_openorca(frac=0.15, max_examples=200)
    o2 = D.load_openorca(frac=0.15, max_examples=200)
    f1, f2 = D.dataset_fingerprint(o1, n=200), D.dataset_fingerprint(o2, n=200)
    g.check("c-openorca-deterministic", expected="len=200 & identical fingerprint",
            actual=f"len={len(o1)} fp={f1}=={f2}", passed=len(o1) == 200 and f1 == f2)

    # 3) Alpaca + build_prompt 精确断言
    a = D.load_alpaca(max_examples=100)
    ex1 = {"instruction": "Q1", "input": "", "output": "A1"}
    ex2 = {"instruction": "Q2", "input": "ctx", "output": "A2"}
    p1 = D.build_prompt(ex1, "alpaca")
    p2 = D.build_prompt(ex2, "alpaca")
    g.check("c-alpaca", expected=(100, "Q1\nA1", "Q2\nctx\nA2"),
            actual=(len(a), p1, p2),
            passed=len(a) == 100 and p1 == "Q1\nA1" and p2 == "Q2\nctx\nA2")

    # 4) 训练/评估文本不重叠
    s_train = {D.build_prompt(e, "openorca") for e in o1}
    s_eval = {D.build_prompt(e, "alpaca") for e in a[:50]}
    g.check("c-disjoint", expected="intersection == 0", actual=len(s_train & s_eval),
            passed=len(s_train & s_eval) == 0)

    # 5) PUPA：字段选择必须是 user_query（SPEC A20）
    pu = D.load_pupa(config="pupa_tnb", max_examples=20)
    row = pu[0]
    prompt = D.build_prompt(row, "pupa")
    used_user_query = prompt.strip() == str(row.get("user_query", "")).strip()
    not_response = prompt.strip() != str(row.get("target_response", "")).strip()
    g.check("c-pupa-load", expected="20 rows & non-empty prompt",
            actual=f"len={len(pu)} prompt_len={len(prompt)}", passed=len(pu) == 20 and len(prompt) > 0)
    g.check("c-pupa-field", expected="build_prompt == user_query (SPEC A20)",
            actual=f"equals_user_query={used_user_query} differs_from_target_response={not_response}",
            passed=used_user_query and not_response,
            detail=f"fields={sorted(row.keys())}")

    # 6) tokenize 往返
    tok = D._default_tokenizer() if hasattr(D, "_default_tokenizer") else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    enc = D.tokenize_examples(o1[:20], tok, max_len=128, dataset="openorca")
    sims = []
    for i, ids in enumerate(enc["input_ids"]):
        txt = tok.decode(ids, skip_special_tokens=True)
        src = D.build_prompt(o1[i], "openorca")
        # 字符级相似度（按较短串归一）
        common = sum(1 for ch in txt if ch in src)
        sims.append(common / max(1, len(txt)))
    g.check("c-tokenize", expected="min char-similarity > 0.9", actual=f"{min(sims):.3f}",
            passed=min(sims) > 0.9)

    # 7) batch 契约
    b1 = next(iter(D.make_batches(o1, tok, batch_size=4, max_len=64, seed=3, dataset_name="openorca")))
    b2 = next(iter(D.make_batches(o1, tok, batch_size=4, max_len=64, seed=3, dataset_name="openorca")))
    import torch
    same = torch.equal(b1["input_ids"], b2["input_ids"])
    shapes_ok = b1["input_ids"].shape == b1["attention_mask"].shape
    pad_ok = bool((b1["attention_mask"][b1["input_ids"] == tok.pad_token_id] == 0).all().item()) if tok.pad_token_id is not None else True
    g.check("c-batches", expected="deterministic, shapes equal, padding masked",
            actual=f"same={same} shapes={tuple(b1['input_ids'].shape)} pad_masked={pad_ok}",
            passed=same and shapes_ok and pad_ok)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
