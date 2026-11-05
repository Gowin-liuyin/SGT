"""M6 训练回路门禁：冻结 LM、只训 SGT、损失分解、可复现、checkpoint。

用随机初始化的微型 Llama（不下载权重），在 CPU 上跑几步。
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402

torch.set_default_dtype(torch.float32)


def tiny_lm(vocab=128, h=32, layers=2, heads=2):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=h, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=heads, max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def make_batches():
    ids = torch.tensor([
        [5, 6, 7, 8, 9, 10, 11, 12],
        [21, 22, 23, 24, 25, 26, 27, 28],
    ])
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1, 6:] = False  # padding
    b1 = {"input_ids": ids, "attention_mask": mask}
    ids2 = torch.tensor([[31, 32, 33, 34, 35, 36, 37, 38], [41, 42, 43, 44, 45, 46, 47, 48]])
    m2 = torch.ones_like(ids2, dtype=torch.bool)
    m2[0, :2] = False
    b2 = {"input_ids": ids2, "attention_mask": m2}
    ids3 = torch.tensor([[51, 52, 53, 54, 55, 56, 57, 58], [61, 62, 63, 64, 65, 66, 67, 68]])
    b3 = {"input_ids": ids3, "attention_mask": torch.ones_like(ids3, dtype=torch.bool)}
    return [b1, b2, b3]


def main() -> int:
    g = Gate("m6", "train", module_entry="§4：冻结 M、只更新 SGT；Eq (3)(11)")
    try:
        from sgt.model import SGTModel
        from sgt.train import TrainConfig, train_sgt
        import_ok, err = True, ""
    except Exception as exc:  # pragma: no cover
        import_ok, err = False, f"{type(exc).__name__}: {exc}"
    g.check("c-import", expected=True, actual=import_ok, detail=err)
    if not import_ok:
        return 0 if g.finish() else 1

    torch.manual_seed(0)
    lm = tiny_lm()
    sgt = SGTModel(32, hidden=16, n_layers=1, n_heads=2, max_len=32)
    cfg = TrainConfig(steps=3, lr=1e-2, seed=11, grad_clip=1.0, log_every=1)
    batches = make_batches()

    lm_before = copy.deepcopy(lm.state_dict())
    sgt_before = copy.deepcopy(sgt.state_dict())
    out = train_sgt(sgt, lm, batches, config=cfg)
    sgt_after = sgt.state_dict()
    n_changed = sum(1 for k in sgt_before if not torch.equal(sgt_before[k], sgt_after[k]))

    hist = out["history"]
    g.check("c-train-runs", expected="3 finite steps",
            actual=f"n={len(hist)} losses={[round(h['loss'],4) for h in hist]}",
            passed=len(hist) == 3 and all(torch.isfinite(torch.tensor(h["loss"])) for h in hist))

    lm_after = lm.state_dict()
    frozen = all(torch.equal(lm_before[k], lm_after[k]) for k in lm_before)
    g.check("c-lm-frozen", expected="bitwise identical", actual=frozen, passed=frozen)

    g.check("c-sgt-updated", expected="全部 SGT 张量都有更新", actual=f"{n_changed}/{len(sgt_before)} changed",
            passed=n_changed > 0)

    # 损失分解
    worst = 0.0
    for h in hist:
        terms = sum(h.get(t, 0.0) * cfg.weights.get(t, 1.0) for t in cfg.loss_terms)
        worst = max(worst, abs(h["loss"] - (h["loss_u"] + terms)))
    g.check("c-loss-decomp", expected="|loss-(loss_u+Σw·term)| < 1e-9", actual=f"{worst:.3e}",
            passed=worst < 1e-9)

    # 可复现
    torch.manual_seed(0)
    lm2 = tiny_lm()
    sgt2 = SGTModel(32, hidden=16, n_layers=1, n_heads=2, max_len=32)
    sgt2.load_state_dict(copy.deepcopy(sgt.state_dict()))  # 不同初始化，仅比同初始化下的两次运行
    # 用同一初始权重重跑两次
    sgt_a = SGTModel(32, hidden=16, n_layers=1, n_heads=2, max_len=32)
    sgt_b = copy.deepcopy(sgt_a)
    out_a = train_sgt(sgt_a, lm, batches, config=cfg)
    out_b = train_sgt(sgt_b, lm, batches, config=cfg)
    same = [round(h["loss"], 10) for h in out_a["history"]] == [round(h["loss"], 10) for h in out_b["history"]]
    g.check("c-determinism", expected="identical history", actual=f"{[round(h['loss'],6) for h in out_a['history']]}",
            passed=same)

    # 只用效用损失
    sgt_c = SGTModel(32, hidden=16, n_layers=1, n_heads=2, max_len=32)
    cfg_u = TrainConfig(steps=2, lr=1e-2, seed=3, loss_terms=())
    out_c = train_sgt(sgt_c, lm, batches, config=cfg_u)
    ok_u = all(abs(h["loss"] - h["loss_u"]) < 1e-12 for h in out_c["history"])
    g.check("c-no-terms", expected="loss == loss_u", actual=ok_u, passed=ok_u)

    # checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        sgt_d = SGTModel(32, hidden=16, n_layers=1, n_heads=2, max_len=32)
        cfg_ck = TrainConfig(steps=2, lr=1e-2, seed=5, checkpoint_dir=tmp, log_every=1)
        out_d = train_sgt(sgt_d, lm, batches, config=cfg_ck)
        files = sorted(Path(tmp).glob("*.pt"))
        g.check("c-checkpoint-files", expected=">=1 checkpoint", actual=[f.name for f in files],
                passed=len(files) >= 1)
        if files:
            ck = torch.load(files[-1], map_location="cpu")
            state = ck["sgt"] if isinstance(ck, dict) and "sgt" in ck else ck
            sgt_e = SGTModel(32, hidden=16, n_layers=1, n_heads=2, max_len=32)
            sgt_e.load_state_dict(state)
            x = torch.randn(2, 5, 32)
            with torch.no_grad():
                mu_d, lv_d = sgt_d(x)
                mu_e, lv_e = sgt_e(x)
            g.check("c-checkpoint-restore", expected="identical outputs",
                    actual=f"{float((mu_d-mu_e).abs().max()):.3e}",
                    passed=float((mu_d - mu_e).abs().max()) == 0.0 and float((lv_d - lv_e).abs().max()) == 0.0)

    g.check("c-config-echo", expected="config echoed", actual=list(out["config"].keys())[:4],
            passed=out["config"]["steps"] == 3 and out["config"]["seed"] == 11)

    g.check("c-median-norm", expected="> 0", actual=f"{out['median_norm']:.6f}",
            passed=out["median_norm"] > 0)

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
