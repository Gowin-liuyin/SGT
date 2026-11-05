"""M6 训练回路的验证（任务单 1–8）：冻结 LM、只训练 SGT、损失恒等式、
严格同 seed 复现、checkpoint 往返、padding 无 NaN、config 原样返回。

全部测试使用**随机初始化的微型 Llama**（不下载模型）+ 微型 SGT，CPU、float32。
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgt.model import SGTModel  # noqa: E402
from sgt.train import TrainConfig, train_sgt  # noqa: E402

VOCAB = 128
HIDDEN = 32
B, T = 2, 8
SGT_MAX_LEN = 32


# --------------------------------------------------------------------------- #
# 固定输入
# --------------------------------------------------------------------------- #
def _make_lm(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def _make_sgt(seed: int = 1) -> SGTModel:
    torch.manual_seed(seed)
    return SGTModel(HIDDEN, hidden=16, n_layers=1, n_heads=2, max_len=SGT_MAX_LEN)


def _make_batches() -> list[dict[str, torch.Tensor]]:
    """两个固定 batch，均含右侧 padding（batch 1 为 3 个、batch 2 为 1 个）。"""
    g = torch.Generator().manual_seed(11)
    ids_a = torch.randint(0, VOCAB, (B, T), generator=g)
    mask_a = torch.ones(B, T, dtype=torch.bool)
    mask_a[1, -3:] = False
    ids_a[~mask_a] = 0

    ids_b = torch.randint(0, VOCAB, (B, T), generator=g)
    mask_b = torch.ones(B, T, dtype=torch.bool)
    mask_b[0, -1] = False
    ids_b[~mask_b] = 0

    return [
        {"input_ids": ids_a, "attention_mask": mask_a},
        {"input_ids": ids_b, "attention_mask": mask_b},
    ]


def _config(**overrides) -> TrainConfig:
    base = {"steps": 3, "lr": 1e-3, "seed": 0}
    base.update(overrides)
    return TrainConfig(**base)


def _run(sgt=None, lm=None, batches=None, cfg=None) -> dict:
    sgt = _make_sgt() if sgt is None else sgt
    lm = _make_lm() if lm is None else lm
    batches = _make_batches() if batches is None else batches
    cfg = _config() if cfg is None else cfg
    return train_sgt(sgt, lm, batches, config=cfg)


def _snapshot(module: torch.nn.Module) -> tuple[dict, dict]:
    params = {name: p.detach().clone() for name, p in module.named_parameters()}
    buffers = {name: b.detach().clone() for name, b in module.named_buffers()}
    return params, buffers


# --------------------------------------------------------------------------- #
# 1 + 8. 冻结 LM、SGT 确实更新；config 原样返回
# --------------------------------------------------------------------------- #
def test_01_lm_frozen_and_sgt_updated():
    lm = _make_lm()
    sgt = _make_sgt()
    lm_params_before, lm_buffers_before = _snapshot(lm)
    sgt_params_before, _ = _snapshot(sgt)

    result = _run(sgt=sgt, lm=lm)

    # LM 的所有参数与 buffer 逐位不变；参数全部 requires_grad=False。
    for name, param in lm.named_parameters():
        assert torch.equal(lm_params_before[name], param), f"lm 参数 {name} 被修改"
        assert param.requires_grad is False, f"lm 参数 {name} 未被冻结"
    for name, buffer in lm.named_buffers():
        assert torch.equal(lm_buffers_before[name], buffer), f"lm buffer {name} 被修改"
    assert lm.training is False, "lm 未处于 eval 模式"

    # SGT 至少一个参数发生变化。
    changed = [
        name
        for name, param in sgt.named_parameters()
        if not torch.equal(sgt_params_before[name], param)
    ]
    assert changed, "训练后 SGT 没有任何参数变化"
    assert result["completed_steps"] == 3


def test_02_history_identity_finite_and_config_returned():
    cfg = _config(steps=3)
    batches = _make_batches()
    result = _run(batches=batches, cfg=cfg)

    # config 原样返回（asdict 语义；含全部超参）。
    assert result["config"] == asdict(cfg)
    assert result["completed_steps"] == cfg.steps
    assert len(result["history"]) == cfg.steps
    assert isinstance(result["median_norm"], float)
    assert math.isfinite(result["median_norm"]) and result["median_norm"] > 0.0

    for index, row in enumerate(result["history"]):
        assert row["step"] == index
        assert row["seed"] == cfg.seed
        assert set(cfg.loss_terms) <= set(row)
        for key in ("loss", "loss_u", *cfg.loss_terms):
            assert isinstance(row[key], float)
            assert math.isfinite(row[key]), f"step {index} 的 {key} 非有限: {row[key]}"

        expected = row["loss_u"] + sum(
            float(cfg.weights.get(term, 1.0)) * row[term] for term in cfg.loss_terms
        )
        relative_error = abs(row["loss"] - expected) / max(1.0, abs(expected))
        assert relative_error < 1e-9, (
            f"step {index}: loss={row['loss']} != loss_u + Σw·term={expected} "
            f"(相对误差 {relative_error:.3e})"
        )


# --------------------------------------------------------------------------- #
# 4. 同 seed 两次运行 history 逐项严格相等
# --------------------------------------------------------------------------- #
def test_03_same_seed_is_strictly_reproducible():
    lm = _make_lm()
    batches = _make_batches()
    cfg = _config(steps=3)
    sgt_a = _make_sgt(seed=7)
    sgt_b = copy.deepcopy(sgt_a)

    # 两次运行前设置**不同**的全局 RNG：训练必须只由 config.seed 的显式 Generator 驱动。
    torch.manual_seed(1234)
    result_a = train_sgt(sgt_a, lm, batches, config=cfg)
    torch.manual_seed(9876)
    result_b = train_sgt(sgt_b, lm, batches, config=cfg)

    assert result_a["history"] == result_b["history"], "同 seed 两次运行的 history 不严格相等"
    assert result_a["median_norm"] == result_b["median_norm"]
    assert result_a["config"] == result_b["config"]
    assert result_a["completed_steps"] == result_b["completed_steps"]


# --------------------------------------------------------------------------- #
# 5. loss_terms=() 时 loss == loss_u
# --------------------------------------------------------------------------- #
def test_04_empty_loss_terms_equals_loss_u():
    cfg = _config(steps=2, loss_terms=())
    sgt = _make_sgt()
    before, _ = _snapshot(sgt)
    result = _run(sgt=sgt, cfg=cfg)

    for row in result["history"]:
        assert row["loss"] == row["loss_u"]
        assert set(row) == {"step", "seed", "loss", "loss_u"}
    assert any(
        not torch.equal(before[name], param) for name, param in sgt.named_parameters()
    ), "仅 L_U 时 SGT 未更新"


# --------------------------------------------------------------------------- #
# 6. checkpoint 落盘 + load_state_dict 后输出逐位一致
# --------------------------------------------------------------------------- #
def test_05_checkpoint_roundtrip(tmp_path):
    lm = _make_lm()
    sgt = _make_sgt()
    batches = _make_batches()
    checkpoint_dir = tmp_path / "ckpt"
    cfg = _config(steps=2, log_every=1, checkpoint_dir=str(checkpoint_dir))

    train_sgt(sgt, lm, batches, config=cfg)

    files = sorted(checkpoint_dir.glob("step_*.pt"))
    assert [path.name for path in files] == ["step_0.pt", "step_1.pt"]

    checkpoint = torch.load(files[-1], map_location="cpu")
    assert {"sgt", "optimizer", "step", "config"} <= set(checkpoint)
    assert checkpoint["step"] == 1
    assert checkpoint["config"] == asdict(cfg)
    assert checkpoint["optimizer"]["state"], "optimizer state 为空"

    fresh = _make_sgt(seed=99)
    fresh.load_state_dict(checkpoint["sgt"])
    fresh.eval()
    sgt.eval()

    ids = batches[0]["input_ids"]
    mask = batches[0]["attention_mask"]
    x = lm.get_input_embeddings().weight.detach()[ids]
    with torch.no_grad():
        mu_trained, lv_trained = sgt(x, attention_mask=mask)
        mu_loaded, lv_loaded = fresh(x, attention_mask=mask)
    assert torch.equal(mu_trained, mu_loaded), "load_state_dict 后 mu 不一致"
    assert torch.equal(lv_trained, lv_loaded), "load_state_dict 后 log_sigma2 不一致"


# --------------------------------------------------------------------------- #
# 7. 带 padding 的 batch 不产生 NaN
# --------------------------------------------------------------------------- #
def test_06_padded_batch_no_nan():
    batches = _make_batches()
    # 前提：这些 batch 确实含 padding（否则该测试没有意义）。
    assert any(
        not bool(value)
        for batch in batches
        for value in batch["attention_mask"].reshape(-1).tolist()
    )

    sgt = _make_sgt()
    result = _run(sgt=sgt, batches=batches, cfg=_config(steps=3))

    for row in result["history"]:
        for key, value in row.items():
            if key in ("step", "seed"):
                continue
            assert math.isfinite(value), f"padding batch 产生非有限值: {row}"
    for name, param in sgt.named_parameters():
        assert torch.isfinite(param).all(), f"padding batch 使 SGT 参数 {name} 出现 NaN/Inf"


# --------------------------------------------------------------------------- #
# 补充：全部五个 loss term + 非 1 权重；padding 位置被 mask
# --------------------------------------------------------------------------- #
def test_07_all_terms_and_weights_identity():
    cfg = _config(
        steps=2,
        loss_terms=("mi", "acs", "mnp", "cge", "bcs"),
        weights={"mi": 0.5, "acs": 2.0, "mnp": 0.25, "cge": 0.125, "bcs": 0.75},
    )
    result = _run(cfg=cfg)

    for row in result["history"]:
        assert set(cfg.loss_terms) <= set(row)
        expected = row["loss_u"] + sum(
            cfg.weights[term] * row[term] for term in cfg.loss_terms
        )
        assert abs(row["loss"] - expected) / max(1.0, abs(expected)) < 1e-9
        assert all(math.isfinite(row[term]) for term in cfg.loss_terms)


# --------------------------------------------------------------------------- #
# 前置条件校验（快速失败，不静默）
# --------------------------------------------------------------------------- #
def test_08_invalid_inputs_rejected():
    lm = _make_lm()
    sgt = _make_sgt()

    with pytest.raises(ValueError, match="batches"):
        train_sgt(sgt, lm, [], config=_config())
    with pytest.raises(ValueError, match="loss_terms"):
        train_sgt(sgt, lm, _make_batches(), config=_config(loss_terms=("unknown",)))
    with pytest.raises(ValueError, match="重复"):
        train_sgt(sgt, lm, _make_batches(), config=_config(loss_terms=("mi", "mi")))
    with pytest.raises(TypeError, match="input_ids"):
        bad = [{"input_ids": torch.zeros(2, T), "attention_mask": torch.ones(2, T)}]
        train_sgt(sgt, lm, bad, config=_config(steps=1))
