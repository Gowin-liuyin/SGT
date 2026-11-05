"""M8a `utility.py` 的验证（主 venv + 微模型，不依赖 lm_eval）。

覆盖任务单 1–5：

1. ``SGTEmbedding``：T>1 输出 ≠ base，T==1 输出 == base；
2. 同 seed 两次调用逐位一致，不同 seed 不同；
3. ``install_sgt_embedding`` 后 ``get_input_embeddings()`` 是 wrapper，模型可前向、
   形状正确、无 NaN，且确实改变 logits（wrapper 真被 forward 调用）；
4. ``utility_scores`` 的 diff_pp / 平均值手算一致，缺失任务进入 ``tasks_missing``
   且不参与平均；
5. ``load_sgt_for_lm`` 往返逐位一致，checkpoint 的 d 与模型不符时报错。
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgt.model import SGTModel  # noqa: E402
from sgt.utility import (  # noqa: E402
    SGTEmbedding,
    install_sgt_embedding,
    load_sgt_for_lm,
    utility_scores,
)

D = 8
VOCAB = 32
HIDDEN = 16
N_LAYERS = 1
N_HEADS = 2
MAX_LEN = 16
B = 2
T = 5


def _base_embedding(seed: int = 0) -> nn.Embedding:
    generator = torch.Generator().manual_seed(seed)
    embedding = nn.Embedding(VOCAB, D)
    with torch.no_grad():
        embedding.weight.copy_(torch.randn(VOCAB, D, generator=generator))
    return embedding


def _trained_like_sgt(
    *,
    d: int = D,
    hidden: int = HIDDEN,
    n_layers: int = N_LAYERS,
    n_heads: int = N_HEADS,
    max_len: int = MAX_LEN,
    seed: int = 0,
) -> SGTModel:
    """构造一个“训练后样子”的 SGT：输出头非零，sigma 固定为 0.5。"""
    torch.manual_seed(seed)
    model = SGTModel(
        d, hidden=hidden, n_layers=n_layers, n_heads=n_heads, max_len=max_len
    )
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.head.weight.copy_(
            0.1 * torch.randn(model.head.weight.shape, generator=generator)
        )
        model.head.bias[:d].copy_(0.05 * torch.randn(d, generator=generator))
        model.head.bias[d:].fill_(math.log(0.5**2))
    return model.eval()


def _ids(batch: int = B, seq_len: int = T) -> torch.Tensor:
    generator = torch.Generator().manual_seed(42)
    return torch.randint(0, VOCAB, (batch, seq_len), generator=generator)


# --------------------------------------------------------------------------- #
# 1. T > 1 施加 SGT；T == 1 直通
# --------------------------------------------------------------------------- #
def test_01_sgt_embedding_only_transforms_multi_token_inputs() -> None:
    base = _base_embedding()
    sgt = _trained_like_sgt()
    embedding = SGTEmbedding(base, sgt, seed=0)
    ids = _ids()

    with torch.no_grad():
        transformed = embedding(ids)
        reference = base(ids)
        single_transformed = embedding(ids[:, :1])
        single_reference = base(ids[:, :1])

    assert transformed.shape == reference.shape == (B, T, D)
    assert not torch.allclose(transformed, reference), "T>1 时 SGT 未改变 embedding"
    # T == 1 必须逐位等于 base（逐 token 解码，查服务端表，不施加 SGT）。
    assert torch.equal(single_transformed, single_reference)
    assert single_transformed.shape == (B, 1, D)


# --------------------------------------------------------------------------- #
# 2. seed 确定性
# --------------------------------------------------------------------------- #
def test_02_seed_is_deterministic_and_effective() -> None:
    base = _base_embedding()
    sgt = _trained_like_sgt()
    embedding_a = SGTEmbedding(base, sgt, seed=7)
    embedding_b = SGTEmbedding(copy.deepcopy(base), sgt, seed=7)
    embedding_c = SGTEmbedding(copy.deepcopy(base), sgt, seed=8)
    ids = _ids()

    with torch.no_grad():
        first = embedding_a(ids)
        second = embedding_a(ids)  # 同一 wrapper 的两次调用
        same_seed = embedding_b(ids)  # 同 seed 的另一 wrapper
        other_seed = embedding_c(ids)

    assert torch.equal(first, second), "同一 seed 两次调用不一致"
    assert torch.equal(first, same_seed), "同 seed 的两个 wrapper 输出不一致"
    assert not torch.equal(first, other_seed), "不同 seed 的输出相同"


# --------------------------------------------------------------------------- #
# 3. install_sgt_embedding：替换真实 LM 的输入 embedding
# --------------------------------------------------------------------------- #
def test_03_install_replaces_embedding_and_model_forward_works() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=D,
        intermediate_size=2 * D,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        max_position_embeddings=MAX_LEN,
    )
    model = LlamaForCausalLM(config).eval()
    ids = _ids()

    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()

    wrapper = install_sgt_embedding(model, _trained_like_sgt(), seed=3)

    assert isinstance(wrapper, SGTEmbedding)
    assert model.get_input_embeddings() is wrapper
    assert model.model.embed_tokens is wrapper
    assert len(list(model.modules())) > 0

    with torch.no_grad():
        after = model(input_ids=ids).logits

    assert after.shape == (B, T, VOCAB)
    assert torch.isfinite(after).all(), "模型前向出现 NaN/Inf"
    assert not torch.equal(after, before), "安装 wrapper 后 logits 未改变，SGT 未被调用"


# --------------------------------------------------------------------------- #
# 4. utility_scores：手算 diff_pp / 平均值 / 缺失任务
# --------------------------------------------------------------------------- #
def test_04_utility_scores_matches_hand_computation() -> None:
    base = {
        "a": {"acc": 0.50},
        "b": {"acc_norm": 0.25},
        "c": {"acc": 0.90},
        "d": {"acc": 0.10, "acc_norm": 0.90},
        "e": {"acc,none": 0.40},  # lm-eval 0.4.13 的键名风格
    }
    sgt = {
        "a": {"acc": 0.55},  # +5 pp
        "b": {"acc_norm": 0.20},  # -5 pp
        "c": {},  # SGT 缺指标 -> missing
        "d": {"acc": 0.15, "acc_norm": 0.00},  # 必须用 acc：+5 pp
        "e": {"acc,none": 0.45},  # +5 pp
    }
    result = utility_scores(base, sgt, ["a", "b", "c", "d", "e", "f"])

    assert list(result["per_task"]) == ["a", "b", "d", "e"]
    assert result["tasks_missing"] == ["c", "f"]
    assert result["per_task"]["a"]["diff_pp"] == pytest.approx(5.0, abs=1e-12)
    assert result["per_task"]["b"]["diff_pp"] == pytest.approx(-5.0, abs=1e-12)
    assert result["per_task"]["d"]["diff_pp"] == pytest.approx(5.0, abs=1e-12)
    assert result["per_task"]["e"]["diff_pp"] == pytest.approx(5.0, abs=1e-12)
    # (5 - 5 + 5 + 5) / 4 = 2.5；c/f 不参与平均
    assert result["utility_pp"] == pytest.approx(2.5, abs=1e-12)

    # 兼容直接传 simple_evaluate 的完整返回值（含顶层 "results"）。
    wrapped = utility_scores({"results": base}, {"results": sgt}, ["a", "b", "c", "d", "e", "f"])
    assert wrapped == result

    # 全部缺失时 utility_pp 为 None（不是 NaN）。
    empty = utility_scores({}, {}, ["x", "y"])
    assert empty == {"per_task": {}, "utility_pp": None, "tasks_missing": ["x", "y"]}


# --------------------------------------------------------------------------- #
# 5. load_sgt_for_lm：往返逐位一致 + d 不匹配报错
# --------------------------------------------------------------------------- #
def test_05_load_sgt_for_lm_roundtrip_and_d_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"hidden_size": D}), encoding="utf-8"
    )
    source = _trained_like_sgt()
    ckpt_path = tmp_path / "sgt.pt"
    torch.save(source.state_dict(), ckpt_path)

    # hidden/n_layers 由 checkpoint 推断（默认 512/2 与 ckpt 不同，应发警告）；
    # n_heads 无法从 state_dict 推断，显式传入。
    with pytest.warns(RuntimeWarning):
        loaded = load_sgt_for_lm(model_dir, ckpt_path, n_heads=N_HEADS)

    assert loaded.d == D
    assert loaded.hidden == HIDDEN
    assert loaded.n_layers == N_LAYERS
    assert loaded.max_len == MAX_LEN

    x = torch.randn(B, T, D, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        source_mu, source_log_sigma2 = source(x)
        loaded_mu, loaded_log_sigma2 = loaded(x)
    assert torch.equal(source_mu, loaded_mu)
    assert torch.equal(source_log_sigma2, loaded_log_sigma2)

    wrong_dir = tmp_path / "wrong_model"
    wrong_dir.mkdir()
    (wrong_dir / "config.json").write_text(
        json.dumps({"hidden_size": D + 4}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="不一致"):
        load_sgt_for_lm(wrong_dir, ckpt_path, n_heads=N_HEADS)
