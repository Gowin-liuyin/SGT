"""M8a `utility.py` + `experiments/run_utility.py` 的验证（主 venv + 微模型，不依赖 lm_eval）。

覆盖任务单 1–5：

1. ``SGTEmbedding``：T>1 输出 ≠ base，T==1 输出 == base；
2. 同 seed 的两个 wrapper 相同调用序列逐位一致（跨 wrapper 可复现），不同 seed 不同；
   同一 wrapper 对同一输入两次调用**不同**（fresh noise，论文的 stochastic SGT）；
   ``fresh_noise=False`` 保留旧的“每次重置”行为（仅消融对照）；
3. ``install_sgt_embedding`` 后 ``get_input_embeddings()`` 是 wrapper，模型可前向、
   形状正确、无 NaN，且确实改变 logits（wrapper 真被 forward 调用）；
4. ``utility_scores`` 的 diff_pp / 平均值手算一致，缺失任务进入 ``tasks_missing``
   且不参与平均；
5. ``load_sgt_for_lm`` 往返逐位一致，checkpoint 的 d 与模型不符时报错。

审计 GAP-6（论文 §7 的 8 任务与 per-task fewshot）额外覆盖（``simple_evaluate`` 全部
monkeypatch，不联网、不跑 harness）：

6. ``SGTMaxLenExceeded`` 长度守卫：T ≤ max_len 行为不变，超长在进入 SGT 前抛哨兵；
7. ``--task-fewshot`` 解析与优先级（CLI > config > 全局）；
8. 按 shot 分组调用次数与合并后的键结构（与单次 ``simple_evaluate`` 一致）；
9. ``--max-len-fallback skip`` 的 JSON 结构（``{"status": "skipped_too_long"}``）；
10. ``--max-len-fallback error`` 时非零退出且不落盘；
11. ``configs/utility.yaml``：论文 8 任务、HellaSwag 10-shot、MMLU limit 是 per-subtask。
"""

from __future__ import annotations

import copy
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import run_utility as U  # noqa: E402
from sgt.model import SGTModel  # noqa: E402
from sgt.utility import (  # noqa: E402
    SGTEmbedding,
    _read_state_dict,
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


def _option_rich_sgt(
    *,
    pos_encoding: str = "sinusoidal",
    final_norm: bool = True,
    mu_scale: float | None = 0.5,
    sigma_floor: float | None = 0.01,
    sigma_max: float | None = 0.05,
    seed: int = 0,
) -> SGTModel:
    """构造带 A1/A21 选项、输出头非零的 SGT（mu/σ 的夹取在重建后必须逐位复现）。"""
    torch.manual_seed(seed)
    model = SGTModel(
        D,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=2,
        max_len=MAX_LEN,
        pos_encoding=pos_encoding,
        final_norm=final_norm,
        mu_scale=mu_scale,
        sigma_floor=sigma_floor,
        sigma_max=sigma_max,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.head.weight.copy_(
            0.1 * torch.randn(model.head.weight.shape, generator=generator)
        )
        model.head.bias[:D].copy_(0.05 * torch.randn(D, generator=generator))
        model.head.bias[D:].fill_(math.log(0.5**2))
    return model.eval()


def _sgt_config(options: dict) -> dict:
    """``experiments/run_ablation.py`` 风格的 ``config["sgt"]`` 块。"""
    config = {
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "ffn_mult": 2,
        "max_len": MAX_LEN,
        "sigma_init": 1.0,
    }
    config.update(options)
    return config


def _write_model_dir(tmp_path: Path, d: int = D) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"hidden_size": d}), encoding="utf-8"
    )
    return model_dir


def _assert_bitwise_equal(reference: SGTModel, loaded: SGTModel) -> None:
    x = torch.randn(B, T, D, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref_mu, ref_log_sigma2 = reference(x)
        loaded_mu, loaded_log_sigma2 = loaded(x)
    assert torch.equal(ref_mu, loaded_mu)
    assert torch.equal(ref_log_sigma2, loaded_log_sigma2)


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
# 2. seed 确定性：跨 wrapper 可复现；同一 wrapper 每次新噪声
# --------------------------------------------------------------------------- #
def test_02_seed_is_deterministic_across_wrappers_and_effective() -> None:
    base = _base_embedding()
    sgt = _trained_like_sgt()
    embedding_a = SGTEmbedding(base, sgt, seed=7)
    embedding_b = SGTEmbedding(copy.deepcopy(base), sgt, seed=7)
    embedding_c = SGTEmbedding(copy.deepcopy(base), sgt, seed=8)
    ids = _ids()

    with torch.no_grad():
        same_seed = embedding_b(ids)  # 同 seed 的另一 wrapper（各自第 1 次调用）
        other_seed = embedding_c(ids)
        first = embedding_a(ids)

    assert torch.equal(first, same_seed), "同 seed 的两个 wrapper 首次调用不一致"
    assert not torch.equal(first, other_seed), "不同 seed 的输出相同"


def test_02a_fresh_noise_makes_repeated_calls_differ() -> None:
    """fresh_noise=True（默认）：同一 wrapper 对同一输入的两次调用必须不同。"""
    embedding = SGTEmbedding(_base_embedding(), _trained_like_sgt(), seed=0)
    ids = _ids()

    with torch.no_grad():
        first = embedding(ids)
        second = embedding(ids)
        third = embedding(ids)

    assert not torch.equal(first, second), "同一输入两次调用输出相同：噪声生成器被重置"
    assert not torch.equal(second, third), "同一输入连续调用复用噪声"


def test_02b_same_seed_wrappers_follow_identical_call_sequences() -> None:
    """两个同 seed 的 wrapper 走相同调用序列时输出逐位一致（可复现）。"""
    base = _base_embedding()
    sgt = _trained_like_sgt()
    wrapper_a = SGTEmbedding(copy.deepcopy(base), sgt, seed=11)
    wrapper_b = SGTEmbedding(copy.deepcopy(base), sgt, seed=11)
    second_ids = _ids(batch=1, seq_len=T)

    with torch.no_grad():
        for step, current in enumerate((_ids(), second_ids, _ids())):
            out_a = wrapper_a(current)
            out_b = wrapper_b(current)
            assert torch.equal(out_a, out_b), f"第 {step} 次调用的输出不一致"


def test_02c_fresh_noise_false_keeps_legacy_reset() -> None:
    """fresh_noise=False 保留旧行为（每次重置到 seed），仅作消融对照。"""
    base = _base_embedding()
    sgt = _trained_like_sgt()
    legacy = SGTEmbedding(base, sgt, seed=5, fresh_noise=False)
    ids = _ids()

    with torch.no_grad():
        first = legacy(ids)
        second = legacy(ids)
        # 新 wrapper 的首次调用 = 旧 wrapper 的任一次调用（都从同一 seed 开始）
        reference = SGTEmbedding(copy.deepcopy(base), sgt, seed=5, fresh_noise=False)(ids)

    assert torch.equal(first, second), "fresh_noise=False 时同一输入两次调用应相同"
    assert torch.equal(first, reference), "fresh_noise=False 时同 seed 首次调用应一致"


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


# --------------------------------------------------------------------------- #
# 6. 生成路径：SGT 只作用于 prompt，逐 token 解码不进入 SGT
# --------------------------------------------------------------------------- #
def test_06_generate_only_transforms_prompt() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=D,
        intermediate_size=2 * D,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        max_position_embeddings=MAX_LEN,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = LlamaForCausalLM(config).eval()
    sgt = _trained_like_sgt()
    install_sgt_embedding(model, sgt, seed=0)

    # 记录 SGT 估计器被调用时的序列长度：prompt 一次 T>1；decode 的 T==1 必须在
    # SGTEmbedding 内直接返回 base，不进入 SGT。
    calls: list[int] = []
    original_forward = sgt.forward

    def spy(x, *args, **kwargs):
        calls.append(int(x.shape[1]))
        return original_forward(x, *args, **kwargs)

    sgt.forward = spy
    ids = torch.randint(2, VOCAB, (1, 6), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        generated = model.generate(
            input_ids=ids,
            max_new_tokens=4,
            min_new_tokens=4,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=1,
        )

    assert generated.shape[0] == 1
    assert generated.shape[1] >= ids.shape[1]
    assert calls == [ids.shape[1]], (
        f"SGT 应只对 prompt 调用一次（T={ids.shape[1]}），实际调用序列 {calls}"
    )


# --------------------------------------------------------------------------- #
# 7. checkpoint 键名兼容：裸 / {"state_dict": ...} / {"sgt": ...}
# --------------------------------------------------------------------------- #
def test_read_state_dict_accepts_three_formats(tmp_path: Path) -> None:
    d, hidden, n_heads, max_len = 8, 8, 2, 8
    torch.manual_seed(0)
    source = SGTModel(
        d, hidden=hidden, n_layers=1, n_heads=n_heads, max_len=max_len
    ).eval()
    with torch.no_grad():  # 输出头非零，否则 mu 恒 0，无法证明权重真的被加载
        source.head.weight.normal_(0.0, 0.1)
        source.head.bias[:d].normal_(0.0, 0.05)
        source.head.bias[d:].fill_(math.log(0.25))

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"hidden_size": d}), encoding="utf-8"
    )

    state = source.state_dict()
    payloads = {
        "bare": state,
        "state_dict": {"state_dict": state},
        "sgt": {"sgt": state, "step": 0, "config": {"n_heads": n_heads}},
    }
    x = torch.randn(B, T, d, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref_mu, ref_log_sigma2 = source(x)

    for name, payload in payloads.items():
        ckpt_path = tmp_path / f"{name}.pt"
        torch.save(payload, ckpt_path)

        # 内部读取：三种封装都解出同一个 state_dict
        loaded_state = _read_state_dict(ckpt_path)
        assert set(loaded_state) == set(state), f"{name}: 解包出的 state_dict 键不一致"

        # 端到端：模型构造 + 权重加载成功，输出与原模型逐位一致
        loaded = load_sgt_for_lm(
            model_dir, ckpt_path, n_heads=n_heads, hidden=hidden, n_layers=1
        )
        with torch.no_grad():
            mu, log_sigma2 = loaded(x)
        assert torch.equal(mu, ref_mu), f"{name}: mu 未逐位一致"
        assert torch.equal(log_sigma2, ref_log_sigma2), f"{name}: log_sigma2 未逐位一致"


# --------------------------------------------------------------------------- #
# 8. checkpoint 结构非法：报错指明缺少 input_proj.weight 与已尝试的封装
# --------------------------------------------------------------------------- #
def test_read_state_dict_rejects_garbage(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "garbage.pt"
    torch.save({"foo": 1}, ckpt_path)
    with pytest.raises(ValueError) as excinfo:
        _read_state_dict(ckpt_path)
    message = str(excinfo.value)
    assert "input_proj.weight" in message
    # 错误信息必须说明尝试过哪些封装键（裸 / state_dict / sgt）。
    assert "state_dict" in message and "sgt" in message



# --------------------------------------------------------------------------- #
# 6. GAP-6：SGT 长度守卫（T ≤ max_len 不变；超长在进入 SGT 前抛哨兵）
# --------------------------------------------------------------------------- #
def test_06_max_len_guard_raises_sentinel() -> None:
    wrapper = SGTEmbedding(_base_embedding(), _trained_like_sgt(), seed=0)
    U._install_max_len_guard(wrapper, MAX_LEN)

    ok_ids = torch.randint(0, VOCAB, (1, MAX_LEN), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = wrapper(ok_ids)
    assert out.shape == (1, MAX_LEN, D)

    long_ids = torch.randint(0, VOCAB, (1, MAX_LEN + 1), generator=torch.Generator().manual_seed(0))
    with pytest.raises(U.SGTMaxLenExceeded) as excinfo:
        wrapper(long_ids)
    assert excinfo.value.max_len == MAX_LEN
    assert excinfo.value.observed_len == MAX_LEN + 1
    assert U.is_too_long_error(excinfo.value)
    # 也识别 SGTModel 的原始报错（guard 未安装时的路径）
    assert U.is_too_long_error(ValueError(f"序列长度 T=99 超过 max_len={MAX_LEN}"))
    assert not U.is_too_long_error(ValueError("别的问题"))
    assert not U.is_too_long_error(RuntimeError("别的问题"))


# --------------------------------------------------------------------------- #
# 7. GAP-6：--task-fewshot 解析与优先级
# --------------------------------------------------------------------------- #
def test_07_task_fewshot_parsing_and_precedence() -> None:
    assert U.parse_task_fewshot("hellaswag=10,mmlu=0") == {"hellaswag": 10, "mmlu": 0}
    assert U.parse_task_fewshot("") == {}
    assert U.parse_task_fewshot(None) == {}
    with pytest.raises(ValueError, match="缺少"):
        U.parse_task_fewshot("hellaswag")
    with pytest.raises(ValueError, match="非负"):
        U.parse_task_fewshot("hellaswag=-1")
    with pytest.raises(ValueError, match="整数"):
        U.parse_task_fewshot("hellaswag=ten")

    shots = U.resolve_task_fewshot(
        ["a", "b", "c"],
        global_num_fewshot=5,
        config_shots={"b": 10},
        cli_shots={"c": 0},
    )
    assert shots == {"a": 5, "b": 10, "c": 0}
    with pytest.raises(ValueError, match="未在任务列表"):
        U.resolve_task_fewshot(["a"], cli_shots={"zzz": 1})
    assert U.group_by_fewshot(["a", "b", "c"], shots) == [(5, ["a"]), (10, ["b"]), (0, ["c"])]


# --------------------------------------------------------------------------- #
# 8. GAP-6：按 shot 分组调用 + 合并结果键结构与单次调用一致
# --------------------------------------------------------------------------- #
def _fake_payload(tasks, shot):
    return {
        "results": {task: {"acc,none": 0.5} for task in tasks},
        "n-samples": {task: {"original": 100, "effective": 100} for task in tasks},
        "n-shot": {task: shot for task in tasks},
        "versions": {task: 1 for task in tasks},
        "group_subtasks": {},
        "configs": {task: {"num_fewshot": shot} for task in tasks},
        "higher_is_better": {task: {"acc": True} for task in tasks},
    }


def test_08_evaluate_tasks_groups_by_fewshot_and_merges_keys() -> None:
    tasks = ["arc_easy", "hellaswag", "mmlu"]
    shots = {"arc_easy": 0, "hellaswag": 10, "mmlu": 0}
    calls: list[dict] = []

    def evaluate_fn(**kwargs):
        calls.append({"tasks": list(kwargs["tasks"]), "num_fewshot": kwargs["num_fewshot"]})
        return _fake_payload(kwargs["tasks"], kwargs["num_fewshot"])

    outcome = U.evaluate_tasks(
        evaluate_fn,
        base_kwargs={"limit": 100},
        tasks=tasks,
        shots=shots,
        max_len_fallback="error",
    )
    assert len(calls) == 2  # 两个 shot 值 -> 两次 simple_evaluate 调用
    assert calls[0] == {"tasks": ["arc_easy", "mmlu"], "num_fewshot": 0}
    assert calls[1] == {"tasks": ["hellaswag"], "num_fewshot": 10}

    merged = outcome["merged"]
    assert outcome["skipped"] == {}
    assert list(merged["results"]) == ["arc_easy", "mmlu", "hellaswag"]
    assert merged["n-shot"] == {"arc_easy": 0, "mmlu": 0, "hellaswag": 10}

    single = _fake_payload(tasks, None)
    assert set(merged) == set(single)  # 顶层键结构与单次调用一致
    assert set(merged["results"]) == set(single["results"])
    for task in tasks:
        assert set(merged["results"][task]) == set(single["results"][task])


def test_08b_merge_eval_results_rejects_conflicting_entries() -> None:
    first = {"results": {"a": {"acc,none": 0.1}}, "n-shot": {"a": 0}}
    second = {"results": {"a": {"acc,none": 0.2}}, "n-shot": {"a": 10}}
    with pytest.raises(ValueError, match="冲突"):
        U.merge_eval_results([first, second])
    assert U.merge_eval_results([first])["results"] == first["results"]


# --------------------------------------------------------------------------- #
# 9–10. GAP-6：main 的 skip / error 行为（mock lm_eval + 微模型，不联网）
# --------------------------------------------------------------------------- #
def _write_tiny_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"hidden_size": D, "vocab_size": VOCAB}), encoding="utf-8"
    )
    return model_dir


def _write_sgt_ckpt(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    sgt = SGTModel(D, hidden=D, n_layers=1, n_heads=N_HEADS, max_len=MAX_LEN)
    ckpt = tmp_path / "sgt.pt"
    torch.save({"sgt": sgt.state_dict()}, ckpt)
    return ckpt


def _install_fake_lm_eval(monkeypatch, evaluate_fn) -> None:
    module = types.ModuleType("lm_eval")
    module.__version__ = "0.4.13-fake"
    module.simple_evaluate = evaluate_fn
    monkeypatch.setitem(sys.modules, "lm_eval", module)


def _patch_from_pretrained(monkeypatch) -> None:
    """把 harness 的模型加载替换成随机初始化微型 Llama（不下载、不联网）。"""
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    def fake_from_pretrained(*fargs, **fkwargs):
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=D,
            intermediate_size=2 * D,
            num_hidden_layers=1,
            num_attention_heads=N_HEADS,
            num_key_value_heads=N_HEADS,
            max_position_embeddings=MAX_LEN,
        )
        return LlamaForCausalLM(config)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)


def _fake_eval_with_hellaswag_too_long(calls: list[dict], *, trigger_injection) -> callable:
    def evaluate_fn(**kwargs):
        tasks = list(kwargs["tasks"])
        calls.append({"tasks": tasks, "num_fewshot": kwargs["num_fewshot"]})
        trigger_injection()
        results = {}
        for task in tasks:
            if task == "hellaswag":
                raise U.SGTMaxLenExceeded(
                    f"输入序列长度 T={MAX_LEN + 10} 超过 SGT max_len={MAX_LEN}",
                    max_len=MAX_LEN,
                    observed_len=MAX_LEN + 10,
                )
            results[task] = {"acc,none": 0.5}
        return _fake_payload(results, kwargs["num_fewshot"])

    return evaluate_fn


def _sgt_main_args(tmp_path: Path, model_dir: Path, ckpt: Path, out: Path, fallback: str):
    return [
        "--model",
        str(model_dir),
        "--tasks",
        "hellaswag,arc_easy",
        "--task-fewshot",
        "hellaswag=10",
        "--sgt-ckpt",
        str(ckpt),
        "--sgt-hidden",
        str(D),
        "--sgt-layers",
        "1",
        "--sgt-heads",
        str(N_HEADS),
        "--max-len-fallback",
        fallback,
        "--limit",
        "100",
        "--out",
        str(out),
    ]


def test_09_main_skip_writes_skipped_task_and_continues(tmp_path, monkeypatch) -> None:
    from transformers import AutoModelForCausalLM

    model_dir = _write_tiny_model_dir(tmp_path)
    ckpt = _write_sgt_ckpt(tmp_path)
    eval_calls: list[dict] = []
    _patch_from_pretrained(monkeypatch)

    def trigger_injection():
        AutoModelForCausalLM.from_pretrained("unused")

    _install_fake_lm_eval(
        monkeypatch, _fake_eval_with_hellaswag_too_long(eval_calls, trigger_injection=trigger_injection)
    )

    out = tmp_path / "utility_skip.json"
    code = U.main(_sgt_main_args(tmp_path, model_dir, ckpt, out, "skip"))

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["results"]["arc_easy"] == {"acc,none": 0.5}
    assert payload["results"]["hellaswag"]["status"] == "skipped_too_long"
    assert payload["results"]["hellaswag"]["max_len"] == MAX_LEN
    assert payload["meta"]["max_len_fallback"] == "skip"
    assert payload["meta"]["sgt_max_len"] == MAX_LEN
    assert payload["meta"]["skipped"]["hellaswag"]["status"] == "skipped_too_long"
    assert payload["meta"]["num_fewshot"] == {"hellaswag": 10, "arc_easy": 0}
    # lm-eval 自己的 n-shot 原样保留（只有成功任务有）
    assert payload["meta"]["n_shot"] == {"arc_easy": 0}
    # 分组尝试 + 逐任务重试 + arc_easy 成功 = 3 次调用
    assert payload["meta"]["n_calls"] == 3
    assert [call["tasks"] for call in payload["meta"]["calls"]] == [
        ["hellaswag"],
        ["hellaswag"],
        ["arc_easy"],
    ]
    assert payload["meta"]["sgt"]["max_len"] == MAX_LEN


def test_10_main_error_mode_exits_nonzero_without_json(tmp_path, monkeypatch) -> None:
    from transformers import AutoModelForCausalLM

    model_dir = _write_tiny_model_dir(tmp_path)
    ckpt = _write_sgt_ckpt(tmp_path)
    _patch_from_pretrained(monkeypatch)
    _install_fake_lm_eval(
        monkeypatch,
        _fake_eval_with_hellaswag_too_long(
            [], trigger_injection=lambda: AutoModelForCausalLM.from_pretrained("unused")
        ),
    )

    out = tmp_path / "utility_error.json"
    code = U.main(_sgt_main_args(tmp_path, model_dir, ckpt, out, "error"))

    assert code == 1
    assert not out.exists()


# --------------------------------------------------------------------------- #
# 11. GAP-6：configs/utility.yaml = 论文 8 任务 + per-task fewshot
# --------------------------------------------------------------------------- #
def test_11_utility_config_matches_paper_protocol() -> None:
    config_path = REPO_ROOT / "configs" / "utility.yaml"
    config = U.load_utility_config(config_path)
    expected_tasks = [
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "piqa",
        "truthfulqa_mc2",
        "ifeval",
        "mmlu",
        "hellaswag",
    ]
    assert config["task_names"] == expected_tasks
    assert config["limit"] == 100
    assert config["num_fewshot"] == 0
    assert config["max_len_fallback"] == "error"
    shots = {name: spec["num_fewshot"] for name, spec in config["tasks"].items()}
    assert shots == {**{name: 0 for name in expected_tasks[:7]}, "hellaswag": 10}
    text = config_path.read_text(encoding="utf-8")
    # MMLU 的 limit 是 per-subtask（57 个子任务各 limit 条）必须写明
    assert "per-subtask" in text
    assert "MMLU" in text and "hellaswag" in text


def test_11b_main_uses_config_for_tasks_and_shots(tmp_path, monkeypatch) -> None:
    model_dir = _write_tiny_model_dir(tmp_path)
    calls: list[dict] = []

    def evaluate_fn(**kwargs):
        calls.append(
            {
                "tasks": list(kwargs["tasks"]),
                "num_fewshot": kwargs["num_fewshot"],
                "limit": kwargs["limit"],
            }
        )
        return _fake_payload(kwargs["tasks"], kwargs["num_fewshot"])

    _install_fake_lm_eval(monkeypatch, evaluate_fn)

    out = tmp_path / "utility_config.json"
    code = U.main(
        [
            "--model",
            str(model_dir),
            "--config",
            str(REPO_ROOT / "configs" / "utility.yaml"),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tasks"] == [
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "piqa",
        "truthfulqa_mc2",
        "ifeval",
        "mmlu",
        "hellaswag",
    ]
    assert payload["limit"] == 100
    shots = payload["meta"]["num_fewshot"]
    assert shots["hellaswag"] == 10
    assert all(shots[task] == 0 for task in payload["tasks"] if task != "hellaswag")
    assert [call["num_fewshot"] for call in calls] == [0, 10]
    assert calls[0]["tasks"] == payload["tasks"][:7]
    assert calls[1]["tasks"] == ["hellaswag"]
    assert calls[0]["limit"] == 100


# --------------------------------------------------------------------------- #
# 12. config["sgt"] 优先：无参数但改变前向行为的选项必须按训练时重建
# --------------------------------------------------------------------------- #
def test_12_load_sgt_for_lm_uses_checkpoint_config_options(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path)
    source = _option_rich_sgt()  # sinusoidal + final_norm + mu_scale + sigma_floor
    sgt_config = _sgt_config(
        {
            "sigma_max": 0.05,
            "sigma_floor": 0.01,
            "mu_scale": 0.5,
            "pos_encoding": "sinusoidal",
            "final_norm": True,
        }
    )
    ckpt_path = tmp_path / "sgt.pt"
    torch.save(
        {"sgt": source.state_dict(), "config": {"sgt": sgt_config, "d": D}},
        ckpt_path,
    )

    loaded, meta = load_sgt_for_lm(
        model_dir,
        ckpt_path,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        return_meta=True,
    )

    # 重建出的选项与保存时一致（sinusoidal 无 pos_emb；final_norm 有 weight/bias）
    assert loaded.pos_encoding == "sinusoidal"
    assert loaded.pos_emb is None
    assert loaded.final_norm is not None
    assert loaded.mu_scale == 0.5
    assert loaded.sigma_floor == 0.01
    assert loaded.sigma_max == 0.05
    assert loaded.max_len == MAX_LEN
    assert loaded.n_heads == N_HEADS

    # meta 记录实际使用的选项（供结果 JSON 追溯）
    assert meta["pos_encoding"] == "sinusoidal"
    assert meta["final_norm"] is True
    assert meta["mu_scale"] == 0.5
    assert meta["sigma_floor"] == 0.01
    assert meta["pos_encoding_source"] == "config"
    assert meta["n_heads_source"] == "config"
    assert meta["inferred_unparameterized_options"] is False
    assert meta["missing_unparameterized_options"] == []

    # 端到端：µ / log σ² 与源模型逐位一致（含 tanh 夹取与 σ 下界）
    _assert_bitwise_equal(source, loaded)

    # 形状推断仍是一致性校验：config 与 state_dict 不一致必须报错，不静默
    bad_config = _sgt_config({"pos_encoding": "sinusoidal", "final_norm": True})
    bad_config["hidden"] = 2 * HIDDEN
    bad_path = tmp_path / "bad_hidden.pt"
    torch.save(
        {"sgt": source.state_dict(), "config": {"sgt": bad_config}}, bad_path
    )
    with pytest.raises(ValueError, match="hidden"):
        load_sgt_for_lm(
            model_dir,
            bad_path,
            hidden=HIDDEN,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
        )

    # pos_encoding 与 state_dict 键不一致也要在加载前报错（指出是哪一项）
    bad_pos = _sgt_config({"pos_encoding": "learned", "final_norm": True})
    bad_pos_path = tmp_path / "bad_pos.pt"
    torch.save({"sgt": source.state_dict(), "config": {"sgt": bad_pos}}, bad_pos_path)
    with pytest.raises(ValueError, match="pos_encoding"):
        load_sgt_for_lm(
            model_dir,
            bad_pos_path,
            hidden=HIDDEN,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
        )


# --------------------------------------------------------------------------- #
# 13. 无 config 的旧式 {"sgt": ...}：按 state_dict 推断，并标记 inferred
# --------------------------------------------------------------------------- #
def test_13_load_sgt_for_lm_infers_options_without_config(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path)

    # 旧式 ckpt（无 config）：sinusoidal + final_norm=False + 无 µ/σ 界
    source = _option_rich_sgt(
        pos_encoding="sinusoidal",
        final_norm=False,
        mu_scale=None,
        sigma_floor=None,
        sigma_max=None,
    )
    ckpt_path = tmp_path / "old.pt"
    torch.save({"sgt": source.state_dict()}, ckpt_path)

    loaded, meta = load_sgt_for_lm(
        model_dir,
        ckpt_path,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        return_meta=True,
    )
    assert loaded.pos_encoding == "sinusoidal"
    assert loaded.final_norm is None
    assert loaded.mu_scale is None
    assert loaded.sigma_floor is None
    assert loaded.sigma_max is None
    # 没有 pos_emb 就推不出训练长度，用 SGTModel 默认（sinusoidal 不受 max_len 限制）
    assert loaded.max_len == 512
    assert meta["pos_encoding_source"] == "inferred"
    assert meta["mu_scale"] is None
    assert meta["sigma_floor"] is None
    assert meta["inferred_unparameterized_options"] is True
    assert sorted(meta["missing_unparameterized_options"]) == [
        "mu_scale",
        "pos_encoding",
        "sigma_floor",
        "sigma_max",
    ]
    _assert_bitwise_equal(source, loaded)

    # 旧式 learned ckpt：pos_emb 存在 → 推断为 learned，max_len 取 pos_emb 行数
    learned = _option_rich_sgt(
        pos_encoding="learned",
        final_norm=True,
        mu_scale=None,
        sigma_floor=None,
        sigma_max=None,
    )
    learned_path = tmp_path / "old_learned.pt"
    torch.save({"sgt": learned.state_dict()}, learned_path)
    loaded_learned, meta_learned = load_sgt_for_lm(
        model_dir,
        learned_path,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        return_meta=True,
    )
    assert loaded_learned.pos_encoding == "learned"
    assert loaded_learned.pos_emb is not None
    assert loaded_learned.max_len == MAX_LEN
    assert loaded_learned.final_norm is not None  # 由 final_norm.weight 是否存在推断
    assert meta_learned["pos_encoding_source"] == "inferred"
    assert meta_learned["final_norm"] is True
    _assert_bitwise_equal(learned, loaded_learned)


# --------------------------------------------------------------------------- #
# 14. config["sgt"] 的 12 个键逐项生效（任务规格指定的最小构造）
# --------------------------------------------------------------------------- #
def test_load_sgt_for_lm_rebuilds_all_config_sgt_options(tmp_path: Path) -> None:
    """``config["sgt"]`` 的 12 项必须全部按 checkpoint 重建（不是按调用方默认值猜）。

    构造 ``sinusoidal`` + ``final_norm`` + µ/σ 上下界的 SGT（无 ``pos_emb.weight``），
    保存成带 ``config["sgt"]`` 的 checkpoint 后用 ``load_sgt_for_lm`` 加载：重建模型的
    ``pos_encoding`` / ``mu_scale`` / ``sigma_floor`` / ``final_norm`` 等属性必须一致，
    同输入输出逐位一致（tanh(µ/M) 与 σ 的 floor/max 夹取都要复现）。
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"hidden_size": 8}), encoding="utf-8"
    )

    torch.manual_seed(0)
    source = SGTModel(
        8,
        hidden=8,
        n_layers=1,
        n_heads=2,
        max_len=8,
        sigma_init=0.01,
        sigma_max=0.02,
        sigma_floor=0.005,
        mu_scale=0.1,
        pos_encoding="sinusoidal",
        final_norm=True,
    )
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():  # 输出头非零，否则 µ ≡ 0，逐位比较证明不了权重/选项生效
        source.head.weight.copy_(
            0.1 * torch.randn(source.head.weight.shape, generator=generator)
        )
        source.head.bias[:8].copy_(0.05 * torch.randn(8, generator=generator))
        source.head.bias[8:].fill_(math.log(0.005**2))
    source.eval()

    sgt_config = {
        "hidden": 8,
        "n_layers": 1,
        "n_heads": 2,
        "ffn_mult": 4,
        "max_len": 8,
        "sigma_init": 0.01,
        "sigma_max": 0.02,
        "sigma_floor": 0.005,
        "mu_scale": 0.1,
        "pos_encoding": "sinusoidal",
        "final_norm": True,
        "pos_emb_init_std": 0.02,
    }
    ckpt_path = tmp_path / "sgt.pt"
    torch.save({"sgt": source.state_dict(), "config": {"sgt": sgt_config}}, ckpt_path)

    # 调用方默认 hidden=512 / n_layers=2 与 ckpt 不同：必须告警并按 ckpt 构造。
    with pytest.warns(RuntimeWarning):
        loaded, meta = load_sgt_for_lm(model_dir, ckpt_path, return_meta=True)

    assert loaded.pos_encoding == "sinusoidal"
    assert loaded.pos_emb is None  # sinusoidal 无 pos_emb.weight
    assert loaded.final_norm is not None
    assert loaded.mu_scale == pytest.approx(0.1)
    assert loaded.sigma_floor == pytest.approx(0.005)
    assert loaded.sigma_max == pytest.approx(0.02)
    assert loaded.sigma_init == pytest.approx(0.01)
    assert (loaded.hidden, loaded.n_layers, loaded.n_heads) == (8, 1, 2)
    assert (loaded.ffn_mult, loaded.max_len) == (4, 8)

    # meta 记录实际使用的这些选项（供结果 JSON 追溯）。
    assert meta["pos_encoding"] == "sinusoidal"
    assert meta["final_norm"] is True
    assert meta["mu_scale"] == pytest.approx(0.1)
    assert meta["sigma_floor"] == pytest.approx(0.005)
    assert meta["sigma_max"] == pytest.approx(0.02)

    x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref_mu, ref_log_sigma2 = source(x)
        mu, log_sigma2 = loaded(x)
    assert torch.equal(mu, ref_mu)
    assert torch.equal(log_sigma2, ref_log_sigma2)


# --------------------------------------------------------------------------- #
# 15. 长度守卫只对 learned 位置编码生效（sinusoidal 是长度泛化选择）
# --------------------------------------------------------------------------- #
def test_15_max_len_guard_only_applies_to_learned_pos_encoding() -> None:
    """``pos_encoding != "learned"`` 时 SGT 前向对任意 T 有效，守卫不得误判超长。

    真实场景：sinusoidal SGT（``max_len`` 只是元数据）跑 HellaSwag 10-shot，prompt
    最长 ~971 token，被旧的 ``max_len`` 守卫误记为 ``skipped_too_long``。
    """
    max_len = 4
    torch.manual_seed(0)
    sinusoidal = SGTModel(
        8, hidden=8, n_layers=1, n_heads=2, max_len=max_len, pos_encoding="sinusoidal"
    ).eval()
    wrapper = SGTEmbedding(_base_embedding(), sinusoidal, seed=0)
    assert wrapper.sgt.pos_encoding == "sinusoidal"

    guarded, meta = U._install_max_len_guard(wrapper, max_len, "sinusoidal")
    assert meta["installed"] is False
    assert meta["reason"] == "pos_encoding=sinusoidal"
    # 未安装守卫：wrapper 原样返回，forward 不被包装（实例 __dict__ 里没有 forward）。
    assert guarded is wrapper
    assert "forward" not in wrapper.__dict__
    long_ids = _ids(batch=1, seq_len=32)
    with torch.no_grad():
        out = guarded(long_ids)
    assert out.shape == (1, 32, D)

    # learned：max_len 是硬上限，T > max_len 必须抛哨兵。
    torch.manual_seed(0)
    learned = SGTModel(8, hidden=8, n_layers=1, n_heads=2, max_len=max_len).eval()
    assert learned.pos_encoding == "learned"
    guarded_learned, learned_meta = U._install_max_len_guard(
        SGTEmbedding(_base_embedding(), learned, seed=0), max_len, "learned"
    )
    assert learned_meta["installed"] is True
    assert learned_meta["max_len"] == max_len
    assert "forward" in guarded_learned.__dict__
    with pytest.raises(U.SGTMaxLenExceeded) as excinfo:
        guarded_learned(_ids(batch=1, seq_len=8))
    assert excinfo.value.max_len == max_len
    assert excinfo.value.observed_len == 8
    assert U.is_too_long_error(excinfo.value)
