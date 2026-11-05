"""M5 数据管线测试：协议计数、条数/确定性、prompt 精确格式、零重叠、PUPA 字段、
tokenize→decode 往返、batch 形状/mask/确定性。

真实数据：OpenOrca/Alpaca 走 datasets-server rows 接口（首次运行写 ``data_cache/``，
已 gitignore，之后走缓存），PUPA 走 ``datasets.load_dataset``（237 条，< 1 MB）。
"""

from __future__ import annotations

import math
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgt import data as D  # noqa: E402

TOKENIZER_NAME = "unsloth/Llama-3.2-1B-Instruct"
PUPA_FIELDS = {
    "conversation_hash",
    "predicted_category",
    "user_query",
    "target_response",
    "pii_units",
    "redacted_query",
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# 1) 论文协议计数 ----------------------------------------------------------- #
def test_01_openorca_counts():
    counts = D.openorca_counts()
    print(f"openorca_counts = {counts}")
    assert counts["frac"] == 0.15
    assert counts["n_total"] > 0
    assert counts["n_selected"] == math.ceil(0.15 * counts["n_total"])
    assert isinstance(counts["n_fetched"], int)
    assert counts["n_fetched"] >= 0


# 2) OpenOrca 200 条：条数 + 指纹确定性 ------------------------------------- #
def test_02_openorca_200_deterministic():
    t0 = time.perf_counter()
    first = D.load_openorca(max_examples=200)
    t1 = time.perf_counter()
    second = D.load_openorca(max_examples=200)
    print(
        f"OpenOrca: 条数={len(first)} n_total={first.meta['n_total']} "
        f"n_fetched={first.meta['n_fetched']} 首载耗时={t1 - t0:.2f}s "
        f"第二次={time.perf_counter() - t1:.3f}s"
    )
    assert isinstance(first, list) and isinstance(second, list)
    assert len(first) == 200 and len(second) == 200
    assert D.dataset_fingerprint(first) == D.dataset_fingerprint(second)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert first.meta["n_total"] > 0
    assert first.meta["n_fetched"] == 200
    print(f"OpenOrca 指纹 = {D.dataset_fingerprint(first)}")


# 3) Alpaca 100 条：条数 + prompt 精确字符串 -------------------------------- #
def test_03_alpaca_100_prompt_format():
    t0 = time.perf_counter()
    examples = D.load_alpaca(max_examples=100)
    dt = time.perf_counter() - t0
    print(
        f"Alpaca: 条数={len(examples)} n_total={examples.meta['n_total']} "
        f"耗时={dt:.2f}s"
    )
    assert len(examples) == 100
    assert all(set(row) == {"instruction", "input", "output", "text"} for row in examples)

    with_input = {
        "instruction": "Translate to French.",
        "input": "Hello",
        "output": "Bonjour",
        "text": "unused",
    }
    without_input = {
        "instruction": "Say hello.",
        "input": "",
        "output": "Hello!",
        "text": "unused",
    }
    # 精确字符串断言（含/不含 input 两条）
    assert D.build_prompt(with_input, "alpaca") == "Translate to French.\nHello\nBonjour"
    assert D.build_prompt(without_input, "alpaca") == "Say hello.\nHello!"

    # 真实样本复核字段拼接
    real_with = next(row for row in examples if row["input"])
    real_without = next(row for row in examples if not row["input"])
    assert D.build_prompt(real_with, "alpaca") == (
        f"{real_with['instruction']}\n{real_with['input']}\n{real_with['output']}"
    )
    assert D.build_prompt(real_without, "alpaca") == (
        f"{real_without['instruction']}\n{real_without['output']}"
    )
    print(f"Alpaca 真实含 input prompt = {D.build_prompt(real_with, 'alpaca')[:120]!r}")
    print(f"Alpaca 真实无 input prompt = {D.build_prompt(real_without, 'alpaca')[:120]!r}")


# 4) 训练集与评估集 prompt 零重叠 ------------------------------------------ #
def test_04_openorca_alpaca_no_overlap():
    openorca = D.load_openorca(max_examples=200)
    alpaca = D.load_alpaca(max_examples=50)
    openorca_prompts = {D.build_prompt(row, "openorca") for row in openorca}
    alpaca_prompts = {D.build_prompt(row, "alpaca") for row in alpaca}
    overlap = openorca_prompts & alpaca_prompts
    print(
        f"OpenOrca prompts={len(openorca_prompts)} Alpaca prompts={len(alpaca_prompts)} "
        f"交集={len(overlap)}"
    )
    assert len(openorca_prompts) == 200
    assert len(alpaca_prompts) == 50
    assert overlap == set()


# 5) PUPA：字段名、非空、build_prompt 非空 --------------------------------- #
def test_05_pupa_20_fields_and_prompt():
    t0 = time.perf_counter()
    examples = D.load_pupa(max_examples=20)
    dt = time.perf_counter() - t0
    fields = list(examples[0].keys())
    print(f"PUPA: 条数={len(examples)} n_total={examples.meta['n_total']} "
          f"source={examples.meta['source']} 耗时={dt:.2f}s")
    print(f"PUPA 字段名 = {fields}")
    print(f"PUPA 选用文本字段 = {D.PUPA_TEXT_FIELD}")

    assert len(examples) == 20
    assert set(fields) == PUPA_FIELDS
    assert examples.meta["n_total"] >= 237
    for row in examples:
        assert row["conversation_hash"]
        assert row["user_query"] and row["user_query"].strip()
        prompt = D.build_prompt(row, "pupa")
        assert prompt and prompt.strip()

    # 首选字段即 target_response；缺失时回退 user_query
    assert D.build_prompt(examples[0], "pupa") == examples[0]["target_response"]
    assert (
        D.build_prompt(
            {"target_response": None, "user_query": "Q", "redacted_query": "R"}, "pupa"
        )
        == "Q"
    )
    print(f"PUPA prompt[0] 长度 = {len(D.build_prompt(examples[0], 'pupa'))}")


# 6) tokenize → decode 字符相似度 ------------------------------------------ #
def test_06_tokenize_decode_roundtrip(tokenizer):
    examples = D.load_alpaca(max_examples=20)
    prompts = [D.build_prompt(row, "alpaca") for row in examples]
    encoded = D.tokenize_examples(examples, tokenizer, max_len=512, dataset="alpaca")
    assert len(encoded["input_ids"]) == len(encoded["attention_mask"]) == 20
    assert all(
        len(ids) == len(mask)
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
    )
    assert all(len(ids) <= 512 for ids in encoded["input_ids"])
    assert all(set(mask) <= {0, 1} for mask in encoded["attention_mask"])

    ratios = []
    for text, ids in zip(prompts, encoded["input_ids"]):
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        ratios.append(SequenceMatcher(None, text, decoded).ratio())
    print(
        f"tokenize→decode 字符相似度: min={min(ratios):.4f} "
        f"mean={sum(ratios) / len(ratios):.4f} (20 条)"
    )
    assert min(ratios) > 0.9

    # 右截断：max_len=128 的结果必须等于完整 ids 的前 128 个
    longest = max(zip(prompts, encoded["input_ids"]), key=lambda item: len(item[1]))
    assert len(longest[1]) > 128
    truncated = D.tokenize_examples([longest[0]], tokenizer, max_len=128)
    assert truncated["input_ids"][0] == longest[1][:128]
    assert truncated["attention_mask"][0] == [1] * 128


# 7) make_batches：同 seed 一致、形状、右 padding、词表范围 ----------------- #
def test_07_make_batches(tokenizer):
    examples = D.load_alpaca(max_examples=100)
    kwargs = dict(
        batch_size=4, max_len=64, seed=0, dataset_name="alpaca", shuffle=True
    )
    batches_a = list(D.make_batches(examples, tokenizer, **kwargs))
    batches_b = list(D.make_batches(examples, tokenizer, **kwargs))
    assert len(batches_a) == len(batches_b) == 25

    first_a, first_b = batches_a[0], batches_b[0]
    assert torch.equal(first_a["input_ids"], first_b["input_ids"])
    assert torch.equal(first_a["attention_mask"], first_b["attention_mask"])
    assert first_a["texts"] == first_b["texts"]
    assert first_a["input_ids"].shape == first_a["attention_mask"].shape

    pad_id = int(tokenizer.pad_token_id)
    total_padded = 0
    for batch in batches_a:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        assert ids.dtype == torch.long and mask.dtype == torch.long
        assert ids.shape == mask.shape
        assert ids.shape[0] <= 4
        assert ids.shape[1] <= 64
        assert int(ids.max()) < len(tokenizer)
        assert set(mask.flatten().tolist()) <= {0, 1}
        pad_positions = mask == 0
        total_padded += int(pad_positions.sum())
        assert torch.all(ids[pad_positions] == pad_id)
        assert torch.all(ids[mask == 1] != pad_id)

    print(
        f"make_batches: batches={len(batches_a)} shape={tuple(first_a['input_ids'].shape)} "
        f"padding_token={pad_id} 总 padding 位置={total_padded} "
        f"vocab={len(tokenizer)}"
    )
    assert total_padded > 0
    assert tuple(first_a["input_ids"].shape) == (4, first_a["input_ids"].shape[1])
