"""M8b PII 指标与 PUPA 评测流程的验证（不下载大模型）。

覆盖任务单 M8b 的验收点：

1. ``pii_fraction`` 手算 + 大小写不敏感；
2. ``pii_mask`` 重叠 / 最长优先 / 不重复计数；
3. ``pii_ratio``：恒等 → 1.0，全脱敏 → 0.0，原文无 PII → 未定义；
4. ``pii_units_recovered``：按片段计。
5. 端到端小样本：V=40/d=8 随机表 + 5 条手工 PUPA 风格样例，走 NN 重建（``run_pupa_eval``，
   不加载 LM、不跑 BeamClean），断言 FR 在 [0, 100] 且结果可 JSON 序列化（无 NaN/Inf）。

另加（同一模块的契约检查，不依赖大模型）：

6. ``make_lm_score_fn`` 的**标准自回归**口径
   ``Σ_{i=2..t} log p(tokens[:, i] | tokens[:, :i])``（用随机初始化微型 Llama 与
   确定性 logits stub 各验一次，前者跑真实 HF 模型、后者与手算逐位一致）；
7. ``load_sgt_from_checkpoint`` 往返与结构反推；
8. ``run_pupa_eval`` 拒绝“未给 lm 却静默跳过 BeamClean”，且只用 ``user_query``；
9. PII-Ratio 的分母 ``T_ref`` 是 max_len 截断后的解码文本（不是完整 ``user_query``）：
   无保护基线且 ``t̃ = t`` 时 ``nn.fr == 0`` 且 ``nn.pii_ratio == 1.0``，同时保留
   完整原文与 ``pii_ratio_full``（旧整文口径）供审计。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sgt import pii  # noqa: E402
from sgt.model import SGTModel  # noqa: E402

import run_pupa  # noqa: E402


# --------------------------------------------------------------------------- #
# 1) pii_fraction：手算与大小写
# --------------------------------------------------------------------------- #
def test_01_pii_fraction_hand_computed():
    # 任务单给的是 "John lives in Paris"：长度 19 字符，PII 字符 4（John）+5（Paris）= 9
    # → 9/19 ≈ 0.4737。任务单同时写 9/20 = 0.45，那对应带句点的 20 字符版本，
    # 两种口径都在这里断言（交付报告里标注这处不一致）。
    text = "John lives in Paris"
    assert len(text) == 19
    assert pii.pii_fraction(text, ["John", "Paris"]) == pytest.approx(9 / 19)

    with_period = "John lives in Paris."
    assert len(with_period) == 20
    assert pii.pii_fraction(with_period, ["John", "Paris"]) == pytest.approx(9 / 20)
    assert pii.pii_fraction(with_period, ["John", "Paris"]) == pytest.approx(0.45)

    # 大小写不敏感（文本侧大写、片段侧小写，反之亦然）
    assert pii.pii_fraction("JOHN lives in paris", ["john", "PARIS"]) == pytest.approx(9 / 19)
    # 空文本 / 空片段表
    assert pii.pii_fraction("", ["John"]) == 0.0
    assert pii.pii_fraction(text, []) == 0.0
    # pii_units 是 PUPA 的 "||" 串时同样可用
    assert pii.pii_fraction(text, "John||Paris") == pytest.approx(9 / 19)


# --------------------------------------------------------------------------- #
# 2) pii_mask：重叠 / 最长优先 / 不重复计数
# --------------------------------------------------------------------------- #
def test_02_pii_mask_longest_first_no_double_count():
    # 重叠片段："York" 已被 "New York" 覆盖，不能再计一次
    mask = pii.pii_mask("New York", ["New York", "York"])
    assert mask == [True] * 8
    assert sum(mask) == 8
    # 与片段顺序无关
    assert pii.pii_mask("New York", ["York", "New York"]) == mask
    # 独立出现也要标记（同一片段多次出现）
    mask2 = pii.pii_mask("York is in New York", ["New York", "York"])
    assert len(mask2) == 19
    assert mask2[:4] == [True] * 4
    assert not any(mask2[4:11])  # " is in "
    assert all(mask2[11:])  # "New York"
    assert sum(mask2) == 12
    # 最长优先的判别性用例：左到右取最长，"abab" 而不是 "aba"
    assert pii.pii_mask("abab", ["aba", "abab"]) == [True] * 4
    # 大小写不敏感
    assert pii.pii_mask("new york", ["New York"]) == [True] * 8
    # 片段表规范化：trim / 去重 / 丢弃空串
    assert pii.split_pii_units(" a || A ||b|| ") == ["a", "b"]
    assert pii.pii_mask("hello", []) == [False] * 5


# --------------------------------------------------------------------------- #
# 3) pii_ratio：1.0 / 0.0 / 未定义
# --------------------------------------------------------------------------- #
def test_03_pii_ratio_identity_redacted_undefined():
    text = "John lives in Paris"
    units = ["John", "Paris"]

    same = pii.pii_ratio(text, text, units)
    assert same["defined"] is True
    assert same["pii_orig"] == pytest.approx(9 / 19)
    assert same["pii_rec"] == pytest.approx(9 / 19)
    assert same["ratio"] == pytest.approx(1.0)
    assert same["ratio"] == 1.0  # 同一占比相除，浮点上也必须精确为 1.0

    redacted = pii.pii_ratio(text, "[REDACTED] lives in [REDACTED]", units)
    assert redacted["defined"] is True
    assert redacted["pii_rec"] == 0.0
    assert redacted["ratio"] == 0.0

    # 只删掉一个 PII：占比之比（不是字符数之比）
    partial = pii.pii_ratio(text, "[REDACTED] lives in Paris", units)
    # pii_rec = 5/25 = 1/5，pii_orig = 9/19 → ratio = 19/45
    assert partial["pii_rec"] == pytest.approx(1 / 5)
    assert partial["ratio"] == pytest.approx(19 / 45)

    # 原文无 PII → 分母为 0 → 未定义（按任务单约定 pii_orig/pii_rec 均报 0.0）
    undefined = pii.pii_ratio("No names here", "No names here", units)
    assert undefined == {
        "ratio": None,
        "pii_orig": 0.0,
        "pii_rec": 0.0,
        "defined": False,
    }


# --------------------------------------------------------------------------- #
# 4) pii_units_recovered：按片段计
# --------------------------------------------------------------------------- #
def test_04_pii_units_recovered_counts_fragments():
    original = "Alice met Bob in Paris."
    reconstructed = "[REDACTED] met Bob in [REDACTED]."
    units = ["Alice", "Bob", "Paris"]

    result = pii.pii_units_recovered(original, reconstructed, units)
    assert result["n_units"] == 3
    assert result["n_recovered"] == 1
    assert result["recovered_frac"] == pytest.approx(1 / 3)

    # 不在原文里的片段不进入分母
    assert pii.pii_units_recovered(original, reconstructed, units + ["Zoe"])["n_units"] == 3
    # 大小写不敏感
    assert pii.pii_units_recovered(original, "ALICE met Bob in Paris.", units)["n_recovered"] == 3
    # 没有片段
    empty = pii.pii_units_recovered(original, reconstructed, [])
    assert empty == {"n_units": 0, "n_recovered": 0, "recovered_frac": 0.0}


# --------------------------------------------------------------------------- #
# 5) 端到端小样本（NN 路径；不加载 LM、不跑 BeamClean）
# --------------------------------------------------------------------------- #
class _CharTokenizer:
    """字符级假 tokenizer：只为驱动 ``sgt.data.make_batches`` 与 ``decode``，不下载模型。

    词表 = 2 个特殊 token（PAD=0、BOS=1）+ 测试样例里出现的 38 个字符，因此
    tokenize→decode 是**无损往返**：干净 embedding 的 NN 重建应当逐 token 复原，
    重建文本与原文本一致 → PII-Ratio 精确为 1.0。这比“只要能跑通”更强的集成检查。
    """

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = 40, alphabet: str | None = None) -> None:
        self.vocab_size = int(vocab_size)
        if alphabet is None:
            alphabet = "".join(sorted(_ALPHABET))
        if len(alphabet) + 2 > self.vocab_size:
            raise ValueError(
                f"字母表 {len(alphabet)} 个字符放不进 vocab_size={self.vocab_size}"
            )
        self.alphabet = alphabet
        self._to_id = {char: 2 + index for index, char in enumerate(alphabet)}
        self._to_char = {2 + index: char for index, char in enumerate(alphabet)}

    def __call__(self, texts, add_special_tokens=True, padding=False, truncation=False):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        for text in texts:
            ids = [int(self._to_id[char]) for char in text]
            if add_special_tokens:
                ids = [self.eos_token_id] + ids
            input_ids.append(ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }

    def decode(self, ids, skip_special_tokens=True) -> str:
        id_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        chars = []
        for token_id in id_list:
            token_id = int(token_id)
            if skip_special_tokens and token_id in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(self._to_char.get(token_id, ""))
        return "".join(chars)


# 5 条 PUPA 风格样例（字段同 PUPA：conversation_hash / predicted_category /
# user_query / target_response / pii_units / redacted_query）
_PUPA_STYLE_EXAMPLES = [
    {
        "conversation_hash": "h1",
        "predicted_category": "job",
        "user_query": "My name is John Smith and I live in Paris.",
        "target_response": "John Smith lives in Paris and works at ACME.",
        "pii_units": "John Smith||Paris||ACME",
        "redacted_query": "My name is [REDACTED] and I live in [REDACTED].",
    },
    {
        "conversation_hash": "h2",
        "predicted_category": "visa",
        "user_query": "Can Alice Chen apply for a visa in Tokyo.",
        "target_response": "Alice Chen can apply in Tokyo.",
        "pii_units": "alice chen||tokyo",
        "redacted_query": "Can [REDACTED] apply for a visa in [REDACTED].",
    },
    {
        # user_query 没有 PII；target_response 里有（若误用 target_response，pii_orig > 0）
        "conversation_hash": "h3",
        "predicted_category": "other",
        "user_query": "Please summarize the attached report.",
        "target_response": "Zoe lives in Berlin and wrote the report.",
        "pii_units": "zoe||berlin",
        "redacted_query": "Please summarize the attached report.",
    },
    {
        "conversation_hash": "h4",
        "predicted_category": "job",
        "user_query": "Email Mary Jones about the HR Technology offer.",
        "target_response": "Mary Jones will receive the offer.",
        "pii_units": "mary jones||hr technology",
        "redacted_query": "Email [REDACTED] about the [REDACTED] offer.",
    },
    {
        "conversation_hash": "h5",
        "predicted_category": "medical",
        "user_query": "The patient Robert Brown is staying in London.",
        "target_response": "Robert Brown is in London.",
        "pii_units": "",
        "redacted_query": "The patient Robert Brown is staying in London.",
    },
]


#: 样例里出现的全部字符（应为 38 个，正好占满 vocab_size=40 的 2..39 号位）。
_ALPHABET = {char for example in _PUPA_STYLE_EXAMPLES for char in example["user_query"]}


@pytest.fixture(scope="module")
def small_table() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20251105)
    return torch.randn(40, 8, generator=generator)


@pytest.fixture(scope="module")
def char_tokenizer() -> _CharTokenizer:
    return _CharTokenizer(vocab_size=40)


def test_05_run_pupa_eval_nn_only_small_sample(small_table, char_tokenizer):
    examples = _PUPA_STYLE_EXAMPLES + [
        {  # 空 user_query：跳过并显式计数（不得静默丢样本）
            "conversation_hash": "h6",
            "predicted_category": "other",
            "user_query": "   ",
            "target_response": "whatever",
            "pii_units": "whatever",
            "redacted_query": "",
        }
    ]

    payload = run_pupa.run_pupa_eval(
        examples,
        tokenizer=char_tokenizer,
        table=small_table,
        lm=None,
        skip_beamclean=True,
        max_len=64,  # 不截断，保证 tokenize→decode 无损往返（见 _CharTokenizer）
        batch_size=2,
        seed=0,
        progress=False,
    )

    assert set(payload) == {"nn", "beamclean", "rows", "meta"}

    # BeamClean 必须显式标记 skipped，且不给数值
    assert payload["beamclean"]["status"] == "skipped"
    assert payload["beamclean"]["fr"] is None
    assert payload["beamclean"]["pii_ratio_mean"] is None
    assert payload["beamclean"]["scripts"] == []
    assert payload["meta"]["beamclean_skipped"] is True

    # NN FR 在 [0, 100]；干净 embedding（无 SGT）的 NN 重建按构造应当逐 token 精确复原
    fr = payload["nn"]["fr"]
    assert isinstance(fr, float) and 0.0 <= fr <= 100.0
    assert fr == 0.0
    assert payload["nn"]["n_tokens"] == sum(row["n_tokens"] for row in payload["rows"])
    assert payload["nn"]["n_errors"] == sum(row["nn"]["n_errors"] for row in payload["rows"])

    # 样本计数：6 条输入，1 条空文本被跳过（显式记录）
    assert payload["meta"]["n_input_examples"] == 6
    assert payload["meta"]["n_used_examples"] == 5
    assert payload["meta"]["n_skipped_empty_text"] == 1
    assert len(payload["rows"]) == 5
    assert len(payload["nn"]["scripts"]) == 5
    assert payload["meta"]["text_field"] == "user_query"

    # 被保护文本必须是 user_query（不是 target_response）
    for row, example in zip(payload["rows"], _PUPA_STYLE_EXAMPLES):
        assert row["user_query"] == example["user_query"]
        assert row["conversation_hash"] == example["conversation_hash"]
        # 未截断（max_len=64 > 所有样本长度）时 T_ref 与完整原文一致
        assert row["pii_reference_text"] == example["user_query"]
        assert payload["meta"]["pii_reference"] == "truncated_decoded"

    # 第 3 条样例的 PII 只在 target_response 里 → user_query 口径下 PII-Ratio 未定义
    third = payload["rows"][2]["nn"]
    assert third["pii_orig"] == 0.0
    assert third["pii_ratio"] is None
    assert third["pii_ratio_defined"] is False
    assert third["pii_ratio_full"] is None
    assert third["pii_ratio_full_defined"] is False

    # 有 PII 的样本：重建文本 == 原文（FR=0）→ PII-Ratio 精确为 1.0、片段全恢复
    defined = [
        row["nn"]["pii_ratio"]
        for row in payload["rows"]
        if row["nn"]["pii_ratio_defined"]
    ]
    assert len(defined) == 3, "h1/h2/h4 的 user_query 含登记的 PII 片段"
    assert defined == [1.0, 1.0, 1.0]
    assert payload["nn"]["pii_ratio_mean"] == 1.0
    assert payload["nn"]["pii_units_recovered_mean"] == 1.0
    # 未截断时整文口径（pii_ratio_full）与主口径相同
    assert payload["nn"]["pii_ratio_full_mean"] == 1.0
    assert payload["nn"]["n_pii_ratio_full_defined"] == 3
    for row in payload["rows"]:
        assert row["nn"]["pii_ratio_full"] == row["nn"]["pii_ratio"]
        assert row["nn"]["pii_ratio_full_defined"] == row["nn"]["pii_ratio_defined"]
        if row["nn"]["n_units"] > 0:
            assert row["nn"]["n_recovered"] == row["nn"]["n_units"]
            assert row["nn"]["recovered_frac"] == 1.0
            assert row["nn"]["reconstructed"] == row["user_query"]

    # 全部浮点有限、可 JSON 序列化（allow_nan=False 遇 NaN/Inf 会直接抛错）
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in serialized and "Infinity" not in serialized
    json.loads(serialized)


def test_06_lm_score_fn_matches_manual_forward():
    """真实 HF 模型上的 shift 校验：``logits[:, i-1]`` 预测 ``tokens[:, i]``。"""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    lm = LlamaForCausalLM(config).eval()
    score_fn = run_pupa.make_lm_score_fn(lm, chunk=2)

    ids = torch.randint(0, 64, (5, 6), generator=torch.Generator().manual_seed(1))
    scores = score_fn(ids)
    assert scores.shape == (5,)
    assert scores.dtype == torch.float64
    assert torch.isfinite(scores).all()

    with torch.no_grad():
        logits = lm(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
        manual = (
            log_probs.gather(-1, ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
            .to(torch.float64)
            .sum(dim=1)
        )
    torch.testing.assert_close(scores, manual, atol=1e-4, rtol=1e-5)

    # 空输入不炸
    assert score_fn(torch.zeros((0, 4), dtype=torch.long)).shape == (0,)


class _StubLogits:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FixedLogitsLM(torch.nn.Module):
    """确定性 LM stub：logits 只由位置决定（与输入 token 无关），便于逐位手算。

    ``forward(input_ids[N, t])`` 返回 ``.logits [N, t, V]``，每行都等于
    ``fixed_logits[:t]``；``make_lm_score_fn`` 只需要 ``parameters()`` /
    ``get_input_embeddings()`` / ``forward().logits``。
    """

    def __init__(self, vocab: int, max_len: int, *, seed: int = 7) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        fixed = torch.randn(max_len, vocab, generator=generator)
        self.register_buffer("fixed_logits", fixed)
        self.embedding = torch.nn.Embedding(vocab, 4)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(self, input_ids, attention_mask=None, **kwargs):
        n_rows, seq_len = input_ids.shape
        logits = self.fixed_logits[:seq_len].unsqueeze(0).expand(
            n_rows, seq_len, self.fixed_logits.shape[1]
        )
        return _StubLogits(logits)


def test_lm_score_fn_shifted():
    """实现与手算 ``Σ_{i=2..t} log_softmax(logits[:, i-1])[tokens[:, i]]`` 一致。"""
    vocab, seq_len = 7, 5
    lm = _FixedLogitsLM(vocab, max_len=16, seed=11)
    score_fn = run_pupa.make_lm_score_fn(lm, chunk=0)

    ids = torch.randint(
        0, vocab, (4, seq_len), generator=torch.Generator().manual_seed(3)
    )
    scores = score_fn(ids)
    assert scores.shape == (4,)
    assert scores.dtype == torch.float64
    assert torch.isfinite(scores).all()

    with torch.no_grad():
        logits = lm(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        step_log_probs = torch.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
        manual = (
            step_log_probs.gather(-1, ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
            .to(torch.float64)
            .sum(dim=1)
        )
        # 旧口径（同位置 gather，包含“看到自己”的项）用于确认 shift 真的生效
        naive = (
            torch.log_softmax(logits.to(torch.float32), dim=-1)
            .gather(-1, ids.unsqueeze(-1))
            .squeeze(-1)
            .to(torch.float64)
            .sum(dim=1)
        )

    # 手算与实现用同一批 logits：float64 求和，逐位一致（容差 1e-9 远大于实际误差 0）
    torch.testing.assert_close(scores, manual, atol=1e-9, rtol=0.0)
    assert not torch.allclose(scores, naive, atol=1e-6), "分数仍是同位置 gather（未 shift）"


def test_lm_score_fn_single_token_is_zero():
    """``t == 1`` 时没有可条件化的 next-token，返回 0。"""
    lm = _FixedLogitsLM(vocab=5, max_len=8, seed=2)
    score_fn = run_pupa.make_lm_score_fn(lm)

    single = torch.tensor([[1], [4], [0]], dtype=torch.long)
    scores = score_fn(single)
    assert scores.shape == (3,)
    assert scores.dtype == torch.float64
    assert torch.equal(scores, torch.zeros(3, dtype=torch.float64))

    # t=0（空前缀）与空批次同样返回 0 / 空张量
    assert torch.equal(
        score_fn(torch.zeros((2, 0), dtype=torch.long)),
        torch.zeros(2, dtype=torch.float64),
    )
    assert score_fn(torch.zeros((0, 3), dtype=torch.long)).shape == (0,)


def test_pii_ratio_uses_truncated_reference(small_table, char_tokenizer):
    """PII-Ratio 分母 = 模型实际看到的截断解码文本 ``T_ref``，不是完整 ``user_query``。

    用 ``h1``（含 "John Smith" / "Paris"）：``max_len=22`` → 有效 ids 解码出
    ``"My name is John Smith"``（截断）。无保护基线（``x̃ = x``）下 NN 逐 token
    精确复原 → ``fr == 0``、重建文本 == ``T_ref``，因此主口径 PII-Ratio 必须正好
    是 1.0；旧口径（分母为完整 42 字符原文）应约为 4/3，说明两个字段都在。
    """
    example = _PUPA_STYLE_EXAMPLES[0]
    full_text = example["user_query"]
    assert len(full_text) == 42
    max_len = 22  # eos + 21 字符 = "My name is John Smith"
    reference_text = full_text[: max_len - 1]
    assert reference_text == "My name is John Smith"
    units = pii.split_pii_units(example["pii_units"])
    assert "John Smith" in units

    payload = run_pupa.run_pupa_eval(
        [example],
        tokenizer=char_tokenizer,
        table=small_table,
        lm=None,
        skip_beamclean=True,
        max_len=max_len,
        batch_size=1,
        seed=0,
        progress=False,
    )
    assert payload["meta"]["pii_reference"] == "truncated_decoded"

    row = payload["rows"][0]
    nn = row["nn"]
    assert row["n_tokens"] == max_len
    assert row["user_query"] == full_text  # 完整原文仍保留在输出里
    assert row["pii_reference_text"] == reference_text

    # 无保护基线逐 token 精确复原
    assert nn["fr"] == 0.0
    assert nn["n_errors"] == 0
    assert nn["reconstructed"] == reference_text

    # 主口径：分母 T_ref == 重建文本 → 恒等文本，PII-Ratio 精确 = 1.0
    assert nn["pii_ratio_defined"] is True
    assert nn["pii_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert nn["pii_orig"] == pytest.approx(10 / 21)  # "John Smith" / 21 字符
    assert payload["nn"]["pii_ratio_mean"] == pytest.approx(1.0, abs=1e-6)

    # 旧口径（分母完整原文）仍然记录，且与 sgt.pii 的整文比值一致
    expected_full = pii.pii_ratio(full_text, reference_text, units)
    assert expected_full["defined"] is True
    assert nn["pii_ratio_full_defined"] is True
    assert nn["pii_ratio_full"] == pytest.approx(expected_full["ratio"], abs=1e-12)
    assert nn["pii_ratio_full"] == pytest.approx(4 / 3)
    assert nn["pii_ratio_full"] != pytest.approx(1.0, abs=1e-3)
    assert payload["nn"]["pii_ratio_full_mean"] == pytest.approx(
        expected_full["ratio"], abs=1e-12
    )
    assert payload["nn"]["n_pii_ratio_full_defined"] == 1
    assert payload["nn"]["scripts"][0]["pii_ratio_full"] == pytest.approx(
        expected_full["ratio"], abs=1e-12
    )

    # 输出仍可 JSON 序列化且无 NaN/Inf
    json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def test_07_sgt_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    sgt = SGTModel(8, hidden=16, n_layers=1, n_heads=2, ffn_mult=2, max_len=16)
    path = tmp_path / "ckpt_demo.pt"
    torch.save({"sgt": sgt.state_dict(), "step": 3}, path)

    loaded, meta = run_pupa.load_sgt_from_checkpoint(
        path, device="cpu", dtype=torch.float32, n_heads=2
    )
    assert meta["d"] == 8
    assert meta["hidden"] == 16
    assert meta["n_layers"] == 1
    assert meta["ffn_mult"] == 2
    assert meta["max_len"] == 16
    assert meta["n_heads"] == 2
    assert meta["n_heads_source"] == "cli"
    assert meta["step"] == 3

    for (name_a, value_a), (name_b, value_b) in zip(
        sgt.state_dict().items(), loaded.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(value_a, value_b)

    # d 不一致直接报错（不静默加载）
    with pytest.raises(ValueError, match="hidden_size"):
        run_pupa.load_sgt_from_checkpoint(
            path, device="cpu", dtype=torch.float32, d=16, n_heads=2
        )


def test_08_run_pupa_eval_rejects_silent_beamclean_skip(small_table, char_tokenizer):
    with pytest.raises(ValueError, match="BeamClean"):
        run_pupa.run_pupa_eval(
            _PUPA_STYLE_EXAMPLES,
            tokenizer=char_tokenizer,
            table=small_table,
            lm=None,  # 没有 lm
            skip_beamclean=False,  # 却要求跑 BeamClean → 必须报错，不能静默跳过
            max_len=24,
            progress=False,
        )

    # 缺 user_query 时也必须报错（防止退化成 target_response）
    with pytest.raises(ValueError, match="user_query"):
        run_pupa.run_pupa_eval(
            [{"conversation_hash": "h", "target_response": "Zoe lives in Berlin."}],
            tokenizer=char_tokenizer,
            table=small_table,
            lm=None,
            skip_beamclean=True,
            max_len=24,
            progress=False,
        )
