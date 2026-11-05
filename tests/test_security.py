"""S1 安全评测（``experiments/run_attack_suite.py``）的验证（不加载 1B 模型）。

覆盖任务单 6 条：

1. fresh noise：同一 wrapper 同一输入两次调用输出不同；两个同 seed wrapper 走相同
   调用序列输出逐位一致。
2. ``--no-obfuscation`` 路径：恒等重构 ⇒ NN-FR = 0、TTR-1 = 0、Hist. Entropy = 0。
3. 常数噪声 σ = 0 与 ``no-obfuscation`` 结果逐键一致。
4. ``fr_by_r``：r = 1 与 ``nn_fr`` 相等；r = |V| 与“最远邻”攻击相等。
5. BeamClean σ → 0 退化为 NN（合成表 + 常量 ``lm_score_fn``）。
6. PUPA 分支：PII-Ratio 用**截断后**参考文本（与 ``run_pupa`` 口径一致），
   恒等重建 = 1.0。

全部使用小合成表 / 小 SGT，不联网、不读 data_cache、不 import transformers 模型。
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_attack_suite as S  # noqa: E402
from sgt import metrics  # noqa: E402
from sgt.attacks import beamclean  # noqa: E402
from sgt.model import SGTModel  # noqa: E402
from sgt.utility import SGTEmbedding  # noqa: E402

D = 8
VOCAB = 64
N_TOKENS = 24
SEED = 20251105


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #
def _table(seed: int = SEED, *, vocab: int = VOCAB, d: int = D) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, d, generator=generator, dtype=torch.float64)


def _targets(seed: int = SEED, *, vocab: int = VOCAB, n: int = N_TOKENS) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randint(0, vocab, (n,), generator=generator)


def _noisy(x: torch.Tensor, sigma: float, *, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return x + sigma * noise


def _trained_like_sgt(*, seed: int = 0) -> SGTModel:
    """非零输出头的 SGT（sigma 固定 0.5，mu 随机），保证 SGT 真起作用。"""
    torch.manual_seed(seed)
    model = SGTModel(D, hidden=16, n_layers=1, n_heads=2, max_len=16)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.head.weight.copy_(0.1 * torch.randn(model.head.weight.shape, generator=generator))
        model.head.bias[:D].copy_(0.05 * torch.randn(D, generator=generator))
        model.head.bias[D:].fill_(math.log(0.5**2))
    return model.eval()


def _base_embedding(*, seed: int = 0) -> nn.Embedding:
    embedding = nn.Embedding(VOCAB, D)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        embedding.weight.copy_(torch.randn(VOCAB, D, generator=generator))
    return embedding


# --------------------------------------------------------------------------- #
# 1. fresh noise
# --------------------------------------------------------------------------- #
def test_01_fresh_noise_same_wrapper_twice_differs() -> None:
    embedding = SGTEmbedding(_base_embedding(), _trained_like_sgt(), seed=0)
    ids = torch.randint(0, VOCAB, (2, 5), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        first = embedding(ids)
        second = embedding(ids)

    assert not torch.equal(first, second), "同一输入两次调用输出相同（generator 被重置）"


def test_01b_same_seed_wrappers_follow_identical_sequences() -> None:
    base = _base_embedding()
    sgt = _trained_like_sgt()
    wrapper_a = SGTEmbedding(copy.deepcopy(base), sgt, seed=13)
    wrapper_b = SGTEmbedding(copy.deepcopy(base), sgt, seed=13)
    first_ids = torch.randint(0, VOCAB, (2, 5), generator=torch.Generator().manual_seed(2))
    second_ids = torch.randint(0, VOCAB, (1, 4), generator=torch.Generator().manual_seed(3))

    with torch.no_grad():
        for step, current in enumerate((first_ids, second_ids, first_ids)):
            out_a = wrapper_a(current)
            out_b = wrapper_b(current)
            assert torch.equal(out_a, out_b), f"第 {step} 次调用输出不一致"


# --------------------------------------------------------------------------- #
# 2. no-obfuscation：恒等重构
# --------------------------------------------------------------------------- #
def test_02_no_obfuscation_identity_reconstruction() -> None:
    table = _table()
    target = _targets()
    x_tilde = table[target]  # x̃ = x

    result = S.evaluate_attacks(x_tilde, target, table)

    assert result["n_tokens"] == N_TOKENS
    assert result["nn_fr"] == 0.0, "恒等变换下 NN-FR 必须为 0"
    assert result["ttr_1"] == 0.0, "恒等变换下 TTR-1 必须为 0"
    assert result["hist_entropy"] == 0.0, "秩全为 1 ⇒ 直方图单点 ⇒ 熵 0"
    assert result["mrp_r_star"] == 1
    assert result["mrp_fr"] == 0.0
    # 无噪声时 Eq (8) 的 1/σ² 发散：MI 未定义，PAC 界极限 δ=1。
    assert result["mi_feature_mean"] is None
    assert result["pac_adv"] == 100.0
    assert result["mi_status"] == "undefined_sigma_zero"


# --------------------------------------------------------------------------- #
# 3. 常数噪声 σ = 0 与 no-obfuscation 一致
# --------------------------------------------------------------------------- #
def test_03_constant_sigma_zero_matches_no_obfuscation() -> None:
    table = _table()
    target = _targets()
    x = table[target]
    x_zero = _noisy(x, 0.0)  # 与主流程 constant 模式同构造：x + 0·u
    assert torch.equal(x_zero, x), "σ=0 的噪声张量必须逐位等于干净 x"

    baseline = S.evaluate_attacks(x, target, table)
    zero_sigma = S.evaluate_attacks(x_zero, target, table)
    assert zero_sigma == baseline, "σ=0 与 no-obfuscation 的指标必须逐键一致"


# --------------------------------------------------------------------------- #
# 4. fr_by_r：r = 1 与 NN；r = |V| 与最远邻
# --------------------------------------------------------------------------- #
def test_04_fr_by_r_r1_equals_nn_and_rv_equals_farthest() -> None:
    table = _table()
    target = _targets()
    x_tilde = _noisy(table[target], 0.6)
    result = S.evaluate_attacks(x_tilde, target, table, return_predictions=True)
    predictions = result["predictions"]

    # r = 1 就是 NN：预测与 FR 都必须相等。
    nn_reference = metrics.nn_reconstruct(x_tilde, table)
    assert torch.equal(predictions["nn"], nn_reference)
    assert result["fr_by_r"]["1"] == result["nn_fr"]
    assert result["nn_fr_at_r1"] == result["nn_fr"]

    # r = |V| 就是“最远邻”攻击：与 metrics 的第 V 近邻、以及显式 argmax 距离一致。
    farthest_reference = metrics.mrp_reconstruct(x_tilde, table, VOCAB)
    assert torch.equal(predictions["fr_by_r"][str(VOCAB)], farthest_reference)
    direct_farthest = torch.cdist(x_tilde, table).argmax(dim=1)
    assert torch.equal(farthest_reference, direct_farthest), "合成数据不应出现距离并列"
    assert result["fr_by_r"][str(VOCAB)] == metrics.failure_rate(direct_farthest, target)

    # MRP 的 r* 预测也必须是第 r* 近邻。
    r_star = result["mrp_r_star"]
    mrp_reference = metrics.mrp_reconstruct(x_tilde, table, r_star)
    assert torch.equal(predictions["mrp"], mrp_reference)
    assert result["mrp_fr"] == metrics.failure_rate(mrp_reference, target)


# --------------------------------------------------------------------------- #
# 5. BeamClean σ → 0 退化为 NN
# --------------------------------------------------------------------------- #
def test_05_beamclean_sigma_to_zero_equals_nn() -> None:
    generator = torch.Generator().manual_seed(SEED)
    vocab, d, length = 48, 8, 6
    table = torch.randn(vocab, d, generator=generator, dtype=torch.float64)
    target = torch.randint(0, vocab, (1, length), generator=generator)
    x_tilde = table[target] + 0.05 * torch.randn(
        1, length, d, generator=generator, dtype=torch.float64
    )
    nn_reference = metrics.nn_reconstruct(x_tilde.reshape(-1, d), table).reshape(1, length)

    def constant_lm_score_fn(prefix_ids) -> torch.Tensor:
        return torch.zeros(int(prefix_ids.shape[0]), dtype=torch.float64)

    for sigma in (0.0, 1e-12):
        prediction, sigma_used = beamclean(
            x_tilde, table, constant_lm_score_fn, sigma=sigma, beam=2, topk=4
        )
        assert torch.equal(prediction, nn_reference), f"σ={sigma} 时 BeamClean 未退化为 NN"
        assert sigma_used == sigma

    # 数值意义上的极小 σ（非显式 0）也必须保持 NN（后验被 top-1 候选支配）。
    prediction_small, _ = beamclean(
        x_tilde, table, constant_lm_score_fn, sigma=1e-6, beam=2, topk=4
    )
    assert torch.equal(prediction_small, nn_reference)


# --------------------------------------------------------------------------- #
# 6. PUPA：PII-Ratio 的分母是截断参考文本
# --------------------------------------------------------------------------- #
class _WordTokenizer:
    """按 id 拼词的最小 tokenizer（只实现 ``decode``）。"""

    def __init__(self) -> None:
        self.words = ["john", " smith", " lives", " at", " home"]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self.words[int(token)] for token in ids)


def test_06_pupa_pii_ratio_uses_truncated_reference() -> None:
    tokenizer = _WordTokenizer()
    # 完整 user_query 含 "home"，但截断后的参考文本只到 "john smith"。
    examples = [
        {
            "user_query": "john smith lives at home",
            "pii_units": "john||smith||home",
        }
    ]
    sequences = [{"example_index": 0, "ids": [0, 1]}]
    reconstructed = [[0, 1]]  # 恒等重建

    block = S.evaluate_pii(sequences, reconstructed, tokenizer, examples)

    assert block["status"] == "ok"
    assert block["pii_reference"] == "truncated_decoded"
    assert block["pii_ratio"] == 1.0, "恒等重建且分母为截断文本 ⇒ PII-Ratio = 1.0"
    assert block["pii_units_recovered"] == 1.0
    assert block["n_pii_ratio_defined"] == 1
    assert block["n_samples_with_pii_units"] == 1

    # 对照：若误用完整 user_query 做分母，比值 ≠ 1（长度/占比不同；这里 > 1）。
    full_text_ratio = S.pii_ratio(
        examples[0]["user_query"], "john smith", ["john", "smith", "home"]
    )
    assert full_text_ratio["ratio"] != 1.0, "测试前提：完整原文口径下比值应 ≠ 1"


# --------------------------------------------------------------------------- #
# 附加：威胁模型块与默认 r 集合
# --------------------------------------------------------------------------- #
def test_07_threat_model_block_and_default_rs() -> None:
    assert S.THREAT_MODEL["attacker_knows"] == ["tokenizer", "embedding_table"]
    assert S.THREAT_MODEL["sgt_weights_hidden"] is True
    assert S.THREAT_MODEL["greybox_rank_histogram"] is True
    assert S.THREAT_MODEL["fresh_noise"] is True
    assert S.THREAT_MODEL["whitebox_weights_tested"] is False

    rs = S.default_fr_rs(128256)
    assert 1 in rs and 10 in rs and 100 in rs and 1000 in rs
    assert 128256 // 2 in rs and 128256 - 1 in rs and 128256 in rs
