#!/usr/bin/env python3
"""S1 安全评测套件：不可信服务端从 SGT 混淆 embedding 重建原始 token（论文 §6 / §7）。

用法::

    ./.venv/bin/python experiments/run_attack_suite.py --model .models/llama-3.2-1b-instruct \
        --dataset alpaca --max-examples 200 --max-len 96 --eval-tokens 600 \
        [--sgt-ckpt results/ckpt_abscos.pt | --constant-sigma 0.4 | --no-obfuscation] \
        [--beamclean-n 10 --beamclean-max-len 24 --beam 4 --topk 8] \
        --device mps --out results/attack_abscos.json

威胁模型（论文安全模型；原样写入结果 JSON 的 ``threat_model`` 块）
--------------------------------------------------------------------
- **对手位置**：不可信的**服务端**。它能看到客户端发来的混淆 embedding ``x̃``，以及
  公开的 tokenizer 与 embedding 表（论文原话：除非 tokenizer / embedding 表保密，
  知道干净 token embedding 等价于知道文本）。
- **对手不知道**客户端的 SGT 权重（权重在客户端侧、私有）——这是**灰盒**而非白盒。
  MRP 攻击额外允许对手拥有秩直方图的先验（“additional greybox information ...
  some knowledge of the rank histogram or ability to run an SGT on a small dataset”）：
  本套件在本评测集的秩直方图上取众数 ``r*`` 作为该先验。
- **攻击目标**：从 ``x̃`` 恢复**原始 token**（token 级重建），指标为 FR（失败率，%）。
- **保密性保证**：PAC-Privacy 界 ``Δ_KL(δ) ≤ MI(X; X̃)``，``δ0 = 1/|V|``；本仓库用
  A7 口径的 Eq (8) 逐特征估计量（``run_ablation.feature_mi``）算 ``MI``，再
  ``metrics.pac_adv`` 反解 ``δ``。注意 F30：连续特征上这个估计量不是严格意义上的
  ``MI(X;X̃)``，只作为同口径可比指标。
- **未测试**：知道 SGT 权重的白盒对手（论文模型也不假设），见 JSON
  ``threat_model.whitebox_weights_tested=false``。

攻击与指标（全部只使用 ``table`` 与 ``x̃``；真 token id 只用于算 FR）
--------------------------------------------------------------------
- **NN**：最近邻 token -> ``nn_fr``；``ttr_1`` 与 ``nn_fr`` 定义上相等（秩 1 ↔ 最近邻正确）。
- **MRP**：``r* = argmax_r P̂(R=r)``（本评测集秩直方图众数）-> 第 ``r*`` 近邻 ->
  ``mrp_fr``；另报 ``nn_fr_at_r1`` 与 ``fr_by_r``（r ∈ {1, 10, 100, 1000, |V|/2,
  |V|−1, |V|}，用于复现“远端攻击更强”的现象）。
- ``ttr_10/50``、``symttr_10/100``、``hist_entropy``、``mi_feature_mean``、``pac_adv``。
- **BeamClean**：前 ``--beamclean-n`` 条序列（截断到 ``--beamclean-max-len``）逐条调用
  :func:`sgt.attacks.beamclean`，``lm_score_fn`` 复用
  ``run_pupa.make_lm_score_fn``（标准自回归 shift 口径）。``--beamclean-n 0`` 时
  ``status="skipped"``（显式，不静默跳过）。
- ``--dataset pupa`` 时额外报 ``pii_ratio``（NN 重建文本 vs **截断后**参考文本，
  ``pii_units``）与 ``pii_units_recovered``（与 ``run_pupa`` 同口径）。

随机性：所有采样由 ``--seed`` 驱动的显式 ``torch.Generator`` 完成；常数噪声一次采样、
SGT 噪声一个 generator 贯穿所有序列。``fresh_noise=True``（论文语义）：同一输入两次
采样应不同，不同输入的噪声相互独立——本套件不在评测中重置 generator 种子。

默认值与 ``configs/security.yaml`` 一致（该 YAML 记录默认参数与降规模声明）。

退出码：0 成功；2 配置/数据/模型/checkpoint 等前置条件不可用；1 评测本身失败。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sgt import metrics  # noqa: E402
from sgt.attacks import beamclean  # noqa: E402
from sgt.data import build_prompt, load_alpaca, load_pupa, tokenize_examples  # noqa: E402
from sgt.pii import pii_ratio, pii_units_recovered, split_pii_units  # noqa: E402

from run_ablation import (  # noqa: E402
    feature_mi,
    json_safe,
    load_lm,
    sgt_forward_sequences,
    write_json,
)
from run_pupa import load_sgt_from_checkpoint, make_lm_score_fn  # noqa: E402

__all__ = [
    "THREAT_MODEL",
    "evaluate_attacks",
    "evaluate_pii",
    "collect_sequences",
    "dataset_texts",
    "load_examples",
    "load_attack_sgt",
    "make_attack_lm_score_fn",
    "run_beamclean",
    "default_fr_rs",
    "parse_args",
    "main",
]

DEFAULT_MODEL = ".models/llama-3.2-1b-instruct"
DEFAULT_MAX_EXAMPLES = 200
DEFAULT_MAX_LEN = 96
DEFAULT_EVAL_TOKENS = 600
DEFAULT_BEAMCLEAN_N = 10
DEFAULT_BEAMCLEAN_MAX_LEN = 24
DEFAULT_N_PAIRS = 2000
DEFAULT_BEAM = 4
DEFAULT_TOPK = 8
DEFAULT_N_ITER = 3

#: 距离/排序分块预算：``[chunk, |V|]`` 的 float32 距离 + int64 排序索引。
#: 与 ``run_ablation.auto_chunk`` 不同，这里把一次 argsort 的索引也算进预算，
#: 避免 128k 词表上一次性分配数 GB 的 int64。
_SWEEP_BUDGET_ELEMS = 8_000_000

#: 论文安全模型的机器可读版本（原样写入结果 JSON）。
THREAT_MODEL: dict[str, Any] = {
    "attacker_position": "untrusted_server",
    "attacker_knows": ["tokenizer", "embedding_table"],
    "attacker_lm": "目标 LM 本身（服务端托管模型；BeamClean 的 LM 先验）",
    "sgt_weights_hidden": True,
    "greybox_rank_histogram": True,
    "mrp_prior": "本评测集的秩直方图众数 r*（任务单口径；SPEC A8 的独立校准集版本未测）",
    "fresh_noise": True,
    "whitebox_weights_tested": False,
    "attacker_goal": "token_reconstruction",
    "privacy_bound": "PAC-Privacy: Δ_KL(δ) ≤ MI(X; X̃), δ0 = 1/|V|（A7 口径 Eq (8)）",
    "note": "未测试知道 SGT 权重的白盒对手（论文模型也不假设）",
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S1 攻击评测：NN / MRP / BeamClean 从 x̃ 重建原始 token"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="本地 LM 目录")
    parser.add_argument("--dataset", choices=("alpaca", "pupa"), default="alpaca")
    parser.add_argument(
        "--pupa-config", choices=("pupa_tnb", "pupa_new"), default="pupa_tnb"
    )
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    obfuscation = parser.add_mutually_exclusive_group(required=True)
    obfuscation.add_argument("--sgt-ckpt", default=None, help="SGT checkpoint（客户端私有）")
    obfuscation.add_argument(
        "--constant-sigma", type=float, default=None, help="常数高斯噪声的 σ（标准差）"
    )
    obfuscation.add_argument(
        "--no-obfuscation", action="store_true", help="基线：x̃ = x（无保护）"
    )
    parser.add_argument(
        "--sgt-n-heads", type=int, default=None, help="覆盖 checkpoint 的 n_heads（默认取 config）"
    )
    parser.add_argument("--sgt-batch", type=int, default=8, help="SGT 前向的序列数/批")
    parser.add_argument(
        "--beamclean-n", type=int, default=DEFAULT_BEAMCLEAN_N,
        help="BeamClean 的序列子集大小；0 = 显式跳过",
    )
    parser.add_argument(
        "--beamclean-max-len", type=int, default=DEFAULT_BEAMCLEAN_MAX_LEN,
        help="BeamClean 每条序列截断长度",
    )
    parser.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER, help="BeamClean σ EM 迭代数")
    parser.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS, help="A7 MI 的 MC 配对数")
    parser.add_argument("--rank-chunk", type=int, default=0, help="攻击距离分块（0 = 自适应）")
    parser.add_argument("--seed", type=int, default=0, help="所有随机采样的种子")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--tag", default=None, help="结果标签（默认取 --out 的文件名）")
    parser.add_argument("--out", default="results/attack_suite.json", help="结果 JSON 路径")
    args = parser.parse_args(argv)

    for name in (
        "max_examples",
        "max_len",
        "eval_tokens",
        "beamclean_max_len",
        "beam",
        "topk",
        "sgt_batch",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    for name in ("beamclean_n", "n_iter", "n_pairs", "seed", "rank_chunk"):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} 必须非负")
    if args.constant_sigma is not None:
        sigma = float(args.constant_sigma)
        if not math.isfinite(sigma) or sigma < 0.0:
            parser.error(f"--constant-sigma 必须是非负有限值，实际 {args.constant_sigma}")
    if args.sgt_n_heads is not None and args.sgt_n_heads <= 0:
        parser.error("--sgt-n-heads 必须为正整数")
    return args


def _fail(message: str, code: int = 2) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- #
# 攻击评测（可独立测试：不加载模型）
# --------------------------------------------------------------------------- #
def default_fr_rs(vocab_size: int) -> list[int]:
    """``fr_by_r`` 的默认 r 集合：{1, 10, 100, 1000, |V|/2, |V|−1, |V|}（去重/越界剔除）。"""
    vocab_size = int(vocab_size)
    if vocab_size < 1:
        raise ValueError(f"vocab_size 必须为正，实际 {vocab_size}")
    candidates = [1, 10, 100, 1000, max(1, vocab_size // 2), max(1, vocab_size - 1), vocab_size]
    return sorted({r for r in candidates if 1 <= r <= vocab_size})


def _auto_sweep_chunk(vocab_size: int, budget_elems: int = _SWEEP_BUDGET_ELEMS) -> int:
    return max(1, int(budget_elems) // max(1, int(vocab_size)))


def _sweep_rank_predictions(
    x: torch.Tensor,
    table: torch.Tensor,
    target: torch.Tensor,
    rs: Sequence[int],
    *,
    chunk: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """一次距离/排序 pass 同时算秩与若干 r 的重建（与 ``metrics.ranks`` 同并列语义）。

    对每个分块：``dist = ‖x−e‖²``（:func:`sgt.metrics._sqdist_chunk`）→ 稳定
    ``argsort``（距离并列按 token id 升序）→ 第 ``r`` 列即第 ``r`` 近邻。
    秩用与 :func:`sgt.metrics.ranks` 完全相同的公式
    ``1 + #{d < d_target} + #{d == d_target 且 id < target}`` 复算，保证语义一致。
    """
    rs = sorted({int(r) for r in rs})
    table_sq = table.square().sum(dim=1)
    vocab = int(table.shape[0])
    token_ids = torch.arange(vocab, device=x.device)
    n = int(x.shape[0])
    ranks = torch.empty(n, dtype=torch.long, device=x.device)
    preds = {r: torch.empty(n, dtype=torch.long, device=x.device) for r in rs}
    step = max(1, int(chunk))
    for start in range(0, n, step):
        stop = min(start + step, n)
        dist = metrics._sqdist_chunk(x[start:stop], table, table_sq)
        tc = target[start:stop]
        # 秩：与 metrics.ranks 逐位一致的并列处理（稳定排序的等价公式）。
        d_target = dist.gather(1, tc.unsqueeze(1))
        closer = (dist < d_target).sum(dim=1)
        tied_before = (
            (dist == d_target) & token_ids.unsqueeze(0).lt(tc.unsqueeze(1))
        ).sum(dim=1)
        ranks[start:stop] = closer + tied_before + 1
        order = torch.argsort(dist, dim=1, stable=True)
        for r in rs:
            preds[r][start:stop] = order[:, r - 1]
    return ranks, preds


def _failure_rate(pred, target) -> float:
    """百分比 FR；在 CPU 上算（``metrics`` 的标量聚合用 float64，MPS 不支持）。"""
    return metrics.failure_rate(pred.detach().cpu(), target.detach().cpu())


def _mi_block(
    x_tilde: torch.Tensor,
    *,
    x_clean: torch.Tensor | None,
    mu: torch.Tensor | None,
    log_sigma2: torch.Tensor | None,
    n_pairs: int,
    seed: int,
    vocab_size: int,
) -> dict[str, Any]:
    """A7 口径 Eq (8) MI + PAC-Adv；σ=0（无噪声）时显式记为无定义。"""
    if log_sigma2 is None:
        # 无噪声：1/σ² 发散，Eq (8) 无定义。Eq (4) 的极限是 δ=1（完美可恢复）。
        return {
            "mi_feature_mean": None,
            "pac_adv": 100.0,
            "mi_status": "undefined_sigma_zero",
            "mi_note": (
                "σ=0（无噪声）：Eq (8) 的 1/σ² 发散，MI 未定义；"
                "PAC 界极限 δ=1，故记 PAC-Adv=100%"
            ),
        }
    if x_clean is None:
        raise ValueError("计算 feature MI 需要干净的 x_clean（独立 batch 的干净侧）")
    log_sigma2 = torch.as_tensor(log_sigma2)
    if not bool(torch.isfinite(log_sigma2).all()):
        raise ValueError("log_sigma2 含非有限值，无法计算 feature MI")
    mu_eff = torch.zeros_like(x_clean) if mu is None else torch.as_tensor(mu)
    mi = feature_mi(
        x_tilde,
        x_clean,
        mu_eff,
        log_sigma2,
        log_sigma2,
        n_pairs=int(n_pairs),
        seed=int(seed),
    )
    return {
        "mi_feature_mean": float(mi["mi_feature_mean"]),
        "pac_adv": float(metrics.pac_adv(float(mi["mi_feature_mean"]), int(vocab_size))),
        "mi_feature_median": float(mi["mi_feature_median"]),
        "mi_sum": float(mi["mi_sum"]),
        "mi_status": "ok",
        "mi_estimator": mi["estimator"],
        "mi_n_pairs": int(n_pairs),
        "mi_seed": int(seed),
    }


def evaluate_attacks(
    x_tilde,
    target_ids,
    table,
    *,
    x_clean=None,
    mu=None,
    log_sigma2=None,
    fr_rs: Sequence[int] | None = None,
    n_pairs: int = DEFAULT_N_PAIRS,
    seed: int = 0,
    chunk: int = 0,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """在只知道 ``table`` 与 ``x̃`` 的前提下跑 NN / MRP / 指标（可独立测试）。

    Args:
        x_tilde: ``[N, d]`` 混淆后 token embedding（展平的有效 token）。
        target_ids: ``[N]`` 真 token id；**只用于算 FR**，不参与任何重建。
        table: ``[V, d]`` 公开 embedding 表。
        x_clean: ``[N, d]`` 干净 embedding（feature MI 的独立 batch 干净侧）。
        mu / log_sigma2: ``[N, d]`` SGT（或常数噪声）的均值/对数方差，用于 Eq (8) MI。
            二者都为 ``None`` 表示无噪声（σ=0）：MI 记为无定义、PAC-Adv=100%。
        fr_rs: ``fr_by_r`` 的 r 集合（默认 :func:`default_fr_rs`；r=1 总会被加入）。
        n_pairs / seed: feature MI 的 MC 配对数/配对种子。
        chunk: 距离分块（0 = 按 ``_SWEEP_BUDGET_ELEMS`` 自适应）。
        return_predictions: 若为 True，额外返回 ``predictions``（CPU 张量），
            供 PII 文本重建使用；JSON 落盘前需弹出。

    Returns:
        指标 dict（见模块 docstring）。``fr_by_r`` 的键是 r 的字符串。
    """
    x = torch.as_tensor(x_tilde)
    table = torch.as_tensor(table)
    if x.ndim != 2:
        raise ValueError(f"x_tilde 必须为 [N, d]，实际 {tuple(x.shape)}")
    if table.ndim != 2:
        raise ValueError(f"table 必须为 [V, d]，实际 {tuple(table.shape)}")
    if x.shape[1] != table.shape[1]:
        raise ValueError(
            f"x_tilde 维度 {x.shape[1]} 与 embedding 表维度 {table.shape[1]} 不一致"
        )
    n = int(x.shape[0])
    vocab = int(table.shape[0])
    if n == 0:
        raise ValueError("x_tilde 不能为空")
    target = torch.as_tensor(target_ids).reshape(-1).long()
    if target.numel() != n:
        raise ValueError(f"target_ids 长度必须为 {n}，实际 {target.numel()}")
    if int(target.min()) < 0 or int(target.max()) >= vocab:
        raise ValueError(f"target_ids 必须在 [0, {vocab - 1}]，实际 "
                         f"[{int(target.min())}, {int(target.max())}]")

    # dtype/device 与 metrics._prepare 一致：半精度上采样，fp64 保持（CPU 数值对照）。
    dtype = torch.promote_types(x.dtype, table.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    elif dtype not in (torch.float32, torch.float64):
        dtype = torch.get_default_dtype()
    x = x.to(dtype=dtype)
    table = table.to(device=x.device, dtype=dtype)
    target = target.to(x.device)

    rs = default_fr_rs(vocab) if fr_rs is None else sorted({1, *(int(r) for r in fr_rs)})
    for r in rs:
        if not 1 <= r <= vocab:
            raise ValueError(f"fr_rs 中的 r 必须在 [1, {vocab}]，实际 {r}")
    step = int(chunk) if int(chunk) > 0 else _auto_sweep_chunk(vocab)

    ranks, preds = _sweep_rank_predictions(x, table, target, rs, chunk=step)
    ranks_cpu = ranks.cpu()
    target_cpu = target.cpu()
    histogram = metrics.rank_histogram(ranks_cpu, vocab)
    r_star = int(metrics.mrp_rank(histogram))
    if r_star not in preds:
        # r* 不在默认集合里：只对这一个 r 补一次 pass（仍是精确的第 r* 近邻）。
        _, extra = _sweep_rank_predictions(x, table, target, [r_star], chunk=step)
        preds[r_star] = extra[r_star]

    fr_by_r = {str(r): _failure_rate(preds[r], target) for r in sorted(preds)}
    mi_block = _mi_block(
        x,
        x_clean=x_clean,
        mu=mu,
        log_sigma2=log_sigma2,
        n_pairs=n_pairs,
        seed=seed,
        vocab_size=vocab,
    )
    result: dict[str, Any] = {
        "n_tokens": n,
        "vocab_size": vocab,
        "nn_fr": float(fr_by_r["1"]),
        "nn_fr_at_r1": float(fr_by_r["1"]),
        "mrp_fr": float(_failure_rate(preds[r_star], target)),
        "mrp_r_star": r_star,
        "ttr_1": float(metrics.ttr(ranks_cpu, 1, vocab)),
        "ttr_10": float(metrics.ttr(ranks_cpu, 10, vocab)),
        "ttr_50": float(metrics.ttr(ranks_cpu, 50, vocab)),
        "symttr_10": float(metrics.symttr(ranks_cpu, 10, vocab)),
        "symttr_100": float(metrics.symttr(ranks_cpu, 100, vocab)),
        "hist_entropy": float(metrics.hist_entropy(ranks_cpu, vocab)),
        "fr_by_r": fr_by_r,
        "fr_by_r_r_values": sorted(int(r) for r in preds),
        "rank_mean": float(ranks_cpu.to(torch.float64).mean()),
        "rank_median": float(ranks_cpu.to(torch.float64).median()),
        **mi_block,
    }
    if return_predictions:
        result["predictions"] = {
            "nn": preds[1].detach().cpu(),
            "mrp": preds[r_star].detach().cpu(),
            "fr_by_r": {str(r): preds[r].detach().cpu() for r in sorted(preds)},
        }
    return result


# --------------------------------------------------------------------------- #
# PUPA：PII-Ratio（NN 重建 vs 截断参考文本）
# --------------------------------------------------------------------------- #
def _decode(tokenizer, ids) -> str:
    if isinstance(ids, torch.Tensor):
        id_list = [int(value) for value in ids.detach().cpu().reshape(-1).tolist()]
    else:
        id_list = [int(value) for value in ids]
    return str(tokenizer.decode(id_list, skip_special_tokens=True))


def evaluate_pii(
    sequences: Sequence[Mapping[str, Any]],
    reconstructed_ids: Sequence[Any],
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """PUPA 分支：NN 重建文本 vs **截断后**参考文本（与 ``run_pupa`` 同口径）。

    Args:
        sequences: ``[{"example_index": int, "ids": list[int]}]``；``ids`` 已按
            ``max_len``/``eval_tokens`` 截断，是模型实际看到的 token。
        reconstructed_ids: 与 ``sequences`` 等长的重建 token id。
        tokenizer: 需提供 ``decode(ids, skip_special_tokens=True)``。
        examples: 原始 PUPA 样本（只用 ``pii_units``）。

    Returns:
        ``status="ok"``、``pii_ratio``/``pii_units_recovered`` 的样本均值（无定义
        样本不参与；全无定义时为 ``None``），以及计数与口径说明。
    """
    if len(sequences) != len(reconstructed_ids):
        raise ValueError(
            f"sequences 与 reconstructed_ids 长度不一致："
            f"{len(sequences)} vs {len(reconstructed_ids)}"
        )
    ratios: list[float] = []
    recovered: list[float] = []
    for seq, pred in zip(sequences, reconstructed_ids):
        target_ids = [int(token) for token in seq["ids"]]
        # 分母 T_ref：模型实际看到的**截断后**有效 token 的解码文本（不是完整 user_query）。
        reference = _decode(tokenizer, target_ids)
        reconstructed = _decode(tokenizer, pred)
        units = split_pii_units(examples[int(seq["example_index"])].get("pii_units") or [])
        ratio = pii_ratio(reference, reconstructed, units)
        units_recovered = pii_units_recovered(reference, reconstructed, units)
        if ratio["defined"] and ratio["ratio"] is not None:
            ratios.append(float(ratio["ratio"]))
        if units_recovered["n_units"] > 0:
            recovered.append(float(units_recovered["recovered_frac"]))
    return {
        "status": "ok",
        "pii_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "pii_units_recovered": (sum(recovered) / len(recovered)) if recovered else None,
        "n_samples": len(sequences),
        "n_pii_ratio_defined": len(ratios),
        "n_samples_with_pii_units": len(recovered),
        "pii_reference": "truncated_decoded",
        "pii_definition": (
            "T_ref = tokenizer.decode(截断后的有效 ids)；T* = NN 重建 ids 的解码文本；"
            "PII 字符匹配（大小写不敏感、最长优先、不重叠），非 PAPILLON judge（A10/D5）"
        ),
    }


# --------------------------------------------------------------------------- #
# 数据 -> 序列
# --------------------------------------------------------------------------- #
def load_examples(
    dataset: str, *, max_examples: int, pupa_config: str
) -> tuple[list[dict], dict[str, Any]]:
    """加载 Alpaca / PUPA，返回 ``(examples, meta)``。"""
    if dataset == "alpaca":
        examples = load_alpaca(max_examples=int(max_examples))
        meta = {
            "name": "tatsu-lab/alpaca",
            "role": "obfuscation-quality evaluation set（论文 §7）",
            "n_examples": len(examples),
            "rows_meta": dict(getattr(examples, "meta", {}) or {}),
        }
    elif dataset == "pupa":
        examples = load_pupa(config=str(pupa_config), max_examples=int(max_examples))
        meta = {
            "name": "Columbia-NLP/PUPA",
            "config": str(pupa_config),
            "text_field": "user_query",
            "note": "SPEC A20：被保护文本 = user_query（不是 target_response）",
            "n_examples": len(examples),
            "rows_meta": dict(getattr(examples, "meta", {}) or {}),
        }
    else:  # pragma: no cover - argparse 已限制
        raise ValueError(f"未知 dataset {dataset!r}")
    return [dict(row) for row in examples], meta


def dataset_texts(
    examples: Sequence[Mapping[str, Any]], dataset: str
) -> tuple[list[str], list[int]]:
    """按数据集口径构造文本，返回 ``(texts, example_indices)``。

    - Alpaca：``sgt.data.build_prompt``（instruction + input + output，SPEC A18）；
    - PUPA：``user_query``（SPEC A20）；空文本样本跳过并保持索引映射。
    """
    texts: list[str] = []
    indices: list[int] = []
    for index, example in enumerate(examples):
        if dataset == "pupa":
            value = example.get("user_query")
            if not isinstance(value, str):
                raise ValueError(
                    f"PUPA 样例 #{index} 缺少字符串字段 'user_query'（SPEC A20）"
                )
            if not value.strip():
                continue
        texts.append(build_prompt(dict(example), dataset))
        indices.append(index)
    if not texts:
        raise ValueError("没有可评测的文本（数据集为空或 PUPA 的 user_query 全为空）")
    return texts, indices


def collect_sequences(
    texts: Sequence[str],
    tokenizer,
    *,
    max_len: int,
    eval_tokens: int,
    example_indices: Sequence[int] | None = None,
) -> tuple[list[dict], int]:
    """右截断 tokenize，汇总到 ``eval_tokens`` 个有效 token（最后一条按需截断）。"""
    if int(max_len) <= 0:
        raise ValueError(f"max_len 必须为正整数，实际 {max_len}")
    if int(eval_tokens) <= 0:
        raise ValueError(f"eval_tokens 必须为正整数，实际 {eval_tokens}")
    encoded = tokenize_examples(list(texts), tokenizer, max_len=int(max_len))
    if example_indices is None:
        example_indices = list(range(len(texts)))
    if len(example_indices) != len(texts):
        raise ValueError("example_indices 与 texts 长度不一致")
    sequences: list[dict] = []
    total = 0
    for position, (ids, mask) in enumerate(
        zip(encoded["input_ids"], encoded["attention_mask"])
    ):
        valid = [int(token) for token, keep in zip(ids, mask) if keep]
        if not valid:
            continue
        take = min(len(valid), int(eval_tokens) - total)
        if take <= 0:
            break
        sequences.append(
            {"example_index": int(example_indices[position]), "ids": valid[:take]}
        )
        total += take
        if total >= int(eval_tokens):
            break
    if not sequences:
        raise ValueError("没有取到任何有效 token（检查 tokenizer 的 add_special_tokens 行为）")
    return sequences, total


# --------------------------------------------------------------------------- #
# SGT checkpoint 载入
# --------------------------------------------------------------------------- #
def load_attack_sgt(
    ckpt_path: str | Path,
    *,
    table: torch.Tensor,
    device: torch.device,
    n_heads: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """载入客户端私有的 SGT（``torch.load(..., weights_only=False)["sgt"]``）。

    重建统一走 :func:`run_pupa.load_sgt_from_checkpoint`：**优先**用 checkpoint 的
    ``config["sgt"]`` 重建（含 ``final_norm`` / ``mu_scale`` / ``sigma_floor`` /
    ``pos_encoding`` / ``sigma_max`` 等无参数但改变前向行为的选项），state_dict 形状只作
    一致性校验；无 config 的旧 checkpoint 按形状推断（``mu_scale`` / ``sigma_floor``
    推断不出 → ``None``，meta 标记 ``inferred_unparameterized_options=True``）。

    ``n_heads`` 无法由形状推断：``--sgt-n-heads`` 显式给出的值覆盖 checkpoint config，
    未给时取 ``config["sgt"]["n_heads"]``，再默认 4。

    返回的 meta 记录实际使用的 ``pos_encoding`` / ``final_norm`` / ``mu_scale`` /
    ``sigma_floor`` 及其来源，供结果 JSON 追溯。
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"SGT checkpoint 不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} 顶层必须是 mapping，实际 {type(payload).__name__}")
    state = payload.get("sgt")
    if not isinstance(state, Mapping):
        raise ValueError(
            f"{path} 缺少 'sgt' state_dict（要求 torch.load(...)[ 'sgt' ]）"
        )
    config = payload.get("config") or {}
    if isinstance(config, Mapping) and config.get("d") is not None:
        if int(config["d"]) != int(table.shape[1]):
            raise ValueError(
                f"checkpoint config 的 d={config['d']} 与 LM hidden_size="
                f"{int(table.shape[1])} 不一致"
            )

    sgt, meta = load_sgt_from_checkpoint(
        path,
        device=device,
        dtype=torch.float32,
        d=int(table.shape[1]),
        n_heads=n_heads,
    )
    sgt.eval()
    for parameter in sgt.parameters():
        parameter.requires_grad_(False)
    return sgt, meta


# --------------------------------------------------------------------------- #
# BeamClean
# --------------------------------------------------------------------------- #
class _CpuScoringAdapter:
    """让 ``run_pupa.make_lm_score_fn`` 的 float64 聚合在 CPU 上执行（MPS 无 float64）。

    ``make_lm_score_fn`` 用 ``next(lm.parameters())`` 判断计算设备，并在打分函数里做
    ``token_log_probs.to(torch.float64)``；MPS 不支持 float64。本适配器不暴露参数，
    因此调用方指定 ``device="cpu"`` 时，softmax/gather/float64 求和都在 CPU；
    只在 LM 前向时把 token 张量临时搬到 LM 设备，再把 logits 搬回 CPU。
    """

    def __init__(self, lm, device: torch.device) -> None:
        self.lm = lm
        self.device = device

    def parameters(self):
        return iter(())

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def __call__(self, input_ids, attention_mask=None):
        kwargs: dict[str, Any] = {"input_ids": input_ids.to(self.device)}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self.device)
        return SimpleNamespace(logits=self.lm(**kwargs).logits.cpu())


def make_attack_lm_score_fn(lm, *, device: torch.device | str):
    """复用 ``run_pupa.make_lm_score_fn``（标准自回归 shift 口径）。

    CPU 设备直接复用；MPS 上套 :class:`_CpuScoringAdapter`，把 float64 聚合留在 CPU，
    避免 ``Cannot convert a MPS Tensor to float64``。两条路径的 LM 前向与打分公式相同。
    """
    if torch.device(device).type == "cpu":
        return make_lm_score_fn(lm, chunk=0)
    adapter = _CpuScoringAdapter(lm, torch.device(device))
    return make_lm_score_fn(adapter, chunk=0, device="cpu")


def run_beamclean(
    sequences: Sequence[Mapping[str, Any]],
    x_tilde: torch.Tensor,
    table: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    lm_score_fn,
    n_sequences: int,
    max_len: int,
    beam: int,
    topk: int,
    n_iter: int,
    offsets: Sequence[int],
) -> dict[str, Any]:
    """前 ``n_sequences`` 条序列的 BeamClean（逐条调用，显式报告，不静默跳过）。

    ``sgt.attacks._topk_sqdist`` 会分配 float64 张量，MPS 不支持 float64；因此
    BeamClean 的 top-k 距离在 CPU 上算（``table`` 只复制一次），LM 打分仍由
    ``lm_score_fn`` 在 LM 所在设备上前向。结果 JSON 会记录 ``compute_device``。
    """
    if int(n_sequences) <= 0:
        return {
            "status": "skipped",
            "fr": None,
            "sigma_hat_mean": None,
            "n_sequences": 0,
            "n_tokens": 0,
            "reason": "--beamclean-n 0：显式跳过（不是静默跳过）",
        }
    if int(max_len) <= 0:
        raise ValueError(f"beamclean max_len 必须为正整数，实际 {max_len}")
    if len(offsets) != len(sequences) + 1:
        raise ValueError("offsets 长度必须为 len(sequences) + 1")

    take = min(int(n_sequences), len(sequences))
    table_cpu = table.detach().to("cpu", torch.float32)
    total_tokens = 0
    errors = 0
    sigma_hats: list[float] = []
    for index in range(take):
        sequence = sequences[index]
        length = min(len(sequence["ids"]), int(max_len))
        start = int(offsets[index])
        rows = x_tilde[start : start + length].detach().to("cpu", torch.float32)
        target = target_ids[start : start + length].detach().cpu().long()
        if rows.shape[0] != length:  # pragma: no cover - offsets 不变式
            raise RuntimeError(
                f"序列 #{index} 的 x̃ 行数 {rows.shape[0]} 与截断长度 {length} 不一致"
            )
        prediction, sigma_hat = beamclean(
            rows.unsqueeze(0),
            table_cpu,
            lm_score_fn,
            beam=int(beam),
            topk=int(topk),
            n_iter=int(n_iter),
        )
        prediction = prediction.reshape(-1).detach().cpu().long()
        errors += int((prediction != target).sum())
        total_tokens += int(target.numel())
        sigma_hats.append(float(sigma_hat))
    if total_tokens == 0:
        raise RuntimeError("BeamClean 没有可评测的 token")
    return {
        "status": "ok",
        "fr": float(100.0 * errors / total_tokens),
        "sigma_hat_mean": float(sum(sigma_hats) / len(sigma_hats)),
        "sigma_hat_values": sigma_hats,
        "n_sequences": take,
        "n_tokens": total_tokens,
        "max_len": int(max_len),
        "beam": int(beam),
        "topk": int(topk),
        "n_iter": int(n_iter),
        "sigma_estimator": "sgt.attacks.beamclean 的 top-k EM 估计（逐序列，n_iter）",
        "lm_scorer": (
            "run_pupa.make_lm_score_fn（标准自回归 shift：logits[:, :-1] 对 tokens[:, 1:]）；"
            "MPS 上 logits 经 CPU 适配器做 float64 聚合（MPS 不支持 float64）"
        ),
        "compute_device": "cpu（attacks._topk_sqdist 需 float64，MPS 不支持）",
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _flatten_sequences(sequences: Sequence[Mapping[str, Any]]) -> list[int]:
    ids: list[int] = []
    for sequence in sequences:
        ids.extend(int(token) for token in sequence["ids"])
    return ids


def _offsets(sequences: Sequence[Mapping[str, Any]]) -> list[int]:
    out = [0]
    for sequence in sequences:
        out.append(out[-1] + len(sequence["ids"]))
    return out


def _sigma_stats(sigma: torch.Tensor) -> dict[str, float]:
    values = sigma.detach().cpu().to(torch.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    mode = (
        "none"
        if args.no_obfuscation
        else ("constant" if args.constant_sigma is not None else "sgt")
    )
    out_path = Path(args.out)
    tag = str(args.tag) if args.tag is not None else out_path.stem
    if args.device == "mps" and not torch.backends.mps.is_available():
        return _fail("--device mps 不可用（torch.backends.mps.is_available() == False）")
    device = torch.device(args.device)

    # ---- 1. 数据 ---------------------------------------------------------- #
    data_start = time.perf_counter()
    try:
        examples, data_meta = load_examples(
            args.dataset,
            max_examples=int(args.max_examples),
            pupa_config=str(args.pupa_config),
        )
        texts, example_indices = dataset_texts(examples, args.dataset)
    except Exception as exc:
        return _fail(f"加载/构造数据集失败: {type(exc).__name__}: {exc}")
    timings["data_s"] = time.perf_counter() - data_start

    # ---- 2. LM + 公开 embedding 表 ---------------------------------------- #
    lm_start = time.perf_counter()
    try:
        tokenizer, lm = load_lm(args.model, device=device, dtype=torch.float32)
        table = lm.get_input_embeddings().weight.detach()
        if not bool(torch.isfinite(table).all()):
            return _fail("embedding 表含非有限值")
    except Exception as exc:
        return _fail(f"加载模型 {args.model} 失败: {type(exc).__name__}: {exc}")
    timings["lm_load_s"] = time.perf_counter() - lm_start

    # ---- 3. 序列（右截断 + eval_tokens 汇总） ------------------------------ #
    seq_start = time.perf_counter()
    try:
        sequences, n_tokens = collect_sequences(
            texts,
            tokenizer,
            max_len=int(args.max_len),
            eval_tokens=int(args.eval_tokens),
            example_indices=example_indices,
        )
    except Exception as exc:
        return _fail(f"tokenize/截断失败: {type(exc).__name__}: {exc}")
    timings["tokenize_s"] = time.perf_counter() - seq_start
    offsets = _offsets(sequences)

    # ---- 4. SGT checkpoint（客户端私有；攻击侧不可见） -------------------- #
    sgt = None
    sgt_meta: dict[str, Any] = {"applied": False}
    if mode == "sgt":
        try:
            sgt, sgt_meta = load_attack_sgt(
                args.sgt_ckpt,
                table=table,
                device=device,
                n_heads=args.sgt_n_heads,
            )
        except Exception as exc:
            traceback.print_exc()
            return _fail(f"载入 SGT checkpoint {args.sgt_ckpt} 失败: {type(exc).__name__}: {exc}")

    # ---- 5. 构造 x̃（fresh 噪声） ----------------------------------------- #
    obf_start = time.perf_counter()
    target_ids: torch.Tensor
    x_clean: torch.Tensor
    x_tilde: torch.Tensor
    mu: torch.Tensor | None = None
    log_sigma2: torch.Tensor | None = None
    sigma_block: dict[str, Any]
    obfuscation: dict[str, Any]
    try:
        if mode == "sgt":
            if max(len(sequence["ids"]) for sequence in sequences) > int(getattr(sgt, "max_len")):
                return _fail(
                    f"最长序列 {max(len(s['ids']) for s in sequences)} 超过 SGT max_len="
                    f"{int(getattr(sgt, 'max_len'))}；请降低 --max-len"
                )
            features = sgt_forward_sequences(
                [sequence["ids"] for sequence in sequences],
                sgt,
                table,
                pad_id=_pad_id(tokenizer),
                batch_seqs=int(args.sgt_batch),
                seed=int(args.seed),
                device=device,
            )
            target_ids = features["ids"]
            x_clean = features["x"]
            x_tilde = features["x_tilde"]
            mu = features["mu"]
            log_sigma2 = features["log_sigma2"]
            sigma = torch.exp(0.5 * log_sigma2)
            sigma_block = _sigma_stats(sigma)
            obfuscation = {
                "mode": "sgt",
                "ckpt": str(args.sgt_ckpt),
                "sigma": None,
                "sigma_mean": sigma_block["mean"],
                "sigma_std": sigma_block["std"],
                "sigma_min": sigma_block["min"],
                "sigma_max": sigma_block["max"],
                "fresh_noise": True,
                "noise": "x + mu + exp(0.5 * log_sigma2) * u（Eq (1)，逐 token 独立）",
                "seed": int(args.seed),
                "sgt_meta": json_safe(sgt_meta),
            }
        else:
            flat_ids = torch.tensor(_flatten_sequences(sequences), dtype=torch.long, device=device)
            target_ids = flat_ids
            x_clean = table[flat_ids]
            if mode == "constant":
                sigma_value = float(args.constant_sigma)
                generator = torch.Generator(device=device).manual_seed(int(args.seed))
                noise = torch.randn(
                    x_clean.shape,
                    generator=generator,
                    dtype=x_clean.dtype,
                    device=device,
                )
                x_tilde = x_clean + sigma_value * noise
                sigma_block = {
                    "mean": sigma_value,
                    "std": 0.0,
                    "min": sigma_value,
                    "max": sigma_value,
                }
                obfuscation = {
                    "mode": "constant",
                    "ckpt": None,
                    "sigma": sigma_value,
                    "sigma_mean": sigma_value,
                    "fresh_noise": True,
                    "noise": "x + sigma * u（逐 token 独立高斯，单次采样；seed 记录在案）",
                    "seed": int(args.seed),
                }
            else:
                x_tilde = x_clean
                sigma_block = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                obfuscation = {
                    "mode": "none",
                    "ckpt": None,
                    "sigma": 0.0,
                    "sigma_mean": 0.0,
                    "fresh_noise": True,
                    "noise": "x̃ = x（无保护基线）",
                    "seed": int(args.seed),
                }
            if mode == "constant" and float(args.constant_sigma) > 0.0:
                log_sigma2 = torch.full_like(x_clean, math.log(float(args.constant_sigma) ** 2))
                mu = torch.zeros_like(x_clean)
    except Exception:
        traceback.print_exc()
        return 1
    timings["obfuscation_s"] = time.perf_counter() - obf_start

    # ---- 6. 攻击（只知道 table 与 x̃） ------------------------------------- #
    attacks_start = time.perf_counter()
    try:
        attack_metrics = evaluate_attacks(
            x_tilde,
            target_ids,
            table,
            x_clean=x_clean,
            mu=mu,
            log_sigma2=log_sigma2,
            n_pairs=int(args.n_pairs),
            seed=int(args.seed),
            chunk=int(args.rank_chunk),
            return_predictions=True,
        )
        predictions = attack_metrics.pop("predictions")
    except Exception:
        traceback.print_exc()
        return 1
    timings["attacks_s"] = time.perf_counter() - attacks_start

    # ---- 7. BeamClean ------------------------------------------------------ #
    beamclean_start = time.perf_counter()
    try:
        if int(args.beamclean_n) > 0:
            lm_score_fn = make_attack_lm_score_fn(lm, device=device)
        else:
            lm_score_fn = None
        beamclean_block = run_beamclean(
            sequences,
            x_tilde,
            table,
            target_ids,
            lm_score_fn=lm_score_fn,
            n_sequences=int(args.beamclean_n),
            max_len=int(args.beamclean_max_len),
            beam=int(args.beam),
            topk=int(args.topk),
            n_iter=int(args.n_iter),
            offsets=offsets,
        )
    except Exception:
        traceback.print_exc()
        return 1
    timings["beamclean_s"] = time.perf_counter() - beamclean_start

    # ---- 8. PUPA PII-Ratio（NN 重建 vs 截断参考文本） --------------------- #
    pii_start = time.perf_counter()
    if args.dataset == "pupa":
        try:
            pii_block = evaluate_pii(
                sequences,
                predictions["nn"].split([len(sequence["ids"]) for sequence in sequences]),
                tokenizer,
                examples,
            )
        except Exception:
            traceback.print_exc()
            return 1
    else:
        pii_block = {
            "status": "not_applicable",
            "pii_ratio": None,
            "pii_units_recovered": None,
            "reason": "PII-Ratio 只在 PUPA 上定义（论文 §7 / SPEC A20）",
        }
    timings["pii_s"] = time.perf_counter() - pii_start
    timings["total_s"] = time.perf_counter() - started

    payload = {
        "tag": tag,
        "model": str(args.model),
        "dataset": args.dataset,
        "n_tokens": int(n_tokens),
        "max_len": int(args.max_len),
        "seed": int(args.seed),
        "threat_model": json_safe(THREAT_MODEL),
        "obfuscation": json_safe(obfuscation),
        "metrics": json_safe(attack_metrics),
        "beamclean": json_safe(beamclean_block),
        "pii": json_safe(pii_block),
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "sigma": json_safe(sigma_block),
        "dataset_meta": json_safe(data_meta),
        "sgt": json_safe(sgt_meta),
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": list(sys.argv),
            "torch_version": torch.__version__,
            "device": str(device),
            "dtype": "float32",
            "n_examples": len(examples),
            "n_sequences": len(sequences),
            "eval_tokens_requested": int(args.eval_tokens),
            "n_pairs": int(args.n_pairs),
            "rank_chunk": int(args.rank_chunk),
            "note": (
                "降规模评测；论文规模见 SPEC D2/D3。BeamClean 为非官方最小实现（D5），"
                "PII 为确定性字符匹配（非 PAPILLON judge，A10）。"
            ),
        },
    }
    try:
        write_json(out_path, payload)
    except Exception as exc:
        return _fail(f"写结果 JSON 失败 {out_path}: {type(exc).__name__}: {exc}", code=1)

    def _fmt(value, digits: int = 2) -> str:
        return "None" if value is None else f"{float(value):.{digits}f}"

    print(
        f"[attack_suite] tag={tag} mode={mode} dataset={args.dataset} "
        f"n_tokens={int(n_tokens)} nn_fr={_fmt(attack_metrics['nn_fr'])} "
        f"mrp_fr={_fmt(attack_metrics['mrp_fr'])} r*={attack_metrics['mrp_r_star']} "
        f"ttr_1={_fmt(attack_metrics['ttr_1'])} ttr_10={_fmt(attack_metrics['ttr_10'])} "
        f"ttr_50={_fmt(attack_metrics['ttr_50'])} "
        f"symttr_10={_fmt(attack_metrics['symttr_10'])} "
        f"symttr_100={_fmt(attack_metrics['symttr_100'])} "
        f"hist_entropy={_fmt(attack_metrics['hist_entropy'], 4)} "
        f"mi={_fmt(attack_metrics['mi_feature_mean'], 4)} "
        f"pac_adv={_fmt(attack_metrics['pac_adv'])} "
        f"sigma_mean={_fmt(sigma_block['mean'], 4)} "
        f"beamclean={_fmt(beamclean_block['fr'])} "
        f"pii_ratio={_fmt(pii_block['pii_ratio'], 4)} "
        f"total_s={timings['total_s']:.1f} -> {out_path}"
    )
    print(
        f"[attack_suite] fr_by_r={ {key: round(value, 2) for key, value in attack_metrics['fr_by_r'].items()} } "
        f"beamclean_status={beamclean_block['status']} pii_status={pii_block['status']}"
    )
    return 0


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad) if pad is not None else 0


if __name__ == "__main__":
    sys.exit(main())
