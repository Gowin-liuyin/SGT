#!/usr/bin/env python3
"""M7 消融实验（SPEC §5；论文 §7.1 Table 1）：训练一个 SGT variant，在 Alpaca 上评估。

用法::

    ./.venv/bin/python experiments/run_ablation.py --config configs/ablation.yaml \
        --variant mi_abscos_norm --device mps --max-steps 300 --max-examples 200 \
        --eval-examples 200 --eval-max-len 64 --eval-tokens 2000 \
        --out results/ablation_mi_abscos_norm.json

流程（任务单 M7-A）：

1. 读 yaml：``variants.<variant>.loss_terms``、``sgt:`` 超参、``d:``、``train:``（A12 复现选择）。
2. 训练数据：优先本地 OpenOrca parquet（``--local-parquet``，默认
   ``.models/openorca_1M.parquet``，取前 N 条，记录文件 sha256）；文件不存在或给了
   ``--force-rows-api`` 时回退 ``sgt.data.load_openorca``（datasets-server rows API，
   走 ``data_cache/`` 缓存）。``data_source`` 如实写入 JSON（``local_parquet`` /
   ``rows_api``），不静默换数据集。
3. 冻结目标 LM（默认 ``.models/llama-3.2-1b-instruct``，fp32；``eval()`` +
   ``requires_grad_(False)``）。
4. ``make_batches(..., dataset_name="openorca")`` 后 **drop_last**（丢弃不足一个
   batch 的尾批，保证 MI ``mode="pair"`` 两侧同形）；另把各 batch 右 padding 到同一 T
   （理由见 :func:`_pad_to_common_width`）。
5. ``train_sgt``（``TrainConfig`` 全部来自 yaml；α 权重来自 yaml ``alphas:``，
   缺省 mi=1/d、acs=1、mnp=1，见 :func:`resolve_weights` 与 SPEC A12）。
6. 保存 ``results/ckpt_<variant>.pt``：``{"sgt": state_dict, "config": ..., "seed": ...}``。
7. Alpaca 评估（Table 1 指标 + A7 特征级 MI），定义见 :func:`rank_metrics` /
   :func:`feature_mi`（两者都是可导入函数，``tests/test_experiments.py`` 用小合成数据
   直接测试）。
8. 写 JSON（严格可序列化、NaN/Inf 记为 ``null``）并打印一行 ``[ablation]`` 汇总。

退出码：0 成功；2 配置/数据/模型等前置条件不可用；1 训练或评估本身失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sgt import metrics  # noqa: E402
from sgt.model import SGTModel  # noqa: E402
from sgt.sampling import sample_sgt  # noqa: E402
from sgt.train import LOSS_TERMS, TrainConfig, train_sgt  # noqa: E402

__all__ = [
    "rank_metrics",
    "feature_mi",
    "auto_chunk",
    "json_safe",
    "write_json",
    "load_lm",
    "load_eval_examples",
    "tokenize_eval_sequences",
    "load_train_examples",
]

#: 目标 LM 与训练数据的本仓库默认路径（任务单 M7-A）。
DEFAULT_MODEL = ".models/llama-3.2-1b-instruct"
DEFAULT_LOCAL_PARQUET = ".models/openorca_1M.parquet"
OPENORCA_FIELDS = ("id", "system_prompt", "question", "response")

#: 距离/排序分块预算：``[chunk, V]`` 的 float32 距离 + int64 排序索引 ≈ 16 B/元素。
_DIST_BUDGET_BYTES = 256 * 1024 * 1024

#: 论文 Table 1（p.9）原值；一律标注为论文值，降规模复现不可直接等同（SPEC §9）。
#: MI 列的 CosSim/AbsCos 是论文印刷值（3.2×10^19 / 6×10^9），如实转录。
TABLE1_REFERENCE: dict[str, dict[str, Any]] = {
    "comp_entropy": {
        "sgt_loss": "Comp. Entr.",
        "utility": -13.84,
        "nn_fr": 88.41,
        "mrp_fr": 88.41,
        "symttr_10": 75.99,
        "symttr_100": 49.75,
        "hist_entropy": 58.60,
        "mi": 22.19,
        "pac_adv": 100.0,
    },
    "cossim": {
        "sgt_loss": "CosSim",
        "utility": -5.49,
        "nn_fr": 99.98,
        "mrp_fr": 3.21,
        "symttr_10": 1.43,
        "symttr_100": 1.07,
        "hist_entropy": 2.47,
        "mi": 3.2e19,
        "pac_adv": 100.0,
    },
    "abscos": {
        "sgt_loss": "AbsCos",
        "utility": -0.01,
        "nn_fr": 94.90,
        "mrp_fr": 94.90,
        "symttr_10": 93.36,
        "symttr_100": 92.86,
        "hist_entropy": 87.62,
        "mi": 6e9,
        "pac_adv": 100.0,
    },
    "mi": {
        "sgt_loss": "MI",
        "utility": -0.72,
        "nn_fr": 34.43,
        "mrp_fr": 34.43,
        "symttr_10": 18.96,
        "symttr_100": 17.39,
        "hist_entropy": 23.96,
        "mi": 1.48,
        "pac_adv": 16.39,
    },
    "mi_abscos": {
        "sgt_loss": "MI+AbsCos",
        "utility": -2.44,
        "nn_fr": 81.23,
        "mrp_fr": 81.23,
        "symttr_10": 71.36,
        "symttr_100": 66.88,
        "hist_entropy": 75.72,
        "mi": 1.91,
        "pac_adv": 20.56,
    },
    "mi_abscos_norm": {
        "sgt_loss": "MI+AbsCos+Norm",
        "utility": -1.97,
        "nn_fr": 93.00,
        "mrp_fr": 93.00,
        "symttr_10": 90.42,
        "symttr_100": 88.15,
        "hist_entropy": 91.90,
        "mi": 1.11,
        "pac_adv": 12.69,
    },
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M7 消融实验：训练一个 Table 1 variant 并在 Alpaca 上评估"
    )
    parser.add_argument("--config", default="configs/ablation.yaml", help="YAML 配置路径")
    parser.add_argument("--variant", required=True, help="variants 下的名字")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 train.steps")
    parser.add_argument(
        "--max-examples", type=int, default=None, help="覆盖 data.max_examples（训练条数）"
    )
    parser.add_argument("--eval-examples", type=int, default=200, help="Alpaca 评估条数")
    parser.add_argument("--eval-max-len", type=int, default=64, help="评估序列截断长度")
    parser.add_argument("--eval-tokens", type=int, default=2000, help="评估有效 token 上限")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="本地 LM 目录")
    parser.add_argument(
        "--local-parquet", default=DEFAULT_LOCAL_PARQUET, help="本地 OpenOrca parquet"
    )
    parser.add_argument(
        "--force-rows-api", action="store_true", help="忽略本地 parquet，强制走 rows API"
    )
    parser.add_argument("--n-pairs", type=int, default=20000, help="A7 MI 的 MC 配对数")
    parser.add_argument("--rank-chunk", type=int, default=1024, help="metrics.ranks 分块")
    parser.add_argument(
        "--eval-batch", type=int, default=8, help="评估时每次 SGT 前向的序列数"
    )
    parser.add_argument("--seed", type=int, default=None, help="覆盖 train.seed")
    parser.add_argument("--ckpt", default=None, help="checkpoint 路径（默认 results/ckpt_<variant>.pt）")
    parser.add_argument("--out", default=None, help="结果 JSON（默认 results/ablation_<variant>.json）")
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps 必须非负")
    for name in ("max_examples", "eval_examples", "eval_max_len", "eval_tokens", "n_pairs",
                 "rank_chunk", "eval_batch"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed 必须非负")
    return args


def _fail(message: str, code: int = 2) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- #
# JSON / 文件工具
# --------------------------------------------------------------------------- #
def json_safe(obj):
    """转成严格 JSON 可序列化对象：非有限浮点 -> ``None``，numpy/Path -> 原生类型。"""
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):  # 结果 JSON 不应含张量；标量张量兜底
        if obj.numel() == 1:
            return json_safe(obj.detach().cpu().item())
        return json_safe(obj.detach().cpu().tolist())
    return str(obj)


def write_json(path: str | Path, payload: Any) -> Path:
    """严格写 JSON（``allow_nan=False``）；NaN/Inf 先经 :func:`json_safe` 记为 null。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _read_json(path: str | Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_sha256(path: str | Path, *, out_path: str | Path | None = None) -> tuple[str, bool]:
    """返回 ``(sha256, 是否复用)``；同路径同大小且旧结果 JSON 已有记录时直接复用。"""
    path = Path(path)
    size = path.stat().st_size
    if out_path is not None:
        previous = _read_json(out_path) or {}
        entry = previous.get("local_parquet")
        if (
            isinstance(entry, dict)
            and entry.get("sha256")
            and entry.get("size_bytes") == size
            and entry.get("path") == str(path)
        ):
            return str(entry["sha256"]), True
    return sha256_file(path), False


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #
def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 的顶层必须是 mapping，实际 {type(raw).__name__}")
    return raw


def resolve_variant(raw: Mapping[str, Any], variant_name: str) -> tuple[dict, tuple[str, ...]]:
    variants = raw.get("variants")
    if not isinstance(variants, Mapping) or variant_name not in variants:
        raise KeyError(f"variant {variant_name!r} 不存在；可选 {sorted(dict(variants or {}))}")
    variant = variants[variant_name] or {}
    if not isinstance(variant, Mapping):
        raise ValueError(f"variants.{variant_name} 必须是 mapping")
    loss_terms = tuple(str(term) for term in (variant.get("loss_terms") or ()))
    if not loss_terms:
        raise ValueError(f"variants.{variant_name}.loss_terms 不能为空")
    unknown = [term for term in loss_terms if term not in LOSS_TERMS]
    if unknown:
        raise ValueError(f"未知 loss_terms {unknown}；允许 {list(LOSS_TERMS)}")
    return dict(variant), loss_terms


def resolve_weights(
    raw: Mapping[str, Any],
    variant: Mapping[str, Any],
    loss_terms: Sequence[str],
    d: int,
    *,
    variant_name: str = "",
) -> tuple[dict[str, float], dict[str, Any]]:
    """α 权重：yaml ``alphas:``；缺省 ``mi=1/d``、``acs=1.0``、``mnp=1.0``。

    论文未给 α（SPEC A12）。``mi=1/d`` 的理由：``losses.mi_loss`` 对 d 个特征求和
    （Eq (8) 的 ``Σ_j``），逐 token 量级为 O(d)，除以 d 后与 ACS/MNP 同为 O(1)。
    **这是复现选择，不是论文值**；实际使用的权重全部写入结果 JSON。
    """
    defaults = {"mi": 1.0 / float(d), "acs": 1.0, "mnp": 1.0}
    configured: dict[str, float] = {}
    sources: list[str] = []
    train = raw.get("train") or {}
    for name, candidate in (
        ("alphas", raw.get("alphas")),
        ("train.alphas", train.get("alphas") if isinstance(train, Mapping) else None),
        (f"variants.{variant_name}.weights", variant.get("weights")),
    ):
        if isinstance(candidate, Mapping):
            configured.update({str(key): float(value) for key, value in candidate.items()})
            sources.append(name)
    weights = {
        term: float(configured.get(term, defaults.get(term, 1.0))) for term in loss_terms
    }
    echo = {
        "weights": weights,
        "defaults": defaults,
        "configured": configured,
        "sources": sources,
        "source": "config alphas" if sources else "default",
        "note": "α 是复现选择（SPEC A12，论文未给）；mi=1/d 用于抵消 Eq (8) 对 d 个特征的求和",
    }
    return weights, echo


def build_train_config(
    raw: Mapping[str, Any],
    *,
    loss_terms: Sequence[str],
    weights: Mapping[str, float],
    max_steps: int | None,
    device: str,
    seed: int,
) -> TrainConfig:
    train = raw.get("train") or {}
    if not isinstance(train, Mapping):
        raise ValueError("train 必须是 mapping")
    steps = int(train.get("steps", 100)) if max_steps is None else int(max_steps)
    grad_clip = train.get("grad_clip", 1.0)
    return TrainConfig(
        steps=steps,
        lr=float(train.get("lr", 1e-4)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        grad_clip=None if grad_clip is None else float(grad_clip),
        seed=int(seed),
        device=device,
        dtype="float32",
        loss_terms=tuple(loss_terms),
        weights=dict(weights),
        log_every=int(train.get("log_every", 1)),
        sigma_min=float(train.get("sigma_min", 1e-6)),
        temperature=float(train.get("temperature", 1.0)),
        checkpoint_dir=train.get("checkpoint_dir"),
    )


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_train_examples(
    *,
    local_parquet: str | Path,
    max_examples: int,
    frac: float,
    cache_dir: str | Path | None,
    out_path: str | Path | None = None,
    force_rows_api: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    """训练数据：本地 parquet 优先，否则 rows API（``sgt.data.load_openorca``）。"""
    from sgt.data import dataset_fingerprint, load_openorca

    if int(max_examples) <= 0:
        raise ValueError(f"max_examples 必须为正整数，实际 {max_examples}")
    path = Path(local_parquet)
    meta: dict[str, Any] = {"frac": float(frac), "max_examples": int(max_examples)}
    if path.is_file() and not force_rows_api:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(path), split="train")
        columns = [field for field in OPENORCA_FIELDS if field in dataset.column_names]
        missing = {"system_prompt", "question", "response"} - set(columns)
        if missing:
            raise ValueError(f"parquet {path} 缺少字段 {sorted(missing)}")
        n_available = len(dataset)
        n = min(int(max_examples), n_available)
        subset = dataset.select_columns(columns).select(range(n))
        examples = [{field: row.get(field) for field in OPENORCA_FIELDS} for row in subset]
        digest, reused = cached_sha256(path, out_path=out_path)
        meta.update(
            {
                "data_source": "local_parquet",
                "local_parquet": {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "sha256_reused_from_out": bool(reused),
                },
                "n_available": int(n_available),
                "n_examples": len(examples),
                "columns": columns,
            }
        )
    else:
        reason = "force_rows_api" if force_rows_api else "local_parquet_missing"
        examples_list = load_openorca(
            frac=float(frac), max_examples=int(max_examples), cache_dir=cache_dir
        )
        examples = [
            {field: row.get(field) for field in OPENORCA_FIELDS} for row in examples_list
        ]
        meta.update(
            {
                "data_source": "rows_api",
                "rows_api_fallback_reason": reason,
                "rows_meta": dict(getattr(examples_list, "meta", {}) or {}),
                "n_examples": len(examples),
            }
        )
    meta["fingerprint"] = dataset_fingerprint(examples)
    return examples, meta


def load_eval_examples(max_examples: int) -> tuple[list[dict], dict[str, Any]]:
    """Alpaca 评估集（前 ``max_examples`` 条，原始顺序）。"""
    from sgt.data import load_alpaca

    examples_list = load_alpaca(max_examples=int(max_examples))
    meta = {
        "dataset": "tatsu-lab/alpaca",
        "n_examples": len(examples_list),
        "rows_meta": dict(getattr(examples_list, "meta", {}) or {}),
    }
    return [dict(row) for row in examples_list], meta


def tokenize_eval_sequences(
    examples: Sequence[Mapping[str, Any]],
    tokenizer,
    *,
    max_len: int,
    n_tokens: int,
) -> tuple[list[list[int]], int]:
    """按序列 tokenize，取前 ``n_tokens`` 个有效 token（最后一条按需截断）。"""
    from sgt.data import tokenize_examples

    if int(max_len) <= 0:
        raise ValueError(f"max_len 必须为正整数，实际 {max_len}")
    if int(n_tokens) <= 0:
        raise ValueError(f"n_tokens 必须为正整数，实际 {n_tokens}")
    encoded = tokenize_examples(
        list(examples), tokenizer, max_len=int(max_len), dataset="alpaca"
    )
    sequences: list[list[int]] = []
    total = 0
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        valid = [int(token) for token, keep in zip(ids, mask) if keep]
        if not valid:
            continue
        take = min(len(valid), int(n_tokens) - total)
        if take <= 0:
            break
        sequences.append(valid[:take])
        total += take
        if total >= int(n_tokens):
            break
    return sequences, total


def auto_chunk(vocab_size: int, *, budget_bytes: int = _DIST_BUDGET_BYTES) -> int:
    """按 ``[chunk, V]`` 距离 + 排序索引的内存预算取分块大小（无固定默认依赖）。"""
    return max(1, int(budget_bytes) // max(1, int(vocab_size) * 16))


def _pad_to_common_width(
    batches: Sequence[Mapping[str, torch.Tensor]], pad_id: int
) -> list[dict[str, torch.Tensor]]:
    """把所有 batch 右 padding 到同一 T（mask=0，不改变有效 token）。

    ``make_batches`` 按每个 batch 内的最长序列补 padding，因此相邻 batch 的 T 可能不同；
    而 ``train_sgt`` 的 MI 项（``mode="pair"``，SPEC A2 默认）要求 x̃ 与 x′ 同形。
    只在 padding 位置补齐，等价于 A17 的右 padding 约定。
    """
    width = max(int(batch["input_ids"].shape[1]) for batch in batches)
    fixed: list[dict[str, torch.Tensor]] = []
    for batch in batches:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        missing = width - int(ids.shape[1])
        if missing > 0:
            ids = torch.cat(
                [ids, torch.full((ids.shape[0], missing), int(pad_id), dtype=ids.dtype)], dim=1
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], missing), dtype=mask.dtype)], dim=1
            )
        fixed.append({"input_ids": ids, "attention_mask": mask})
    return fixed


def build_train_batches(
    examples: Sequence[Mapping[str, Any]],
    tokenizer,
    *,
    batch_size: int,
    max_len: int,
    seed: int,
    pad_id: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """``make_batches`` + drop_last + 统一 T（见 :func:`_pad_to_common_width`）。"""
    from sgt.data import make_batches

    raw_batches = list(
        make_batches(
            list(examples),
            tokenizer,
            batch_size=int(batch_size),
            max_len=int(max_len),
            seed=int(seed),
            dataset_name="openorca",
        )
    )
    if not raw_batches:
        raise ValueError("make_batches 没有产出任何 batch")
    dropped_last = int(raw_batches[-1]["input_ids"].shape[0]) < int(batch_size)
    if dropped_last:
        raw_batches = raw_batches[:-1]
    if not raw_batches:
        raise ValueError(
            f"drop_last 之后没有完整 batch（batch_size={batch_size}，样本数 {len(examples)}）"
        )
    batches = _pad_to_common_width(raw_batches, pad_id)
    info = {
        "n_batches": len(batches),
        "dropped_last": dropped_last,
        "batch_size": int(batch_size),
        "max_len": int(max_len),
        "padded_width": int(batches[0]["input_ids"].shape[1]),
        "n_examples": len(examples),
        "tokens_per_epoch": int(
            sum(int(batch["attention_mask"].sum()) for batch in batches)
        ),
        "seed": int(seed),
        "note": "drop_last + 统一右 padding（MI pair 模式要求两侧同形）",
    }
    return batches, info


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad) if pad is not None else 0


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def load_lm(model_path: str | Path, *, device: torch.device, dtype: torch.dtype):
    """加载 LM+tokenizer；``eval()`` + 冻结全部参数。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    lm = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype)
    lm = lm.to(device)
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    return tokenizer, lm


# --------------------------------------------------------------------------- #
# 评估：SGT 前向 + 秩指标 + A7 特征级 MI
# --------------------------------------------------------------------------- #
def sgt_forward_sequences(
    sequences: Sequence[Sequence[int]],
    sgt,
    embedding_table: torch.Tensor,
    *,
    pad_id: int,
    batch_seqs: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """按序列前向 SGT 并采样（Eq (1)），返回展平后的有效 token 特征。

    Returns:
        ``{"x","mu","log_sigma2","x_tilde"}`` 均为 ``[N, d]``，``"ids"`` 为 ``[N]``；
        ``N`` = 有效 token 数。``x_tilde = sample_sgt(x, mu, exp(0.5·log_sigma2))``，
        采样由 ``torch.Generator(device).manual_seed(seed)`` 驱动（固定种子）。
    """
    if not sequences:
        raise ValueError("sequences 不能为空")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    collected: dict[str, list[torch.Tensor]] = {
        "x": [],
        "mu": [],
        "log_sigma2": [],
        "x_tilde": [],
        "ids": [],
    }
    with torch.no_grad():
        for start in range(0, len(sequences), int(batch_seqs)):
            chunk = sequences[start : start + int(batch_seqs)]
            width = max(len(sequence) for sequence in chunk)
            ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.bool)
            for row, sequence in enumerate(chunk):
                ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                mask[row, : len(sequence)] = True
            ids = ids.to(device)
            mask = mask.to(device)
            x = embedding_table[ids]
            mu, log_sigma2 = sgt(x, attention_mask=mask)
            sigma = torch.exp(0.5 * log_sigma2)
            x_tilde = sample_sgt(x, mu, sigma, generator=generator)
            collected["x"].append(x[mask])
            collected["mu"].append(mu[mask])
            collected["log_sigma2"].append(log_sigma2[mask])
            collected["x_tilde"].append(x_tilde[mask])
            collected["ids"].append(ids[mask])
    return {key: torch.cat(values, dim=0) for key, values in collected.items()}


def rank_metrics(
    x_tilde: torch.Tensor,
    target_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    chunk: int = 1024,
    aux_chunk: int | None = None,
    mrp_r: int | None = None,
) -> dict[str, Any]:
    """秩指标聚合（Table 1 的 NN-FR / MRP-FR / TTR / SymTTR / Hist. Entropy）。

    - ``ranks`` 用 ``chunk``（任务单固定 1024，V=128256 时 ``[chunk, V]`` 约 0.5 GB）；
      NN/MRP 的逐行 ``argsort`` 另用较小的 ``aux_chunk``（默认
      :func:`auto_chunk`），避免 int64 索引把显存打满。
    - MRP 的 ``r*`` = 秩直方图众数（``metrics.mrp_rank``，SPEC A8）；也可显式给
      ``mrp_r`` 复算。
    """
    x_tilde = torch.as_tensor(x_tilde)
    embedding_table = torch.as_tensor(embedding_table)
    if x_tilde.ndim != 2:
        raise ValueError(f"x_tilde 必须为 [N, d]，实际 {tuple(x_tilde.shape)}")
    if embedding_table.ndim != 2:
        raise ValueError(f"embedding_table 必须为 [V, d]，实际 {tuple(embedding_table.shape)}")
    n = int(x_tilde.shape[0])
    if n == 0:
        raise ValueError("x_tilde 不能为空")
    vocab_size = int(embedding_table.shape[0])
    aux = auto_chunk(vocab_size) if aux_chunk is None else int(aux_chunk)
    if aux <= 0:
        raise ValueError(f"aux_chunk 必须为正，实际 {aux}")

    ranks = metrics.ranks(x_tilde, target_ids, embedding_table, chunk=int(chunk))
    nn_pred = metrics.nn_reconstruct(x_tilde, embedding_table, chunk=aux)
    # metrics 的标量聚合内部用 float64（MPS 不支持 float64），统一在 CPU 副本上做；
    # 距离/排序仍在原设备（MPS）上算，排名结果与 CPU 路径逐位一致。
    ranks_cpu = ranks.detach().cpu()
    target_cpu = torch.as_tensor(target_ids).detach().cpu()
    nn_pred_cpu = nn_pred.detach().cpu()
    histogram = metrics.rank_histogram(ranks_cpu, vocab_size)
    r_star = int(mrp_r) if mrp_r is not None else metrics.mrp_rank(histogram)
    mrp_pred = metrics.mrp_reconstruct(x_tilde, embedding_table, r_star, chunk=aux)
    ranks_f64 = ranks_cpu.to(torch.float64)
    return {
        "n_tokens": n,
        "vocab_size": vocab_size,
        "nn_fr": metrics.failure_rate(nn_pred_cpu, target_cpu),
        "mrp_fr": metrics.failure_rate(mrp_pred.detach().cpu(), target_cpu),
        "mrp_r_star": r_star,
        "ttr_10": metrics.ttr(ranks_cpu, 10, vocab_size),
        "ttr_50": metrics.ttr(ranks_cpu, 50, vocab_size),
        "symttr_10": metrics.symttr(ranks_cpu, 10, vocab_size),
        "symttr_100": metrics.symttr(ranks_cpu, 100, vocab_size),
        "hist_entropy": metrics.hist_entropy(ranks_cpu, vocab_size),
        "rank_mean": float(ranks_f64.mean()),
        "rank_median": float(ranks_f64.median()),
    }


def feature_mi(
    x_tilde: torch.Tensor,
    x_prime: torch.Tensor,
    mu_prime: torch.Tensor,
    log_sigma2_prime: torch.Tensor,
    log_sigma2: torch.Tensor,
    *,
    n_pairs: int,
    seed: int,
    chunk: int = 1024,
) -> dict[str, Any]:
    """A7 特征级 MI：Eq (8) 逐特征项的 MC 平均（评测用，不是训练损失的复读）。

    均匀**有放回**抽 ``n_pairs`` 对 ``(i, ℓ')``（``i`` 索引 SGT 侧 ``x̃_i``，
    ``ℓ'`` 索引独立干净侧 ``x_{ℓ'}, µ_{ℓ'}, σ²_{ℓ'}``），每对

        v = mean_j [ log σ²_{ℓ',j} − log σ²_{i,j}
                     + (x̃_{i,j} − x_{ℓ',j} − µ_{ℓ',j})² / σ²_{ℓ',j} ]

    返回 ``mi_feature_mean = mean(v)``、``mi_sum = mi_feature_mean · d``、
    ``mi_feature_median``（**先在配对上按特征求均值，再对 d 个特征取中位数**，SPEC A7），
    以及逐特征均值的 std/min/max 供诊断。

    数值：``1/σ²`` 用 ``exp(-log σ²)``；逐块用输入精度（float32）计算，逐特征均值
    在 CPU float64 上累加。出现非有限值（σ² 极端）时显式报错，不静默写 null。
    """
    x_tilde = torch.as_tensor(x_tilde)
    x_prime = torch.as_tensor(x_prime)
    mu_prime = torch.as_tensor(mu_prime)
    log_sigma2_prime = torch.as_tensor(log_sigma2_prime)
    log_sigma2 = torch.as_tensor(log_sigma2)
    for name, tensor in (
        ("x_tilde", x_tilde),
        ("x_prime", x_prime),
        ("mu_prime", mu_prime),
        ("log_sigma2_prime", log_sigma2_prime),
        ("log_sigma2", log_sigma2),
    ):
        if tensor.ndim != 2:
            raise ValueError(f"{name} 必须为 [N, d]，实际 {tuple(tensor.shape)}")
    d = int(x_tilde.shape[1])
    if x_prime.shape[1] != d or mu_prime.shape[1] != d:
        raise ValueError("x_prime/mu_prime 的特征维必须与 x_tilde 一致")
    if log_sigma2_prime.shape[1] != d or log_sigma2.shape[1] != d:
        raise ValueError("log_sigma2/log_sigma2_prime 的特征维必须为 d")
    if log_sigma2.shape[0] != x_tilde.shape[0]:
        raise ValueError(
            f"log_sigma2 行数 {log_sigma2.shape[0]} 与 x_tilde 行数 {x_tilde.shape[0]} 不一致"
        )
    n = int(x_tilde.shape[0])
    m = int(x_prime.shape[0])
    if n == 0 or m == 0:
        raise ValueError("x_tilde/x_prime 不能为空")
    if int(n_pairs) <= 0:
        raise ValueError(f"n_pairs 必须为正，实际 {n_pairs}")
    if int(chunk) <= 0:
        raise ValueError(f"chunk 必须为正，实际 {chunk}")
    device = x_tilde.device
    x_tilde = x_tilde.to(device=device)
    x_prime = x_prime.to(device=device)
    mu_prime = mu_prime.to(device=device)
    log_sigma2_prime = log_sigma2_prime.to(device=device)
    log_sigma2 = log_sigma2.to(device=device)

    # 配对索引在 CPU 上用显式 Generator 抽样（设备无关、可复现），再搬到计算设备。
    index_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    index_i = torch.randint(0, n, (int(n_pairs),), generator=index_generator)
    index_p = torch.randint(0, m, (int(n_pairs),), generator=index_generator)

    feature_sum = torch.zeros(d, dtype=torch.float64)
    for start in range(0, int(n_pairs), int(chunk)):
        stop = min(start + int(chunk), int(n_pairs))
        rows_i = index_i[start:stop].to(device=device)
        rows_p = index_p[start:stop].to(device=device)
        lv_p = log_sigma2_prime[rows_p]
        lv_i = log_sigma2[rows_i]
        residual = x_tilde[rows_i] - x_prime[rows_p] - mu_prime[rows_p]
        term = (lv_p - lv_i) + residual.square() * torch.exp(-lv_p)
        if not bool(torch.isfinite(term).all()):
            raise ValueError(
                "feature_mi 出现非有限值：σ² 极端（log σ² 范围 "
                f"[{float(log_sigma2.min()):.2f}, {float(log_sigma2.max()):.2f}]）"
            )
        # 逐特征**求和**（再除以 n_pairs），不能累加块均值——块大小不同会引入权重偏差。
        feature_sum += term.sum(dim=0).detach().cpu().to(torch.float64)
    feature_mean = feature_sum / float(n_pairs)
    mi_feature_mean = float(feature_mean.mean())
    return {
        "estimator": "Eq(8) 逐特征 MC 均值（A7）",
        "n_pairs": int(n_pairs),
        "n_tokens": n,
        "n_prime": m,
        "n_features": d,
        "seed": int(seed),
        "chunk": int(chunk),
        "mi_feature_mean": mi_feature_mean,
        "mi_sum": mi_feature_mean * float(d),
        "mi_feature_median": float(feature_mean.median()),
        "mi_feature_std": float(feature_mean.std(unbiased=False)),
        "mi_feature_min": float(feature_mean.min()),
        "mi_feature_max": float(feature_mean.max()),
    }


def evaluate_sgt(
    sgt,
    embedding_table: torch.Tensor,
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    eval_batch: int,
    seed: int,
    rank_chunk: int,
    n_pairs: int,
) -> dict[str, Any]:
    """Alpaca 评估：秩指标 + A7 特征级 MI + PAC-Adv。"""
    features = sgt_forward_sequences(
        sequences,
        sgt,
        embedding_table,
        pad_id=pad_id,
        batch_seqs=eval_batch,
        seed=seed,
        device=embedding_table.device,
    )
    metrics_block = rank_metrics(
        features["x_tilde"], features["ids"], embedding_table, chunk=rank_chunk
    )
    # Eq (8) 的“独立 batch”：评测端用同一 Alpaca token 集的有放回抽样近似同分布独立采样。
    mi_block = feature_mi(
        features["x_tilde"],
        features["x"],
        features["mu"],
        features["log_sigma2"],
        features["log_sigma2"],
        n_pairs=n_pairs,
        seed=seed,
    )
    mi_block["pac_adv"] = metrics.pac_adv(mi_block["mi_feature_mean"], metrics_block["vocab_size"])
    mi_block["prime_source"] = "同一 Alpaca token 集的有放回抽样（近似 Eq (8) 的独立 batch）"
    metrics_block.update(
        {
            key: mi_block[key]
            for key in ("mi_feature_mean", "mi_sum", "mi_feature_median", "pac_adv")
        }
    )
    sigma = torch.exp(0.5 * features["log_sigma2"]).detach().cpu().to(torch.float64)
    sigma_block = {
        "mean": float(sigma.mean()),
        "std": float(sigma.std()),
        "min": float(sigma.min()),
        "max": float(sigma.max()),
        "log_sigma2_min": float(features["log_sigma2"].detach().cpu().min()),
        "log_sigma2_max": float(features["log_sigma2"].detach().cpu().max()),
    }
    return {
        "n_tokens": metrics_block["n_tokens"],
        "metrics": metrics_block,
        "feature_mi": mi_block,
        "sigma": sigma_block,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _save_checkpoint(path: Path, *, sgt, config: Mapping[str, Any], seed: int, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sgt": {key: value.detach().cpu() for key, value in sgt.state_dict().items()},
            "config": dict(config),
            "seed": int(seed),
            "variant": variant,
        },
        path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    # ---- 1. 配置 ---------------------------------------------------------- #
    try:
        raw = read_yaml(args.config)
        variant, loss_terms = resolve_variant(raw, args.variant)
    except Exception as exc:
        return _fail(f"读取/解析配置 {args.config} 失败: {type(exc).__name__}: {exc}")
    if not raw.get("model"):
        return _fail("configs 缺少 model")
    d = int(raw.get("d", 0))
    if d <= 0:
        return _fail(f"configs 的 d 必须为正整数，实际 {d}")
    sgt_raw = raw.get("sgt") or {}
    data_raw = raw.get("data") or {}
    train_raw = raw.get("train") or {}
    if not all(isinstance(block, Mapping) for block in (sgt_raw, data_raw, train_raw)):
        return _fail("sgt / data / train 必须是 mapping")
    seed = int(args.seed if args.seed is not None else train_raw.get("seed", 0))
    if seed < 0:
        return _fail(f"seed 必须非负，实际 {seed}")
    weights, weights_echo = resolve_weights(
        raw, variant, loss_terms, d, variant_name=args.variant
    )
    try:
        train_config = build_train_config(
            raw,
            loss_terms=loss_terms,
            weights=weights,
            max_steps=args.max_steps,
            device=args.device,
            seed=seed,
        )
    except Exception as exc:
        return _fail(f"训练配置不合法: {type(exc).__name__}: {exc}")
    if int(train_raw.get("max_len", 128)) > int(sgt_raw.get("max_len", 512)):
        return _fail("train.max_len 超过 sgt.max_len，SGT 无法前向")

    max_examples = int(
        args.max_examples if args.max_examples is not None else data_raw.get("max_examples", 4000)
    )
    frac = float(data_raw.get("frac", 0.15))
    cache_dir = data_raw.get("cache_dir")
    out_path = Path(args.out or f"results/ablation_{args.variant}.json")
    ckpt_path = Path(args.ckpt or f"results/ckpt_{args.variant}.pt")

    if args.device == "mps" and not torch.backends.mps.is_available():
        return _fail("--device mps 不可用（torch.backends.mps.is_available() == False）")
    device = torch.device(args.device)
    torch_dtype = torch.float32

    # ---- 2. 训练数据 ------------------------------------------------------ #
    data_start = time.perf_counter()
    try:
        examples, data_meta = load_train_examples(
            local_parquet=args.local_parquet,
            max_examples=max_examples,
            frac=frac,
            cache_dir=cache_dir,
            out_path=out_path,
            force_rows_api=args.force_rows_api,
        )
    except Exception as exc:
        return _fail(f"训练数据加载失败: {type(exc).__name__}: {exc}")
    timings["data_s"] = time.perf_counter() - data_start

    # ---- 3. 目标 LM ------------------------------------------------------- #
    lm_start = time.perf_counter()
    try:
        tokenizer, lm = load_lm(args.model, device=device, dtype=torch_dtype)
    except Exception as exc:
        return _fail(f"加载模型 {args.model} 失败: {type(exc).__name__}: {exc}")
    timings["lm_load_s"] = time.perf_counter() - lm_start
    hidden_size = int(getattr(lm.config, "hidden_size", -1))
    if hidden_size != d:
        return _fail(f"config d={d} 与 lm.config.hidden_size={hidden_size} 不一致")
    embedding_table = lm.get_input_embeddings().weight.detach()
    if not bool(torch.isfinite(embedding_table).all()):
        return _fail("embedding 表含非有限值")

    # ---- 4. SGT 初始化 ---------------------------------------------------- #
    init_start = time.perf_counter()
    torch.manual_seed(seed)  # SGT 初始化由 seed 决定（训练内部采样另有显式 Generator）
    try:
        sgt = SGTModel(
            d,
            hidden=int(sgt_raw.get("hidden", 512)),
            n_layers=int(sgt_raw.get("n_layers", 2)),
            n_heads=int(sgt_raw.get("n_heads", 4)),
            ffn_mult=int(sgt_raw.get("ffn_mult", 4)),
            max_len=int(sgt_raw.get("max_len", 512)),
            sigma_init=float(sgt_raw.get("sigma_init", 1.0)),
            sigma_min=train_config.sigma_min,
        ).to(device=device, dtype=torch_dtype)
    except Exception as exc:
        return _fail(f"构造 SGTModel 失败: {type(exc).__name__}: {exc}")
    timings["sgt_init_s"] = time.perf_counter() - init_start

    # ---- 5. batches ------------------------------------------------------- #
    try:
        batches, batch_info = build_train_batches(
            examples,
            tokenizer,
            batch_size=int(train_raw.get("batch_size", 2)),
            max_len=int(train_raw.get("max_len", 128)),
            seed=seed,
            pad_id=_pad_id(tokenizer),
        )
    except Exception as exc:
        return _fail(f"构造训练 batch 失败: {type(exc).__name__}: {exc}")

    # ---- 6. 训练 + checkpoint -------------------------------------------- #
    train_start = time.perf_counter()
    try:
        result = train_sgt(sgt, lm, batches, config=train_config)
    except Exception:
        traceback.print_exc()
        return 1
    timings["train_s"] = time.perf_counter() - train_start
    ckpt_start = time.perf_counter()
    try:
        _save_checkpoint(
            ckpt_path,
            sgt=sgt,
            config={
                "variant": args.variant,
                "yaml_path": str(args.config),
                "yaml": json_safe(raw),
                "loss_terms": list(loss_terms),
                "weights": weights,
                "train_config": json_safe(asdict(train_config)),
                "sgt": dict(sgt_raw),
                "d": d,
                "model": str(args.model),
                "eval": {
                    "examples": int(args.eval_examples),
                    "max_len": int(args.eval_max_len),
                    "tokens": int(args.eval_tokens),
                    "n_pairs": int(args.n_pairs),
                    "seed": seed,
                },
            },
            seed=seed,
            variant=args.variant,
        )
    except Exception as exc:
        return _fail(f"保存 checkpoint 失败: {type(exc).__name__}: {exc}", code=1)
    timings["ckpt_s"] = time.perf_counter() - ckpt_start

    # ---- 7. 评估（Alpaca） ------------------------------------------------ #
    eval_start = time.perf_counter()
    try:
        eval_examples, eval_meta = load_eval_examples(int(args.eval_examples))
        sequences, n_tokens = tokenize_eval_sequences(
            eval_examples, tokenizer, max_len=int(args.eval_max_len), n_tokens=int(args.eval_tokens)
        )
        if n_tokens == 0:
            return _fail("评估集没有取到任何有效 token", code=1)
        sgt.eval()
        eval_block = evaluate_sgt(
            sgt,
            embedding_table,
            sequences,
            pad_id=_pad_id(tokenizer),
            eval_batch=int(args.eval_batch),
            seed=seed,
            rank_chunk=int(args.rank_chunk),
            n_pairs=int(args.n_pairs),
        )
    except Exception:
        traceback.print_exc()
        return 1
    timings["eval_s"] = time.perf_counter() - eval_start
    timings["total_s"] = time.perf_counter() - total_start

    metrics_block = eval_block["metrics"]
    table_row = TABLE1_REFERENCE.get(args.variant)
    payload = {
        "variant": args.variant,
        "model": str(args.model),
        "device": args.device,
        "data_source": data_meta["data_source"],
        "n_train_examples": int(data_meta["n_examples"]),
        "steps": int(result["completed_steps"]),
        "train_history": result["history"],
        "eval": {
            "n_tokens": int(eval_block["n_tokens"]),
            "metrics": metrics_block,
            "feature_mi": eval_block["feature_mi"],
            "sigma": eval_block["sigma"],
            "dataset": "tatsu-lab/alpaca",
            "eval_examples": int(eval_meta["n_examples"]),
            "eval_max_len": int(args.eval_max_len),
            "eval_tokens_requested": int(args.eval_tokens),
            "sequences": len(sequences),
        },
        "config_echo": {
            "yaml_path": str(args.config),
            "raw_yaml": json_safe(raw),
            "variant": args.variant,
            "loss_terms": list(loss_terms),
            "weights": weights,
            "weights_echo": weights_echo,
            "sgt": dict(sgt_raw),
            "d": d,
            "train": json_safe(asdict(train_config)),
            "median_norm_T": float(result["median_norm"]),
            "eval": {
                "examples": int(args.eval_examples),
                "max_len": int(args.eval_max_len),
                "tokens": int(args.eval_tokens),
                "n_pairs": int(args.n_pairs),
                "rank_chunk": int(args.rank_chunk),
                "aux_chunk": auto_chunk(int(embedding_table.shape[0])),
                "eval_batch": int(args.eval_batch),
                "seed": seed,
                "sampling": "x̃ = x + µ + exp(0.5·log σ²) ⊙ u（Eq (1)，固定 seed）",
            },
            "model": {"path": str(args.model), "dtype": "float32", "frozen": True},
            "seed": seed,
        },
        "data": data_meta,
        "train_batches": batch_info,
        "checkpoint": str(ckpt_path),
        "table1_reference": {
            "source": "arXiv:2506.09452v1 Table 1（p.9，paper.txt 转录）",
            "scale_note": "论文在 A100 上 300 步以上/6h 训练；本复现降规模（max_examples/steps 见 "
            "config_echo），数值不可直接与论文对照（SPEC §9 判定规则）。",
            "variant": args.variant,
            "row": table_row,
            "all_variants": TABLE1_REFERENCE,
        },
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": list(sys.argv),
            "torch_version": torch.__version__,
            "torch_device": str(device),
            "torch_dtype": str(torch_dtype),
        },
    }
    try:
        write_json(out_path, payload)
    except Exception as exc:
        return _fail(f"写结果 JSON 失败 {out_path}: {type(exc).__name__}: {exc}", code=1)

    print(
        f"[ablation] variant={args.variant} steps={payload['steps']} "
        f"nn_fr={metrics_block['nn_fr']:.2f} mrp_fr={metrics_block['mrp_fr']:.2f} "
        f"symttr_10={metrics_block['symttr_10']:.2f} "
        f"hist_entropy={metrics_block['hist_entropy']:.4f} "
        f"mi_feature_mean={metrics_block['mi_feature_mean']:.4f} "
        f"pac_adv={metrics_block['pac_adv']:.4f} "
        f"train_s={timings['train_s']:.1f} total_s={timings['total_s']:.1f} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
