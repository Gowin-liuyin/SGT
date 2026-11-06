#!/usr/bin/env python3
"""**非论文语义**的对照实验：按 Protopia 官方文档的训练配方训练一个 SGT。

本脚本实现的损失/优化配方来自厂商文档
https://docs.protopia.ai/engine/1.2.1/recommendations_for_training_llms
（F36），**不是**论文 arXiv:2506.09452v1 的 Eq (8)。用途：检验「厂商实际配方」能否
达到论文声称的效用-隐私权衡；因此本脚本产出的结果**不得**当作论文值的复现。

与论文语义的差异（逐条，写给下游审计）：

- 损失：文档给 ``(1-α)(w1·L_M1 + w2·L_M2) + α(β·L_N + γ·L_S)``。本脚本 ``w1=0``，
  不实现 L_M1，也**不实现论文 Eq (8) 的 Mahalanobis 项**；``L_N`` 是去掉了
  Mahalanobis 的 std-log-ratio（论文 Eq (8) 的第一项），``L_S`` 是论文 Eq (9) 的绝对
  余弦。换言之：**Mahalanobis MI 损失完全不参与训练**。
- 架构：文档要求「与基座同类型的 transformer（mean/std 两个 estimator）」。本机没有
  Stained Glass Core，用仓库已有的 :class:`sgt.model.SGTModel` **近似**（同为
  transformer 编码器 + µ/Σ 两个头，但层数/宽度/位置编码/因果掩码均不同）。这是**近似**，
  不是文档架构的实现。
- σ 参数化：文档 ``scale=(1e-3, 1.0)``（``directly_learn_stds=False``）。本脚本用
  ``sigma_min=1e-3`` / ``sigma_max=1.0`` 的夹逼参数化近似（``SGTModel`` 的 A21 选择）。
  注意：``SGTModel.forward`` 只按 ``sigma_max`` 夹**上界**；``sigma_min`` 只在
  ``sample()``（推理）里生效，所以训练中 ``σ = exp(0.5·logσ²)`` 没有显式下界夹逼。

文档逐条对应的实现（任务单给定公式）：

===============  =========================================================
文档项            本脚本实现
===============  =========================================================
``L_M2``         蒸馏层隐藏态 L2：``mean_t ||h_clean - h_sgt||^2 / d``
``L_N``          std-log-ratio：两个独立 batch，
                 ``mean_{i,ℓ'} mean_j (log σ²_{ℓ',j} − log σ²_{i,j})``
``L_S``          输入 embedding 与 x̃ 的逐 token 绝对余弦均值（Eq (9)）
``α, β, γ``      0.7 / 0.01 / 1.00
``w1, w2``       0.0 / 1.4
优化              AdamW lr=3e-4, betas=(0.9,0.95), eps=1e-5, wd=0,
                 linear warmup 5% + cosine 到 ``0.1*lr``，grad clip 1.0
精度              SGT fp32、基座 bf16（文档要求）
===============  =========================================================

用法::

    ./.venv/bin/python experiments/run_documented_recipe.py \
        --config configs/documented_recipe.yaml --device mps --steps 800 \
        --out results/recipe_sgt.json --ckpt results/ckpt_recipe.pt

退出码：0 成功；2 配置/数据/模型等前置条件不可用；1 训练本身失败。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sgt.model import SGTModel  # noqa: E402

__all__ = [
    "VARIANT",
    "NOTE",
    "DOC_URL",
    "lr_multiplier",
    "recipe_total_loss",
    "distillation_l2_loss",
    "std_log_ratio_loss",
    "input_embedding_similarity_loss",
    "build_sgt_config",
    "build_sgt",
    "build_checkpoint_payload",
    "format_example",
    "resolve_run_config",
    "load_lm",
    "load_train_examples",
    "tokenize_texts",
    "collate",
    "step_batches",
    "write_json",
    "main",
]

#: 结果 JSON / checkpoint 的 ``variant`` 标记（下游据此与论文语义 run 区分）。
VARIANT = "documented_recipe"
#: 结果 JSON 的 ``note``：任何读者第一眼就该看到这不是论文语义。
NOTE = "非论文语义：按厂商文档配方"
DOC_URL = "https://docs.protopia.ai/engine/1.2.1/recommendations_for_training_llms"

DEFAULT_CONFIG = "configs/documented_recipe.yaml"
DEFAULT_MODEL = ".models/llama-3.2-1b-instruct"
DEFAULT_PARQUET = ".models/openorca_1M.parquet"

#: ``SGTModel.__init__`` 的全部默认值（checkpoint 的 ``config["sgt"]`` 要含全部构造参数）。
_SGT_SIGNATURE = inspect.signature(SGTModel.__init__)
_SGT_KWARGS = tuple(
    name for name in _SGT_SIGNATURE.parameters if name not in ("self", "d")
)
_SGT_DEFAULTS: dict[str, Any] = {
    name: _SGT_SIGNATURE.parameters[name].default
    for name in _SGT_KWARGS
    if _SGT_SIGNATURE.parameters[name].default is not inspect.Parameter.empty
}

_LM_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------------------- #
# 损失（文档配方；纯函数，可单独导入测试）
# --------------------------------------------------------------------------- #
def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """对 ``[B, T]`` 的逐 token 值取有效 token 均值；``mask=None`` 表示全有效。"""
    if mask is None:
        if values.numel() == 0:
            raise ValueError("输入为空，无法归约")
        return values.mean()
    m = torch.as_tensor(mask).to(device=values.device).bool()
    if tuple(m.shape) != tuple(values.shape):
        raise ValueError(f"mask 形状必须为 {tuple(values.shape)}，实际 {tuple(m.shape)}")
    if not bool(m.any()):
        raise ValueError("mask 未选中任何有效 token")
    return values[m].mean()


def _masked_feature_mean(
    values: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """对 ``[B, T, d]`` 取「有效 token × batch」的逐特征均值，返回 ``[d]``。"""
    if values.dim() != 3:
        raise ValueError(f"values 必须为 [B, T, d]，实际 {tuple(values.shape)}")
    if mask is None:
        return values.mean(dim=(0, 1))
    m = torch.as_tensor(mask).to(device=values.device).bool()
    if tuple(m.shape) != tuple(values.shape[:2]):
        raise ValueError(
            f"mask 形状必须为 {tuple(values.shape[:2])}，实际 {tuple(m.shape)}"
        )
    count = m.sum()
    if int(count) == 0:
        raise ValueError("mask 未选中任何有效 token")
    return (values * m.unsqueeze(-1)).sum(dim=(0, 1)) / count


def distillation_l2_loss(
    h_clean: torch.Tensor,
    h_sgt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``L_M2``：蒸馏层隐藏态的 L2 距离 ``mean_t ||h_clean - h_sgt||^2 / d``。

    文档的 ``distillation_layer_l2_distance_loss_weight`` 项；文档推荐用最后一层
    （``truncated_layer_index`` 取最后一层）的隐藏态。梯度只经 ``h_sgt`` 回传
    （``h_clean`` 侧由调用方 ``no_grad``）。
    """
    h_clean = torch.as_tensor(h_clean)
    h_sgt = torch.as_tensor(h_sgt)
    if tuple(h_clean.shape) != tuple(h_sgt.shape):
        raise ValueError(
            f"h_clean/h_sgt 形状必须一致：{tuple(h_clean.shape)} vs {tuple(h_sgt.shape)}"
        )
    if h_sgt.dim() != 3:
        raise ValueError(f"隐藏态必须为 [B, T, d]，实际 {tuple(h_sgt.shape)}")
    d = int(h_sgt.shape[-1])
    # 逐 token 平方 L2 在 fp32 下累加：bf16 的差平方容易在 2048 维上丢精度。
    per_token = (h_sgt.float() - h_clean.float()).pow(2).sum(dim=-1)
    return _masked_mean(per_token, mask) / d


def std_log_ratio_loss(
    log_sigma2_a: torch.Tensor,
    log_sigma2_b: torch.Tensor,
    mask_a: torch.Tensor | None = None,
    mask_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """``L_N``（std-log-ratio，论文 Eq (8) 第一项去掉 Mahalanobis）。

    文档/任务单公式（两个**独立 batch**）::

        mean_{i,ℓ'} mean_j (log σ²_{ℓ',j} − log σ²_{i,j})

    实现为「batch b 的逐特征均值（对有效 token 取均值）− batch a 的同口径均值」再对
    特征维 ``j`` 取均值——由线性性，这与上式的 ``mean_{i,ℓ'} mean_j`` 等价（``B=1``
    时逐字相等；``B>1`` 时逐 token 的均值按有效 token 总数加权）。

    注意：按此公式 ``L_N`` 是**有符号**量，交换两个 batch 会变号（反对称）；两个 batch
    相同时为 0。它不是标准差、也不是非负距离。
    """
    log_sigma2_a = torch.as_tensor(log_sigma2_a)
    log_sigma2_b = torch.as_tensor(log_sigma2_b)
    if log_sigma2_a.dim() != 3 or log_sigma2_b.dim() != 3:
        raise ValueError("log_sigma2 必须为 [B, T, d]")
    if int(log_sigma2_a.shape[-1]) != int(log_sigma2_b.shape[-1]):
        raise ValueError(
            f"两个 batch 的 d 必须一致：{int(log_sigma2_a.shape[-1])} vs "
            f"{int(log_sigma2_b.shape[-1])}"
        )
    mean_a = _masked_feature_mean(log_sigma2_a.float(), mask_a)
    mean_b = _masked_feature_mean(log_sigma2_b.float(), mask_b)
    return (mean_b - mean_a).mean()


def input_embedding_similarity_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``L_S``：输入 embedding 与 x̃ 的逐 token 绝对余弦均值（论文 Eq (9)）。

    与 :func:`sgt.losses.abs_cos_loss` 同语义（逐 token ``|cos|`` 后对有效 token 平均），
    本脚本内联实现以保持「文档配方」自成一体。
    """
    x = torch.as_tensor(x)
    x_tilde = torch.as_tensor(x_tilde)
    if tuple(x.shape) != tuple(x_tilde.shape):
        raise ValueError(f"x/x_tilde 形状必须一致：{tuple(x.shape)} vs {tuple(x_tilde.shape)}")
    if x.dim() != 3:
        raise ValueError(f"x 必须为 [B, T, d]，实际 {tuple(x.shape)}")
    x = x.float()
    x_tilde = x_tilde.float()
    numer = (x * x_tilde).sum(dim=-1)
    denom = (x.norm(dim=-1) * x_tilde.norm(dim=-1)).clamp_min(1e-12)
    return _masked_mean((numer / denom).abs(), mask)


def recipe_total_loss(
    loss_m2: torch.Tensor | float,
    loss_n: torch.Tensor | float,
    loss_s: torch.Tensor | float,
    alpha: float,
    beta: float,
    gamma: float,
    w1: float = 0.0,
    w2: float = 1.4,
    loss_m1: torch.Tensor | float | None = None,
) -> torch.Tensor | float:
    """文档总损失 ``(1-α)(w1·L_M1 + w2·L_M2) + α(β·L_N + γ·L_S)``。

    ``loss_m1=None`` 表示本次配方不实现 ``L_M1``（``w1=0`` 时它对总损失无贡献；传入
    ``loss_m1`` 只用于让公式可被独立验证）。参数/返回值可以是 float 或 0 维 tensor。
    """
    for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma), ("w1", w1), ("w2", w2)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} 必须是有限值，实际 {value!r}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1]，实际 {alpha!r}")
    for name, value in (("beta", beta), ("gamma", gamma), ("w1", w1), ("w2", w2)):
        if float(value) < 0.0:
            raise ValueError(f"{name} 必须非负，实际 {value!r}")
    if loss_m1 is None:
        if isinstance(loss_m2, torch.Tensor):
            loss_m1 = torch.zeros((), dtype=loss_m2.dtype, device=loss_m2.device)
        else:
            loss_m1 = 0.0
    return (1.0 - float(alpha)) * (
        float(w1) * loss_m1 + float(w2) * loss_m2
    ) + float(alpha) * (float(beta) * loss_n + float(gamma) * loss_s)


# --------------------------------------------------------------------------- #
# 学习率计划（warmup 5% + cosine 到 lr_min_factor）
# --------------------------------------------------------------------------- #
def lr_multiplier(
    step: int,
    total: int,
    warmup_fraction: float = 0.05,
    min_factor: float = 0.1,
) -> float:
    """线性 warmup + cosine 衰减的 lr 倍率，返回 ``lr_multiplier * lr`` 的系数。

    - ``step == 0`` → 0（warmup 起点）；
    - ``step == warmup`` → 1.0（``warmup = round(warmup_fraction * total)``）；
    - 之后余弦衰减，**最后一个训练步 ``step == total-1`` 恰为 ``min_factor``**；
      ``step >= total`` 同样返回 ``min_factor``（函数在端点外是常数）。
    """
    total = int(total)
    step = int(step)
    if total <= 0:
        raise ValueError(f"total 必须为正整数，实际 {total}")
    if not 0.0 <= float(warmup_fraction) < 1.0:
        raise ValueError(f"warmup_fraction 必须在 [0, 1)，实际 {warmup_fraction!r}")
    if not 0.0 <= float(min_factor) <= 1.0:
        raise ValueError(f"min_factor 必须在 [0, 1]，实际 {min_factor!r}")
    if step <= 0:
        return 0.0
    warmup = int(round(float(warmup_fraction) * total))
    if warmup == 0:
        return 1.0 if step < total else float(min_factor)
    if step <= warmup:
        return step / warmup
    span = (total - 1) - warmup
    if span <= 0:
        return float(min_factor)
    progress = min(1.0, (step - warmup) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_factor) + (1.0 - float(min_factor)) * cosine


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #
def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 mapping，实际 {type(value).__name__}")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}={value!r} 不是数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限值，实际 {number!r}")
    return number


def build_sgt_config(
    sgt_raw: Mapping[str, Any], *, d: int | None = None
) -> dict[str, Any]:
    """把 yaml ``sgt:`` 段补全成**全部构造参数**（checkpoint 的 ``config["sgt"]``）。

    未在 yaml 中给出的键用 :class:`SGTModel` 的构造默认值；未知键直接报错（拼写错误
    不应被静默忽略）。``d`` 可以放在 ``sgt:`` 段里，也可以显式传入。
    """
    sgt_raw = _require_mapping(sgt_raw, "sgt")
    unknown = sorted(set(sgt_raw) - set(_SGT_KWARGS) - {"d"})
    if unknown:
        raise ValueError(f"sgt: 段含未知键 {unknown}；允许 {list(_SGT_KWARGS) + ['d']}")
    d_value = int(sgt_raw.get("d", d if d is not None else 0))
    if d_value <= 0:
        raise ValueError(f"sgt.d 必须为正整数，实际 {sgt_raw.get('d', d)!r}")
    config: dict[str, Any] = {"d": d_value}
    for key in _SGT_KWARGS:
        config[key] = sgt_raw[key] if key in sgt_raw else _SGT_DEFAULTS[key]
    return config


def build_sgt(config: Mapping[str, Any]) -> SGTModel:
    """按 :func:`build_sgt_config` 的输出构造 :class:`SGTModel`（全部参数显式传入）。"""
    kwargs = {key: config[key] for key in _SGT_KWARGS}
    return SGTModel(int(config["d"]), **kwargs)


def resolve_run_config(
    raw: Mapping[str, Any],
    *,
    steps: int | None = None,
    max_examples: int | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """合并 yaml 与 CLI 覆盖，返回本次 run 的完整配置（validation 在此集中）。"""
    raw = _require_mapping(raw, "config")
    model_value = model_path if model_path is not None else raw.get("model", DEFAULT_MODEL)
    if isinstance(model_value, Mapping):
        model_value = model_value.get("path", DEFAULT_MODEL)
    model_value = str(model_value)

    sgt_raw = _require_mapping(raw.get("sgt") or {}, "sgt")
    sgt_config = build_sgt_config(sgt_raw)

    train_raw = _require_mapping(raw.get("train") or {}, "train")
    train = {
        "lr": _finite_float(train_raw.get("lr", 3.0e-4), "train.lr"),
        "betas": [float(x) for x in train_raw.get("betas", (0.9, 0.95))],
        "eps": _finite_float(train_raw.get("eps", 1.0e-5), "train.eps"),
        "weight_decay": _finite_float(
            train_raw.get("weight_decay", 0.0), "train.weight_decay"
        ),
        "steps": int(train_raw.get("steps", 800) if steps is None else steps),
        "batch_size": int(train_raw.get("batch_size", 1)),
        "max_len": int(train_raw.get("max_len", 256)),
        "seed": int(train_raw.get("seed", 0)),
        "grad_clip": _finite_float(train_raw.get("grad_clip", 1.0), "train.grad_clip"),
        "warmup_fraction": _finite_float(
            train_raw.get("warmup_fraction", 0.05), "train.warmup_fraction"
        ),
        "lr_min_factor": _finite_float(
            train_raw.get("lr_min_factor", 0.1), "train.lr_min_factor"
        ),
        "lm_dtype": str(train_raw.get("lm_dtype", "bfloat16")),
        "log_every": int(train_raw.get("log_every", 50)),
    }
    if len(train["betas"]) != 2:
        raise ValueError(f"train.betas 必须有两个元素，实际 {train['betas']}")
    for beta_value in train["betas"]:
        if not 0.0 <= beta_value < 1.0:
            raise ValueError(f"train.betas 元素必须在 [0, 1)，实际 {train['betas']}")
    if train["eps"] <= 0.0:
        raise ValueError(f"train.eps 必须为正，实际 {train['eps']}")
    if train["weight_decay"] < 0.0:
        raise ValueError(f"train.weight_decay 必须非负，实际 {train['weight_decay']}")
    if train["steps"] <= 0:
        raise ValueError(f"train.steps 必须为正，实际 {train['steps']}")
    if train["batch_size"] <= 0:
        raise ValueError(f"train.batch_size 必须为正，实际 {train['batch_size']}")
    if train["max_len"] <= 0:
        raise ValueError(f"train.max_len 必须为正，实际 {train['max_len']}")
    if not 0.0 <= train["warmup_fraction"] < 1.0:
        raise ValueError(
            f"train.warmup_fraction 必须在 [0, 1)，实际 {train['warmup_fraction']}"
        )
    if not 0.0 <= train["lr_min_factor"] <= 1.0:
        raise ValueError(
            f"train.lr_min_factor 必须在 [0, 1]，实际 {train['lr_min_factor']}"
        )
    if train["grad_clip"] <= 0.0:
        raise ValueError(f"train.grad_clip 必须为正，实际 {train['grad_clip']}")
    if train["log_every"] < 1:
        raise ValueError(f"train.log_every 必须 ≥ 1，实际 {train['log_every']}")
    if train["lm_dtype"] not in _LM_DTYPES:
        raise ValueError(
            f"train.lm_dtype 必须是 {sorted(_LM_DTYPES)} 之一，实际 {train['lm_dtype']!r}"
        )

    recipe_raw = _require_mapping(raw.get("recipe") or {}, "recipe")
    recipe = {
        "alpha": _finite_float(recipe_raw.get("alpha", 0.7), "recipe.alpha"),
        "beta": _finite_float(recipe_raw.get("beta", 0.01), "recipe.beta"),
        "gamma": _finite_float(recipe_raw.get("gamma", 1.0), "recipe.gamma"),
        "w1": _finite_float(recipe_raw.get("w1", 0.0), "recipe.w1"),
        "w2": _finite_float(recipe_raw.get("w2", 1.4), "recipe.w2"),
    }
    if not 0.0 <= recipe["alpha"] <= 1.0:
        raise ValueError(f"recipe.alpha 必须在 [0, 1]，实际 {recipe['alpha']}")
    for key in ("beta", "gamma", "w1", "w2"):
        if recipe[key] < 0.0:
            raise ValueError(f"recipe.{key} 必须非负，实际 {recipe[key]}")

    data_raw = _require_mapping(raw.get("data") or {}, "data")
    data = {
        "parquet": str(data_raw.get("parquet", DEFAULT_PARQUET)),
        "max_examples": int(
            data_raw.get("max_examples", 20000) if max_examples is None else max_examples
        ),
        "frac": _finite_float(data_raw.get("frac", 0.15), "data.frac"),
        "dataset": str(data_raw.get("dataset", "Open-Orca/OpenOrca")),
    }
    if data["max_examples"] <= 0:
        raise ValueError(f"data.max_examples 必须为正，实际 {data['max_examples']}")
    if not 0.0 < data["frac"] <= 1.0:
        raise ValueError(f"data.frac 必须在 (0, 1]，实际 {data['frac']}")

    return {
        "model": model_value,
        "sgt": sgt_config,
        "train": train,
        "recipe": recipe,
        "data": data,
    }


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def format_example(example: Mapping[str, Any]) -> str:
    """OpenOrca 文本：``system_prompt + "\\n" + question + "\\n" + response``。

    任务单要求逐字拼接三个字段（空字段也保留分隔符，不跳过）。
    """
    parts = (example.get("system_prompt"), example.get("question"), example.get("response"))
    return "\n".join("" if part is None else str(part) for part in parts)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_train_examples(
    data_config: Mapping[str, Any], *, max_examples: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """加载训练样本：优先本地 parquet，缺失时回退 ``sgt.data.load_openorca``。

    Returns:
        ``(examples, source)``；``source`` 如实记录实际走了哪条路径（不静默换数据集）。
    """
    limit = int(data_config.get("max_examples", 20000) if max_examples is None else max_examples)
    if limit <= 0:
        raise ValueError(f"max_examples 必须为正，实际 {limit}")
    parquet_path = _resolve_path(str(data_config.get("parquet", DEFAULT_PARQUET)))
    if parquet_path.is_file():
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
        available = len(dataset)
        take = min(limit, available)
        subset = dataset.select(range(take))
        examples = [dict(row) for row in subset]
        source = {
            "kind": "local_parquet",
            "path": str(parquet_path),
            "rows_available": int(available),
            "max_examples": int(take),
        }
    else:
        from sgt.data import load_openorca

        loaded = load_openorca(
            frac=float(data_config.get("frac", 0.15)),
            max_examples=limit,
        )
        examples = [dict(example) for example in loaded]
        meta = {
            key: value
            for key, value in dict(getattr(loaded, "meta", {}) or {}).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        source = {
            "kind": "rows_api",
            "dataset": str(data_config.get("dataset", "Open-Orca/OpenOrca")),
            "max_examples": int(limit),
            "n_fetched": int(len(examples)),
            "missing_parquet": str(parquet_path),
            "meta": meta,
        }
    if not examples:
        raise ValueError(f"没有加载到任何训练样本（source={source}）")
    return examples, source


def tokenize_texts(
    texts: Sequence[str], tokenizer, *, max_len: int, batch_size: int = 512
) -> list[list[int]]:
    """右截断到 ``max_len`` 并返回 token id 列表（去掉空序列）。"""
    if int(max_len) <= 0:
        raise ValueError(f"max_len 必须为正，实际 {max_len}")
    sequences: list[list[int]] = []
    for start in range(0, len(texts), int(batch_size)):
        chunk = list(texts[start : start + int(batch_size)])
        encoded = tokenizer(
            chunk, truncation=True, max_length=int(max_len), add_special_tokens=True
        )
        sequences.extend([list(ids) for ids in encoded["input_ids"]])
    return [sequence for sequence in sequences if sequence]


def collate(
    sequences: Sequence[Sequence[int]], *, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """把一批变长序列右 padding 成 ``ids [B, T]`` + ``mask [B, T]``（True=有效）。"""
    if not sequences:
        raise ValueError("sequences 不能为空")
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = True
    return ids, mask


def step_batches(
    sequences: Sequence[Sequence[int]], *, steps: int, batch_size: int
) -> Iterable[tuple[list[list[int]], list[list[int]]]]:
    """按固定顺序循环取数：每步返回 ``(主 batch, 第二个独立 batch)``。

    数据顺序是**确定性的**：按 parquet/数据集原始顺序循环，不 shuffle（``train.seed``
    只控制噪声与 SGT 初始化）。每个 batch 的样本互不重叠（相邻位置），走到末尾后回到
    开头继续。第二个 batch 只用于 ``L_N``（std-log-ratio 需要两个独立 batch）。
    """
    if not sequences:
        raise ValueError("sequences 不能为空")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size 必须为正，实际 {batch_size}")
    total = len(sequences)
    batch_size = int(batch_size)
    cursor = 0
    for _ in range(int(steps)):
        main = [list(sequences[(cursor + i) % total]) for i in range(batch_size)]
        cursor += batch_size
        aux = [list(sequences[(cursor + i) % total]) for i in range(batch_size)]
        cursor += batch_size
        yield main, aux


# --------------------------------------------------------------------------- #
# 模型 / checkpoint
# --------------------------------------------------------------------------- #
def load_lm(model_path: str | Path, *, device: torch.device, dtype: torch.dtype):
    """加载 LM+tokenizer；``eval()`` + 全参数 ``requires_grad=False``，返回冻结的
    embedding 表（detach，且不再要求梯度）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype)
    lm = lm.to(device)
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    embedding_table = lm.get_input_embeddings().weight.detach()
    embedding_table.requires_grad_(False)
    return tokenizer, lm, embedding_table


def build_checkpoint_payload(
    sgt: torch.nn.Module,
    *,
    sgt_config: Mapping[str, Any],
    recipe: Mapping[str, Any],
    train: Mapping[str, Any],
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造与 ``run_ablation`` 兼容的 checkpoint（加载器读 ``config["sgt"]``）。

    ``config["recipe"]`` 里带上文档配方的全部加权与优化超参，便于下游区分「论文语义」
    与「文档配方」的 checkpoint。
    """
    recipe_block = {
        "alpha": float(recipe["alpha"]),
        "beta": float(recipe["beta"]),
        "gamma": float(recipe["gamma"]),
        "w1": float(recipe["w1"]),
        "w2": float(recipe["w2"]),
        "lr": float(train["lr"]),
        "betas": [float(value) for value in train["betas"]],
        "eps": float(train["eps"]),
        "weight_decay": float(train["weight_decay"]),
        "steps": int(train["steps"]),
        "batch_size": int(train["batch_size"]),
        "max_len": int(train["max_len"]),
        "seed": int(train["seed"]),
        "grad_clip": float(train["grad_clip"]),
        "warmup_fraction": float(train["warmup_fraction"]),
        "lr_min_factor": float(train["lr_min_factor"]),
        "lm_dtype": str(train["lm_dtype"]),
    }
    config: dict[str, Any] = {
        "variant": VARIANT,
        "note": NOTE,
        "paper_semantics": False,
        "sgt": dict(sgt_config),
        "recipe": recipe_block,
    }
    if extra:
        for key, value in extra.items():
            if key in config:
                raise ValueError(f"checkpoint config 键重复：{key!r}")
            config[key] = value
    return {
        "sgt": {key: value.detach().cpu() for key, value in sgt.state_dict().items()},
        "config": config,
        "seed": int(seed),
        "variant": VARIANT,
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """严格 JSON（``allow_nan=False``：history 里出现 NaN/Inf 时直接失败，不写坏数据）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# CLI / 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="非论文语义：按厂商文档配方训练 SGT 的对照实验",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="yaml 配置路径")
    parser.add_argument("--device", default="auto", help="cpu / mps / cuda / auto")
    parser.add_argument("--steps", type=int, default=None, help="覆盖 train.steps")
    parser.add_argument("--max-examples", type=int, default=None, help="覆盖 data.max_examples")
    parser.add_argument("--model", default=None, help="覆盖 model 路径")
    parser.add_argument("--out", default="results/recipe_sgt.json", help="结果 JSON 路径")
    parser.add_argument(
        "--ckpt", default="results/ckpt_recipe.pt", help="checkpoint 路径"
    )
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("请求 mps 但当前环境不可用")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 cuda 但当前环境不可用")
    return device


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 顶层必须是 mapping，实际 {type(raw).__name__}")
    return raw


def _json_safe(value: Any) -> Any:
    """把 yaml 里的 Path/元组等转成可 JSON 序列化的对象（只用于配置回显）。"""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    # ---- 0. 配置 ---------------------------------------------------------- #
    try:
        raw = _load_yaml(args.config)
        run = resolve_run_config(
            raw,
            steps=args.steps,
            max_examples=args.max_examples,
            model_path=args.model,
        )
        device = resolve_device(args.device)
    except Exception as exc:
        print(f"[documented_recipe] 配置错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    sgt_config = run["sgt"]
    train = run["train"]
    recipe = run["recipe"]
    steps = int(train["steps"])
    batch_size = int(train["batch_size"])
    max_len = int(train["max_len"])
    seed = int(train["seed"])
    lm_dtype = _LM_DTYPES[str(train["lm_dtype"])]

    print(
        f"[documented_recipe] 非论文语义（厂商文档配方）config={args.config} "
        f"device={device} steps={steps} batch_size={batch_size} max_len={max_len} "
        f"alpha={recipe['alpha']} beta={recipe['beta']} gamma={recipe['gamma']} "
        f"w1={recipe['w1']} w2={recipe['w2']} lr={train['lr']:g}"
    )

    # ---- 1. LM / tokenizer / embedding 表 --------------------------------- #
    lm_start = time.perf_counter()
    try:
        tokenizer, lm, embedding_table = load_lm(
            run["model"], device=device, dtype=lm_dtype
        )
    except Exception as exc:
        print(
            f"[documented_recipe] 加载 LM 失败 ({run['model']}): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    timings: dict[str, Any] = {"load_lm_s": time.perf_counter() - lm_start}

    # ---- 2. 数据 ---------------------------------------------------------- #
    data_start = time.perf_counter()
    try:
        examples, data_source = load_train_examples(run["data"])
        texts = [format_example(example) for example in examples]
        sequences = tokenize_texts(texts, tokenizer, max_len=max_len)
    except Exception as exc:
        print(
            f"[documented_recipe] 加载/处理训练数据失败: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if not sequences:
        print("[documented_recipe] tokenize 后没有任何非空序列", file=sys.stderr)
        return 2
    timings["load_data_s"] = time.perf_counter() - data_start
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    # ---- 3. SGT ----------------------------------------------------------- #
    sgt_start = time.perf_counter()
    try:
        torch.manual_seed(seed)
        sgt = build_sgt(sgt_config).to(device=device, dtype=torch.float32)
        sgt.train()
    except Exception as exc:
        print(f"[documented_recipe] 构造 SGTModel 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    timings["sgt_init_s"] = time.perf_counter() - sgt_start
    n_parameters = sum(parameter.numel() for parameter in sgt.parameters())

    optimizer = torch.optim.AdamW(
        sgt.parameters(),
        lr=float(train["lr"]),
        betas=tuple(float(value) for value in train["betas"]),
        eps=float(train["eps"]),
        weight_decay=float(train["weight_decay"]),
    )

    # ---- 4. 训练循环 ------------------------------------------------------ #
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    train_start = time.perf_counter()
    for step, (main_sequences, aux_sequences) in enumerate(
        step_batches(sequences, steps=steps, batch_size=batch_size)
    ):
        step_start = time.perf_counter()
        lr_now = float(train["lr"]) * lr_multiplier(
            step,
            steps,
            warmup_fraction=float(train["warmup_fraction"]),
            min_factor=float(train["lr_min_factor"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        ids, mask = collate(main_sequences, pad_id=pad_id)
        ids = ids.to(device)
        mask = mask.to(device)
        aux_ids, aux_mask = collate(aux_sequences, pad_id=pad_id)
        aux_ids = aux_ids.to(device)
        aux_mask = aux_mask.to(device)

        x = embedding_table[ids].float()  # [B, T, d]，fp32（SGT 恒为 fp32）
        mu, log_sigma2 = sgt(x, attention_mask=mask)
        sigma = torch.exp(0.5 * log_sigma2)
        generator = torch.Generator(device=device).manual_seed(
            seed * 100003 + int(step)
        )
        noise = torch.randn(
            x.shape, generator=generator, device=device, dtype=torch.float32
        )
        x_tilde = x + mu + sigma * noise

        # 第二份独立 batch：只用于 L_N（std-log-ratio）。
        aux_x = embedding_table[aux_ids].float()
        _, aux_log_sigma2 = sgt(aux_x, attention_mask=aux_mask)

        with torch.no_grad():
            h_clean = lm(
                inputs_embeds=x.to(lm_dtype),
                attention_mask=mask.long(),
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1]
        h_sgt = lm(
            inputs_embeds=x_tilde.to(lm_dtype),
            attention_mask=mask.long(),
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[-1]

        loss_m2 = distillation_l2_loss(h_clean, h_sgt, mask=mask)
        loss_n = std_log_ratio_loss(
            log_sigma2, aux_log_sigma2, mask_a=mask, mask_b=aux_mask
        )
        loss_s = input_embedding_similarity_loss(x, x_tilde, mask=mask)
        loss = recipe_total_loss(
            loss_m2,
            loss_n,
            loss_s,
            alpha=float(recipe["alpha"]),
            beta=float(recipe["beta"]),
            gamma=float(recipe["gamma"]),
            w1=float(recipe["w1"]),
            w2=float(recipe["w2"]),
        )

        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            print(f"[documented_recipe] 第 {step} 步 loss 非有限值，中止", file=sys.stderr)
            return 1
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sgt.parameters(), float(train["grad_clip"])
        )
        if not bool(torch.isfinite(grad_norm)):
            print(
                f"[documented_recipe] 第 {step} 步梯度范数非有限值，中止", file=sys.stderr
            )
            return 1
        optimizer.step()

        sigma_mean = float(sigma[mask].detach().mean())
        mu_norm_mean = float(mu[mask].detach().norm(dim=-1).mean())
        record = {
            "step": int(step),
            "loss": float(loss.detach()),
            "loss_m2": float(loss_m2.detach()),
            "loss_n": float(loss_n.detach()),
            "loss_s": float(loss_s.detach()),
            "sigma_mean": sigma_mean,
            "mu_norm_mean": mu_norm_mean,
            "lr": lr_now,
            "grad_norm": float(grad_norm),
        }
        for key, value in record.items():
            if not math.isfinite(value):
                print(
                    f"[documented_recipe] 第 {step} 步 {key}={value!r} 非有限值，中止",
                    file=sys.stderr,
                )
                return 1
        history.append(record)

        elapsed = time.perf_counter() - step_start
        step_times.append(elapsed)
        if step % int(train["log_every"]) == 0 or step == steps - 1:
            print(
                f"[documented_recipe] step {step}/{steps} loss={record['loss']:.6f} "
                f"m2={record['loss_m2']:.6f} n={record['loss_n']:.6f} "
                f"s={record['loss_s']:.6f} sigma={record['sigma_mean']:.6f} "
                f"mu_norm={record['mu_norm_mean']:.6f} grad_norm={record['grad_norm']:.4f} "
                f"lr={lr_now:.3e} step_s={elapsed:.3f}"
            )
        del x, mu, log_sigma2, sigma, noise, x_tilde, h_clean, h_sgt
        del loss_m2, loss_n, loss_s, loss, aux_x, aux_log_sigma2

    timings["train_s"] = time.perf_counter() - train_start
    timings["mean_step_s"] = float(sum(step_times) / len(step_times)) if step_times else 0.0
    timings["steps_per_s"] = 1.0 / timings["mean_step_s"] if timings["mean_step_s"] > 0 else 0.0
    timings["per_step_s"] = [float(value) for value in step_times]

    # ---- 5. checkpoint ---------------------------------------------------- #
    ckpt_start = time.perf_counter()
    try:
        payload = build_checkpoint_payload(
            sgt,
            sgt_config=sgt_config,
            recipe=recipe,
            train=train,
            seed=seed,
            extra={
                "yaml_path": str(args.config),
                "yaml": _json_safe(raw),
                "model": run["model"],
                "data": data_source,
                "device": str(device),
                "n_parameters": int(n_parameters),
            },
        )
        ckpt_path = Path(args.ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt_path)
    except Exception as exc:
        print(
            f"[documented_recipe] 保存 checkpoint 失败: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    timings["ckpt_s"] = time.perf_counter() - ckpt_start
    timings["total_s"] = time.perf_counter() - started

    # ---- 6. 结果 JSON ----------------------------------------------------- #
    result = {
        "variant": VARIANT,
        "steps": steps,
        "config": {
            "yaml_path": str(args.config),
            "yaml": _json_safe(raw),
            "model": run["model"],
            "sgt": sgt_config,
            "train": train,
            "recipe": recipe,
            "data": {**run["data"], "source": data_source, "n_examples": len(examples),
                     "n_sequences": len(sequences)},
            "device": str(device),
            "n_parameters": int(n_parameters),
            "checkpoint": str(args.ckpt),
        },
        "history": history,
        "dtypes": {
            "lm": str(train["lm_dtype"]),
            "sgt": "float32",
            "input_embeddings": str(embedding_table.dtype).replace("torch.", ""),
        },
        "timings": timings,
        "note": NOTE,
    }
    try:
        write_json(args.out, result)
    except Exception as exc:
        print(f"[documented_recipe] 写结果 JSON 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    last = history[-1]
    print(
        f"[documented_recipe] done steps={steps} variant={VARIANT} "
        f"loss={last['loss']:.6f} m2={last['loss_m2']:.6f} n={last['loss_n']:.6f} "
        f"s={last['loss_s']:.6f} sigma_mean={last['sigma_mean']:.6f} "
        f"mu_norm_mean={last['mu_norm_mean']:.6f} lr={last['lr']:.3e} "
        f"mean_step_s={timings['mean_step_s']:.3f} total_s={timings['total_s']:.1f} "
        f"out={args.out} ckpt={args.ckpt}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
