"""M6 训练入口：读 ``configs/ablation.yaml`` 的一个 variant，冻结 LM 训练 SGT。

用法::

    ./.venv/bin/python experiments/train_sgt.py \
        --config configs/ablation.yaml --variant mi_abscos_norm \
        --max-steps 10 --device cpu --out results/train_mi_abscos_norm.json

数据依赖 ``sgt.data``（M5）：``load_openorca`` + ``make_batches``。数据模块/缓存/网络
不可用时打印原因并以退出码 **2** 结束（任务单要求）；训练本身失败以 1 结束。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sgt.model import SGTModel  # noqa: E402
from sgt.train import TrainConfig, train_sgt  # noqa: E402

_DTYPE_NAMES = ("float32", "float16", "bfloat16", "float64")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M6：冻结目标 LLM，在 OpenOrca 上训练 SGT（Table 1 variant）"
    )
    parser.add_argument(
        "--config", default="configs/ablation.yaml", help="YAML 配置路径"
    )
    parser.add_argument("--variant", required=True, help="variants 下的名字")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="覆盖 train.steps（调试用）"
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps"), default="cpu", help="LM 与 SGT 的运行设备"
    )
    parser.add_argument(
        "--out", default=None, help="结果 JSON 路径（默认 results/train_<variant>.json）"
    )
    return parser.parse_args(argv)


def _fail(message: str, code: int = 2) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 的顶层必须是 mapping，实际 {type(raw).__name__}")
    return raw


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_NAMES:
        raise ValueError(f"dtype 只能是 {_DTYPE_NAMES}，实际 {name!r}")
    return getattr(torch, name)


def _build_train_config(
    raw: dict[str, Any], variant_name: str, *, max_steps: int | None, device: str
) -> tuple[TrainConfig, dict[str, Any], dict[str, Any]]:
    variants = raw.get("variants")
    if not isinstance(variants, dict) or variant_name not in variants:
        raise KeyError(
            f"variant {variant_name!r} 不存在；可选 {sorted((variants or {}).keys())}"
        )
    variant = variants[variant_name] or {}
    train_raw = raw.get("train") or {}
    if not isinstance(train_raw, dict):
        raise ValueError("train 必须是 mapping")
    if not isinstance(variant, dict):
        raise ValueError(f"variants.{variant_name} 必须是 mapping")

    dtype_name = str(train_raw.get("dtype", "float32"))
    _resolve_dtype(dtype_name)  # 尽早报错

    steps = int(train_raw.get("steps", 100)) if max_steps is None else int(max_steps)
    grad_clip_raw = train_raw.get("grad_clip", 1.0)
    train_config = TrainConfig(
        steps=steps,
        lr=float(train_raw.get("lr", 1e-4)),
        weight_decay=float(train_raw.get("weight_decay", 0.0)),
        grad_clip=None if grad_clip_raw is None else float(grad_clip_raw),
        seed=int(train_raw.get("seed", 0)),
        device=device,
        dtype=dtype_name,
        loss_terms=tuple(str(term) for term in (variant.get("loss_terms") or ())),
        weights={
            str(key): float(value)
            for key, value in {**(train_raw.get("weights") or {}), **(variant.get("weights") or {})}.items()
        },
        log_every=int(train_raw.get("log_every", 1)),
        sigma_min=float(train_raw.get("sigma_min", 1e-6)),
        temperature=float(train_raw.get("temperature", 1.0)),
        checkpoint_dir=train_raw.get("checkpoint_dir"),
    )
    return train_config, train_raw, variant


def _load_lm(model_name: str, *, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    lm = lm.to(device)
    return tokenizer, lm


def _load_batches(
    data_raw: dict[str, Any], train_raw: dict[str, Any], tokenizer
) -> list[dict[str, torch.Tensor]] | None:
    """从 sgt.data 取 OpenOrca batches；不可用时返回 None（调用方退出 2）。"""
    try:
        from sgt.data import load_openorca, make_batches
    except Exception as exc:  # 数据模块未就绪 / 依赖缺失
        print(f"ERROR: 无法导入 sgt.data（M5 数据管线不可用）: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None

    try:
        examples = load_openorca(
            frac=float(data_raw.get("frac", 0.15)),
            max_examples=data_raw.get("max_examples"),
            seed=int(train_raw.get("seed", 0)),
            cache_dir=data_raw.get("cache_dir"),
        )
        batches = list(
            make_batches(
                examples,
                tokenizer,
                batch_size=int(train_raw.get("batch_size", 2)),
                max_len=int(train_raw.get("max_len", 128)),
                seed=int(train_raw.get("seed", 0)),
                dataset_name="openorca",
            )
        )
    except Exception as exc:
        print(f"ERROR: 数据加载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    if not batches:
        print("ERROR: sgt.data 没有产出任何 batch", file=sys.stderr)
        return None
    return batches


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad) if pad is not None else 0


def _pad_to_common_width(
    batches: Sequence[dict[str, torch.Tensor]], pad_id: int
) -> list[dict[str, torch.Tensor]]:
    """把所有 batch 右 padding 到同一 T。

    M5 的 ``make_batches`` 按**每个 batch** 内的最长序列补 padding，因此相邻 batch 的 T
    可能不同；而 ``train_sgt`` 的 MI 项（``mode="pair"``）要求 x̃ 与 x′ 同形。这里只在
    padding 位置补齐（mask=0），不改变有效 token，等价于 A17 的右 padding 约定。
    """
    width = max(int(batch["input_ids"].shape[1]) for batch in batches)
    fixed: list[dict[str, torch.Tensor]] = []
    for batch in batches:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        missing = width - int(ids.shape[1])
        if missing > 0:
            ids = torch.cat(
                [ids, torch.full((ids.shape[0], missing), pad_id, dtype=ids.dtype)], dim=1
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], missing), dtype=mask.dtype)], dim=1
            )
        fixed.append({"input_ids": ids, "attention_mask": mask})
    return fixed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        raw = _read_yaml(args.config)
    except Exception as exc:
        return _fail(f"读取配置 {args.config} 失败: {type(exc).__name__}: {exc}")

    try:
        train_config, train_raw, variant = _build_train_config(
            raw, args.variant, max_steps=args.max_steps, device=args.device
        )
    except Exception as exc:
        return _fail(f"配置/variant 不合法: {type(exc).__name__}: {exc}")

    if args.device == "mps" and not torch.backends.mps.is_available():
        return _fail("--device mps 不可用（torch.backends.mps.is_available() == False）")
    device = torch.device(args.device)
    torch_dtype = _resolve_dtype(train_config.dtype)
    if device.type == "mps" and torch_dtype == torch.float64:
        return _fail("MPS 不支持 float64；请改用 float32/bfloat16")

    model_name = str(raw.get("model", ""))
    if not model_name:
        return _fail("configs 缺少 model")
    d = int(raw.get("d", 0))
    sgt_raw = raw.get("sgt") or {}
    data_raw = raw.get("data") or {}
    if not isinstance(sgt_raw, dict) or not isinstance(data_raw, dict):
        return _fail("sgt / data 必须是 mapping")
    if int(train_raw.get("max_len", 128)) > int(sgt_raw.get("max_len", 512)):
        return _fail("train.max_len 超过 sgt.max_len，SGT 无法前向")

    try:
        tokenizer, lm = _load_lm(model_name, device=device, dtype=torch_dtype)
    except Exception as exc:
        return _fail(f"加载模型 {model_name} 失败: {type(exc).__name__}: {exc}")

    hidden_size = int(getattr(lm.config, "hidden_size", -1))
    if hidden_size != d:
        return _fail(f"config d={d} 与 lm.config.hidden_size={hidden_size} 不一致")

    # SGT 初始化由 seed 决定（训练内部的采样另有显式 Generator）。
    torch.manual_seed(train_config.seed)
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

    batches = _load_batches(data_raw, train_raw, tokenizer)
    if batches is None:
        return 2  # 任务单：sgt.data 不可用 -> 退出码 2
    batches = _pad_to_common_width(batches, _pad_id(tokenizer))

    try:
        start = time.perf_counter()
        result = train_sgt(sgt, lm, batches, config=train_config)
        elapsed = time.perf_counter() - start
    except Exception:
        traceback.print_exc()
        return 1

    payload = {
        "variant": args.variant,
        "model": model_name,
        "device": args.device,
        "elapsed_seconds": elapsed,
        **result,
    }
    out_path = Path(args.out or f"results/train_{args.variant}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(
        f"[train_sgt] variant={args.variant} steps={result['completed_steps']} "
        f"batches={len(batches)} elapsed={elapsed:.2f}s -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
