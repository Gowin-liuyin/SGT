#!/usr/bin/env python3
"""M8a Utility 评测入口：lm-evaluation-harness（8 benchmark）+ 可选 SGT 注入。

用法（lm venv，仓库根目录）::

    .venv-lm/bin/python experiments/run_utility.py \
      --model .models/llama-3.2-1b-instruct \
      --tasks arc_easy,arc_challenge,openbookqa,piqa,truthfulqa_mc2,ifeval,mmlu,hellaswag \
      --limit 100 --batch-size 4 --device mps \
      [--sgt-ckpt results/ckpt_abscos.pt --sgt-seed 0] \
      --out results/utility_<tag>.json

注入方式：在 ``simple_evaluate`` 之前 monkeypatch
``transformers.AutoModelForCausalLM.from_pretrained``，对 harness 加载出的模型调用
``install_sgt_embedding``（替换输入 embedding 层）。SGT 以 fp32 计算，base 模型可以是
bf16；wrapper 负责转换后转回。任何注入失败都会抛错并以非零码退出，**不**回退基线。

结果 JSON 结构::

    {"tag", "model", "tasks", "limit", "device", "sgt_ckpt", "results": {...}, "meta": {...}}

``results`` 直接存 ``simple_evaluate(...)["results"]``（lm-eval 0.4.13 的键名为
``acc,none`` / ``acc_norm,none``）；``meta`` 记录版本、样本数、SGT 架构等。
所有非有限浮点（NaN/Inf）写为 ``null``，保证 JSON 可被严格解析。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import numbers
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from sgt.utility import install_sgt_embedding, load_sgt_for_lm  # noqa: E402

DEFAULT_TASKS = [
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "piqa",
    "truthfulqa_mc2",
    "ifeval",
    "mmlu",
    "hellaswag",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGT Utility 评测（lm-evaluation-harness，可选 SGT embedding 注入）"
    )
    parser.add_argument("--model", required=True, help="本地模型目录或 HF 模型 id")
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="逗号分隔的 lm-eval 任务名（默认论文的 8 个 benchmark）",
    )
    parser.add_argument("--limit", type=int, default=100, help="每个任务的样本数上限")
    parser.add_argument("--batch-size", type=int, default=4, help="harness batch size")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument(
        "--sgt-ckpt", default=None, help="SGT checkpoint（state_dict）；不给则跑基线"
    )
    parser.add_argument("--sgt-seed", type=int, default=0, help="SGT 采样种子")
    parser.add_argument("--sgt-hidden", type=int, default=512, help="SGT 隐藏维度")
    parser.add_argument("--sgt-layers", type=int, default=2, help="SGT encoder 层数")
    parser.add_argument("--sgt-heads", type=int, default=4, help="SGT 注意力头数")
    parser.add_argument(
        "--sgt-sigma-init", type=float, default=1.0, help="SGT 输出头 sigma 初始化"
    )
    parser.add_argument(
        "--tag", default=None, help="结果 tag（默认取 --out 文件名去掉 .json）"
    )
    parser.add_argument("--out", required=True, help="结果 JSON 路径")
    args = parser.parse_args(argv)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        parser.error("--tasks 不能为空")
    args.tasks = tasks
    if args.limit <= 0:
        parser.error("--limit 必须为正整数（任务单要求 limit 不得忽略）")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正整数")
    if args.sgt_ckpt is not None and not Path(args.sgt_ckpt).is_file():
        parser.error(f"--sgt-ckpt 不存在: {args.sgt_ckpt}")
    if args.sgt_seed < 0:
        parser.error("--sgt-seed 必须非负")
    return args


def _json_safe(obj):
    """转成严格 JSON 可序列化对象：非有限浮点 -> None，numpy 标量 -> Python 标量。"""
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_model_config(model: str) -> dict:
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "tie_word_embeddings")
    return {key: config[key] for key in keys if key in config}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        import lm_eval
        import transformers
        from lm_eval import simple_evaluate
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境错误
        print(
            f"[run_utility] ERROR: 当前解释器缺少依赖（{exc}）；请用 .venv-lm/bin/python 运行",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out)
    tag = args.tag if args.tag is not None else out_path.stem

    # 预加载 SGT：在跑 harness 之前就暴露 ckpt 损坏 / d 与模型不符等问题（fail fast）。
    sgt_template = None
    if args.sgt_ckpt is not None:
        try:
            sgt_template = load_sgt_for_lm(
                args.model,
                args.sgt_ckpt,
                device="cpu",
                hidden=args.sgt_hidden,
                n_layers=args.sgt_layers,
                n_heads=args.sgt_heads,
                sigma_init=args.sgt_sigma_init,
            )
            sgt_template.float()  # 统一 fp32；wrapper 会把 base 的 bf16/fp16 转换后转回
        except Exception as exc:
            print(f"[run_utility] ERROR: 加载 SGT checkpoint 失败: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    injection = {"count": 0, "info": None}
    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*fargs, **fkwargs):
        model = original_from_pretrained(*fargs, **fkwargs)
        if sgt_template is None:
            return model
        try:
            sgt = copy.deepcopy(sgt_template)
            install_sgt_embedding(model, sgt, seed=args.sgt_seed)
        except Exception as exc:
            # 必须显式失败：静默返回未注入的模型会污染 Utility 结论。
            raise RuntimeError(
                f"SGT 注入失败（ckpt={args.sgt_ckpt}）：{exc}"
            ) from exc
        parameter = next(sgt.parameters())
        injection["count"] += 1
        injection["info"] = {
            "d": sgt.d,
            "hidden": sgt.hidden,
            "n_layers": sgt.n_layers,
            "n_heads": sgt.n_heads,
            "max_len": sgt.max_len,
            "sigma_init": sgt.sigma_init,
            "sigma_min": sgt.sigma_min,
            "seed": args.sgt_seed,
            "dtype": str(parameter.dtype),
            "device_at_install": str(parameter.device),
        }
        return model

    if sgt_template is not None:
        transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained

    started = time.time()
    try:
        results = simple_evaluate(
            model="hf",
            model_args=f"pretrained={args.model}",
            tasks=args.tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception:
        print("[run_utility] ERROR: lm-eval 评测失败", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
    duration = time.time() - started

    if not isinstance(results, dict) or not isinstance(results.get("results"), dict):
        print("[run_utility] ERROR: simple_evaluate 未返回 results 字典", file=sys.stderr)
        return 1
    if args.sgt_ckpt is not None and injection["count"] == 0:
        print(
            "[run_utility] ERROR: 指定了 --sgt-ckpt，但 harness 未通过 "
            "AutoModelForCausalLM.from_pretrained 加载模型，SGT 实际未注入；"
            "拒绝输出基线结果",
            file=sys.stderr,
        )
        return 1

    payload = {
        "tag": tag,
        "model": args.model,
        "tasks": args.tasks,
        "limit": args.limit,
        "device": args.device,
        "sgt_ckpt": str(args.sgt_ckpt) if args.sgt_ckpt is not None else None,
        "results": results["results"],
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 3),
            "batch_size": args.batch_size,
            "argv": list(sys.argv),
            "lm_eval_version": getattr(lm_eval, "__version__", None),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "model_config": _read_model_config(args.model),
            "n_samples": results.get("n-samples"),
            "n_shot": results.get("n-shot"),
            "versions": results.get("versions"),
            "group_subtasks": results.get("group_subtasks"),
            "sgt": injection["info"],
        },
    }

    try:
        payload = _json_safe(payload)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        print(f"[run_utility] ERROR: 写结果 JSON 失败: {out_path}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(
        f"[run_utility] done: tag={tag} tasks={args.tasks} limit={args.limit} "
        f"device={args.device} sgt={'yes' if args.sgt_ckpt else 'no'} "
        f"duration={duration:.1f}s -> {out_path}"
    )
    if injection["info"] is not None:
        print(f"[run_utility] SGT: {injection['info']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
