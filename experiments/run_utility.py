#!/usr/bin/env python3
"""M8a Utility 评测入口：lm-evaluation-harness（论文 8 benchmark）+ 可选 SGT 注入。

用法（lm venv，仓库根目录）::

    .venv-lm/bin/python experiments/run_utility.py \
      --model .models/llama-3.2-1b-instruct \
      --config configs/utility.yaml --limit 100 --batch-size 4 --device mps \
      [--sgt-ckpt results/ckpt_abscos_best.pt --sgt-seed 0] \
      --out results/utility_abscos.json

    # 不用配置文件时等价的 CLI（GAP-6：HellaSwag 10-shot、其余 0-shot）：
    .venv-lm/bin/python experiments/run_utility.py \
      --model .models/llama-3.2-1b-instruct \
      --tasks arc_challenge,arc_easy,openbookqa,piqa,truthfulqa_mc2,ifeval,mmlu,hellaswag \
      --task-fewshot "hellaswag=10" --num-fewshot 0 --limit 100 \
      --max-len-fallback skip --out results/utility_abscos.json

注入方式：在 ``simple_evaluate`` 之前 monkeypatch
``transformers.AutoModelForCausalLM.from_pretrained``，对 harness 加载出的模型调用
``install_sgt_embedding``（替换输入 embedding 层）。SGT 以 fp32 计算，base 模型可以是
bf16；wrapper 负责转换后转回。任何注入失败都会抛错并以非零码退出，**不**回退基线。

GAP-6（论文 §7 / CONFORMANCE CF-31）：

- ``--num-fewshot``：全局 shot 数（默认 0，可被 ``--config`` 覆盖）。
- ``--task-fewshot "hellaswag=10,mmlu=0"``：按任务覆盖 shot 数。实现为**按 shot 值分组**
  调用 ``simple_evaluate(..., num_fewshot=k)``，再合并结果；合并后的 ``results`` 键结构与
  单次调用完全一致（另合并 groups / group_subtasks / configs / versions / n-shot /
  higher_is_better / n-samples）。
- ``--max-len-fallback {error,skip}``：SGT 的 ``max_len`` 小于输入序列长度时，
  ``error`` 直接非零退出（旧行为）；``skip`` 把该任务记为
  ``{"status": "skipped_too_long", "max_len": ...}`` 写进 JSON 而不中断其它任务。
  ``skip`` 模式下分组调用若因超长失败，会退化为逐任务调用以定位超长任务。
  长度守卫只对 ``pos_encoding="learned"`` 安装：``sinusoidal`` / ``none`` 是 A1 的长度
  泛化选择，前向对任意 T 有效，``max_len`` 只是元数据，不能据此判超长（否则 HellaSwag
  10-shot 等长 prompt 会被误跳过）；实际是否安装记在 ``meta.max_len_guard``。
- ``meta.num_fewshot`` 记录每个任务**实际请求**的 shot；lm-eval 自己的 ``n-shot`` 原样保留。

结果 JSON 结构::

    {"tag", "model", "tasks", "limit", "device", "sgt_ckpt", "config_path",
     "results": {...}, "meta": {...}}

``results`` 存合并后的 ``simple_evaluate(...)["results"]``（lm-eval 0.4.13 的键名为
``acc,none`` / ``acc_norm,none``；group 任务如 ``mmlu`` 另含 group 聚合项）；超长被跳过的
任务以 ``skipped_too_long`` 条目出现。``meta`` 记录版本、样本数、每任务 shot、SGT 架构、
SGT 实际选项（``meta.sgt_options``：``pos_encoding`` / ``final_norm`` / ``mu_scale`` /
``sigma_floor`` / ``sigma_max``）以及长度守卫是否安装（``meta.max_len_guard``，只有
``pos_encoding="learned"`` 才安装）。
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
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from sgt.utility import install_sgt_embedding, load_sgt_for_lm  # noqa: E402

#: 论文 §7（CF-31）的 8 个 benchmark（顺序即 Table 1 Utility 列的任务集）。
DEFAULT_TASKS = [
    "arc_challenge",
    "arc_easy",
    "openbookqa",
    "piqa",
    "truthfulqa_mc2",
    "ifeval",
    "mmlu",
    "hellaswag",
]
DEFAULT_LIMIT = 100
DEFAULT_NUM_FEWSHOT = 0
MAX_LEN_FALLBACK_CHOICES = ("error", "skip")

#: ``merge_eval_results`` 会逐键合并的 dict 型顶层键（与 lm-eval 0.4.13 的输出一致）。
_MERGE_DICT_KEYS = (
    "results",
    "groups",
    "group_subtasks",
    "configs",
    "versions",
    "n-shot",
    "higher_is_better",
    "n-samples",
)
#: 其余非 dict 型顶层键（环境信息）：取第一个出现的值，不参与合并。
_PASSTHROUGH_KEYS = ("git_hash", "date", "config")


# --------------------------------------------------------------------------- #
# GAP-6：异常类型与 fewshot / 配置解析
# --------------------------------------------------------------------------- #
class SGTMaxLenExceeded(RuntimeError):
    """SGT 输入序列长度超过 ``max_len``（``--max-len-fallback`` 的判据）。

    由 :func:`_install_max_len_guard` 在 SGT embedding wrapper 入口抛出；``error`` 模式
    直接失败，``skip`` 模式把对应任务记为 ``skipped_too_long``。
    """

    def __init__(self, message: str, *, max_len: int, observed_len: int) -> None:
        super().__init__(message)
        self.max_len = int(max_len)
        self.observed_len = int(observed_len)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
        raise ValueError(f"{name} 必须为正整数，实际 {value!r}")
    return int(value)


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise ValueError(f"{name} 必须为非负整数，实际 {value!r}")
    return int(value)


def parse_task_fewshot(spec: str | None) -> dict[str, int]:
    """解析 ``"hellaswag=10,mmlu=0"`` -> ``{"hellaswag": 10, "mmlu": 0}``。"""
    if spec is None:
        return {}
    text = str(spec).strip()
    if not text:
        return {}
    parsed: dict[str, int] = {}
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f'--task-fewshot 片段 {piece!r} 缺少 "="（格式 "task=shot,..."）')
        name, value = piece.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"--task-fewshot 片段 {piece!r} 的任务名为空")
        try:
            shot = int(value)
        except ValueError as exc:
            raise ValueError(f"--task-fewshot 的 {name!r} shot 数不是整数: {value!r}") from exc
        if shot < 0:
            raise ValueError(f"--task-fewshot 的 {name!r} shot 数必须非负，实际 {shot}")
        if name in parsed and parsed[name] != shot:
            raise ValueError(f"--task-fewshot 对 {name!r} 给了冲突的 shot 数")
        parsed[name] = shot
    return parsed


def load_utility_config(path: str | Path) -> dict[str, Any]:
    """读取 ``configs/utility.yaml``：论文 8 任务 + per-task fewshot + limit。"""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 顶层必须是 mapping，实际 {type(raw).__name__}")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"{path} 缺少非空的 tasks mapping")
    default_shot = _non_negative_int(raw.get("num_fewshot", DEFAULT_NUM_FEWSHOT), "num_fewshot")
    normalized: dict[str, dict[str, int]] = {}
    for name, spec in tasks.items():
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            raise ValueError(f"{path} 的 tasks.{name} 必须是 mapping，实际 {type(spec).__name__}")
        shot = _non_negative_int(
            spec.get("num_fewshot", default_shot), f"tasks.{name}.num_fewshot"
        )
        normalized[str(name)] = {"num_fewshot": shot}
    limit = _positive_int(raw.get("limit", DEFAULT_LIMIT), "limit")
    fallback = str(raw.get("max_len_fallback", "error"))
    if fallback not in MAX_LEN_FALLBACK_CHOICES:
        raise ValueError(
            f"{path} 的 max_len_fallback 必须是 {MAX_LEN_FALLBACK_CHOICES}，实际 {fallback!r}"
        )
    return {
        "path": str(path),
        "tasks": normalized,
        "task_names": list(normalized),
        "limit": limit,
        "num_fewshot": default_shot,
        "max_len_fallback": fallback,
        "raw": raw,
    }


def resolve_task_fewshot(
    tasks: Sequence[str],
    *,
    global_num_fewshot: int = DEFAULT_NUM_FEWSHOT,
    config_shots: Mapping[str, int] | None = None,
    cli_shots: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """逐任务 shot 数：CLI 覆盖 > 配置文件 > 全局值。"""
    tasks = list(tasks)
    config_shots = dict(config_shots or {})
    cli_shots = dict(cli_shots or {})
    unknown = sorted(set(cli_shots) - set(tasks))
    if unknown:
        raise ValueError(
            f"--task-fewshot 含未在任务列表中的任务 {unknown}；任务列表 {tasks}"
        )
    global_num_fewshot = _non_negative_int(global_num_fewshot, "num_fewshot")
    shots: dict[str, int] = {}
    for task in tasks:
        if task in cli_shots:
            shots[task] = _non_negative_int(cli_shots[task], f"task-fewshot[{task}]")
        elif task in config_shots:
            shots[task] = _non_negative_int(config_shots[task], f"config tasks.{task}")
        else:
            shots[task] = global_num_fewshot
    return shots


def group_by_fewshot(
    tasks: Sequence[str], shots: Mapping[str, int]
) -> list[tuple[int, list[str]]]:
    """按 shot 值分组（保持任务列表里首次出现的顺序，确定性）。"""
    groups: dict[int, list[str]] = {}
    for task in tasks:
        shot = int(shots[task])
        groups.setdefault(shot, []).append(task)
    return list(groups.items())


def is_too_long_error(exc: BaseException) -> bool:
    """是否为「输入长度超过 SGT max_len」。识别本脚本的哨兵与 SGTModel 的原始报错。"""
    if isinstance(exc, SGTMaxLenExceeded):
        return True
    return isinstance(exc, ValueError) and "超过 max_len" in str(exc)


# --------------------------------------------------------------------------- #
# GAP-6：SGT 长度守卫 + 结果合并 + 分组评测
# --------------------------------------------------------------------------- #
#: 受 ``max_len`` 约束的 ``pos_encoding``：只有可学习绝对位置编码有 T 上限；
#: ``sinusoidal`` / ``none`` 按序列长度动态计算（A1 长度泛化），对任意 T 有效。
_GUARDED_POS_ENCODINGS = frozenset({"learned"})
#: :class:`~sgt.model.SGTModel` 支持的位置编码（非法值必须报错，不能静默不装守卫）。
_KNOWN_POS_ENCODINGS = frozenset({"learned", "sinusoidal", "none"})


def _install_max_len_guard(wrapper, max_len: int, pos_encoding: str = "learned"):
    """在已安装的 SGT embedding wrapper 入口加长度检查，超限抛 :class:`SGTMaxLenExceeded`。

    守卫只对 ``pos_encoding="learned"`` 安装：可学习绝对位置编码只有 ``max_len`` 行，
    ``T > max_len`` 时 :class:`~sgt.model.SGTModel` 本身就会失败。``sinusoidal`` / ``none``
    是 A1 的长度泛化选择，前向对任意 T 有效（``max_len`` 只是元数据），此时**不安装守卫**
    ——否则 HellaSwag 10-shot（prompt 最长 ~971 token）这类合法长输入会被误判成超长。

    守卫只在实例上包一层 ``forward``（不改 ``src/sgt/utility.py``，该模块保持原语义）；
    T 不超过 ``max_len`` 时与原实现逐位相同，且在进入 SGT 前就失败（避免超长 prompt 白算
    一次）。

    Args:
        wrapper: ``install_sgt_embedding`` 返回的 embedding wrapper。
        max_len: SGT 的 ``max_len``（``pos_encoding != "learned"`` 时不再校验为正整数，
            因为它只是元数据）。
        pos_encoding: SGT 实际使用的位置编码（``learned`` / ``sinusoidal`` / ``none``）。
            默认 ``"learned"`` 保持旧行为（不传时按需要守卫处理）。

    Returns:
        ``(wrapper, meta)``；``meta`` 为 ``{"installed": bool, "pos_encoding": ...}``，
        安装时另有 ``"max_len"``，未安装时另有 ``"reason"``（写进结果 JSON 的
        ``meta.max_len_guard``）。未安装时 wrapper 原样返回、``forward`` 不被包装。
    """
    if not hasattr(wrapper, "forward"):
        raise TypeError(f"wrapper 必须有 forward，实际 {type(wrapper)!r}")
    pos_encoding = str(pos_encoding)
    if pos_encoding not in _KNOWN_POS_ENCODINGS:
        raise ValueError(
            f"pos_encoding 必须为 {sorted(_KNOWN_POS_ENCODINGS)} 之一，实际 {pos_encoding!r}"
        )
    if pos_encoding not in _GUARDED_POS_ENCODINGS:
        # 长度无关的位置编码：T 任意有效，安装守卫会误伤合法长 prompt。
        return wrapper, {
            "installed": False,
            "pos_encoding": pos_encoding,
            "reason": f"pos_encoding={pos_encoding}",
        }
    max_len = _positive_int(max_len, "sgt.max_len")
    original_forward = wrapper.forward

    def guarded_forward(ids, *fargs, **fkwargs):
        tensor = torch.as_tensor(ids)
        if tensor.dim() != 2:
            raise ValueError(f"ids 必须为 [B, T]，实际 {tuple(tensor.shape)}")
        length = int(tensor.shape[1])
        if length > max_len:
            raise SGTMaxLenExceeded(
                f"输入序列长度 T={length} 超过 SGT max_len={max_len}",
                max_len=max_len,
                observed_len=length,
            )
        return original_forward(ids, *fargs, **fkwargs)

    wrapper.forward = guarded_forward
    return wrapper, {
        "installed": True,
        "pos_encoding": pos_encoding,
        "max_len": max_len,
    }


def merge_eval_results(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """合并多次 ``simple_evaluate`` 的返回，键结构与单次调用完全一致。

    逐键合并 dict 型顶层键；同一键在不同分组里出现且内容不同时**报错**（正常不会发生：
    每个任务只属于一个 shot 分组，group 任务（如 ``mmlu``）也不会被拆开）。非 dict 的
    环境信息（``git_hash`` / ``date`` / ``config``）取第一个出现的值。
    """
    if not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError("merge_eval_results 需要非空的 simple_evaluate 结果列表")
    merged: dict[str, Any] = {}
    for key in _MERGE_DICT_KEYS:
        collected: dict[str, Any] = {}
        present = False
        for index, part in enumerate(parts):
            if key not in part:
                continue
            present = True
            value = part.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise TypeError(
                    f"第 {index} 个结果的 {key} 必须是 dict，实际 {type(value).__name__}"
                )
            for name, entry in value.items():
                if name in collected and repr(collected[name]) != repr(entry):
                    raise ValueError(
                        f"合并 simple_evaluate 结果冲突：{key}.{name} 在不同 shot 分组里不同；"
                        "同一任务只能属于一个 shot 分组（检查 --task-fewshot 是否拆开了 group 任务）"
                    )
                collected[name] = entry
        # 单次调用里存在的（可能为空的）键，合并结果里也必须存在，保持键结构逐项一致；
        # ``groups`` 这类只有存在 group 任务时才出现的键，所有分组都没有时就不补空键。
        if present:
            merged[key] = collected
    for key in _PASSTHROUGH_KEYS:
        for part in parts:
            if part.get(key) is not None:
                merged[key] = part[key]
                break
    return merged


def evaluate_tasks(
    evaluate_fn: Callable[..., Mapping[str, Any]],
    *,
    base_kwargs: Mapping[str, Any],
    tasks: Sequence[str],
    shots: Mapping[str, int],
    max_len_fallback: str = "error",
    sgt_max_len: int | None = None,
) -> dict[str, Any]:
    """按 shot 分组调用 ``evaluate_fn``（即 ``lm_eval.simple_evaluate``）并合并结果。

    Args:
        evaluate_fn: ``simple_evaluate``（测试里可注入 mock）。
        base_kwargs: 每次调用共享的参数（model / model_args / limit / batch_size / device）。
        tasks: 任务名列表（决定结果顺序）。
        shots: 每任务 shot 数（:func:`resolve_task_fewshot`）。
        max_len_fallback: ``error`` 或 ``skip``。
        sgt_max_len: SGT 的 ``max_len``（写进 skipped 条目；无 SGT 时为 None）。

    Returns:
        ``{"merged": 合并后的 simple_evaluate 结果, "calls": [{tasks,num_fewshot}, ...],
        "skipped": {task: {"status": "skipped_too_long", "max_len": ...}}}``。
        ``merged["results"]`` 里被跳过的任务即上述 skipped 条目。
    """
    if max_len_fallback not in MAX_LEN_FALLBACK_CHOICES:
        raise ValueError(
            f"max_len_fallback 必须是 {MAX_LEN_FALLBACK_CHOICES}，实际 {max_len_fallback!r}"
        )
    tasks = list(tasks)
    calls: list[dict[str, Any]] = []
    skipped: dict[str, dict[str, Any]] = {}
    parts: list[Mapping[str, Any]] = []

    def _invoke(group_tasks: Sequence[str], shot: int) -> Mapping[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs["tasks"] = list(group_tasks)
        kwargs["num_fewshot"] = int(shot)
        # 先记录再调用：失败的尝试也留在 calls 里（诊断超长任务时看得到试过什么）。
        calls.append({"tasks": list(group_tasks), "num_fewshot": int(shot)})
        part = evaluate_fn(**kwargs)
        if not isinstance(part, Mapping) or not isinstance(part.get("results"), Mapping):
            raise RuntimeError("simple_evaluate 未返回 results 字典")
        return part

    groups = group_by_fewshot(tasks, shots)
    for shot, group_tasks in groups:
        if max_len_fallback == "error":
            parts.append(_invoke(group_tasks, shot))
            continue
        # skip：先整组调用（少调几次）；若因超长失败，退化为逐任务以定位并跳过超长任务。
        try:
            parts.append(_invoke(group_tasks, shot))
            continue
        except Exception as exc:
            if not is_too_long_error(exc):
                raise
        for task in group_tasks:
            try:
                parts.append(_invoke([task], shot))
            except Exception as exc:
                if not is_too_long_error(exc):
                    raise
                skipped[task] = {
                    "status": "skipped_too_long",
                    "max_len": None if sgt_max_len is None else int(sgt_max_len),
                    "observed_error": str(exc),
                }

    if parts:
        merged = merge_eval_results(parts)
    else:  # 全部任务都因超长被跳过：仍返回与单次调用同构的空骨架
        merged = {key: {} for key in _MERGE_DICT_KEYS if key != "groups"}
    merged["results"] = {**dict(merged.get("results") or {}), **skipped}
    return {"merged": merged, "calls": calls, "skipped": skipped}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGT Utility 评测（lm-evaluation-harness，可选 SGT embedding 注入）"
    )
    parser.add_argument("--model", required=True, help="本地模型目录或 HF 模型 id")
    parser.add_argument(
        "--config",
        default=None,
        help="utility 配置（如 configs/utility.yaml：论文 8 任务 + per-task fewshot）",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="逗号分隔的 lm-eval 任务名（默认取 --config 的 tasks，否则论文 8 任务）",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="每个任务的样本数上限（默认取配置/100）"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="全局 num_fewshot（默认取配置/0；GAP-6）",
    )
    parser.add_argument(
        "--task-fewshot",
        default=None,
        help='按任务覆盖 shot 数，如 "hellaswag=10,mmlu=0"（GAP-6）',
    )
    parser.add_argument(
        "--max-len-fallback",
        choices=MAX_LEN_FALLBACK_CHOICES,
        default=None,
        help="输入超过 SGT max_len 时的行为：error=直接失败（默认），skip=记为 skipped_too_long",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="harness batch size")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument(
        "--sgt-ckpt", default=None, help="SGT checkpoint（state_dict；建议用 *_best.pt）"
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

    if args.tasks is not None:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        if not tasks:
            parser.error("--tasks 不能为空")
        args.tasks = tasks
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正整数（任务单要求 limit 不得忽略）")
    if args.num_fewshot is not None and args.num_fewshot < 0:
        parser.error("--num-fewshot 必须非负")
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

    # ---- GAP-6：配置 + 每任务 shot 解析 ----------------------------------- #
    config: dict[str, Any] | None = None
    if args.config is not None:
        try:
            config = load_utility_config(args.config)
        except Exception as exc:
            print(
                f"[run_utility] ERROR: 读取配置 {args.config} 失败: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
    tasks = list(args.tasks) if args.tasks is not None else (
        list(config["task_names"]) if config is not None else list(DEFAULT_TASKS)
    )
    if not tasks:
        print("[run_utility] ERROR: 任务列表为空", file=sys.stderr)
        return 2
    limit = args.limit if args.limit is not None else (
        int(config["limit"]) if config is not None else DEFAULT_LIMIT
    )
    fallback = args.max_len_fallback if args.max_len_fallback is not None else (
        str(config["max_len_fallback"]) if config is not None else "error"
    )
    global_num_fewshot = args.num_fewshot if args.num_fewshot is not None else (
        int(config["num_fewshot"]) if config is not None else DEFAULT_NUM_FEWSHOT
    )
    try:
        cli_shots = parse_task_fewshot(args.task_fewshot)
        config_shots = (
            {name: spec["num_fewshot"] for name, spec in config["tasks"].items()}
            if config is not None
            else {}
        )
        shots = resolve_task_fewshot(
            tasks,
            global_num_fewshot=global_num_fewshot,
            config_shots=config_shots,
            cli_shots=cli_shots,
        )
    except (TypeError, ValueError) as exc:
        print(f"[run_utility] ERROR: fewshot 配置不合法: {exc}", file=sys.stderr)
        return 2

    # ---- SGT 预加载：在跑 harness 之前暴露 ckpt 损坏 / d 与模型不符等问题 --- #
    sgt_template = None
    sgt_max_len: int | None = None
    sgt_meta: dict[str, Any] | None = None
    sgt_options: dict[str, Any] | None = None
    if args.sgt_ckpt is not None:
        try:
            sgt_template, sgt_meta = load_sgt_for_lm(
                args.model,
                args.sgt_ckpt,
                device="cpu",
                hidden=args.sgt_hidden,
                n_layers=args.sgt_layers,
                n_heads=args.sgt_heads,
                sigma_init=args.sgt_sigma_init,
                return_meta=True,
            )
            sgt_template.float()  # 统一 fp32；wrapper 会把 base 的 bf16/fp16 转换后转回
            sgt_max_len = int(sgt_template.max_len)
            # 从加载器 meta 取实际生效的选项（无参数但改变前向行为，必须可追溯）。
            sgt_options = {
                key: sgt_meta[key]
                for key in ("pos_encoding", "final_norm", "mu_scale", "sigma_floor", "sigma_max")
            }
            sgt_options["pos_encoding_source"] = sgt_meta["pos_encoding_source"]
            sgt_options["inferred_unparameterized_options"] = sgt_meta[
                "inferred_unparameterized_options"
            ]
        except Exception as exc:
            print(f"[run_utility] ERROR: 加载 SGT checkpoint 失败: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    injection = {"count": 0, "info": None, "guard": None}
    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*fargs, **fkwargs):
        model = original_from_pretrained(*fargs, **fkwargs)
        if sgt_template is None:
            return model
        try:
            sgt = copy.deepcopy(sgt_template)
            wrapper = install_sgt_embedding(model, sgt, seed=args.sgt_seed)
            # GAP-6：learned 位置编码的超长输入在进入 SGT 前抛 SGTMaxLenExceeded
            # （error/skip 共用同一判据）；sinusoidal/none 长度无关，不安装守卫。
            wrapper, guard_meta = _install_max_len_guard(
                wrapper, sgt.max_len, sgt.pos_encoding
            )
            injection["guard"] = guard_meta
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
        outcome = evaluate_tasks(
            simple_evaluate,
            base_kwargs={
                "model": "hf",
                "model_args": f"pretrained={args.model}",
                "limit": limit,
                "batch_size": args.batch_size,
                "device": args.device,
            },
            tasks=tasks,
            shots=shots,
            max_len_fallback=fallback,
            sgt_max_len=sgt_max_len,
        )
    except SGTMaxLenExceeded as exc:
        print(
            f"[run_utility] ERROR: {exc}；--max-len-fallback=error 直接失败"
            "（可用 --max-len-fallback skip 把该任务记为 skipped_too_long）",
            file=sys.stderr,
        )
        return 1
    except Exception:
        print("[run_utility] ERROR: lm-eval 评测失败", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
    duration = time.time() - started

    merged = outcome["merged"]
    skipped = outcome["skipped"]
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
        "tasks": tasks,
        "limit": limit,
        "device": args.device,
        "sgt_ckpt": str(args.sgt_ckpt) if args.sgt_ckpt is not None else None,
        "config_path": None if args.config is None else str(args.config),
        "results": merged.get("results", {}),
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 3),
            "batch_size": args.batch_size,
            "argv": list(sys.argv),
            "config_path": None if args.config is None else str(args.config),
            "lm_eval_version": getattr(lm_eval, "__version__", None),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "model_config": _read_model_config(args.model),
            # GAP-6：每任务**实际请求**的 shot；lm-eval 自己的 n-shot 原样保留。
            "num_fewshot": shots,
            "n_shot": merged.get("n-shot"),
            "shot_groups": [
                {"num_fewshot": shot, "tasks": group}
                for shot, group in group_by_fewshot(tasks, shots)
            ],
            "n_calls": len(outcome["calls"]),
            "calls": outcome["calls"],
            "max_len_fallback": fallback,
            "sgt_max_len": sgt_max_len,
            # SGT 实际生效的选项（来自加载器 meta）与长度守卫是否安装（learned 才装）。
            "sgt_options": sgt_options,
            "max_len_guard": injection["guard"],
            "skipped": skipped,
            "n_samples": merged.get("n-samples"),
            "versions": merged.get("versions"),
            "group_subtasks": merged.get("group_subtasks"),
            "configs": merged.get("configs"),
            "higher_is_better": merged.get("higher_is_better"),
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

    skipped_names = sorted(skipped)
    print(
        f"[run_utility] done: tag={tag} tasks={tasks} limit={limit} "
        f"fewshot={shots} calls={len(outcome['calls'])} "
        f"skipped={skipped_names} fallback={fallback} "
        f"device={args.device} sgt={'yes' if args.sgt_ckpt else 'no'} "
        f"duration={duration:.1f}s -> {out_path}"
    )
    if injection["info"] is not None:
        print(f"[run_utility] SGT: {injection['info']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
