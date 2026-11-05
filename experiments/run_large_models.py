#!/usr/bin/env python3
"""M9 大模型训练脚本（SPEC §5 D4；论文 arXiv:2506.09452v1 §7.1 Table 2）：配置回显 + 启动命令生成 + 本地 dry-run。

论文 Table 2（p.10 "Large Models Fully Trained"）设置（原文转录，见 ``configs/large_models.yaml``）::

    目标模型：meta-llama/Llama-3.3-70B-Instruct、Qwen/Qwen3-32B、
              deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    损失：full MI + AbsCos + Norm（= train_sgt 的 loss_terms ("mi","acs","mnp")）
    数据：完整 OpenOrca 训练 2 个 epoch；结果报在最多 0.5% 的验证子集上
    硬件：最多 64 节点 × 8 × A100-80G；FSDP2 × Tensor-Parallelism；最多 2 天

本仓库**无法执行**上述训练（SPEC D4：硬件不可得，Table 2 全部记 BLOCKED）。因此本脚本：

1. 读 yaml，按 ``--model`` 取模型配置（``repo_id``、``d``、``sgt_max_len``、并行默认值），
   原样回显到结果 JSON 的 ``config_echo``；
2. 生成 ``torchrun --nnodes=N --nproc-per-node=G -- ... --fsdp2 --tp-size S ...`` 形式的启动
   命令字符串，只写入 ``launch_command``，**不执行**；
3. ``--dry-run``：用 ``--dry-run-model``（默认 ``.models/tiny-llama``）真实跑 1 步
   ``train_sgt``（loss_terms 固定 ``("mi","acs","mnp")``、steps=1、batch=1、max_len=16），
   验证脚本端到端可运行；
4. 非 dry-run：``torch.cuda.device_count() < 8`` 或找不到 ``torchrun`` 时写
   ``status="blocked"``、``reason="no multi-GPU cluster"`` 并 exit 3 —— **绝不**静默降级到
   单卡跑 70B。

用法::

    ./.venv/bin/python experiments/run_large_models.py --config configs/large_models.yaml \\
        --model llama-3.3-70b --out results/large_llama-3.3-70b.json
    ./.venv/bin/python experiments/run_large_models.py --config configs/large_models.yaml \\
        --model llama-3.3-70b --dry-run --dry-run-model .models/tiny-llama \\
        --out results/large_dry_run.json

结果 JSON 顶层字段（顺序固定）：``model_key, repo_id, mode, status, launch_command,
world_size, config_echo, timestamp``；dry-run 另含 ``history, elapsed_seconds``（以及
``median_norm`` / ``d_model`` / ``train_config`` 等诊断字段）；blocked/ready 另含
``reason / status_detail / llm_loaded / env``。

退出码：0 成功（dry-run 完成，或启动命令就绪）；1 运行期失败（训练/写文件）；2 配置或参数
错误（含未知 ``--model``）；3 前置条件不可用（无多卡集群 → Table 2 BLOCKED）。

范围说明（重要）：``--worker`` 是 ``launch_command`` 指向的集群侧入口。本复现只交付启动
契约与配置（SPEC D4），**不含**经集群验证的 FSDP2×TP 训练体：worker 校验启动参数与环境后
显式以 exit 3 拒绝，不静默降级、也不假装训练。分布式训练体需在有集群的环境中实现与验证，
在此之前 Table 2 保持 BLOCKED。
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import shlex
import shutil
import sys
import time
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

from sgt.train import LOSS_TERMS  # noqa: E402

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_DRY_RUN_MODEL",
    "EXIT_BLOCKED",
    "EXIT_CONFIG",
    "EXIT_FAIL",
    "EXIT_OK",
    "MIN_GPUS_PER_NODE",
    "build_launch_command",
    "dry_run",
    "find_torchrun",
    "json_safe",
    "main",
    "parse_args",
    "probe_cluster",
    "read_yaml",
    "resolve_model",
    "resolve_parallel",
    "resolve_weights",
    "train_worker",
    "write_json",
]

#: 默认配置与 dry-run 模型（任务单 M9）。
DEFAULT_CONFIG = "configs/large_models.yaml"
DEFAULT_DRY_RUN_MODEL = ".models/tiny-llama"

#: 论文 Table 2 的硬性下限：每节点 8 张 A100-80G。
MIN_GPUS_PER_NODE = 8
DEFAULT_TP_SIZE = 8
DEFAULT_NODES = 64
DEFAULT_GPUS_PER_NODE = 8

#: dry-run 固定协议（任务单 M9 第 3 条）：1 步、batch=1、max_len=16、三个 loss 项。
DRY_RUN_STEPS = 1
DRY_RUN_BATCH_SIZE = 1
DRY_RUN_MAX_LEN = 16
DRY_RUN_LOSS_TERMS = ("mi", "acs", "mnp")

#: dry-run 用的本地合成语料（不联网、不读 data_cache）：只验证脚本端到端可运行，
#: 不用于任何数值结论，结果 JSON 里 ``data_source="synthetic"``。
DRY_RUN_TEXTS = (
    "The stained glass transform obfuscates embeddings before they reach the target model.",
    "Privacy risk is measured with nearest neighbour reconstruction failure rates.",
    "Mutual information between clean and obfuscated embeddings must stay bounded.",
    "We train on two epochs of OpenOrca and validate on a small holdout split.",
)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_BLOCKED = 3


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M9 大模型训练脚本：Table 2 配置回显 + torchrun 启动命令生成 + 本地 dry-run"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置路径")
    parser.add_argument("--model", required=True, help="models 下的 key（如 llama-3.3-70b）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="用 --dry-run-model 真实跑 1 步 train_sgt（本地可执行，不加载 70B）",
    )
    parser.add_argument(
        "--dry-run-model", default=DEFAULT_DRY_RUN_MODEL, help="dry-run 用的小模型目录"
    )
    parser.add_argument("--nodes", type=int, default=None, help="覆盖 parallel.nodes")
    parser.add_argument(
        "--gpus-per-node", type=int, default=None, help="覆盖 parallel.gpus_per_node"
    )
    parser.add_argument("--tp-size", type=int, default=None, help="覆盖 parallel.tp_size")
    parser.add_argument(
        "--out", default=None, help="结果 JSON（默认 results/large_<model_key>.json）"
    )
    # 以下为 launch_command 指向的内部入口参数（集群侧使用，本机不执行）。
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fsdp2", action="store_true", help="worker：启用 FSDP2 分片（论文 Table 2 设置）"
    )
    parser.add_argument("--epochs", type=int, default=None, help="worker：训练 epoch 数")
    args = parser.parse_args(argv)
    for name in ("nodes", "gpus_per_node", "tp_size", "epochs"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    return args


def _fail(message: str, code: int) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- #
# JSON / 文件工具（与 experiments/run_ablation.py 同口径）
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


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 的顶层必须是 mapping，实际 {type(raw).__name__}")
    return raw


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #
class UnknownModelKey(KeyError):
    """``--model`` 不在 yaml ``models`` 下。``args[0]`` 为可选 key 列表。"""


def resolve_model(raw: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    """取 ``models.<model_key>``；未知 key 抛 :class:`UnknownModelKey`（含可选值）。"""
    models = raw.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("配置缺少 models（mapping）")
    if model_key not in models:
        raise UnknownModelKey(sorted(str(key) for key in models))
    model_cfg = models[model_key] or {}
    if not isinstance(model_cfg, Mapping):
        raise ValueError(f"models.{model_key} 必须是 mapping")
    repo_id = str(model_cfg.get("repo_id") or "").strip()
    if not repo_id:
        raise ValueError(f"models.{model_key}.repo_id 不能为空")
    d = int(model_cfg.get("d", 0))
    if d <= 0:
        raise ValueError(f"models.{model_key}.d 必须为正整数，实际 {d}")
    return dict(model_cfg)


def resolve_parallel(
    raw: Mapping[str, Any],
    *,
    nodes: int | None = None,
    gpus_per_node: int | None = None,
    tp_size: int | None = None,
) -> dict[str, Any]:
    """并行设置：yaml ``parallel`` 为默认值，CLI 覆盖；返回含派生 world_size/dp_size。"""
    parallel = raw.get("parallel") or {}
    if not isinstance(parallel, Mapping):
        raise ValueError("parallel 必须是 mapping")
    resolved_nodes = int(parallel.get("nodes", DEFAULT_NODES) if nodes is None else nodes)
    resolved_gpus = int(
        parallel.get("gpus_per_node", DEFAULT_GPUS_PER_NODE)
        if gpus_per_node is None
        else gpus_per_node
    )
    resolved_tp = int(parallel.get("tp_size", DEFAULT_TP_SIZE) if tp_size is None else tp_size)
    fsdp2 = bool(parallel.get("fsdp2", True))
    for name, value in (
        ("nodes", resolved_nodes),
        ("gpus_per_node", resolved_gpus),
        ("tp_size", resolved_tp),
    ):
        if value <= 0:
            raise ValueError(f"parallel.{name} 必须为正整数，实际 {value}")
    world_size = resolved_nodes * resolved_gpus
    if world_size % resolved_tp != 0:
        raise ValueError(f"world_size={world_size} 必须能被 tp_size={resolved_tp} 整除")
    return {
        "fsdp2": fsdp2,
        "tp_size": resolved_tp,
        "nodes": resolved_nodes,
        "gpus_per_node": resolved_gpus,
        "world_size": world_size,
        "dp_size": world_size // resolved_tp,
        "tp_source": "yaml" if tp_size is None else "cli",
        "nodes_source": "yaml" if nodes is None else "cli",
        "gpus_source": "yaml" if gpus_per_node is None else "cli",
    }


def resolve_weights(
    raw: Mapping[str, Any], *, d: int, loss_terms: Sequence[str]
) -> tuple[dict[str, float], dict[str, Any]]:
    """α 权重：``train.alphas`` 优先；缺省 ``mi=1/d``、``acs=1.0``、``mnp=1.0``。

    α 是复现选择（SPEC A12，论文未给）。本配置沿用 ``configs/ablation.yaml`` 的
    ``mi=4.8828125e-4 = 1/2048``（ablation 的 d=2048）；大模型 d=8192 时 1/d 为
    ``1.220703125e-4``。脚本不回改配置值，只把实际使用值与偏差写进结果 JSON。
    """
    train = raw.get("train") or {}
    if not isinstance(train, Mapping):
        raise ValueError("train 必须是 mapping")
    defaults = {"mi": 1.0 / float(d), "acs": 1.0, "mnp": 1.0}
    alphas = train.get("alphas")
    configured = (
        {str(key): float(value) for key, value in alphas.items()}
        if isinstance(alphas, Mapping)
        else {}
    )
    weights = {
        term: float(configured.get(term, defaults.get(term, 1.0))) for term in loss_terms
    }
    echo = {
        "weights": weights,
        "defaults": defaults,
        "configured": configured,
        "source": "train.alphas" if configured else "default",
        "one_over_d": 1.0 / float(d),
        "mi_equals_one_over_d": bool(
            "mi" in weights and math.isclose(weights["mi"], 1.0 / float(d), rel_tol=1e-12)
        ),
        "note": "α 是复现选择（SPEC A12）；alphas 沿用 configs/ablation.yaml 的值",
    }
    return weights, echo


# --------------------------------------------------------------------------- #
# 启动命令 / 环境探测
# --------------------------------------------------------------------------- #
def _script_rel_path() -> str:
    """仓库内脚本路径（集群上以仓库根为 cwd 启动，故用相对路径）。"""
    path = Path(__file__).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_launch_command(
    *,
    config_path: str | Path,
    model_key: str,
    parallel: Mapping[str, Any],
    epochs: int,
) -> str:
    """生成 ``torchrun --nnodes=N --nproc-per-node=G -- <脚本> --worker ...`` 命令字符串。

    只生成、不执行。FSDP2 与 TP 作为显式 CLI 参数传给内部 ``--worker`` 入口；
    多节点弹性 rendezvous 用 c10d，``MASTER_ADDR/MASTER_PORT`` 由环境提供
    （默认 localhost:29500，单节点验证时无需设置）。
    """
    parts = [
        "torchrun",
        f"--nnodes={int(parallel['nodes'])}",
        f"--nproc-per-node={int(parallel['gpus_per_node'])}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}",
        "--",
        _script_rel_path(),
        "--worker",
        "--config",
        shlex.quote(str(config_path)),
        "--model",
        model_key,
    ]
    if parallel.get("fsdp2"):
        parts.append("--fsdp2")
    parts += [
        "--tp-size",
        str(int(parallel["tp_size"])),
        "--epochs",
        str(int(epochs)),
        "--nodes",
        str(int(parallel["nodes"])),
        "--gpus-per-node",
        str(int(parallel["gpus_per_node"])),
    ]
    return " ".join(parts)


def find_torchrun() -> str | None:
    """优先用当前解释器同目录的 torchrun（本仓库 .venv/bin/torchrun），否则查 PATH。

    注意：不能对 ``sys.executable`` 调 ``Path.resolve()`` —— venv 的 python 是符号链接，
    resolve 之后会落到基础解释器目录，从而漏掉 ``.venv/bin/torchrun``。
    """
    candidate = Path(sys.executable).absolute().with_name("torchrun")
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("torchrun")


def probe_cluster(*, gpus_per_node: int) -> dict[str, Any]:
    """探测「多卡集群」前置条件；``ok=False`` 时调用方必须 exit 3（不得降级到单卡）。"""
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    torchrun = find_torchrun()
    required = max(MIN_GPUS_PER_NODE, int(gpus_per_node))
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "required_gpus": required,
        "torchrun": torchrun,
        "torchrun_found": torchrun is not None,
        "ok": device_count >= required and torchrun is not None,
    }


# --------------------------------------------------------------------------- #
# 配置回显
# --------------------------------------------------------------------------- #
def build_config_echo(
    *,
    raw: Mapping[str, Any],
    config_path: str | Path,
    model_key: str,
    model_cfg: Mapping[str, Any],
    parallel: Mapping[str, Any],
    weights: Mapping[str, float],
    weights_echo: Mapping[str, Any],
    loss_terms: Sequence[str],
    sgt_max_len: int,
    n_layers_sgt: int,
    dry_run_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train = raw.get("train") or {}
    sgt = raw.get("sgt") or {}
    repo_id = str(model_cfg["repo_id"])
    d = int(model_cfg["d"])
    return {
        "yaml_path": str(config_path),
        "raw_yaml": json_safe(raw),
        "model_key": model_key,
        "model": json_safe(dict(model_cfg)),
        "resolved": {
            "repo_id": repo_id,
            "d": d,
            "n_layers_sgt": int(n_layers_sgt),
            "sgt_max_len": int(sgt_max_len),
            "world_size": int(parallel["world_size"]),
            "dp_size": int(parallel["dp_size"]),
            "tp_size": int(parallel["tp_size"]),
            "nodes": int(parallel["nodes"]),
            "gpus_per_node": int(parallel["gpus_per_node"]),
        },
        "sgt": json_safe(dict(sgt)),
        "train": {
            "epochs": int(train.get("epochs", 2)),
            "dataset": str(train.get("dataset", "Open-Orca/OpenOrca")),
            "frac": float(train.get("frac", 1.0)),
            "validation_fraction": float(train.get("validation_fraction", 0.005)),
            "loss_terms": list(loss_terms),
            "alphas": dict(weights),
        },
        "parallel": dict(parallel),
        "weights": dict(weights),
        "weights_echo": dict(weights_echo),
        "paper": {
            "table": "Table 2 (arXiv:2506.09452v1 p.10)",
            "models": [
                "meta-llama/Llama-3.3-70B-Instruct",
                "Qwen/Qwen3-32B",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            ],
            "loss": "MI+AbsCos+Norm",
            "epochs": 2,
            "dataset": "Open-Orca/OpenOrca (full)",
            "validation_fraction": 0.005,
            "hardware": "up to 64 nodes x 8 x A100-80G",
            "parallel": "FSDP2 x Tensor-Parallelism",
            "max_train_time": "up to 2 days",
            "note": "论文值；本仓库不可执行（SPEC D4），Table 2 记 BLOCKED",
        },
        "dry_run_override": json_safe(dict(dry_run_override)) if dry_run_override else None,
    }


# --------------------------------------------------------------------------- #
# dry-run：用 tiny 模型真实跑 1 步 train_sgt
# --------------------------------------------------------------------------- #
def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad) if pad is not None else 0


def _pad_to_width(
    batches: Sequence[Mapping[str, torch.Tensor]], width: int, pad_id: int
) -> list[dict[str, torch.Tensor]]:
    """把每个 batch 右 padding 到固定宽度（mask=0，不改变有效 token）。

    MI 项默认 ``mode="pair"``，要求 x̃ 与 x′ 同形；``make_batches`` 的 T 随 batch 内
    最长序列变化，故 dry-run 统一补齐到 ``max_len``（与 run_ablation 的 A17 右 padding
    口径一致）。SGTModel 的 ``max_len`` 必须 ≥ 该宽度。
    """
    fixed: list[dict[str, torch.Tensor]] = []
    for batch in batches:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        missing = int(width) - int(ids.shape[1])
        if missing < 0:
            raise ValueError(f"batch 宽度 {ids.shape[1]} 超过 max_len={width}")
        if missing > 0:
            ids = torch.cat(
                [ids, torch.full((ids.shape[0], missing), int(pad_id), dtype=ids.dtype)], dim=1
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], missing), dtype=mask.dtype)], dim=1
            )
        fixed.append({"input_ids": ids, "attention_mask": mask})
    return fixed


def dry_run(
    *,
    model_path: str | Path,
    raw: Mapping[str, Any],
    weights: Mapping[str, float],
    loss_terms: Sequence[str],
    seed: int = 0,
) -> dict[str, Any]:
    """加载 tiny LM，跑 1 步 ``train_sgt``（steps=1、batch=1、max_len=16）。

    数据为本地合成文本（``data_source="synthetic"``），不联网、不读 data_cache；
    结果只用于证明脚本端到端可运行，**不**产生任何与 Table 2 可比的数值。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sgt.data import make_batches
    from sgt.model import SGTModel
    from sgt.train import TrainConfig, train_sgt

    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"dry-run 模型目录不存在: {model_path}")
    train_raw = raw.get("train") or {}
    sgt_raw = raw.get("sgt") or {}
    if not isinstance(train_raw, Mapping) or not isinstance(sgt_raw, Mapping):
        raise ValueError("train / sgt 必须是 mapping")
    if not loss_terms:
        raise ValueError("loss_terms 不能为空")
    unknown = [term for term in loss_terms if term not in LOSS_TERMS]
    if unknown:
        raise ValueError(f"未知 loss_terms {unknown}；允许 {list(LOSS_TERMS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    lm = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.float32)
    lm = lm.to(device)
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)

    d = int(getattr(lm.config, "hidden_size", 0))
    if d <= 0:
        raise ValueError(f"{model_path} 的 hidden_size 无效: {d}")
    pad_id = _pad_id(tokenizer)
    raw_batches = list(
        make_batches(
            list(DRY_RUN_TEXTS),
            tokenizer,
            batch_size=DRY_RUN_BATCH_SIZE,
            max_len=DRY_RUN_MAX_LEN,
            seed=int(seed),
            dataset_name=None,
        )
    )
    if not raw_batches:
        raise ValueError("make_batches 没有产出任何 batch")
    batches = _pad_to_width(raw_batches, DRY_RUN_MAX_LEN, pad_id)

    torch.manual_seed(int(seed))  # SGT 初始化由 seed 决定（训练内部采样另有显式 Generator）
    sgt = SGTModel(
        d,
        hidden=int(sgt_raw.get("hidden", 512)),
        n_layers=int(sgt_raw.get("n_layers", 2)),
        n_heads=int(sgt_raw.get("n_heads", 4)),
        ffn_mult=int(sgt_raw.get("ffn_mult", 4)),
        max_len=DRY_RUN_MAX_LEN,
        sigma_init=float(sgt_raw.get("sigma_init", 1.0)),
        sigma_min=float(train_raw.get("sigma_min", 1e-6)),
    ).to(device=device, dtype=torch.float32)

    config = TrainConfig(
        steps=DRY_RUN_STEPS,
        lr=float(train_raw.get("lr", 1e-4)),
        weight_decay=float(train_raw.get("weight_decay", 0.0)),
        grad_clip=float(train_raw.get("grad_clip", 1.0)),
        seed=int(seed),
        device=str(device),
        dtype="float32",
        loss_terms=tuple(loss_terms),
        weights=dict(weights),
        log_every=1,
        sigma_min=float(train_raw.get("sigma_min", 1e-6)),
        temperature=float(train_raw.get("temperature", 1.0)),
        checkpoint_dir=None,
    )
    result = train_sgt(
        sgt,
        lm,
        batches,
        config=config,
        embedding_table=lm.get_input_embeddings().weight.detach(),
    )
    history = list(result["history"])
    for record in history:
        if not math.isfinite(record["loss"]):
            raise ValueError(f"dry-run loss 非有限: step={record['step']} loss={record['loss']}")
    return {
        "history": history,
        "median_norm": float(result["median_norm"]),
        "completed_steps": int(result["completed_steps"]),
        "device": str(device),
        "d_model": d,
        "n_batches": len(batches),
        "batch_shape": list(batches[0]["input_ids"].shape),
        "max_len": DRY_RUN_MAX_LEN,
        "batch_size": DRY_RUN_BATCH_SIZE,
        "loss_terms": list(loss_terms),
        "weights": dict(weights),
        "data_source": "synthetic",
        "dry_run_model": str(model_path),
        "train_config": json_safe(asdict(config)),
    }


# --------------------------------------------------------------------------- #
# 结果 JSON
# --------------------------------------------------------------------------- #
def _base_payload(
    *,
    model_key: str,
    repo_id: str,
    mode: str,
    status: str,
    launch_command: str,
    world_size: int,
    config_echo: Mapping[str, Any],
    llm_loaded: bool,
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "repo_id": repo_id,
        "mode": mode,
        "status": status,
        "launch_command": launch_command,
        "world_size": int(world_size),
        "config_echo": config_echo,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_loaded": bool(llm_loaded),
    }


def _env_block() -> dict[str, Any]:
    torchrun = find_torchrun()
    return {
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torchrun": torchrun,
        "argv": list(sys.argv),
    }


# --------------------------------------------------------------------------- #
# worker：launch_command 指向的集群侧入口
# --------------------------------------------------------------------------- #
def train_worker(args: argparse.Namespace) -> int:
    """``launch_command`` 的 ``--worker`` 入口（集群侧）。

    本复现只交付启动契约与配置（SPEC D4：硬件不可得，Table 2 记 BLOCKED），不含经集群
    验证的 FSDP2×TP 训练体。这里校验启动参数/环境后**显式 exit 3**：不静默降级到单卡，
    也不假装训练完成。参数解析与 ``build_launch_command`` 保持逐项对应，便于后续替换
    训练体时命令无需改动。
    """
    try:
        raw = read_yaml(args.config)
        model_cfg = resolve_model(raw, args.model)
        parallel = resolve_parallel(
            raw, nodes=args.nodes, gpus_per_node=args.gpus_per_node, tp_size=args.tp_size
        )
    except UnknownModelKey as exc:
        available = ", ".join(exc.args[0])
        return _fail(f"未知 --model {args.model!r}；可选值: {available}", EXIT_CONFIG)
    except Exception as exc:
        return _fail(f"worker 配置不合法: {type(exc).__name__}: {exc}", EXIT_CONFIG)

    if not args.fsdp2:
        return _fail("论文 Table 2 要求 FSDP2，但 worker 未收到 --fsdp2", EXIT_CONFIG)
    world_size = int(os.environ.get("WORLD_SIZE", "0") or 0)
    if world_size and world_size != int(parallel["world_size"]):
        return _fail(
            f"torchrun WORLD_SIZE={world_size} 与配置 world_size={parallel['world_size']} 不一致"
            f"（nodes={parallel['nodes']} × gpus_per_node={parallel['gpus_per_node']}）",
            EXIT_CONFIG,
        )
    try:
        import torch.distributed as dist  # noqa: F401

        distributed_available = bool(dist.is_available())
    except Exception:  # pragma: no cover - torch 无分布式支持时
        distributed_available = False
    gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if not distributed_available or gpu_count < MIN_GPUS_PER_NODE:
        return _fail(
            "worker 需要 torch.distributed + 每节点 ≥"
            f"{MIN_GPUS_PER_NODE} 张 GPU（实际 cuda_device_count={gpu_count}）",
            EXIT_BLOCKED,
        )
    print(
        "[worker] BLOCKED: 本复现未交付 FSDP2×TP 分布式训练体（SPEC D4：仅脚本/配置，"
        "Table 2 记 BLOCKED）。启动契约校验通过："
        f"model={args.model}（{model_cfg['repo_id']}） "
        f"world_size={parallel['world_size']} "
        f"dp={parallel['dp_size']} tp={parallel['tp_size']} epochs={args.epochs}。"
        "训练体需在有集群的环境中实现并验证后才能运行；worker 不静默降级、不假装训练。",
        file=sys.stderr,
    )
    return EXIT_BLOCKED


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return train_worker(args)
    start = time.perf_counter()

    # ---- 1. 配置 ---------------------------------------------------------- #
    try:
        raw = read_yaml(args.config)
    except Exception as exc:
        return _fail(f"读取配置 {args.config} 失败: {type(exc).__name__}: {exc}", EXIT_CONFIG)
    try:
        model_cfg = resolve_model(raw, args.model)
    except UnknownModelKey as exc:
        available = ", ".join(exc.args[0])
        return _fail(f"未知 --model {args.model!r}；可选值: {available}", EXIT_CONFIG)
    except Exception as exc:
        return _fail(f"配置不合法: {type(exc).__name__}: {exc}", EXIT_CONFIG)

    train_raw = raw.get("train") or {}
    sgt_raw = raw.get("sgt") or {}
    if not isinstance(train_raw, Mapping) or not isinstance(sgt_raw, Mapping):
        return _fail("train / sgt 必须是 mapping", EXIT_CONFIG)
    try:
        parallel = resolve_parallel(
            raw, nodes=args.nodes, gpus_per_node=args.gpus_per_node, tp_size=args.tp_size
        )
    except Exception as exc:
        return _fail(f"并行配置不合法: {type(exc).__name__}: {exc}", EXIT_CONFIG)

    d = int(model_cfg["d"])
    repo_id = str(model_cfg["repo_id"])
    loss_terms = tuple(str(term) for term in (train_raw.get("loss_terms") or DRY_RUN_LOSS_TERMS))
    if not loss_terms:
        return _fail("train.loss_terms 不能为空", EXIT_CONFIG)
    unknown_terms = [term for term in loss_terms if term not in LOSS_TERMS]
    if unknown_terms:
        return _fail(f"未知 loss_terms {unknown_terms}；允许 {list(LOSS_TERMS)}", EXIT_CONFIG)
    try:
        weights, weights_echo = resolve_weights(raw, d=d, loss_terms=loss_terms)
    except Exception as exc:
        return _fail(f"α 权重不合法: {type(exc).__name__}: {exc}", EXIT_CONFIG)

    epochs = int(train_raw.get("epochs", 2))
    if epochs <= 0:
        return _fail(f"train.epochs 必须为正整数，实际 {epochs}", EXIT_CONFIG)
    sgt_max_len = int(model_cfg.get("sgt_max_len", sgt_raw.get("max_len", 4096)))
    n_layers_sgt = int(model_cfg.get("n_layers_sgt", sgt_raw.get("n_layers", 2)))

    try:
        launch_command = build_launch_command(
            config_path=args.config,
            model_key=args.model,
            parallel=parallel,
            epochs=epochs,
        )
    except Exception as exc:
        return _fail(f"生成启动命令失败: {type(exc).__name__}: {exc}", EXIT_CONFIG)

    out_path = Path(args.out or f"results/large_{args.model}.json")
    dry_override = None
    if args.dry_run:
        dry_override = {
            "model_path": str(args.dry_run_model),
            "steps": DRY_RUN_STEPS,
            "batch_size": DRY_RUN_BATCH_SIZE,
            "max_len": DRY_RUN_MAX_LEN,
            "loss_terms": list(DRY_RUN_LOSS_TERMS),
            "data_source": "synthetic",
        }
    config_echo = build_config_echo(
        raw=raw,
        config_path=args.config,
        model_key=args.model,
        model_cfg=model_cfg,
        parallel=parallel,
        weights=weights,
        weights_echo=weights_echo,
        loss_terms=loss_terms,
        sgt_max_len=sgt_max_len,
        n_layers_sgt=n_layers_sgt,
        dry_run_override=dry_override,
    )

    # ---- 2. dry-run：真实跑 1 步 ------------------------------------------ #
    if args.dry_run:
        payload = _base_payload(
            model_key=args.model,
            repo_id=repo_id,
            mode="dry_run",
            status="ok",
            launch_command=launch_command,
            world_size=parallel["world_size"],
            config_echo=config_echo,
            llm_loaded=True,
        )
        payload["env"] = _env_block()
        try:
            result = dry_run(
                model_path=args.dry_run_model,
                raw=raw,
                weights=weights,
                loss_terms=DRY_RUN_LOSS_TERMS,
            )
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            try:
                write_json(out_path, payload)
            except Exception:
                pass
            return _fail(f"dry-run 失败: {type(exc).__name__}: {exc}", EXIT_FAIL)
        payload.update(result)
        payload["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        try:
            write_json(out_path, payload)
        except Exception as exc:
            return _fail(f"写结果 JSON 失败 {out_path}: {type(exc).__name__}: {exc}", EXIT_FAIL)
        record = payload["history"][0]
        print(
            f"[large] model={args.model} mode=dry_run steps={len(payload['history'])} "
            f"loss={record['loss']:.6f} loss_u={record['loss_u']:.6f} "
            f"d_model={payload['d_model']} device={payload['device']} "
            f"elapsed_s={payload['elapsed_seconds']:.1f} -> {out_path}"
        )
        return EXIT_OK

    # ---- 3. 非 dry-run：只生成命令，不执行 -------------------------------- #
    probe = probe_cluster(gpus_per_node=parallel["gpus_per_node"])
    blocked = not probe["ok"]
    payload = _base_payload(
        model_key=args.model,
        repo_id=repo_id,
        mode="launch",
        status="blocked" if blocked else "ready",
        launch_command=launch_command,
        world_size=parallel["world_size"],
        config_echo=config_echo,
        llm_loaded=False,
    )
    payload["env"] = {**_env_block(), **probe}
    payload["notes"] = [
        "launch_command 只生成、不执行：需要运维在集群上手动运行（见模块 docstring）。",
        "worker 入口（--worker）目前只校验启动契约：本复现未交付 FSDP2×TP 训练体（SPEC D4）。",
        "Table 2 数值在本仓库记为 BLOCKED；本 JSON 不含任何训练结果。",
    ]
    if blocked:
        payload["reason"] = "no multi-GPU cluster"
        payload["status_detail"] = (
            f"torch.cuda.device_count()={probe['cuda_device_count']} "
            f"(需要 ≥{probe['required_gpus']})，torchrun={probe['torchrun']!r}"
        )
    payload["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    try:
        write_json(out_path, payload)
    except Exception as exc:
        return _fail(f"写结果 JSON 失败 {out_path}: {type(exc).__name__}: {exc}", EXIT_FAIL)

    if blocked:
        print(
            f"[large] model={args.model} status=blocked reason=no multi-GPU cluster "
            f"(cuda_device_count={probe['cuda_device_count']}, "
            f"torchrun={'found' if probe['torchrun_found'] else 'missing'}) -> {out_path}",
            file=sys.stderr,
        )
        return EXIT_BLOCKED

    print(
        f"[large] model={args.model} status=ready world_size={payload['world_size']} "
        f"launch_command written -> {out_path}"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
