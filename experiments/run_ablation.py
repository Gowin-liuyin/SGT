#!/usr/bin/env python3
"""M7 消融实验（SPEC §5；论文 §7.1 Table 1）：训练一个 SGT variant，在 Alpaca 上评估。

用法::

    ./.venv/bin/python experiments/run_ablation.py --config configs/ablation.yaml \
        --variant mi_abscos_norm --device mps --max-steps 300 --max-examples 200 \
        --eval-examples 200 --eval-max-len 64 --eval-tokens 2000 \
        --out results/ablation_mi_abscos_norm.json

流程（任务单 M7-A + 审计 GAP-4 / GAP-5 / GAP-7）：

1. 读 yaml：``variants.<variant>.loss_terms``、``sgt:`` 超参、``d:``、``train:``、
   ``selection:``、``eval:``（A12 复现选择）。
2. 训练数据：优先本地 OpenOrca parquet（``--local-parquet``，默认
   ``.models/openorca_1M.parquet``，取前 N 条，记录文件 sha256）；文件不存在或给了
   ``--force-rows-api`` 时回退 ``sgt.data.load_openorca``（datasets-server rows API，
   走 ``data_cache/`` 缓存）。``data_source`` 如实写入 JSON（``local_parquet`` /
   ``rows_api``），不静默换数据集。
3. 冻结目标 LM（默认 ``.models/llama-3.2-1b-instruct``；``eval()`` +
   ``requires_grad_(False)``）。LM 侧精度由 ``train.lm_dtype`` 决定（GAP-5，默认 fp32；
   SGT **恒为 fp32**）。请求非 fp32 时先用微模型跑 1 步 ``train_sgt`` 探测当前训练器的
   混合精度支持（A22/GAP-8 契约），不可用则**显式**退回 fp32 并把原因写入 JSON。
4. ``make_batches(..., dataset_name="openorca")`` 后 **drop_last**（丢弃不足一个
   batch 的尾批，保证 MI ``mode="pair"`` 两侧同形）；另把各 batch 右 padding 到同一 T
   （理由见 :func:`_pad_to_common_width`）。
5. ``train_sgt``（``TrainConfig`` 全部来自 yaml；α 权重来自 yaml ``alphas:``，
   缺省 mi=1/d、acs=1、mnp=1，见 :func:`resolve_weights` 与 SPEC A12）。
   GAP-7：``selection.enabled`` 时把 ``selection.every`` 映射到 ``train_sgt`` 的
   ``checkpoint_dir``/``log_every``，训练结束后逐 checkpoint 算 SymTTR-100 并选最优。
   评测发生在训练之后、直接读 ``train_sgt`` 落盘的同一步 checkpoint（与「每 N 步在线
   评测」看到的权重逐位相同），因此**完全不触碰**训练 RNG 与优化器状态（同 seed 可复现）。
6. 保存最后一个 checkpoint ``results/ckpt_<variant>.pt``：
   ``{"sgt": state_dict, "config": ..., "seed": ...}``；若 SymTTR 最优步不是最后一步，
   另存 ``results/ckpt_<variant>_best.pt``（下游 Utility 必须用这个 checkpoint）。
7. 评测集切分（两者互不重叠）：
   - GAP-7：Alpaca **前** ``selection.calibration_examples`` 条只用作 SymTTR 校准集；
   - GAP-4：其余 Alpaca 条 tokenize 成评测集后，再按 ``eval.calibration_fraction``
     （默认 0.25）切成 calibration / test：MRP 的 ``r*`` **只**从 calibration 的秩直方图
     取众数（``--mrp-r`` 可显式覆盖），NN/MRP/TTR/SymTTR/Hist.Entropy 只在 test 上算。
     calibration 不足 ``MIN_MRP_CALIBRATION_TOKENS`` 个 token 时直接报错（退出码 2）。
8. 最终评测用 SymTTR 最优 checkpoint（``--no-select-by-symttr`` 时用最后一个），算
   Table 1 指标 + A7 特征级 MI，定义见 :func:`rank_metrics` / :func:`feature_mi`。
   MI 在 σ 过小（``log_sigma2 < log(1e-12)``）或结果非有限时**不崩溃**：返回
   ``mi_status="undefined_sigma_too_small"`` 的占位块（``mi_trigger`` 区分
   ``sigma_below_floor`` / ``nonfinite_term``），MI/PAC 记为 ``null``，并由 ``mi_gate``
   显式标注「本次 run 的 MI/PAC 判定不可用」。
9. 写 JSON（严格可序列化、NaN/Inf 记为 ``null``），含 ``selection`` 块与
   ``mrp_r_star_source`` / ``n_calibration_tokens`` / ``n_test_tokens``，并打印一行
   ``[ablation]`` 汇总。
10. ``--skip-eval``（训练探针）：训练 + checkpoint 后跳过上述评测（不加载 Alpaca、不跑
   SymTTR 选择、不算 MI/PAC），仍写出完整 JSON（``eval`` 块为 ``{"status": "skipped"}``，
   含 ``train_history`` / ``selection`` / ``timings``）并返回 0。

退出码：0 成功；2 配置/数据/模型等前置条件不可用（含 MRP 校准集过小）；1 训练或评估本身失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    "mrp_calibration_split",
    "symttr_on_sequences",
    "plan_selection_steps",
    "build_selection",
    "selected_step",
    "best_checkpoint_path",
    "resolve_selection",
    "resolve_eval_options",
    "resolve_sgt_options",
    "build_sgt_model",
    "resolve_lm_dtype",
    "probe_mixed_precision",
    "MI_LOG_SIGMA2_FLOOR",
    "MI_STATUS_OK",
    "MI_STATUS_SIGMA_TOO_SMALL",
]

#: 目标 LM 与训练数据的本仓库默认路径（任务单 M7-A）。
DEFAULT_MODEL = ".models/llama-3.2-1b-instruct"
DEFAULT_LOCAL_PARQUET = ".models/openorca_1M.parquet"
OPENORCA_FIELDS = ("id", "system_prompt", "question", "response")

#: 距离/排序分块预算：``[chunk, V]`` 的 float32 距离 + int64 排序索引 ≈ 16 B/元素。
_DIST_BUDGET_BYTES = 256 * 1024 * 1024

#: GAP-7：论文 §7 "select the best performer based on SymTTR" 的缺省选择节奏与校准条数。
DEFAULT_SELECTION_EVERY = 50
DEFAULT_CALIBRATION_EXAMPLES = 50
#: GAP-4：MRP 的 r* 校准集占评测 token 的比例（SPEC A8；论文未给具体比例）。
DEFAULT_CALIBRATION_FRACTION = 0.25
#: GAP-4：calibration 低于该 token 数时直接报错，禁止静默改用 test 自身求 r*。
MIN_MRP_CALIBRATION_TOKENS = 100
#: GAP-7：论文的模型选择指标是 SymTTR-100（Table 1 第 9 列）。
SELECTION_METRIC = "SymTTR-100"
SELECTION_K = 100
#: GAP-5：论文 §7 训练协议（CONFORMANCE CF-29）：OpenOrca 15% = 635,089 条 / 全量 4,233,927。
PAPER_OPENORCA_FRAC = 0.15
PAPER_OPENORCA_TRAIN_ROWS = 4233927
PAPER_N_SELECTED_REFERENCE = 635089
#: LM 侧可请求的训练精度（GAP-5/A22）；SGT 恒为 float32。
LM_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
DEFAULT_LM_DTYPE = "float32"

#: 位置编码初始化标准差（SPEC A1 实现选择；论文未给初始化）。与
#: :class:`sgt.model.SGTModel` 的默认值一致；yaml ``sgt.pos_emb_init_std`` 可覆盖
#: （``1.0`` = nn.Embedding 原生 N(0,1) 旧行为，仅用于对照）。
DEFAULT_POS_EMB_INIT_STD = 0.02

#: A7 数值门（健壮性）：σ² 低于 1e-12（σ < 1e-6，即 log σ² < log(1e-12) ≈ −27.63）时
#: 1/σ² 已失去数值意义——float32 下 ``exp(-log σ²)`` 会先溢出成 inf、再让结果变 NaN。
#: 触发时 **不抛错**：MI/PAC 记 null，``mi_status`` 标注原因，门禁据此跳过 MI/PAC 判定。
#: 注意 σ→0 本身是 ACS / cge 等目标可能学到的退化（见 SPEC A21：只给了上界）。
MI_LOG_SIGMA2_FLOOR = math.log(1e-12)
MI_STATUS_OK = "ok"
MI_STATUS_SIGMA_TOO_SMALL = "undefined_sigma_too_small"

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
    parser.add_argument(
        "--select-by-symttr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"按 {SELECTION_METRIC} 选 checkpoint（GAP-7；默认开启，"
        "--no-select-by-symttr 用最后一个 checkpoint）",
    )
    parser.add_argument(
        "--select-every",
        type=int,
        default=None,
        help=f"每多少步算一次 {SELECTION_METRIC}（默认取 yaml selection.every，"
        f"缺省 {DEFAULT_SELECTION_EVERY}）",
    )
    parser.add_argument(
        "--calibration-examples",
        type=int,
        default=None,
        help="SymTTR 校准用 Alpaca 前 N 条（默认取 yaml selection.calibration_examples，"
        f"缺省 {DEFAULT_CALIBRATION_EXAMPLES}）",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=None,
        help="GAP-4：MRP 的 r* 校准集占评测 token 的比例（默认取 yaml "
        f"eval.calibration_fraction，缺省 {DEFAULT_CALIBRATION_FRACTION}）",
    )
    parser.add_argument(
        "--mrp-r",
        type=int,
        default=None,
        help="GAP-4：显式覆盖 MRP 的 r*（1-based；不给则从 calibration 秩直方图取众数）",
    )
    parser.add_argument(
        "--lm-dtype",
        choices=("auto", *LM_DTYPES),
        default="auto",
        help="LM 侧精度；auto = 取 yaml train.lm_dtype（非 fp32 时先探测混合精度）",
    )
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
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="训练探针：训练 + checkpoint 后跳过评测/MI/PAC 与 SymTTR 选择，"
        "仍写出完整 JSON（eval 块为 {status: skipped}）并返回 0",
    )
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
    for name in ("select_every", "calibration_examples", "mrp_r"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.calibration_fraction is not None and not (
        math.isfinite(args.calibration_fraction) and 0.0 <= args.calibration_fraction < 1.0
    ):
        parser.error("--calibration-fraction 必须在 [0, 1) 内")
    return args


def _fail(message: str, code: int = 2) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def _fmt_or_null(value: float | None, digits: int = 4) -> str:
    """日志格式化：MI/PAC 在 ``mi_status != "ok"`` 时是 None，不能直接套 ``:.4f``。"""
    return "null" if value is None else f"{float(value):.{digits}f}"


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
    lm_dtype: str = DEFAULT_LM_DTYPE,
    checkpoint_dir: str | None = None,
    log_every: int | None = None,
) -> TrainConfig:
    """构造 ``TrainConfig``；``lm_dtype`` / ``checkpoint_dir`` / ``log_every`` 可被调用方
    覆盖（GAP-5 混合精度、GAP-7 选择节奏）。``TrainConfig.dtype`` 是**记录字段**：
    这里记录 LM 侧精度，SGT 恒为 float32（A22）。"""
    train = raw.get("train") or {}
    if not isinstance(train, Mapping):
        raise ValueError("train 必须是 mapping")
    steps = int(train.get("steps", 100)) if max_steps is None else int(max_steps)
    grad_clip = train.get("grad_clip", 1.0)
    log_every_value = int(train.get("log_every", 1)) if log_every is None else int(log_every)
    checkpoint_value = train.get("checkpoint_dir") if checkpoint_dir is None else checkpoint_dir
    if log_every_value < 1:
        raise ValueError(f"log_every 必须 ≥ 1，实际 {log_every_value}")
    return TrainConfig(
        steps=steps,
        lr=float(train.get("lr", 1e-4)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        grad_clip=None if grad_clip is None else float(grad_clip),
        seed=int(seed),
        device=device,
        dtype=str(lm_dtype),
        loss_terms=tuple(loss_terms),
        weights=dict(weights),
        log_every=log_every_value,
        sigma_min=float(train.get("sigma_min", 1e-6)),
        temperature=float(train.get("temperature", 1.0)),
        checkpoint_dir=checkpoint_value,
    )


def resolve_sgt_options(sgt_raw: Mapping[str, Any]) -> dict[str, Any]:
    """解析 ``sgt:`` 段里 :class:`SGTModel` 的 4 个结构超参（A1 / A21）。

    全部按 yaml 原值透传，缺省回落到 :class:`SGTModel` 的默认值：
    ``final_norm=True``、``mu_scale=None``、``sigma_floor=None``、``pos_encoding="learned"``。
    ``pos_encoding`` 的取值校验（``learned|sinusoidal|none``）交给 ``SGTModel``，
    ``mu_scale`` / ``sigma_floor`` 的数值校验同样由模型负责（不在脚本里重复实现）。
    """
    return {
        "final_norm": sgt_raw.get("final_norm", True),
        "mu_scale": sgt_raw.get("mu_scale"),
        "sigma_floor": sgt_raw.get("sigma_floor"),
        "pos_encoding": sgt_raw.get("pos_encoding", "learned"),
    }


def build_sgt_model(
    d: int,
    sgt_raw: Mapping[str, Any],
    *,
    sigma_min: float = 1e-6,
    sigma_max: float | None = None,
    pos_emb_init_std: float = DEFAULT_POS_EMB_INIT_STD,
) -> SGTModel:
    """按 yaml ``sgt:`` 段构造 :class:`SGTModel`（唯一构造入口，便于测试捕获 kwargs）。

    ``sigma_min`` / ``sigma_max`` / ``pos_emb_init_std`` 由调用方传入（分别来自
    ``TrainConfig.sigma_min``、``sgt.sigma_max`` 与 ``sgt.pos_emb_init_std`` 的解析结果）；
    A1 / A21 的 4 个结构超参见 :func:`resolve_sgt_options`。
    """
    return SGTModel(
        d,
        hidden=int(sgt_raw.get("hidden", 512)),
        n_layers=int(sgt_raw.get("n_layers", 2)),
        n_heads=int(sgt_raw.get("n_heads", 4)),
        ffn_mult=int(sgt_raw.get("ffn_mult", 4)),
        max_len=int(sgt_raw.get("max_len", 512)),
        sigma_init=float(sgt_raw.get("sigma_init", 1.0)),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        pos_emb_init_std=pos_emb_init_std,
        **resolve_sgt_options(sgt_raw),
    )


# --------------------------------------------------------------------------- #
# GAP-4 / GAP-5 / GAP-7 选项解析
# --------------------------------------------------------------------------- #
def resolve_selection(
    raw: Mapping[str, Any],
    *,
    enabled: bool | None = None,
    every: int | None = None,
    calibration_examples: int | None = None,
) -> dict[str, Any]:
    """解析 ``selection:``（GAP-7）：CLI 覆盖 yaml，再退回任务单缺省（50 / 50 / 开启）。

    ``enabled`` 的语义是「CLI 未关闭 **且** yaml 未关闭」：``--no-select-by-symttr`` 或
    ``selection.enabled: false`` 任一为假即整块关闭。
    """
    cfg = raw.get("selection") or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("selection 必须是 mapping")
    enabled_value = bool(cfg.get("enabled", True)) and (True if enabled is None else bool(enabled))
    every_value = int(cfg.get("every", DEFAULT_SELECTION_EVERY)) if every is None else int(every)
    calibration_value = (
        int(cfg.get("calibration_examples", DEFAULT_CALIBRATION_EXAMPLES))
        if calibration_examples is None
        else int(calibration_examples)
    )
    if every_value < 1:
        raise ValueError(f"selection.every 必须为正整数，实际 {every_value}")
    if calibration_value < 1:
        raise ValueError(f"selection.calibration_examples 必须为正整数，实际 {calibration_value}")
    return {
        "enabled": bool(enabled_value),
        "every": every_value,
        "calibration_examples": calibration_value,
        "metric": SELECTION_METRIC,
        "defaults": {
            "every": DEFAULT_SELECTION_EVERY,
            "calibration_examples": DEFAULT_CALIBRATION_EXAMPLES,
        },
        "config": dict(cfg),
    }


def resolve_eval_options(
    raw: Mapping[str, Any],
    *,
    calibration_fraction: float | None = None,
    mrp_r: int | None = None,
) -> dict[str, Any]:
    """解析 ``eval.calibration_fraction``（GAP-4）与 ``--mrp-r`` 覆盖。"""
    cfg = raw.get("eval") or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("eval 必须是 mapping")
    fraction = (
        float(cfg.get("calibration_fraction", DEFAULT_CALIBRATION_FRACTION))
        if calibration_fraction is None
        else float(calibration_fraction)
    )
    if not (math.isfinite(fraction) and 0.0 <= fraction < 1.0):
        raise ValueError(f"eval.calibration_fraction 必须在 [0, 1) 内，实际 {fraction!r}")
    if mrp_r is not None and int(mrp_r) < 1:
        raise ValueError(f"--mrp-r 必须 ≥ 1（1-based），实际 {mrp_r}")
    return {
        "calibration_fraction": fraction,
        "mrp_r": None if mrp_r is None else int(mrp_r),
        "min_calibration_tokens": MIN_MRP_CALIBRATION_TOKENS,
        "source": "yaml eval.calibration_fraction" if calibration_fraction is None else "CLI",
    }


def resolve_lm_dtype(
    requested: str | None,
    *,
    probe: Callable[[], tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    """决定 LM 侧精度：请求 fp32 直接用；否则先探测混合精度，不可用则**显式**退回 fp32。

    ``probe`` 返回 ``(是否支持, 原因)``；只在请求非 fp32 时调用（由 :func:`probe_mixed_precision`
    传入）。返回块会原样写入结果 JSON，便于审计「配置要 bf16、实际跑了什么」。
    """
    name = DEFAULT_LM_DTYPE if requested is None else str(requested)
    if name not in LM_DTYPES:
        raise ValueError(f"未知 lm_dtype {name!r}；允许 {sorted(LM_DTYPES)}")
    if name == "float32":
        return {
            "requested": name,
            "effective": "float32",
            "mixed_precision": False,
            "reason": "请求 fp32：LM 与 SGT 同精度（无需混合精度探测）",
        }
    if probe is None:
        raise ValueError(f"lm_dtype={name} 需要微模型探测混合精度，但未提供 probe")
    supported, reason = probe()
    if supported:
        return {
            "requested": name,
            "effective": name,
            "mixed_precision": True,
            "reason": reason,
        }
    return {
        "requested": name,
        "effective": "float32",
        "mixed_precision": False,
        "reason": f"退回 fp32（显式记录，不静默）：{reason}",
    }


def probe_mixed_precision(
    device: torch.device,
    *,
    lm_dtype: torch.dtype,
    loss_terms: Sequence[str],
    weights: Mapping[str, float],
) -> tuple[bool, str]:
    """用微模型跑 1 步 ``train_sgt``，验证「LM bf16/fp16 + SGT fp32」在本机能跑通。

    这是 GAP-5 的**前置门**：训练侧混合精度（A22/GAP-8）由另一 worker 提供，探测失败时
    主流程显式退回 fp32 并把原因写进 JSON（绝不带着未知精度静默训练）。探测用与本次
    variant 相同的 ``loss_terms``/``weights``，覆盖实际会走的代码路径。
    """
    tiny = None
    tiny_sgt = None
    try:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)  # 探测不污染调用方：探测后主流程会重新 manual_seed(seed)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
        )
        tiny = LlamaForCausalLM(config).to(device=device, dtype=lm_dtype).eval()
        tiny_sgt = SGTModel(
            8, hidden=8, n_layers=1, n_heads=2, ffn_mult=2, max_len=8
        ).to(device=device, dtype=torch.float32)
        ids = torch.randint(0, 32, (1, 8), device=device)
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
        }
        probe_config = TrainConfig(
            steps=1,
            lr=1e-3,
            seed=0,
            device=str(device),
            dtype=str(lm_dtype),
            loss_terms=tuple(loss_terms),
            weights=dict(weights),
        )
        train_sgt(tiny_sgt, tiny, [batch], config=probe_config)
        return True, f"微模型 1 步 train_sgt 通过（LM {lm_dtype} + SGT float32）"
    except Exception as exc:  # 探测失败只影响精度选择，不改变训练语义
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        del tiny, tiny_sgt
        if device.type == "mps":
            torch.mps.empty_cache()


# --------------------------------------------------------------------------- #
# GAP-7：按 SymTTR 选 checkpoint
# --------------------------------------------------------------------------- #
def plan_selection_steps(steps: int, every: int) -> list[int]:
    """返回要评测 SymTTR 的「已完成步数」列表：``every, 2·every, …``，并保证含末步。

    ``steps == 0`` 时返回空列表（没有可选的 checkpoint）。末步不是 ``every`` 的整数倍时
    补一个末步，使「最后一个 checkpoint」与「选择轨迹」口径一致。
    """
    steps = int(steps)
    every = int(every)
    if every < 1:
        raise ValueError(f"select_every 必须为正整数，实际 {every}")
    if steps < 0:
        raise ValueError(f"steps 必须非负，实际 {steps}")
    if steps == 0:
        return []
    plan = list(range(every, steps + 1, every))
    if not plan or plan[-1] != steps:
        plan.append(steps)
    return plan


def build_selection(
    evaluate: Callable[[int], float],
    *,
    steps: int,
    every: int,
    enabled: bool = True,
) -> dict[str, Any]:
    """按 :func:`plan_selection_steps` 逐点评测 SymTTR-100，返回论文要求的选择块。

    ``evaluate(step) -> float`` 由调用方注入（真实流程里是「载入该步 checkpoint → 在
    Alpaca 校准集上算 SymTTR-100」，测试里是合成序列）。并列时取**较晚**的步（``>=``
    比较）：SymTTR 在退化情形（所有 rank=1）下全部为 0，此时不应把训练步数截断到第一个
    采样点；非有限值被跳过（JSON 里经 :func:`json_safe` 记为 null）。
    """
    if not enabled:
        return {"enabled": False, "checkpoint": "last"}
    trace: list[dict[str, Any]] = []
    best_step: int | None = None
    best_value = -math.inf
    for step in plan_selection_steps(steps, every):
        value = float(evaluate(step))
        trace.append({"step": int(step), "symttr_100": value})
        if math.isfinite(value) and value >= best_value:
            best_step, best_value = int(step), value
    if best_step is None:
        raise ValueError(
            f"没有任何步的 {SELECTION_METRIC} 是有限值，无法按论文准则选择 checkpoint"
        )
    return {
        "enabled": True,
        "checkpoint": "symttr-best",
        "metric": SELECTION_METRIC,
        "every": int(every),
        "trace": trace,
        "best_step": int(best_step),
        "best_symttr_100": float(best_value),
    }


def selected_step(selection: Mapping[str, Any], *, steps: int) -> int:
    """最终评测用哪一步：SymTTR 最优步；选择关闭时用最后一个 checkpoint（``steps``）。"""
    if not selection.get("enabled"):
        return int(steps)
    return int(selection["best_step"])


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
    n_tokens: int | None,
) -> tuple[list[list[int]], int]:
    """按序列 tokenize，取前 ``n_tokens`` 个有效 token（最后一条按需截断）。

    ``n_tokens=None`` 表示不设 token 上限（GAP-7 的 SymTTR 校准集要按同一 ``max_len``
    截断全部 calibration 条）；``n_tokens`` 为正整数时行为与旧版一致。
    """
    from sgt.data import tokenize_examples

    if int(max_len) <= 0:
        raise ValueError(f"max_len 必须为正整数，实际 {max_len}")
    limit = None if n_tokens is None else int(n_tokens)
    if limit is not None and limit <= 0:
        raise ValueError(f"n_tokens 必须为正整数或 None，实际 {n_tokens}")
    encoded = tokenize_examples(
        list(examples), tokenizer, max_len=int(max_len), dataset="alpaca"
    )
    sequences: list[list[int]] = []
    total = 0
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        valid = [int(token) for token, keep in zip(ids, mask) if keep]
        if not valid:
            continue
        take = len(valid) if limit is None else min(len(valid), limit - total)
        if take <= 0:
            break
        sequences.append(valid[:take])
        total += take
        if limit is not None and total >= limit:
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
            ids, mask = _pad_sequences_batch(chunk, pad_id=pad_id, device=device)
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


def _pad_sequences_batch(
    sequences: Sequence[Sequence[int]], *, pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """把一批长度不等的序列右 padding 成 ``ids [B,T]`` + ``mask [B,T]``（True=有效）。"""
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = True
    return ids.to(device), mask.to(device)


def symttr_on_sequences(
    sgt,
    embedding_table: torch.Tensor,
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    batch_seqs: int,
    seed: int,
    rank_chunk: int,
    k: int = SELECTION_K,
) -> dict[str, Any]:
    """在给定序列上算 SymTTR-k（GAP-7 的 checkpoint 选择指标，论文 Table 1 的 SymTTR-100）。

    只保留「秩计数」，不收集特征张量（校准集可达上万 token，收集 ``x̃/µ/σ`` 会浪费显存）。
    采样 Generator 用同一 ``seed`` 重新播种，因此**每个 checkpoint 共用同一噪声流**，
    选择结果只反映权重差异，且同 seed 可逐位复现。
    """
    from sgt import metrics as sgt_metrics

    if not sequences:
        raise ValueError("SymTTR 校准序列不能为空")
    embedding_table = torch.as_tensor(embedding_table)
    if embedding_table.ndim != 2:
        raise ValueError(f"embedding_table 必须为 [V, d]，实际 {tuple(embedding_table.shape)}")
    vocab_size = int(embedding_table.shape[0])
    k = int(k)
    if k < 0:
        raise ValueError(f"k 必须非负，实际 {k}")
    generator = torch.Generator(device=embedding_table.device).manual_seed(int(seed))
    n_total = 0
    n_sym = 0
    with torch.no_grad():
        for start in range(0, len(sequences), int(batch_seqs)):
            chunk = sequences[start : start + int(batch_seqs)]
            ids, mask = _pad_sequences_batch(
                chunk, pad_id=pad_id, device=embedding_table.device
            )
            x = embedding_table[ids]
            mu, log_sigma2 = sgt(x, attention_mask=mask)
            sigma = torch.exp(0.5 * log_sigma2)
            x_tilde = sample_sgt(x, mu, sigma, generator=generator)
            ranks = sgt_metrics.ranks(
                x_tilde[mask], ids[mask], embedding_table, chunk=int(rank_chunk)
            ).detach().cpu()
            n_total += int(ranks.numel())
            n_sym += int(((ranks > k) & (ranks < vocab_size - k)).sum())
    if n_total == 0:
        raise ValueError("SymTTR 校准集没有取到任何有效 token")
    return {
        "metric": f"SymTTR-{k}",
        "k": k,
        "symttr": 100.0 * float(n_sym) / float(n_total),
        "n_tokens": n_total,
        "vocab_size": vocab_size,
        "seed": int(seed),
    }


def mrp_calibration_split(
    n_tokens: int,
    calibration_fraction: float,
    *,
    min_calibration_tokens: int = MIN_MRP_CALIBRATION_TOKENS,
) -> dict[str, int]:
    """GAP-4：把评测 token 按 ``calibration_fraction`` 位置切成 calibration / test。

    - ``fraction == 0``：不切分（``n_calibration=0``、test=全体），保留库级旧行为；
    - ``0 < fraction < 1``：calibration = 前 ``floor(n·fraction)`` 个 token（确定性、无 RNG，
      因此 test 指标只由 test token 决定，不受 calibration 数据影响）；
    - calibration 少于 ``min_calibration_tokens`` 时**直接报错**，绝不静默改用 test。

    Returns:
        ``{"n_calibration", "n_test", "start_test", "n_total"}``。
    """
    n = int(n_tokens)
    if n < 1:
        raise ValueError(f"评测 token 数必须为正，实际 {n}")
    fraction = float(calibration_fraction)
    if not (math.isfinite(fraction) and 0.0 <= fraction < 1.0):
        raise ValueError(f"calibration_fraction 必须在 [0, 1) 内，实际 {fraction!r}")
    if fraction == 0.0:
        return {"n_calibration": 0, "n_test": n, "start_test": 0, "n_total": n}
    n_calibration = int(math.floor(n * fraction))
    if n_calibration < int(min_calibration_tokens):
        raise ValueError(
            f"MRP 校准集太小：calibration_fraction={fraction} × {n} token 只有 "
            f"{n_calibration} 个 token，低于下限 {min_calibration_tokens}；"
            "请增大 --eval-tokens 或调大 --calibration-fraction"
            "（禁止静默改用 test 自身求 r*，GAP-4/SPEC A8）"
        )
    n_test = n - n_calibration
    if n_test < 1:
        raise ValueError(
            f"切分后 test 为空（n={n}，fraction={fraction}）；请减小 --calibration-fraction"
        )
    return {
        "n_calibration": n_calibration,
        "n_test": n_test,
        "start_test": n_calibration,
        "n_total": n,
    }


def rank_metrics(
    x_tilde: torch.Tensor,
    target_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    chunk: int = 1024,
    aux_chunk: int | None = None,
    mrp_r: int | None = None,
    calibration_fraction: float = 0.0,
    min_calibration_tokens: int = MIN_MRP_CALIBRATION_TOKENS,
) -> dict[str, Any]:
    """秩指标聚合（Table 1 的 NN-FR / MRP-FR / TTR / SymTTR / Hist. Entropy）。

    - ``ranks`` 用 ``chunk``（任务单固定 1024，V=128256 时 ``[chunk, V]`` 约 0.5 GB）；
      NN/MRP 的逐行 ``argsort`` 另用较小的 ``aux_chunk``（默认
      :func:`auto_chunk`），避免 int64 索引把显存打满。
    - GAP-4：``calibration_fraction > 0`` 时按 :func:`mrp_calibration_split` 切分，
      MRP 的 ``r*`` **只**由 calibration 的秩直方图众数决定（SPEC A8），``nn_fr`` /
      ``mrp_fr`` / TTR / SymTTR / Hist. Entropy / 秩均值全部**只在 test 上**算。
      ``mrp_r`` 显式覆盖校准值；``calibration_fraction=0`` 保留旧行为（r* 与指标都取自
      全体评测集，仅供库级/回归测试，实验主流程固定传 yaml 的 0.25）。
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
    target_cpu = torch.as_tensor(target_ids).reshape(-1).detach().cpu()
    if target_cpu.numel() != n:
        raise ValueError(
            f"target_ids 长度必须为 {n}，实际 {target_cpu.numel()}"
        )
    split = mrp_calibration_split(
        n,
        calibration_fraction,
        # 显式 --mrp-r 时不需要从 calibration 求 r*，只要求切分本身成立。
        min_calibration_tokens=(1 if mrp_r is not None else int(min_calibration_tokens)),
    )
    start = int(split["start_test"])
    n_calibration = int(split["n_calibration"])
    n_test = int(split["n_test"])

    ranks = metrics.ranks(x_tilde, target_cpu, embedding_table, chunk=int(chunk))
    # metrics 的标量聚合内部用 float64（MPS 不支持 float64），统一在 CPU 副本上做；
    # 距离/排序仍在原设备（MPS）上算，排名结果与 CPU 路径逐位一致。
    ranks_cpu = ranks.detach().cpu()
    if n_calibration > 0:
        histogram = metrics.rank_histogram(ranks_cpu[:n_calibration], vocab_size)
        r_star = int(mrp_r) if mrp_r is not None else metrics.mrp_rank(histogram)
        r_star_source = "override" if mrp_r is not None else "calibration"
    else:
        histogram = metrics.rank_histogram(ranks_cpu, vocab_size)
        r_star = int(mrp_r) if mrp_r is not None else metrics.mrp_rank(histogram)
        r_star_source = "override" if mrp_r is not None else "evaluation_set"

    x_test = x_tilde[start:]
    target_test = target_cpu[start:]
    ranks_test = ranks_cpu[start:]
    nn_pred = metrics.nn_reconstruct(x_test, embedding_table, chunk=aux)
    mrp_pred = metrics.mrp_reconstruct(x_test, embedding_table, r_star, chunk=aux)
    ranks_f64 = ranks_test.to(torch.float64)
    return {
        "n_tokens": n_test,
        "n_test_tokens": n_test,
        "n_calibration_tokens": n_calibration,
        "n_tokens_total": n,
        "calibration_fraction": float(calibration_fraction),
        "min_calibration_tokens": int(min_calibration_tokens),
        "mrp_r_star_source": r_star_source,
        "vocab_size": vocab_size,
        "nn_fr": metrics.failure_rate(nn_pred.detach().cpu(), target_test),
        "mrp_fr": metrics.failure_rate(mrp_pred.detach().cpu(), target_test),
        "mrp_r_star": r_star,
        "ttr_10": metrics.ttr(ranks_test, 10, vocab_size),
        "ttr_50": metrics.ttr(ranks_test, 50, vocab_size),
        "symttr_10": metrics.symttr(ranks_test, 10, vocab_size),
        "symttr_100": metrics.symttr(ranks_test, 100, vocab_size),
        "hist_entropy": metrics.hist_entropy(ranks_test, vocab_size),
        "rank_mean": float(ranks_f64.mean()),
        "rank_median": float(ranks_f64.median()),
    }


def _undefined_mi_block(
    status: str, note: str, *, trigger: str, **diagnostics: Any
) -> dict[str, Any]:
    """MI 不可用时的**显式**占位块（绝不把「不可定义」写成 0、也绝不静默丢弃）。

    键名与 :func:`feature_mi` 的正常返回一致，四个统计量一律 ``None``；
    ``mi_status != "ok"`` 时下游门禁必须把 MI/PAC 判为「不可用」而不是「通过」
    （见 :func:`evaluate_sgt` 的 ``mi_gate`` 与结果 JSON 的 ``eval.mi_gate``）。
    """
    block: dict[str, Any] = {
        "mi_feature_mean": None,
        "mi_sum": None,
        "mi_feature_median": None,
        "pac_adv": None,
        "mi_status": status,
        "mi_trigger": trigger,
        "mi_note": note,
    }
    block.update(diagnostics)
    return block


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

    数值（健壮性契约）：``1/σ²`` 用 ``exp(-log σ²)``；逐块用输入精度（float32）计算，
    逐特征均值在 CPU float64 上累加。σ 退化（σ→0，A21 只给了上界）时 1/σ² 溢出，
    此时**不抛错**，改用 :func:`_undefined_mi_block` 的占位块
    （``mi_status="undefined_sigma_too_small"``、四个统计量与 ``pac_adv`` 均为 ``None``）：

    - 预检：``min(log σ², log σ²′) < log(1e-12)``（σ < 1e-6，float32 下 1/σ² 已溢出）；
    - 计算中出现非有限值（σ² 极端导致 ``exp(-log σ²)·residual²`` 溢出）。

    其它错误照旧抛出：形状不匹配、输入含 NaN/Inf（那是调用方 bug，不是 σ 退化）。
    正常路径返回 ``mi_status="ok"`` / ``mi_trigger="ok"``。
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

    # NaN/Inf 输入是调用方 bug：显式拒绝，不能被下面的 σ 门当成「σ 退化」吞掉。
    for name, tensor in (
        ("x_tilde", x_tilde),
        ("x_prime", x_prime),
        ("mu_prime", mu_prime),
        ("log_sigma2_prime", log_sigma2_prime),
        ("log_sigma2", log_sigma2),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"feature_mi 的输入 {name} 含非有限值（NaN/Inf），拒绝计算")

    # A7 数值门：σ 过小时 1/σ² 溢出，MI 在数学上是 +∞，float32 无法表示。
    # 返回占位块（不抛错），让训练探针/实验落盘，并由 mi_status 显式门禁。
    log_sigma2_min = float(log_sigma2.min())
    log_sigma2_prime_min = float(log_sigma2_prime.min())
    sigma_floor = min(log_sigma2_min, log_sigma2_prime_min)
    diagnostics = {
        "n_pairs": int(n_pairs),
        "n_tokens": n,
        "n_prime": m,
        "n_features": d,
        "seed": int(seed),
        "chunk": int(chunk),
        "log_sigma2_min": log_sigma2_min,
        "log_sigma2_prime_min": log_sigma2_prime_min,
    }
    if sigma_floor < MI_LOG_SIGMA2_FLOOR:
        return _undefined_mi_block(
            MI_STATUS_SIGMA_TOO_SMALL,
            f"A7 的 1/σ² 在 σ 过小时溢出：min(log σ², log σ²′)={sigma_floor:.2f} < "
            f"log(1e-12)={MI_LOG_SIGMA2_FLOOR:.2f}（σ < 1e-6）。MI/PAC 判为 undefined"
            "（不是 0；也不代表通过）；如需数值，请先约束 σ（sigma_min 或 A21 下界）。",
            trigger="sigma_below_floor",
            **diagnostics,
        )

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
            # 与 σ 预检同一占位块（任务单口径）；mi_trigger 区分真正的触发路径。
            return _undefined_mi_block(
                MI_STATUS_SIGMA_TOO_SMALL,
                "Eq(8) 的逐对项出现非有限值：σ² 极端导致 exp(-log σ²)·residual² 溢出"
                f"（min(log σ², log σ²′)={sigma_floor:.2f}，log σ² 范围 "
                f"[{log_sigma2_min:.2f}, {float(log_sigma2.max()):.2f}]）。"
                "MI/PAC 判为 undefined（不是 0；也不代表通过）。",
                trigger="nonfinite_term",
                **diagnostics,
            )
        # 逐特征**求和**（再除以 n_pairs），不能累加块均值——块大小不同会引入权重偏差。
        feature_sum += term.sum(dim=0).detach().cpu().to(torch.float64)
    feature_mean = feature_sum / float(n_pairs)
    mi_feature_mean = float(feature_mean.mean())
    return {
        "estimator": "Eq(8) 逐特征 MC 均值（A7）",
        "mi_status": MI_STATUS_OK,
        "mi_trigger": "ok",
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
        "log_sigma2_min": log_sigma2_min,
        "log_sigma2_prime_min": log_sigma2_prime_min,
    }


def _mi_block(
    x_tilde: torch.Tensor,
    x_prime: torch.Tensor,
    mu_prime: torch.Tensor,
    log_sigma2_prime: torch.Tensor,
    log_sigma2: torch.Tensor,
    *,
    n_pairs: int,
    seed: int,
    vocab_size: int,
    chunk: int = 1024,
) -> dict[str, Any]:
    """A7 特征级 MI + PAC-Adv 的统一封装（:func:`evaluate_sgt` 的唯一调用入口）。

    - ``mi_status == "ok"``：算 ``pac_adv = 100·δ``（Eq (4)，``δ`` 由 :func:`metrics.pac_adv`
      反解），``pac_adv_status="ok"``；
    - 否则（σ 过小/非有限）：``pac_adv`` 保持 ``None``，``pac_adv_status`` 标注跳过原因，
      **不**把不可定义写成 0。门禁侧据此把本 run 的 MI/PAC 判为不可用（见 ``mi_gate``）。
    """
    block = feature_mi(
        x_tilde,
        x_prime,
        mu_prime,
        log_sigma2_prime,
        log_sigma2,
        n_pairs=int(n_pairs),
        seed=int(seed),
        chunk=int(chunk),
    )
    if block.get("mi_status") != MI_STATUS_OK:
        block["pac_adv"] = None
        block["pac_adv_status"] = "skipped_mi_undefined"
        return block
    block["pac_adv"] = float(metrics.pac_adv(block["mi_feature_mean"], int(vocab_size)))
    block["pac_adv_status"] = "ok"
    return block


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
    calibration_fraction: float = 0.0,
    min_calibration_tokens: int = MIN_MRP_CALIBRATION_TOKENS,
    mrp_r: int | None = None,
) -> dict[str, Any]:
    """Alpaca 评估：秩指标（MRP 的 r* 来自独立 calibration，GAP-4）+ A7 特征级 MI + PAC-Adv。"""
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
        features["x_tilde"],
        features["ids"],
        embedding_table,
        chunk=rank_chunk,
        mrp_r=mrp_r,
        calibration_fraction=calibration_fraction,
        min_calibration_tokens=min_calibration_tokens,
    )
    # Eq (8) 的“独立 batch”：评测端用同一 Alpaca token 集的有放回抽样近似同分布独立采样。
    mi_block = _mi_block(
        features["x_tilde"],
        features["x"],
        features["mu"],
        features["log_sigma2"],
        features["log_sigma2"],
        n_pairs=n_pairs,
        seed=seed,
        vocab_size=int(metrics_block["vocab_size"]),
    )
    mi_block["prime_source"] = "同一 Alpaca token 集的有放回抽样（近似 Eq (8) 的独立 batch）"
    mi_gate = {
        "mi_status": mi_block["mi_status"],
        "mi_trigger": mi_block.get("mi_trigger"),
        "mi_and_pac_conclusive": mi_block["mi_status"] == MI_STATUS_OK,
        "note": "mi_status != 'ok' 时本次 run 的 MI/PAC 为 undefined（null）："
        "不得当作 0/通过，也不得与 Table 1 的 MI 列对照；只有 mi_status == 'ok' 才可判定",
    }
    metrics_block.update(
        {
            key: mi_block[key]
            for key in ("mi_feature_mean", "mi_sum", "mi_feature_median", "pac_adv")
        }
    )
    metrics_block["mi_status"] = mi_block["mi_status"]
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
        "mi_gate": mi_gate,
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


def _selection_checkpoint_state(
    checkpoint_dir: str | Path,
    step: int,
    every: int,
    final_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """取「已完成 ``step`` 步」时的 SGT state_dict（GAP-7）。

    ``train_sgt`` 在完成第 ``s`` 步（0-based）时保存 ``step_{s}.pt``，所以已完成
    ``step`` 步对应 ``step_{step-1}.pt``。若 ``step`` 不是 ``every`` 的整数倍（选择轨迹的
    末点），用内存里的最终模型——两者在末步上逐位相同。整数倍步的 checkpoint 缺失时
    **报错**（不能用最终模型冒充该步，否则选择轨迹会失真）。
    """
    step = int(step)
    every = int(every)
    if every > 0 and step % every == 0:
        path = Path(checkpoint_dir) / f"step_{step - 1}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"SymTTR 选择缺少第 {step} 步的 checkpoint {path}；"
                "train_sgt 的 checkpoint 节奏必须等于 selection.every（GAP-7）"
            )
        # 本地自产文件（train_sgt 写入的 state_dict + optimizer），按可信文件读取。
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("sgt") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError(f"{path} 缺少 SGT state_dict（键 'sgt'）")
        return dict(state)
    return dict(final_state)


def best_checkpoint_path(ckpt_path: str | Path) -> Path:
    """SymTTR 最优 checkpoint 的落盘路径：``ckpt_<variant>_best.pt``（GAP-7）。"""
    path = Path(ckpt_path)
    return path.with_name(f"{path.stem}_best{path.suffix}")


def paper_scale_block(
    data_meta: Mapping[str, Any], *, steps: int, n_batches: int, max_examples: int
) -> dict[str, Any]:
    """GAP-5：把本机训练规模折算成论文协议（OpenOrca 15% / 1B）的比例。

    论文比例的分母固定用**论文训练集**（OpenOrca train 全量 4,233,927 / 15% 子集
    635,089，CONFORMANCE CF-29），不用本地 parquet 的行数（那只是 1M 条子集，会把比例
    算错）；本地数据量另记在 ``reproduction`` 里。
    """
    n_examples = int(data_meta.get("n_examples", max_examples))
    rows_meta = data_meta.get("rows_meta") or {}
    block: dict[str, Any] = {
        "paper_protocol": {
            "train_dataset": "Open-Orca/OpenOrca（train split，原始顺序前 N 条）",
            "train_frac": PAPER_OPENORCA_FRAC,
            "openorca_train_rows_reference": PAPER_OPENORCA_TRAIN_ROWS,
            "paper_n_selected_reference": PAPER_N_SELECTED_REFERENCE,
            "target_model": "Llama 3.2 1B Instruct",
            "note": "论文 §7 协议：OpenOrca 的 15%（635,089 条 = 0.15×4,233,927）训练；"
            "1×A100 80GB ≈ 6h/变体（D2 降规模）",
        },
        "reproduction": {
            "n_train_examples": n_examples,
            "steps": int(steps),
            "n_batches": int(n_batches),
            "epochs_covered": (
                float(steps) / float(n_batches) if int(n_batches) > 0 else None
            ),
            "frac_of_paper_subset": float(n_examples) / float(PAPER_N_SELECTED_REFERENCE),
            "frac_of_openorca_train": float(n_examples) / float(PAPER_OPENORCA_TRAIN_ROWS),
        },
    }
    if data_meta.get("data_source") == "local_parquet":
        block["reproduction"]["local_parquet_rows"] = data_meta.get("n_available")
    elif rows_meta.get("n_total") is not None:
        block["reproduction"]["rows_api_n_total"] = int(rows_meta["n_total"])
    return block


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
        selection_options = resolve_selection(
            raw,
            enabled=args.select_by_symttr,
            every=args.select_every,
            calibration_examples=args.calibration_examples,
        )
        eval_options = resolve_eval_options(
            raw,
            calibration_fraction=args.calibration_fraction,
            mrp_r=args.mrp_r,
        )
    except Exception as exc:
        return _fail(f"selection/eval 配置不合法: {type(exc).__name__}: {exc}")
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

    # ---- 1b. LM 侧精度（GAP-5 / A22） ------------------------------------- #
    requested_lm_dtype = (
        str(train_raw.get("lm_dtype", DEFAULT_LM_DTYPE))
        if args.lm_dtype == "auto"
        else str(args.lm_dtype)
    )
    try:
        probe: Callable[[], tuple[bool, str]] | None = None
        if requested_lm_dtype != "float32":
            if requested_lm_dtype not in LM_DTYPES:
                raise ValueError(
                    f"未知 train.lm_dtype {requested_lm_dtype!r}；允许 {sorted(LM_DTYPES)}"
                )
            probe = lambda: probe_mixed_precision(  # noqa: E731
                device,
                lm_dtype=LM_DTYPES[requested_lm_dtype],
                loss_terms=loss_terms,
                weights=weights,
            )
        lm_dtype_info = resolve_lm_dtype(requested_lm_dtype, probe=probe)
    except Exception as exc:
        return _fail(f"LM 精度选择失败: {type(exc).__name__}: {exc}")
    torch_dtype = LM_DTYPES[lm_dtype_info["effective"]]
    sgt_dtype = torch.float32  # 论文式 (1) 的 µ/Σ 参数恒以 fp32 计算（A22：不静默降精度）

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
    lm_embedding_table = lm.get_input_embeddings().weight.detach()
    # 评测/秩指标统一在 fp32 上算：bf16→fp32 无损，保证换算路径与 fp32 训练逐位可比。
    embedding_table = (
        lm_embedding_table
        if lm_embedding_table.dtype == torch.float32
        else lm_embedding_table.float()
    )
    if not bool(torch.isfinite(embedding_table).all()):
        return _fail("embedding 表含非有限值")

    # ---- 3b. 训练配置 ----------------------------------------------------- #
    try:
        train_config = build_train_config(
            raw,
            loss_terms=loss_terms,
            weights=weights,
            max_steps=args.max_steps,
            device=args.device,
            seed=seed,
            lm_dtype=lm_dtype_info["effective"],
        )
    except Exception as exc:
        return _fail(f"训练配置不合法: {type(exc).__name__}: {exc}")

    # ---- 4. SGT 初始化（A21：sigma_max 上界用于 Comp.Entr. 可训练） -------- #
    sigma_max_raw = sgt_raw.get("sigma_max")
    sigma_max = None if sigma_max_raw is None else float(sigma_max_raw)
    pos_emb_init_std = float(
        sgt_raw.get("pos_emb_init_std", DEFAULT_POS_EMB_INIT_STD)
    )
    # A1 / A21：final_norm / mu_scale / sigma_floor / pos_encoding（缺省用模型默认值）。
    sgt_options = resolve_sgt_options(sgt_raw)
    init_start = time.perf_counter()
    torch.manual_seed(seed)  # SGT 初始化由 seed 决定（训练内部采样另有显式 Generator）
    try:
        sgt = build_sgt_model(
            d,
            sgt_raw,
            sigma_min=train_config.sigma_min,
            sigma_max=sigma_max,
            pos_emb_init_std=pos_emb_init_std,
        ).to(device=device, dtype=sgt_dtype)
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
    # GAP-7：把 selection.every 映射到 train_sgt 的 checkpoint 节奏（log_every 只控制
    # checkpoint 落盘，history 仍逐步记录），训练结束后再离线评测这些 step checkpoint。
    select_dir: Path | None = None
    log_every_from_yaml = int(train_config.log_every)
    if selection_options["enabled"]:
        select_dir = Path(tempfile.mkdtemp(prefix=f"sgt_select_{args.variant}_"))
        train_config.checkpoint_dir = str(select_dir)
        train_config.log_every = int(selection_options["every"])

    train_start = time.perf_counter()
    try:
        result = train_sgt(sgt, lm, batches, config=train_config)
    except Exception:
        traceback.print_exc()
        if select_dir is not None:
            shutil.rmtree(select_dir, ignore_errors=True)
        return 1
    timings["train_s"] = time.perf_counter() - train_start
    completed_steps = int(result["completed_steps"])

    ckpt_start = time.perf_counter()
    checkpoint_config = {
        "variant": args.variant,
        "yaml_path": str(args.config),
        "yaml": json_safe(raw),
        "loss_terms": list(loss_terms),
        "weights": weights,
        "train_config": json_safe(asdict(train_config)),
        "sgt": dict(sgt_raw),
        "d": d,
        "model": str(args.model),
        "lm_dtype": json_safe(lm_dtype_info),
        "eval": {
            "examples": int(args.eval_examples),
            "max_len": int(args.eval_max_len),
            "tokens": int(args.eval_tokens),
            "n_pairs": int(args.n_pairs),
            "calibration_fraction": float(eval_options["calibration_fraction"]),
            "seed": seed,
        },
    }
    try:
        _save_checkpoint(
            ckpt_path,
            sgt=sgt,
            config=checkpoint_config,
            seed=seed,
            variant=args.variant,
        )
    except Exception as exc:
        if select_dir is not None:
            shutil.rmtree(select_dir, ignore_errors=True)
        return _fail(f"保存 checkpoint 失败: {type(exc).__name__}: {exc}", code=1)
    timings["ckpt_s"] = time.perf_counter() - ckpt_start

    # ---- 7. 评测（--skip-eval 时整段跳过） -------------------------------- #
    if args.skip_eval:
        # 训练探针：只训练 + checkpoint；不加载 Alpaca、不跑 SymTTR 选择、不算 MI/PAC。
        selection_block = {
            "enabled": False,
            "checkpoint": "last",
            "skip_eval": True,
            "selected_step": int(completed_steps),
            "selection_configured_enabled": bool(selection_options["enabled"]),
            "reason": "--skip-eval：跳过评测/MI/PAC 与 SymTTR checkpoint 选择，"
            "selected_step 取最后一步（用于观察训练轨迹）",
        }
        if select_dir is not None:
            # 训练期的 step checkpoint 只服务于 SymTTR 选择，探针模式不再需要。
            shutil.rmtree(select_dir, ignore_errors=True)
        selected_step_value = int(completed_steps)
        eval_payload: dict[str, Any] = {"status": "skipped"}
        timings["total_s"] = time.perf_counter() - total_start
    else:
        # ---- 7a. 评测数据（GAP-7 校准偏移 + GAP-4 MRP 切分） -------------- #
        eval_start = time.perf_counter()
        selection_active = bool(selection_options["enabled"]) and completed_steps > 0
        selection_examples = int(selection_options["calibration_examples"]) if selection_active else 0
        calibration_examples: list[dict] = []
        calibration_sequences: list[list[int]] = []
        calibration_tokens = 0
        try:
            eval_rows_requested = selection_examples + int(args.eval_examples)
            eval_examples_all, eval_meta = load_eval_examples(eval_rows_requested)
            if selection_active:
                calibration_examples = list(eval_examples_all[:selection_examples])
                if len(calibration_examples) < selection_examples:
                    return _fail(
                        f"Alpaca 只有 {len(eval_examples_all)} 条，无法划出前 "
                        f"{selection_examples} 条作 SymTTR 校准集",
                        code=2,
                    )
                calibration_sequences, calibration_tokens = tokenize_eval_sequences(
                    calibration_examples, tokenizer, max_len=int(args.eval_max_len), n_tokens=None
                )
                if not calibration_sequences:
                    return _fail("SymTTR 校准集为空（检查 --calibration-examples/--eval-max-len）", code=2)
                test_examples = list(
                    eval_examples_all[selection_examples : selection_examples + int(args.eval_examples)]
                )
            else:
                test_examples = list(eval_examples_all[: int(args.eval_examples)])
            sequences, n_tokens = tokenize_eval_sequences(
                test_examples,
                tokenizer,
                max_len=int(args.eval_max_len),
                n_tokens=int(args.eval_tokens),
            )
            if not sequences or n_tokens == 0:
                return _fail("评估集没有取到任何有效 token", code=1)
            # GAP-4 前置检查：校准集过小要**在训练结束后、正式评测前**就报错（退出码 2）。
            if float(eval_options["calibration_fraction"]) > 0.0:
                mrp_calibration_split(
                    n_tokens,
                    float(eval_options["calibration_fraction"]),
                    min_calibration_tokens=(
                        1 if eval_options["mrp_r"] is not None else MIN_MRP_CALIBRATION_TOKENS
                    ),
                )
        except ValueError as exc:
            return _fail(f"评测集/MRP 切分不合法: {exc}")
        except Exception:
            traceback.print_exc()
            return 1

        # ---- 7b. GAP-7：按 SymTTR 选 checkpoint -------------------------- #
        selection_block: dict[str, Any]
        selected_step_value = completed_steps
        best_ckpt_path: Path | None = None
        if not selection_options["enabled"]:
            selection_block = {"enabled": False, "checkpoint": "last"}
        elif completed_steps == 0:
            selection_block = {
                "enabled": False,
                "checkpoint": "last",
                "reason": "steps=0：没有可选的 checkpoint，使用最后一个（初始）模型",
            }
            if select_dir is not None:
                shutil.rmtree(select_dir, ignore_errors=True)
        else:
            selection_start = time.perf_counter()
            try:
                final_state = {key: value.detach().clone() for key, value in sgt.state_dict().items()}
                sgt.eval()

                def _evaluate_at(step: int) -> float:
                    sgt.load_state_dict(
                        _selection_checkpoint_state(
                            select_dir, step, selection_options["every"], final_state
                        )
                    )
                    return float(
                        symttr_on_sequences(
                            sgt,
                            embedding_table,
                            calibration_sequences,
                            pad_id=_pad_id(tokenizer),
                            batch_seqs=int(args.eval_batch),
                            seed=seed,
                            rank_chunk=int(args.rank_chunk),
                        )["symttr"]
                    )

                selection_block = build_selection(
                    _evaluate_at,
                    steps=completed_steps,
                    every=int(selection_options["every"]),
                )
                selected_step_value = int(selection_block["best_step"])
                sgt.load_state_dict(
                    _selection_checkpoint_state(
                        select_dir, selected_step_value, selection_options["every"], final_state
                    )
                )
                if selected_step_value != completed_steps:
                    best_ckpt_path = best_checkpoint_path(ckpt_path)
                    _save_checkpoint(
                        best_ckpt_path,
                        sgt=sgt,
                        config=checkpoint_config,
                        seed=seed,
                        variant=args.variant,
                    )
                selection_block.update(
                    {
                        "checkpoint": str(best_ckpt_path or ckpt_path),
                        "calibration_examples": int(selection_options["calibration_examples"]),
                        "calibration_tokens": int(calibration_tokens),
                        "eval_offset": int(selection_examples),
                        "checkpoint_dir": str(select_dir),
                        "log_every_from_yaml": log_every_from_yaml,
                        "sampling": "每个 checkpoint 用同一 seed 重新采样（Eq (1)），"
                        "选择只反映权重差异",
                        "note": "训练结束后离线评测 train_sgt 落盘的 step checkpoint；"
                        "不触碰训练 RNG/优化器状态（同 seed 训练结果不变）",
                    }
                )
            except Exception:
                traceback.print_exc()
                return 1
            finally:
                timings["selection_s"] = time.perf_counter() - selection_start
                shutil.rmtree(select_dir, ignore_errors=True)

        # ---- 7c. 最终评测（SymTTR 最优 checkpoint；GAP-4 只在 test 上算指标） - #
        try:
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
                calibration_fraction=float(eval_options["calibration_fraction"]),
                min_calibration_tokens=MIN_MRP_CALIBRATION_TOKENS,
                mrp_r=eval_options["mrp_r"],
            )
        except Exception:
            traceback.print_exc()
            return 1
        timings["eval_s"] = time.perf_counter() - eval_start
        timings["total_s"] = time.perf_counter() - total_start

        metrics_block = eval_block["metrics"]
        eval_payload = {
            "n_tokens": int(eval_block["n_tokens"]),
            "metrics": metrics_block,
            "feature_mi": eval_block["feature_mi"],
            "mi_gate": eval_block["mi_gate"],
            "sigma": eval_block["sigma"],
            "dataset": "tatsu-lab/alpaca",
            "eval_examples": len(test_examples),
            "eval_rows_requested": int(eval_rows_requested),
            "eval_offset": int(selection_examples),
            "selection_calibration_examples": int(selection_examples),
            "selection_calibration_tokens": int(calibration_tokens),
            "eval_max_len": int(args.eval_max_len),
            "eval_tokens_requested": int(args.eval_tokens),
            "sequences": len(sequences),
            "selected_step": int(selected_step_value),
            "mrp_calibration": {
                "calibration_fraction": float(eval_options["calibration_fraction"]),
                "min_calibration_tokens": MIN_MRP_CALIBRATION_TOKENS,
                "r_star_source": metrics_block["mrp_r_star_source"],
                "n_calibration_tokens": int(metrics_block["n_calibration_tokens"]),
                "n_test_tokens": int(metrics_block["n_test_tokens"]),
                "mrp_r_override": eval_options["mrp_r"],
                "note": "r* 只从 calibration 的秩直方图取众数；所有指标只在 test 上算"
                "（GAP-4 / SPEC A8）",
            },
        }

    table_row = TABLE1_REFERENCE.get(args.variant)
    train_dtypes = result.get("dtypes") if isinstance(result, Mapping) else None
    payload = {
        "variant": args.variant,
        "model": str(args.model),
        "device": args.device,
        "data_source": data_meta["data_source"],
        "n_train_examples": int(data_meta["n_examples"]),
        "steps": completed_steps,
        "selected_step": int(selected_step_value),
        "train_history": result["history"],
        "train_dtypes": json_safe(train_dtypes),
        "selection": json_safe(selection_block),
        "skip_eval": bool(args.skip_eval),
        "eval": eval_payload,
        "config_echo": {
            "yaml_path": str(args.config),
            "raw_yaml": json_safe(raw),
            "variant": args.variant,
            "loss_terms": list(loss_terms),
            "weights": weights,
            "weights_echo": weights_echo,
            "sgt": {
                **dict(sgt_raw),
                **sgt_options,
                "sigma_max_effective": sigma_max,
                "pos_emb_init_std_effective": pos_emb_init_std,
                "dtype": str(sgt_dtype),
            },
            "d": d,
            "train": json_safe(asdict(train_config)),
            "selection": {
                "options": json_safe(
                    {key: value for key, value in selection_options.items() if key != "config"}
                ),
                "yaml": json_safe(selection_options["config"]),
                "checkpoint_dir": str(select_dir) if select_dir is not None else None,
                "log_every_from_yaml": log_every_from_yaml,
                "log_every_effective": int(train_config.log_every),
                "note": "GAP-7：log_every 只控制 checkpoint 落盘；训练结束后离线算 SymTTR-100",
            },
            "median_norm_T": float(result["median_norm"]),
            "eval": {
                "skip_eval": bool(args.skip_eval),
                "examples": int(args.eval_examples),
                "max_len": int(args.eval_max_len),
                "tokens": int(args.eval_tokens),
                "n_pairs": int(args.n_pairs),
                "rank_chunk": int(args.rank_chunk),
                "aux_chunk": auto_chunk(int(embedding_table.shape[0])),
                "eval_batch": int(args.eval_batch),
                "seed": seed,
                "calibration_fraction": float(eval_options["calibration_fraction"]),
                "mrp_r": eval_options["mrp_r"],
                "min_calibration_tokens": MIN_MRP_CALIBRATION_TOKENS,
                "selection_calibration_examples": int(selection_options["calibration_examples"]),
                "selection_every": int(selection_options["every"]),
                "sampling": "x̃ = x + µ + exp(0.5·log σ²) ⊙ u（Eq (1)，固定 seed）",
            },
            "model": {
                "path": str(args.model),
                "dtype": lm_dtype_info["effective"],
                "sgt_dtype": str(sgt_dtype),
                "frozen": True,
                "lm_dtype": json_safe(lm_dtype_info),
                "embedding_table_eval_dtype": str(embedding_table.dtype),
            },
            "seed": seed,
        },
        "data": data_meta,
        "train_batches": batch_info,
        "paper_scale": paper_scale_block(
            data_meta,
            steps=completed_steps,
            n_batches=int(batch_info["n_batches"]),
            max_examples=max_examples,
        ),
        "checkpoint": str(ckpt_path),
        "table1_reference": {
            "source": "arXiv:2506.09452v1 Table 1（p.9，paper.txt 转录）",
            "scale_note": "论文在 A100 上 300 步以上/6h 训练；本复现降规模（max_examples/steps 见 "
            "config_echo 与 paper_scale），数值不可直接与论文对照（SPEC §9 判定规则）。",
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
            "sgt_dtype": str(sgt_dtype),
            "lm_dtype": json_safe(lm_dtype_info),
        },
    }
    try:
        write_json(out_path, payload)
    except Exception as exc:
        return _fail(f"写结果 JSON 失败 {out_path}: {type(exc).__name__}: {exc}", code=1)

    if args.skip_eval:
        print(
            f"[ablation] variant={args.variant} steps={completed_steps} "
            f"selected_step={selected_step_value} select=off eval=skipped "
            f"lm_dtype={lm_dtype_info['effective']} "
            f"train_s={timings['train_s']:.1f} total_s={timings['total_s']:.1f} -> {out_path}"
        )
        return 0

    selection_summary = (
        f"select_step={selection_block['best_step']} "
        f"select_symttr100={selection_block['best_symttr_100']}"
        if selection_block.get("enabled")
        else "select=off"
    )
    print(
        f"[ablation] variant={args.variant} steps={completed_steps} "
        f"selected_step={selected_step_value} {selection_summary} "
        f"mrp_r_star={metrics_block['mrp_r_star']}"
        f"({metrics_block['mrp_r_star_source']}, "
        f"n_cal={metrics_block['n_calibration_tokens']}, "
        f"n_test={metrics_block['n_test_tokens']}) "
        f"nn_fr={metrics_block['nn_fr']:.2f} mrp_fr={metrics_block['mrp_fr']:.2f} "
        f"symttr_10={metrics_block['symttr_10']:.2f} "
        f"hist_entropy={metrics_block['hist_entropy']:.4f} "
        f"mi_feature_mean={_fmt_or_null(metrics_block['mi_feature_mean'])} "
        f"pac_adv={_fmt_or_null(metrics_block['pac_adv'])} "
        f"mi_status={metrics_block['mi_status']} "
        f"lm_dtype={lm_dtype_info['effective']} "
        f"train_s={timings['train_s']:.1f} total_s={timings['total_s']:.1f} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
