"""M8b PUPA 隐私评测：NN / BeamClean 重建 + PII-Ratio（论文 §7 p.9；SPEC §4/§5、A9/A10/A20）。

用法::

    ./.venv/bin/python experiments/run_pupa.py \
        --model .models/llama-3.2-1b-instruct \
        --max-examples 100 --max-len 128 --device cpu \
        [--sgt-ckpt results/ckpt_mi_abscos_norm.pt] [--skip-beamclean] \
        --out results/pupa_<tag>.json

流程（任务单 M8b）：

1. ``sgt.data.load_pupa("pupa_tnb", max_examples=...)``；**被保护文本 = ``user_query``**
   （SPEC A20：SGT 保护的是用户输入；``target_response`` 是模型输出，不参与本评测），
   ``pii_units`` 按 ``||`` 切分。
2. tokenizer + LM（fp32，或 ``--dtype auto`` 时用模型 config 的 dtype）；
   ``table = lm.get_input_embeddings().weight.detach()``，``[V, d]``。
3. 逐批 tokenize（右 padding）→ ``ids`` / ``attention_mask`` → ``x = table[ids]``。
4. 有 ``--sgt-ckpt`` 时 ``mu, lv = sgt(x, mask)``，``x̃ = sample_sgt(x, mu, exp(0.5·lv))``
   （固定 seed 的显式 ``torch.Generator``）；否则 ``x̃ = x``（无保护的基线）。
5. NN：``nn_attack(x̃.reshape(-1, d), table)``，只与 ``attention_mask=1`` 的有效 token 比较得到 FR。
6. BeamClean：``beamclean`` + 由 ``make_lm_score_fn`` 提供的**标准自回归**前缀对数似然
   ``Σ_{i=2..t} log p(token_i | token_<i)``（``logits[:, :-1]`` 对 ``tokens[:, 1:]``，
   ``t == 1`` 为 0）；逐样本调用（每样本只取有效 token 位置，padding 不进入攻击），
   记录 ``sigma_hat``。``--skip-beamclean`` 时跳过，但结果 JSON 里显式记
   ``status="skipped"``，不静默。
7. 重建 ids 与原始 ids 都只解码有效 token（``skip_special_tokens=True``）；PII-Ratio 的
   分母参考文本 ``T_ref`` 用**模型实际看到的**截断后解码文本（不是完整 ``user_query``），
   逐样本另记完整原文 ``user_query``、``pii_reference_text``（= ``T_ref``）与
   ``pii_ratio_full``（对完整原文的比值，审计对照）。
8. 结果写 ``--out``（JSON），逐样本明细（含原始文本与重建文本）写同名 ``_rows.jsonl``。

指标口径（本文件内统一）：

- FR = ``100 · #{重建 ≠ 真 token} / #{有效 token}``（padding 不计；跨样本按 token 汇总，
  即逐样本 FR 的 token 数加权平均），复用 :func:`sgt.metrics.failure_rate` 的公式。
- PII-Ratio 与片段恢复率的定义见 :mod:`sgt.pii`；分母参考文本 ``T_ref`` 是
  ``tokenizer.decode(截断到 max_len 后的有效 token ids, skip_special_tokens=True)``，
  即模型实际看到的文本（``meta.pii_reference = "truncated_decoded"``）。**不是**
  PAPILLON 的 LLM-as-judge 口径（A10），结果 JSON 的 ``meta.pii_definition`` 会写明，
  报告里同样声明。
- 所有浮点在写盘前做有限性清洗：``NaN/±Inf`` 统一变成 ``null``（``None``）。

BeamClean 的成本：每个样本约 ``T × beam × topk`` 次 LM 前向（``_beam_decode_one`` 每步按
整段前缀重算 LM 分数），本机 CPU 上 1B 模型非常慢；因此按 logits 预算自适应分块
（``--lm-chunk``，默认 0 = 自适应 1 GiB），必要时用 ``--skip-beamclean`` 只跑 NN。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sgt import metrics  # noqa: E402
from sgt.attacks import beamclean, nn_attack  # noqa: E402
from sgt.data import load_pupa, make_batches  # noqa: E402
from sgt.model import SGTModel  # noqa: E402
from sgt.pii import pii_ratio, pii_units_recovered, split_pii_units  # noqa: E402
from sgt.sampling import sample_sgt  # noqa: E402

__all__ = [
    "TEXT_FIELD",
    "DEFAULT_MAX_EXAMPLES",
    "DEFAULT_MAX_LEN",
    "make_lm_score_fn",
    "load_sgt_from_checkpoint",
    "run_pupa_eval",
    "parse_args",
    "main",
]

#: 被保护文本字段（SPEC A20）：用户输入，绝不是 ``target_response``。
TEXT_FIELD = "user_query"
DEFAULT_MAX_EXAMPLES = 100
DEFAULT_MAX_LEN = 128
DEFAULT_BATCH_SIZE = 4

#: NN 距离矩阵的元素预算（按 |V| 自适应分块，约 256 MB / 块）。
_NN_BUDGET_BYTES = 256 * 1024 * 1024
#: BeamClean LM 打分的 logits 预算（按 (t, |V|) 自适应分块，约 1 GiB / 块）。
_LM_BUDGET_BYTES = 1024 * 1024 * 1024

_DTYPE_CHOICES = ("float32", "bfloat16", "float16", "auto")


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def _fail(message: str, code: int = 2) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def _auto_nn_chunk(vocab_size: int, budget_bytes: int = _NN_BUDGET_BYTES) -> int:
    """NN 攻击的分块大小：``[chunk, |V|]`` 距离矩阵不超过约 ``budget_bytes``。"""
    return max(1, int(budget_bytes) // max(1, int(vocab_size) * 4))


def _auto_lm_chunk(
    seq_len: int, vocab_size: int, budget_bytes: int = _LM_BUDGET_BYTES
) -> int:
    """LM 打分的分块大小：``[chunk, t, |V|]`` 的 float32 logits 不超过约 ``budget_bytes``。"""
    per_row = max(1, int(seq_len) * int(vocab_size) * 4)
    return max(1, int(budget_bytes) // per_row)


def _module_device(module: torch.nn.Module) -> torch.device:
    param = next(module.parameters(), None)
    return param.device if param is not None else torch.device("cpu")


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    param = next(module.parameters(), None)
    return param.dtype if param is not None else torch.get_default_dtype()


def _decode(tokenizer, ids) -> str:
    """把 token ids 解码为文本（``skip_special_tokens=True``）。"""
    if isinstance(ids, torch.Tensor):
        id_list = [int(value) for value in ids.detach().cpu().reshape(-1).tolist()]
    else:
        id_list = [int(value) for value in ids]
    return str(tokenizer.decode(id_list, skip_special_tokens=True))


def _finite_or_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitize(obj):
    """递归把张量/NumPy 标量转成 Python 值，并把 NaN/±Inf 换成 ``None``。"""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _finite_or_none(float(obj))
    if isinstance(obj, torch.Tensor):
        return _sanitize(obj.detach().cpu().tolist())
    if isinstance(obj, Mapping):
        return {str(key): _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    if hasattr(obj, "item"):  # NumPy 标量
        return _sanitize(obj.item())
    return str(obj)


def _pii_fields(
    reference: str,
    reconstructed: str,
    units: Sequence[str],
    *,
    full_text: str | None = None,
) -> dict:
    """一次算齐 PII-Ratio 与片段恢复率（供逐样本记录使用）。

    Args:
        reference: 分母参考文本 ``T_ref``——模型实际看到的有效 token 解码出的文本
            （``tokenizer.decode(有效 ids, skip_special_tokens=True)``）。
            主字段 ``pii_ratio`` / ``pii_orig`` / ``pii_rec`` / 片段恢复率都基于它。
        reconstructed: 重建文本 ``T*``。
        units: PUPA 的 ``pii_units``。
        full_text: 完整 ``user_query``；给出时额外记录 ``pii_ratio_full``（分母为完整
            原文，旧口径），只用于审计对照，不参与 ``pii_ratio``。

    Returns:
        ``pii_orig/pii_rec/pii_ratio/pii_ratio_defined/n_units/n_recovered/
        recovered_frac``，以及（``full_text`` 非空时）``pii_ratio_full`` /
        ``pii_ratio_full_defined``。
    """
    ratio = pii_ratio(reference, reconstructed, units)
    recovered = pii_units_recovered(reference, reconstructed, units)
    fields = {
        "pii_orig": float(ratio["pii_orig"]),
        "pii_rec": float(ratio["pii_rec"]),
        "pii_ratio": ratio["ratio"],
        "pii_ratio_defined": bool(ratio["defined"]),
        "n_units": int(recovered["n_units"]),
        "n_recovered": int(recovered["n_recovered"]),
        "recovered_frac": float(recovered["recovered_frac"]),
    }
    if full_text is not None:
        full = pii_ratio(full_text, reconstructed, units)
        fields["pii_ratio_full"] = full["ratio"]
        fields["pii_ratio_full_defined"] = bool(full["defined"])
    return fields


# --------------------------------------------------------------------------- #
# LM 打分函数（BeamClean 的 `lm_score_fn`）
# --------------------------------------------------------------------------- #
def make_lm_score_fn(lm, *, chunk: int = 0, device=None):
    """构造 BeamClean 用的 LM 打分函数 ``prefix_ids [N, t] -> [N]``。

    打分口径是**标准自回归对数似然**（已按主控裁决修正，不再是同位置 gather）::

        score(tokens[:, :t]) = Σ_{i=2..t} log p(tokens[:, i] | tokens[:, :i])

    实现：整段前缀喂给 LM（``input_ids=prefix``、``attention_mask=ones``），用
    ``logits[:, :-1]``（位置 ``i-1`` 的预测）对 ``tokens[:, 1:]``（待预测的 ``i``
    号 token）做 ``log_softmax`` + ``gather``，再沿位置求和；``t == 1`` 时没有可
    条件化的 next-token，直接返回 0（``t == 0`` 同理）。旧实现用同位置 gather
    （``logits[:, t]`` 对 ``tokens[:, t]``），分数里包含“看到自己”的项，语义错误。

    前缀不含 padding（BeamClean 每步传入的前缀长度就是有效 token 数），所以
    ``attention_mask`` 全 1；求和前仍显式乘 mask，保证 padding 位置不计分。

    Args:
        lm: ``AutoModelForCausalLM`` 或任何返回 ``.logits [N, t, V]`` 的模块。
        chunk: 每次前向的前缀条数；``0`` 表示按 ``_LM_BUDGET_BYTES`` 自适应。
        device: 缺省取 ``lm`` 参数所在设备。

    Returns:
        ``lm_score_fn(prefix_ids) -> torch.Tensor [N]``（float64，CPU，不参与梯度）。
    """
    param = next(lm.parameters(), None)
    model_device = param.device if param is not None else torch.device(device or "cpu")
    vocab = int(lm.get_input_embeddings().weight.shape[0])
    chunk = int(chunk)
    if chunk < 0:
        raise ValueError(f"chunk 必须非负（0 = 自适应），实际 {chunk}")

    @torch.no_grad()
    def lm_score_fn(prefix_ids) -> torch.Tensor:
        ids = torch.as_tensor(prefix_ids, dtype=torch.long)
        if ids.dim() != 2:
            raise ValueError(f"prefix_ids 必须为 [N, t]，实际 {tuple(ids.shape)}")
        n_rows, seq_len = int(ids.shape[0]), int(ids.shape[1])
        if n_rows == 0:
            return torch.zeros(0, dtype=torch.float64)
        if seq_len <= 1:
            # t == 0 / t == 1：没有可条件化的 next-token，按约定返回 0。
            return torch.zeros(n_rows, dtype=torch.float64)
        step = chunk if chunk > 0 else _auto_lm_chunk(seq_len, vocab)
        scores = torch.empty(n_rows, dtype=torch.float64)
        for start in range(0, n_rows, step):
            block = ids[start : start + step].to(model_device)
            mask = torch.ones_like(block)  # 前缀全是有效 token，无 padding
            logits = lm(input_ids=block, attention_mask=mask).logits
            # 标准自回归：位置 i-1 的 logits 预测的才是 token i（i = 2..t），
            # 因此 logits 去掉最后一个位置、target 去掉第一个位置。
            step_logits = logits[:, :-1, :]
            targets = block[:, 1:]
            step_mask = mask[:, 1:].to(torch.float32)
            # float32 上做 log_softmax：bf16/fp16 的直接 log_softmax 精度不足。
            log_probs = torch.log_softmax(step_logits.to(torch.float32), dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            # 显式 mask 掉 padding（前缀无 padding 时 mask 全 1，结果不变）。
            masked = token_log_probs.to(torch.float64) * step_mask.to(torch.float64)
            scores[start : start + step] = masked.sum(dim=1).cpu()
        return scores

    return lm_score_fn


# --------------------------------------------------------------------------- #
# SGT checkpoint 载入
# --------------------------------------------------------------------------- #
def _infer_sgt_dims(state: Mapping[str, torch.Tensor]) -> dict:
    """从 state_dict 的键/形状反推 SGT 结构（``n_heads`` 不可反推）。"""
    required = ("input_proj.weight", "pos_emb.weight", "head.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(
            f"SGT state_dict 缺少键 {missing}；收到的前 3 个键：{list(state)[:3]}"
        )
    hidden, dim = (int(value) for value in state["input_proj.weight"].shape)
    head_rows = int(state["head.weight"].shape[0])
    if head_rows != 2 * dim:
        raise ValueError(
            f"head.weight 形状 {tuple(state['head.weight'].shape)} 与 input_proj "
            f"的 d={dim} 不一致（应为 [2d, hidden]）"
        )
    layer_ids = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("encoder.layers.") and len(key.split(".")) > 2
    }
    ffn_rows = int(state["encoder.layers.0.linear1.weight"].shape[0])
    return {
        "d": dim,
        "hidden": hidden,
        "n_layers": max(layer_ids) + 1 if layer_ids else 1,
        "ffn_mult": ffn_rows // hidden,
        "max_len": int(state["pos_emb.weight"].shape[0]),
    }


def load_sgt_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    d: int | None = None,
    n_heads: int | None = None,
) -> tuple[SGTModel, dict]:
    """载入 :meth:`sgt.train.train_sgt` 保存的 checkpoint（``{"sgt": state_dict, ...}``）。

    结构（d / hidden / n_layers / ffn_mult / max_len）从 state_dict 形状反推；
    ``n_heads`` 无法从参数形状反推（``in_proj_weight`` 与头数无关）：调用方显式给的
    ``n_heads`` 优先；未给时取 checkpoint 里的 ``sgt_config["n_heads"]``；都没有则默认 4
    （与 ``configs/ablation.yaml`` 的 ``sgt.n_heads`` 一致）。来源会记进返回的
    ``meta["n_heads_source"]``（``"cli"`` / ``"checkpoint"`` / ``"default"``），供结果 JSON
    声明。反推的其它结构（d / hidden / n_layers / ffn_mult / max_len）来自 state_dict 形状，
    若与实际训练结构不符，``strict=True`` 的 load 会直接报错。

    Raises:
        FileNotFoundError / KeyError / ValueError / RuntimeError: 路径、键、形状不符或
            ``d`` 与实际不一致（不静默加载）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SGT checkpoint 不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "sgt" in payload:
        state = payload["sgt"]
        step = payload.get("step")
        extra = payload.get("sgt_config") or {}
    else:  # 兼容裸 state_dict
        state = payload
        step = None
        extra = {}
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint 里的 sgt 必须是 state_dict，实际 {type(state).__name__}")
    state = {
        (key[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }

    dims = _infer_sgt_dims(state)
    if d is not None and int(dims["d"]) != int(d):
        raise ValueError(
            f"checkpoint 的 d={dims['d']} 与目标 LM 的 hidden_size={int(d)} 不一致"
        )
    if n_heads is not None:
        heads, n_heads_source = int(n_heads), "cli"
    elif isinstance(extra, Mapping) and extra.get("n_heads") is not None:
        heads, n_heads_source = int(extra["n_heads"]), "checkpoint"
    else:
        heads, n_heads_source = 4, "default"

    model = SGTModel(
        dims["d"],
        hidden=dims["hidden"],
        n_layers=dims["n_layers"],
        n_heads=heads,
        ffn_mult=dims["ffn_mult"],
        max_len=dims["max_len"],
    ).to(device=torch.device(device), dtype=dtype)
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:  # 形状不匹配多半是 n_heads/n_layers 推断错
        raise RuntimeError(
            f"载入 SGT checkpoint 失败（推断结构 {dims}，n_heads={heads}）: {exc}"
        ) from exc
    model.eval()

    meta = {
        "path": str(path),
        "step": None if step is None else int(step),
        "d": int(dims["d"]),
        "hidden": int(dims["hidden"]),
        "n_layers": int(dims["n_layers"]),
        "n_heads": heads,
        "n_heads_source": n_heads_source,
        "ffn_mult": int(dims["ffn_mult"]),
        "max_len": int(dims["max_len"]),
        "dtype": str(dtype),
    }
    return model, meta


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _collect_examples(examples: Sequence[Mapping[str, Any]]) -> tuple[list[dict], int]:
    """抽取 ``user_query`` 与 ``pii_units``；空文本样本跳过并计数（显式记录）。"""
    records: list[dict] = []
    skipped = 0
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise TypeError(
                f"样例 #{index} 必须是 mapping，实际 {type(example).__name__}"
            )
        text = example.get(TEXT_FIELD)
        if not isinstance(text, str):
            raise ValueError(
                f"样例 #{index} 缺少字符串字段 {TEXT_FIELD!r}（SPEC A20：被保护文本必须"
                f"是 {TEXT_FIELD}，不能用 target_response）"
            )
        if not text.strip():
            skipped += 1
            continue
        records.append(
            {
                "index": index,
                "conversation_hash": example.get("conversation_hash"),
                "predicted_category": example.get("predicted_category"),
                "text": text,
                "units": split_pii_units(example.get("pii_units") or []),
            }
        )
    return records, skipped


def _aggregate(entries: Sequence[Mapping[str, Any]], *, sigma_hats=None) -> dict:
    """按 token 汇总 FR、按样本汇总 PII-Ratio；无定义值时返回 ``None``。

    主口径 ``pii_ratio_mean`` 用 ``T_ref``（模型实际看到的截断文本）做分母；
    ``pii_ratio_full_mean`` 是旧的整文口径，只用于审计对照。
    """
    n_tokens = sum(int(entry["n_tokens"]) for entry in entries)
    n_errors = sum(int(entry["n_errors"]) for entry in entries)
    ratios = [
        float(entry["pii_ratio"])
        for entry in entries
        if entry.get("pii_ratio_defined") and entry.get("pii_ratio") is not None
    ]
    ratios_full = [
        float(entry["pii_ratio_full"])
        for entry in entries
        if entry.get("pii_ratio_full_defined")
        and entry.get("pii_ratio_full") is not None
    ]
    recovered = [
        float(entry["recovered_frac"])
        for entry in entries
        if int(entry.get("n_units", 0)) > 0
    ]
    summary = {
        "fr": (100.0 * n_errors / n_tokens) if n_tokens else None,
        "pii_ratio_mean": (sum(ratios) / len(ratios)) if ratios else None,
        "pii_ratio_full_mean": (
            (sum(ratios_full) / len(ratios_full)) if ratios_full else None
        ),
        "n_samples": len(entries),
        "n_tokens": n_tokens,
        "n_errors": n_errors,
        "n_pii_ratio_defined": len(ratios),
        "n_pii_ratio_undefined": len(entries) - len(ratios),
        "n_pii_ratio_full_defined": len(ratios_full),
        "pii_units_recovered_mean": (sum(recovered) / len(recovered)) if recovered else None,
        "n_samples_with_pii_units": len(recovered),
    }
    if sigma_hats is not None:
        summary["sigma_hat_mean"] = (
            sum(sigma_hats) / len(sigma_hats) if sigma_hats else None
        )
        summary["sigma_hat_min"] = min(sigma_hats) if sigma_hats else None
        summary["sigma_hat_max"] = max(sigma_hats) if sigma_hats else None
    return summary


def run_pupa_eval(
    examples: Sequence[Mapping[str, Any]],
    *,
    tokenizer,
    table: torch.Tensor,
    lm=None,
    sgt: torch.nn.Module | None = None,
    sgt_meta: Mapping[str, Any] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    skip_beamclean: bool = False,
    beam: int = 8,
    topk: int = 32,
    n_iter: int = 3,
    lm_chunk: int = 0,
    nn_chunk: int = 0,
    progress: bool = True,
) -> dict[str, Any]:
    """PUPA 隐私评测主流程（NN / BeamClean 重建 + PII-Ratio），返回可 JSON 序列化的 dict。

    Args:
        examples: PUPA 样本（mapping），只用 ``user_query`` 与 ``pii_units``；
            ``target_response`` **不参与**任何计算。
        tokenizer: 至少支持 ``__call__(texts, add_special_tokens=..., padding=False,
            truncation=False)`` 与 ``decode(ids, skip_special_tokens=True)``。
        table: embedding 表 ``[V, d]``（通常 ``lm.get_input_embeddings().weight``）；
            所有张量的设备/dtype 以它为基准。
        lm: 因果 LM，仅 BeamClean 需要；``skip_beamclean=False`` 且未提供时**直接报错**。
        sgt: 已载入的 :class:`sgt.model.SGTModel`；``None`` 表示无保护基线（``x̃ = x``）。
        sgt_meta: 写进 ``meta.sgt`` 的附加信息（checkpoint 路径/结构等）。
        max_len: tokenize 的最大长度（右截断）；PII-Ratio 的分母参考文本是截断后的
            有效 token 解码文本 ``T_ref``，不是完整 ``user_query``。
        batch_size: 批大小（批内右 padding）。
        seed: SGT 采样 ``torch.Generator`` 的种子（固定 → 结果可复现）。
        skip_beamclean: 显式跳过 BeamClean；结果里记 ``status="skipped"``。
        beam / topk / n_iter: 透传给 :func:`sgt.attacks.beamclean`。
        lm_chunk: LM 打分分块（0 = 自适应）。
        nn_chunk: NN 攻击分块（0 = 自适应）。
        progress: 是否向 stderr 打印进度。

    Returns:
        ``{"nn": {...}, "beamclean": {...}, "rows": [...], "meta": {...}}``；
        所有浮点已清洗为有限值或 ``None``（不会有 NaN/Inf）。逐样本 ``rows`` 含完整
        ``user_query``、``pii_reference_text``（= ``T_ref``），以及 NN / BeamClean 各自的
        ``pii_ratio``（分母 ``T_ref``）与 ``pii_ratio_full``（分母完整原文，审计对照）。
    """
    started = time.time()
    table = torch.as_tensor(table)
    if table.ndim != 2:
        raise ValueError(f"table 必须为 [V, d]，实际 {tuple(table.shape)}")
    if not table.is_floating_point():
        raise TypeError(f"table 必须是浮点张量，实际 dtype={table.dtype}")
    if int(max_len) <= 0:
        raise ValueError(f"max_len 必须为正整数，实际 {max_len}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size 必须为正整数，实际 {batch_size}")
    if not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer 必须提供 decode(ids, skip_special_tokens=True)")
    if not skip_beamclean and lm is None:
        raise ValueError(
            "未提供 lm，无法运行 BeamClean；如确要跳过必须显式传 skip_beamclean=True"
            "（不得静默跳过并当已跑）"
        )

    records, n_skipped = _collect_examples(examples)
    if not records:
        raise ValueError("没有可评测的样例（examples 为空或 user_query 全为空串）")

    device = table.device
    vocab, dim = int(table.shape[0]), int(table.shape[1])
    nn_chunk_eff = int(nn_chunk) if int(nn_chunk) > 0 else _auto_nn_chunk(vocab)

    sgt_device = None
    sgt_dtype = None
    if sgt is not None:
        sgt_device = _module_device(sgt)
        sgt_dtype = _module_dtype(sgt)
        if sgt_device != device:
            raise ValueError(
                f"SGT 在 {sgt_device}，embedding 表在 {device}；请放到同一设备（不隐式搬运）"
            )
        sgt_max_len = int(getattr(sgt, "max_len", max_len))
        if sgt_max_len < int(max_len):
            raise ValueError(
                f"SGT 的 max_len={sgt_max_len} 小于 max_len={int(max_len)}，无法前向"
            )
    generator = (
        torch.Generator(device=device).manual_seed(int(seed)) if sgt is not None else None
    )
    lm_score_fn = None if skip_beamclean else make_lm_score_fn(lm, chunk=lm_chunk)

    texts = [record["text"] for record in records]
    batches = list(
        make_batches(
            texts,
            tokenizer,
            batch_size=int(batch_size),
            max_len=int(max_len),
            seed=int(seed),
            shuffle=False,  # 保持与 examples 同序，逐样本记录可对齐
        )
    )

    rows: list[dict] = []
    sigma_chunks: list[torch.Tensor] = []
    sigma_hats: list[float] = []
    cursor = 0
    for batch_index, batch in enumerate(batches):
        ids_cpu = batch["input_ids"].to(torch.long)
        mask_cpu = batch["attention_mask"].to(torch.long)
        size = int(ids_cpu.shape[0])
        batch_records = records[cursor : cursor + size]
        cursor += size

        ids = ids_cpu.to(device)
        mask_bool = mask_cpu.to(device).bool()
        x = table[ids]  # [B, T, d]
        if sgt is not None:
            x_in = x.to(dtype=sgt_dtype)
            with torch.no_grad():  # 评测不反传：x̃ 不需要图（也避免下游 float() 告警）
                mu, log_sigma2 = sgt(x_in, mask_bool)
                sigma = torch.exp(0.5 * log_sigma2)
                sigma_chunks.append(sigma[mask_bool].reshape(-1).float().cpu())
                x_tilde = sample_sgt(x_in, mu, sigma, generator=generator)
        else:
            x_tilde = x

        pred_nn = nn_attack(x_tilde.reshape(-1, dim), table, chunk=nn_chunk_eff)
        pred_nn = pred_nn.reshape(size, -1).cpu()

        for row_index, record in enumerate(batch_records):
            valid = mask_cpu[row_index].bool()
            target_ids = ids_cpu[row_index][valid]
            n_tokens = int(target_ids.numel())
            if n_tokens == 0:
                raise RuntimeError(
                    f"样例 #{record['index']} tokenize 后没有有效 token（检查 tokenizer 的 "
                    "add_special_tokens 行为）"
                )
            nn_ids = pred_nn[row_index][valid].to(torch.long)
            nn_errors = int((nn_ids != target_ids).sum())
            nn_text = _decode(tokenizer, nn_ids)
            # PII-Ratio 的参考文本 T_ref：模型实际看到的有效 token 的解码文本
            # （max_len 截断后），不是完整 user_query；完整原文另记 pii_ratio_full。
            reference_text = _decode(tokenizer, target_ids)

            row = {
                "index": int(record["index"]),
                "conversation_hash": record["conversation_hash"],
                "predicted_category": record["predicted_category"],
                "n_tokens": n_tokens,
                TEXT_FIELD: record["text"],
                "pii_reference_text": reference_text,
                "pii_units": list(record["units"]),
                "nn": {
                    "n_tokens": n_tokens,
                    "n_errors": nn_errors,
                    "fr": metrics.failure_rate(nn_ids, target_ids),
                    "reconstructed": nn_text,
                    **_pii_fields(
                        reference_text,
                        nn_text,
                        record["units"],
                        full_text=record["text"],
                    ),
                },
                "beamclean": {"status": "skipped"},
            }

            if lm_score_fn is not None:
                x_row = x_tilde[row_index][mask_bool[row_index]].unsqueeze(0)
                bc_ids, sigma_hat = beamclean(
                    x_row,
                    table,
                    lm_score_fn,
                    beam=int(beam),
                    topk=int(topk),
                    n_iter=int(n_iter),
                )
                bc_ids = bc_ids.reshape(-1).to(torch.long).cpu()
                bc_errors = int((bc_ids != target_ids).sum())
                bc_text = _decode(tokenizer, bc_ids)
                sigma_hats.append(float(sigma_hat))
                row["beamclean"] = {
                    "status": "ok",
                    "n_tokens": n_tokens,
                    "n_errors": bc_errors,
                    "fr": metrics.failure_rate(bc_ids, target_ids),
                    "sigma_hat": float(sigma_hat),
                    "reconstructed": bc_text,
                    **_pii_fields(
                        reference_text,
                        bc_text,
                        record["units"],
                        full_text=record["text"],
                    ),
                }

            rows.append(row)

        if progress:
            print(
                f"[run_pupa] batch {batch_index + 1}/{len(batches)} "
                f"samples={len(rows)}/{len(records)}",
                file=sys.stderr,
            )

    if cursor != len(records):  # pragma: no cover - make_batches 保证覆盖全部
        raise RuntimeError(f"batch 覆盖 {cursor} 条，与 {len(records)} 条样例不一致")

    nn_scripts = [
        {
            "index": row["index"],
            "conversation_hash": row["conversation_hash"],
            "n_tokens": row["nn"]["n_tokens"],
            "n_errors": row["nn"]["n_errors"],
            **_only_pii(row["nn"]),
        }
        for row in rows
    ]
    nn_summary = {"status": "ok", "scripts": nn_scripts, **_aggregate(nn_scripts)}

    if skip_beamclean:
        beamclean_summary = {
            "status": "skipped",
            "fr": None,
            "pii_ratio_mean": None,
            "pii_ratio_full_mean": None,
            "scripts": [],
            "sigma_hat_mean": None,
            "n_samples": 0,
        }
    else:
        bc_scripts = [
            {
                "index": row["index"],
                "conversation_hash": row["conversation_hash"],
                "n_tokens": row["beamclean"]["n_tokens"],
                "n_errors": row["beamclean"]["n_errors"],
                "sigma_hat": row["beamclean"]["sigma_hat"],
                **_only_pii(row["beamclean"]),
            }
            for row in rows
        ]
        beamclean_summary = {
            "status": "ok",
            "scripts": bc_scripts,
            **_aggregate(bc_scripts, sigma_hats=sigma_hats),
        }

    sigma_stats = None
    if sigma_chunks:
        sigma_all = torch.cat(sigma_chunks).to(torch.float64)
        sigma_stats = {
            "mean": float(sigma_all.mean()),
            "std": float(sigma_all.std(unbiased=False)),
            "min": float(sigma_all.min()),
            "max": float(sigma_all.max()),
            "n": int(sigma_all.numel()),
        }

    meta = {
        "text_field": TEXT_FIELD,
        "n_input_examples": len(examples),
        "n_used_examples": len(records),
        "n_skipped_empty_text": n_skipped,
        "max_len": int(max_len),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "device": str(device),
        "table_shape": [vocab, dim],
        "table_dtype": str(table.dtype),
        "nn_chunk": nn_chunk_eff,
        "lm_chunk": int(lm_chunk),
        "beamclean_skipped": bool(skip_beamclean),
        "beam": int(beam),
        "topk": int(topk),
        "n_iter": int(n_iter),
        "sgt": (
            {"applied": True, **dict(sgt_meta or {})}
            if sgt is not None
            else {"applied": False}
        ),
        "sigma": sigma_stats,
        "fr_definition": (
            "100 * #{reconstructed != target} / #{valid tokens}；padding 不计；"
            "跨样本按 token 汇总（逐样本 FR 的 token 加权平均）"
        ),
        "pii_definition": (
            "PII-Ratio = (T* 中 PII 字符占比) / (T_ref 中 PII 字符占比)；PII 用 PUPA 的 "
            "pii_units 做大小写不敏感、最长优先、不重叠的字符匹配（非 PAPILLON "
            "LLM-as-judge，见 SPEC A10）"
        ),
        "pii_reference": "truncated_decoded",
        "pii_reference_definition": (
            "T_ref = tokenizer.decode(截断到 max_len 后的有效 token ids, "
            "skip_special_tokens=True)，即模型实际看到的文本；"
            "T* = 重建 ids 的解码文本。逐样本另记 pii_reference_text（T_ref 全文）与 "
            "pii_ratio_full（分母为完整 user_query 的旧口径，仅审计对照，不参与主指标）"
        ),
        "duration_seconds": float(time.time() - started),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": _versions(),
    }

    return _sanitize(
        {"nn": nn_summary, "beamclean": beamclean_summary, "rows": rows, "meta": meta}
    )


def _only_pii(entry: Mapping[str, Any]) -> dict:
    keys = (
        "pii_orig",
        "pii_rec",
        "pii_ratio",
        "pii_ratio_defined",
        "pii_ratio_full",
        "pii_ratio_full_defined",
        "n_units",
        "n_recovered",
        "recovered_frac",
    )
    return {key: entry[key] for key in keys}


def _versions() -> dict:
    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover - 仅记录用
        versions["transformers"] = None
    return versions


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M8b：PUPA 上的 NN / BeamClean 重建与 PII-Ratio"
    )
    parser.add_argument("--model", required=True, help="目标 LM（本地目录或 HF id）")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help=f"PUPA 取前 N 条（默认 {DEFAULT_MAX_EXAMPLES}；全量 237）",
    )
    parser.add_argument(
        "--max-len", type=int, default=DEFAULT_MAX_LEN, help="tokenize 最大长度"
    )
    parser.add_argument(
        "--pupa-config",
        default="pupa_tnb",
        choices=("pupa_tnb", "pupa_new"),
        help="PUPA 子集（默认 pupa_tnb：237 条）",
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps"), default="cpu", help="LM/SGT/攻击的运行设备"
    )
    parser.add_argument(
        "--dtype",
        choices=_DTYPE_CHOICES,
        default="float32",
        help="LM 精度；auto = 用模型 config 指定的 dtype（默认 float32）",
    )
    parser.add_argument(
        "--sgt-ckpt",
        default=None,
        help="SGT checkpoint（train_sgt 的 {sgt: state_dict} 格式）；缺省为无保护基线",
    )
    parser.add_argument(
        "--sgt-heads",
        type=int,
        default=None,
        help="SGT 注意力头数（state_dict 无法反推；checkpoint 无 sgt_config 时默认 4）",
    )
    parser.add_argument(
        "--skip-beamclean", action="store_true", help="显式跳过 BeamClean（结果里标记 skipped）"
    )
    parser.add_argument("--beam", type=int, default=8, help="BeamClean beam 宽度")
    parser.add_argument("--topk", type=int, default=32, help="BeamClean 每位置候选数")
    parser.add_argument("--n-iter", type=int, default=3, help="BeamClean sigma EM 迭代数")
    parser.add_argument("--seed", type=int, default=0, help="SGT 采样与数据顺序的种子")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="tokenize 批大小"
    )
    parser.add_argument(
        "--lm-chunk", type=int, default=0, help="LM 打分分块（0 = 按显存预算自适应）"
    )
    parser.add_argument(
        "--nn-chunk", type=int, default=0, help="NN 攻击分块（0 = 按词表大小自适应）"
    )
    parser.add_argument("--tag", default=None, help="结果标签（默认由 --sgt-ckpt 文件名推导）")
    parser.add_argument(
        "--out",
        default=None,
        help="结果 JSON 路径（默认 results/pupa_<tag>.json；逐样本明细为同名 _rows.jsonl）",
    )
    return parser.parse_args(argv)


def _default_tag(sgt_ckpt: str | None) -> str:
    if not sgt_ckpt:
        return "baseline"
    stem = Path(sgt_ckpt).stem
    return stem[len("ckpt_") :] if stem.startswith("ckpt_") else stem


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_CHOICES:
        raise ValueError(f"dtype 只能是 {_DTYPE_CHOICES}，实际 {name!r}")
    return getattr(torch, name)


def _load_lm(model_name: str, *, device: torch.device, dtype_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dtype_name == "auto":
        lm = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=_resolve_dtype(dtype_name)
        )
    lm = lm.to(device).eval()
    for param in lm.parameters():
        param.requires_grad_(False)
    return tokenizer, lm


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tag = args.tag or _default_tag(args.sgt_ckpt)
    out_path = Path(args.out or f"results/pupa_{tag}.json")
    rows_path = out_path.with_name(f"{out_path.stem}_rows.jsonl")

    if args.device == "mps" and not torch.backends.mps.is_available():
        return _fail("--device mps 不可用（torch.backends.mps.is_available() == False）")
    device = torch.device(args.device)

    try:
        examples = load_pupa(config=args.pupa_config, max_examples=args.max_examples)
    except Exception as exc:
        return _fail(f"加载 PUPA 失败: {type(exc).__name__}: {exc}")

    try:
        tokenizer, lm = _load_lm(args.model, device=device, dtype_name=args.dtype)
    except Exception as exc:
        return _fail(f"加载模型 {args.model} 失败: {type(exc).__name__}: {exc}")

    table = lm.get_input_embeddings().weight.detach()
    dim = int(table.shape[1])

    sgt = None
    sgt_meta: dict = {}
    if args.sgt_ckpt:
        try:
            sgt, sgt_meta = load_sgt_from_checkpoint(
                args.sgt_ckpt,
                device=device,
                dtype=table.dtype,
                d=dim,
                n_heads=args.sgt_heads,
            )
        except Exception as exc:
            return _fail(f"载入 SGT checkpoint 失败: {type(exc).__name__}: {exc}")

    try:
        result = run_pupa_eval(
            examples,
            tokenizer=tokenizer,
            table=table,
            lm=lm,
            sgt=sgt,
            sgt_meta=sgt_meta,
            max_len=int(args.max_len),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            skip_beamclean=bool(args.skip_beamclean),
            beam=int(args.beam),
            topk=int(args.topk),
            n_iter=int(args.n_iter),
            lm_chunk=int(args.lm_chunk),
            nn_chunk=int(args.nn_chunk),
        )
    except Exception:
        traceback.print_exc()
        return 1

    payload = {
        "tag": tag,
        "model": args.model,
        "sgt_ckpt": args.sgt_ckpt,
        "max_examples": int(args.max_examples),
        "max_len": int(args.max_len),
        "nn": result["nn"],
        "beamclean": result["beamclean"],
        "meta": {
            **result["meta"],
            "pupa_config": args.pupa_config,
            # M5 的 load_pupa(...).meta 原样记录（数据来源/条数/抓取时间等）；
            # 本评测直接读 user_query，不依赖 M5 的 PUPA_TEXT_FIELD / build_prompt。
            "pupa_dataset_meta": getattr(examples, "meta", {}),
            "dtype": args.dtype,
            "tag": tag,
        },
    }
    payload = _sanitize(payload)

    try:
        _write_json(out_path, payload)
        _write_jsonl(rows_path, result["rows"])
    except Exception:
        traceback.print_exc()
        return 1

    def _fmt(value):
        return "None" if value is None else f"{value:.4f}"

    nn_metrics = payload["nn"]
    bc_metrics = payload["beamclean"]
    print(
        f"[run_pupa] tag={tag} model={args.model} "
        f"n={payload['meta']['n_used_examples']}/{payload['meta']['n_input_examples']} "
        f"max_len={args.max_len} sgt={'yes' if sgt is not None else 'no'}"
    )
    print(
        f"[run_pupa] nn        fr={_fmt(nn_metrics['fr'])}% "
        f"pii_ratio={_fmt(nn_metrics['pii_ratio_mean'])} "
        f"(defined {nn_metrics['n_pii_ratio_defined']}/{nn_metrics['n_samples']})"
    )
    if bc_metrics["status"] == "skipped":
        print("[run_pupa] beamclean SKIPPED (--skip-beamclean)")
    else:
        print(
            f"[run_pupa] beamclean fr={_fmt(bc_metrics['fr'])}% "
            f"pii_ratio={_fmt(bc_metrics['pii_ratio_mean'])} "
            f"sigma_hat={_fmt(bc_metrics['sigma_hat_mean'])}"
        )
    print(f"[run_pupa] wrote {out_path}, {rows_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
