#!/usr/bin/env python3
"""M7 常数噪声实验（论文 §7.2 / Figure 4）：不训练，只加常数高斯噪声。

用法::

    ./.venv/bin/python experiments/run_constant_noise.py --model .models/llama-3.2-1b-instruct \
        --eval-examples 200 --eval-max-len 64 --eval-tokens 2000 \
        --sigmas 1.65,2.72,4.48,7.39,12.18,20.09,33.12 --device mps \
        --out results/constnoise.json

对每个 σ（标准差）：

- ``x̃ = x + σ·u``，``u ~ N(0, I)`` 逐 token（固定 seed，显式 Generator）；NN-FR 用
  :func:`sgt.metrics.nn_reconstruct` + :func:`sgt.metrics.failure_rate`（与消融同一实现）。
- ``epsilon = metrics.eps_dp(σ, sensitivity=Δ, delta=1/|V|)``（SPEC A14 的高斯机制
  ε-DP），``Δ`` = embedding 表行范数的 **99 分位**（``sensitivity_rule="norm_p99"``，
  复现选择：论文未定义 Δ）。``log_eps = log(epsilon)``，与 Fig 4 的 x 轴对应。

输出 JSON::

    {"sigmas": [...], "rows": [{"sigma","log_eps","epsilon","nn_fr","table_norm_p99"}, ...],
     "utility": null, ...}

``--utility-cmd`` 只记录不执行（Utility 由 M8 的 lm-eval 管线负责，这里保持
``utility: null``，不用常数噪声冒充 Utility 结论）。

退出码：0 成功；2 配置/数据/模型等前置条件不可用；1 评测本身失败。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sgt import metrics  # noqa: E402

from run_ablation import (  # noqa: E402  （同目录复用；两个脚本保持同一 JSON/评测口径）
    auto_chunk,
    load_eval_examples,
    load_lm,
    tokenize_eval_sequences,
    write_json,
)

__all__ = ["constant_noise", "nn_fr_for_sigma", "table_norm_percentile", "main"]

DEFAULT_MODEL = ".models/llama-3.2-1b-instruct"
DEFAULT_SIGMAS = "1.65,2.72,4.48,7.39,12.18,20.09,33.12"
DEFAULT_OUT = "results/constnoise.json"

#: 计算 embedding 行范数时每块的元素预算（float64 约 32 MB/块）。
_NORM_CHUNK_ELEMS = 4_000_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M7 常数噪声：σ 曲线上的 NN-FR 与 ε-DP（Figure 4）"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="本地 LM 目录")
    parser.add_argument("--eval-examples", type=int, default=200, help="Alpaca 评估条数")
    parser.add_argument("--eval-max-len", type=int, default=64, help="评估序列截断长度")
    parser.add_argument("--eval-tokens", type=int, default=2000, help="评估有效 token 上限")
    parser.add_argument("--sigmas", default=DEFAULT_SIGMAS, help="逗号分隔的 σ（标准差）")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="噪声采样种子")
    parser.add_argument("--chunk", type=int, default=1024, help="NN 重建分块")
    parser.add_argument("--utility-cmd", default=None, help="（可选）Utility 命令模板；仅记录")
    parser.add_argument("--out", default=DEFAULT_OUT, help="结果 JSON 路径")
    args = parser.parse_args(argv)

    try:
        sigmas = [float(part) for part in str(args.sigmas).split(",") if part.strip()]
    except ValueError as exc:
        parser.error(f"--sigmas 解析失败: {exc}")
    if not sigmas:
        parser.error("--sigmas 不能为空")
    if any(not math.isfinite(sigma) or sigma <= 0.0 for sigma in sigmas):
        parser.error("--sigmas 必须全是正的有限值（σ=0 没有 ε-DP 定义）")
    args.sigma_values = sigmas
    if args.eval_examples <= 0 or args.eval_max_len <= 0 or args.eval_tokens <= 0:
        parser.error("--eval-examples/--eval-max-len/--eval-tokens 必须为正整数")
    if args.seed < 0:
        parser.error("--seed 必须非负")
    if args.chunk <= 0:
        parser.error("--chunk 必须为正整数")
    return args


def _fail(message: str, code: int = 2) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- #
# 常数噪声与 NN-FR
# --------------------------------------------------------------------------- #
def constant_noise(
    x: torch.Tensor, sigma: float, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """逐 token 常数高斯噪声：``x + σ·u``，``u ~ N(0, I)``（Eq (1) 的 µ=0、σ 常数特例）。"""
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma 必须是非负有限值，实际 {sigma}")
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return x + sigma * noise


def nn_fr_for_sigma(
    x_clean: torch.Tensor,
    target_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    sigma: float,
    *,
    seed: int = 0,
    chunk: int | None = None,
) -> float:
    """给定 σ 的常数噪声 NN 重建失败率（百分比）；固定 seed 逐次可复现。"""
    x_clean = torch.as_tensor(x_clean)
    if x_clean.ndim != 2:
        raise ValueError(f"x_clean 必须为 [N, d]，实际 {tuple(x_clean.shape)}")
    vocab_size = int(torch.as_tensor(embedding_table).shape[0])
    aux = auto_chunk(vocab_size) if chunk is None else int(chunk)
    if aux <= 0:
        raise ValueError(f"chunk 必须为正，实际 {aux}")
    generator = torch.Generator(device=x_clean.device).manual_seed(int(seed))
    x_tilde = constant_noise(x_clean, sigma, generator=generator)
    prediction = metrics.nn_reconstruct(x_tilde, embedding_table, chunk=aux)
    # metrics.failure_rate 内部转 float64（MPS 不支持 float64）：在 CPU 副本上比较。
    return metrics.failure_rate(
        prediction.detach().cpu(), torch.as_tensor(target_ids).detach().cpu()
    )


def table_norm_percentile(embedding_table: torch.Tensor, q: float = 99.0) -> float:
    """embedding 表全部行 L2 范数的 q 分位数（CPU、float64、分块）。

    MPS 不支持 float64（``.to(torch.float64)`` 会报 Undefined type Double），
    必须先 ``.cpu()`` 再升精度；分块避免 128256×2048 的 float64 临时内存。
    """
    table = torch.as_tensor(embedding_table)
    if table.ndim != 2:
        raise ValueError(f"embedding_table 必须为 [V, d]，实际 {tuple(table.shape)}")
    if not 0.0 < float(q) < 100.0:
        raise ValueError(f"q 必须在 (0, 100) 内，实际 {q}")
    row_size = max(1, _NORM_CHUNK_ELEMS // max(1, table.shape[1]))
    norms: list[torch.Tensor] = []
    for start in range(0, table.shape[0], row_size):
        block = table[start : start + row_size].detach().cpu().to(torch.float64)
        norms.append(torch.linalg.vector_norm(block, dim=-1))
    return float(torch.quantile(torch.cat(norms), float(q) / 100.0))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    if args.device == "mps" and not torch.backends.mps.is_available():
        return _fail("--device mps 不可用（torch.backends.mps.is_available() == False）")
    device = torch.device(args.device)
    out_path = Path(args.out)

    lm_start = time.perf_counter()
    try:
        tokenizer, lm = load_lm(args.model, device=device, dtype=torch.float32)
    except Exception as exc:
        return _fail(f"加载模型 {args.model} 失败: {type(exc).__name__}: {exc}")
    timings["lm_load_s"] = time.perf_counter() - lm_start

    embedding_layer = lm.get_input_embeddings()
    if embedding_layer is None or getattr(embedding_layer, "weight", None) is None:
        return _fail("lm.get_input_embeddings() 必须返回带 weight 的 embedding 层")
    embedding_table = embedding_layer.weight.detach()
    vocab_size = int(embedding_table.shape[0])
    hidden_size = int(embedding_table.shape[1])

    data_start = time.perf_counter()
    try:
        eval_examples, eval_meta = load_eval_examples(int(args.eval_examples))
        sequences, n_tokens = tokenize_eval_sequences(
            eval_examples, tokenizer, max_len=int(args.eval_max_len), n_tokens=int(args.eval_tokens)
        )
    except Exception as exc:
        return _fail(f"评估数据加载/预处理失败: {type(exc).__name__}: {exc}")
    if not sequences or n_tokens <= 0:
        return _fail("评估集没有取到任何有效 token", code=1)
    timings["data_s"] = time.perf_counter() - data_start

    token_ids = torch.tensor(
        [token for sequence in sequences for token in sequence], dtype=torch.long, device=device
    )
    if int(token_ids.numel()) != n_tokens:
        return _fail(f"token 数不一致：{int(token_ids.numel())} vs {n_tokens}", code=1)
    x_clean = embedding_table[token_ids]

    sensitivity_start = time.perf_counter()
    try:
        sensitivity = table_norm_percentile(embedding_table, 99.0)
    except Exception as exc:
        return _fail(f"计算 sensitivity（norm_p99）失败: {type(exc).__name__}: {exc}")
    timings["sensitivity_s"] = time.perf_counter() - sensitivity_start
    if not math.isfinite(sensitivity) or sensitivity <= 0.0:
        return _fail(f"sensitivity 必须是正的有限值，实际 {sensitivity}", code=1)
    delta_dp = 1.0 / float(vocab_size)

    rows: list[dict[str, Any]] = []
    eval_start = time.perf_counter()
    try:
        for index, sigma in enumerate(args.sigma_values):
            noise_seed = int(args.seed) * 100003 + int(index)
            nn_fr = nn_fr_for_sigma(
                x_clean,
                token_ids,
                embedding_table,
                sigma,
                seed=noise_seed,
                chunk=int(args.chunk),
            )
            epsilon = metrics.eps_dp(sigma, sensitivity=sensitivity, delta=delta_dp)
            rows.append(
                {
                    "sigma": float(sigma),
                    "log_eps": float(math.log(epsilon)) if epsilon > 0.0 else None,
                    "epsilon": float(epsilon),
                    "nn_fr": float(nn_fr),
                    "table_norm_p99": float(sensitivity),
                    "noise_seed": noise_seed,
                    "n_tokens": int(n_tokens),
                }
            )
    except Exception:
        traceback.print_exc()
        return 1
    timings["eval_s"] = time.perf_counter() - eval_start
    timings["total_s"] = time.perf_counter() - total_start

    payload = {
        "sigmas": [float(sigma) for sigma in args.sigma_values],
        "rows": rows,
        "utility": None,
        "utility_cmd": args.utility_cmd,
        "utility_note": "未实现：Utility 由 M8 的 lm-eval 管线负责，常数噪声不提供 Utility 结论",
        "model": str(args.model),
        "device": args.device,
        "n_tokens": int(n_tokens),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "sensitivity_rule": "norm_p99",
        "sensitivity_delta": float(sensitivity),
        "delta_dp": float(delta_dp),
        "noise": "x̃ = x + σ·u, u~N(0,I)（逐 token，固定 seed）",
        "eval": {
            "dataset": "tatsu-lab/alpaca",
            "examples": int(eval_meta["n_examples"]),
            "max_len": int(args.eval_max_len),
            "tokens_requested": int(args.eval_tokens),
            "tokens": int(n_tokens),
            "sequences": len(sequences),
            "seed": int(args.seed),
            "chunk": int(args.chunk),
        },
        "data": eval_meta,
        "epsilon_dp_note": (
            "ε = Δ·sqrt(2 ln(1.25/δ_dp))/σ，Δ=embedding 行范数 99 分位（SPEC A14 复现选择；"
            "论文未定义 Δ，Fig 4 的 log(ε) 轴映射有 BLOCKED 风险）"
        ),
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": list(sys.argv),
            "torch_version": torch.__version__,
            "torch_device": str(device),
            "torch_dtype": str(embedding_table.dtype),
        },
    }
    try:
        write_json(out_path, payload)
    except Exception as exc:
        return _fail(f"写结果 JSON 失败 {out_path}: {type(exc).__name__}: {exc}", code=1)

    print(
        f"[const_noise] model={args.model} n_tokens={n_tokens} V={vocab_size} "
        f"sensitivity(norm_p99)={sensitivity:.6f} delta_dp={delta_dp:.3e} "
        f"eval_s={timings['eval_s']:.1f} total_s={timings['total_s']:.1f} -> {out_path}"
    )
    for row in rows:
        log_eps = "None" if row["log_eps"] is None else f"{row['log_eps']:.4f}"
        print(
            f"[const_noise] sigma={row['sigma']:.2f} log_eps={log_eps} "
            f"epsilon={row['epsilon']:.4f} nn_fr={row['nn_fr']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
