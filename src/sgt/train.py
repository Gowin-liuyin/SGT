"""M6 训练回路：冻结目标 LLM，只训练 SGT（SPEC §3.5, §6；A12/A17/A19）。

训练目标（逐 step）：

    x         = E[input_ids]                         # embedding 查表，detach
    mu, lv    = SGT(x, mask)
    sigma     = exp(0.5·lv).clamp_min(sigma_min)
    x̃         = x + mu + sigma ⊙ u,  u ~ N(0,I)     # Eq (1)，显式 Generator
    L_U       = CE(softmax(z_clean/τ) ‖ softmax(z̃/τ))  # 只对有效 token 平均
    L         = L_U + Σ_t w_t · L_t                   # Eq (3)/(11)
    L.backward() -> clip -> AdamW.step()              # 梯度只到 SGT / x̃

约定与实现要点：

- 冻结：``lm.eval()`` + 所有参数 ``requires_grad_(False)``；训练器从不调用
  ``lm.to()/train()/zero_grad()``，也不读写 lm 的参数或 buffer（因此 BN/dropout 均关闭，
  buffer 逐位不变；见 ``tests/test_train.py``）。
- 独立批（A19）：第 ``s`` 步用 ``batches[s % n]``，其下一批 ``batches[(s+1) % n]`` 作为
  ``X_b'``；``x'``、``mu'``、``lv'`` 全部在 ``torch.no_grad()`` 下计算（不反传）。
- 随机性：采样 u 只由 ``torch.Generator(device=x.device).manual_seed(seed*100003+step)``
  决定，不使用全局 RNG。SGTModel 默认 ``dropout=0``，因此 ``sgt.train()`` 不引入额外随机性。
- 设备/精度：训练器不搬模型、不做隐式 dtype 转换，按传入 ``lm``/``sgt`` 的设备运行；
  会校验 ``sgt``、embedding 表、lm 输入 embedding 三者的 device/dtype 一致（不一致直接
  报错，避免静默降精度）。``TrainConfig.device`` / ``dtype`` 是记录字段，不作为强制转换
  指令（冻结的 lm 不允许被本模块搬动或换精度）。
- 中位数范数 T（Eq (10)）：对 embedding 表**全部行**的 L2 范数取 ``torch.median``
  （PAPER_ANALYSIS P26 口径）。范数在 CPU 上以 float64 分块计算（MPS 不支持 float64；
  分块避免 128256×2048 的 float64 全表约 2.1 GB 临时内存），只算一次。
- 日志精度：``loss`` 由 float64 的 ``loss_u`` 与各 term 值按同一公式在 Python 中重建，
  因此 ``history[i]["loss"] == loss_u + Σ w_t·term_t`` 在 double 下逐项成立（<1e-9），
  不受参数 dtype（float32/bf16）的求和舍入影响；反传用的总损失仍按各 term 的自然
  dtype 累加（MPS 不支持 float64）。
- ``log_every``：每 ``log_every`` 个 step（step 计数从 0 开始，完成第 s 步时判断
  ``(s+1) % log_every == 0``）保存一次 checkpoint 到 ``step_<s>.pt``；history 仍逐步记录。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .losses import (
    abs_cos_loss,
    comp_entropy_loss,
    cos_sim_loss,
    median_norm_loss,
    mi_loss,
    utility_loss,
)
from .sampling import sample_sgt

__all__ = ["TrainConfig", "train_sgt"]

#: 允许的混淆损失项（对应 losses.py 中的五个函数）
LOSS_TERMS: tuple[str, ...] = ("mi", "acs", "mnp", "cge", "bcs")

#: 计算 embedding 范数中位数时每块的元素预算（float64 约 32 MB/块）
_MEDIAN_CHUNK_ELEMS = 4_000_000


@dataclass
class TrainConfig:
    """M6 训练超参（A12：论文未给出的复现选择，全部写入结果 JSON）。"""

    steps: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    seed: int = 0
    device: str = "cpu"  # 记录字段；实际设备由传入 lm/sgt 决定
    dtype: str = "float32"  # 记录字段；实际精度由传入 lm/sgt/embedding 决定
    loss_terms: tuple[str, ...] = ("mi", "acs", "mnp")
    weights: dict[str, float] = field(default_factory=dict)  # 缺省 1.0
    log_every: int = 1
    sigma_min: float = 1e-6
    temperature: float = 1.0
    checkpoint_dir: str | None = None


# --------------------------------------------------------------------------- #
# 校验与工具
# --------------------------------------------------------------------------- #
def _validate_config(config: TrainConfig) -> None:
    if config.steps < 0:
        raise ValueError(f"steps 必须 ≥ 0，实际 {config.steps}")
    if not (math.isfinite(config.lr) and config.lr > 0.0):
        raise ValueError(f"lr 必须为正的有限值，实际 {config.lr!r}")
    if not (math.isfinite(config.weight_decay) and config.weight_decay >= 0.0):
        raise ValueError(f"weight_decay 必须 ≥ 0，实际 {config.weight_decay!r}")
    if config.grad_clip is not None and not (
        math.isfinite(config.grad_clip) and config.grad_clip > 0.0
    ):
        raise ValueError(f"grad_clip 必须是正的有限值或 None，实际 {config.grad_clip!r}")
    if config.log_every < 1:
        raise ValueError(f"log_every 必须 ≥ 1，实际 {config.log_every}")
    if not (math.isfinite(config.sigma_min) and config.sigma_min > 0.0):
        raise ValueError(f"sigma_min 必须为正的有限值，实际 {config.sigma_min!r}")
    if not (math.isfinite(config.temperature) and config.temperature > 0.0):
        raise ValueError(f"temperature 必须为正的有限值，实际 {config.temperature!r}")

    terms = tuple(config.loss_terms)
    unknown = [name for name in terms if name not in LOSS_TERMS]
    if unknown:
        raise ValueError(f"未知的 loss_terms {unknown}；允许 {list(LOSS_TERMS)}")
    if len(set(terms)) != len(terms):
        raise ValueError(f"loss_terms 不能重复，实际 {terms}")


def _first_parameter(module: torch.nn.Module, name: str) -> torch.nn.Parameter:
    for param in module.parameters():
        return param
    raise ValueError(f"{name} 没有任何参数，无法确定 device/dtype")


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )


def _batch_tensors(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """取出并规范化 ``input_ids``/``attention_mask``，搬到 embedding 表所在设备。"""
    if not isinstance(batch, Mapping):
        raise TypeError(f"batches[{index}] 必须是 dict，实际 {type(batch).__name__}")
    missing = [key for key in ("input_ids", "attention_mask") if key not in batch]
    if missing:
        raise ValueError(f"batches[{index}] 缺少字段 {missing}")

    ids = torch.as_tensor(batch["input_ids"])
    mask = torch.as_tensor(batch["attention_mask"])
    if ids.dim() != 2:
        raise ValueError(f"batches[{index}]['input_ids'] 必须为 [B, T]，实际 {tuple(ids.shape)}")
    if not _is_integer_dtype(ids.dtype):
        raise TypeError(
            f"batches[{index}]['input_ids'] 必须是整数张量，实际 dtype={ids.dtype}"
        )
    if tuple(mask.shape) != tuple(ids.shape):
        raise ValueError(
            f"batches[{index}] 的 attention_mask 形状必须为 {tuple(ids.shape)}，"
            f"实际 {tuple(mask.shape)}"
        )
    if ids.numel() == 0:
        raise ValueError(f"batches[{index}] 为空 batch（B·T = 0）")
    return (
        ids.to(device=device, dtype=torch.long),
        mask.to(device=device, dtype=torch.bool),
    )


def _embedding_median_norm(embedding_table: torch.Tensor) -> float:
    """T = median(‖e‖₂) 对 embedding 表全部行，μ 计算在 CPU float64 上分块完成。

    注意 MPS 张量不能直接 ``.to(torch.float64)``（会报 "Undefined type Double"），
    必须先 ``.cpu()`` 再升精度。
    """
    row_size = max(1, _MEDIAN_CHUNK_ELEMS // max(1, embedding_table.shape[1]))
    chunks: list[torch.Tensor] = []
    for start in range(0, embedding_table.shape[0], row_size):
        block = embedding_table[start : start + row_size].detach().cpu().to(torch.float64)
        chunks.append(torch.linalg.vector_norm(block, dim=-1))
    norms = torch.cat(chunks)
    return float(torch.median(norms))


def _to_float(value: torch.Tensor) -> float:
    """标量张量 -> Python float（先搬 CPU 再升 float64，兼容 MPS）。"""
    return float(value.detach().cpu().to(torch.float64).item())


def _history_float(value: torch.Tensor) -> float:
    """记录到 history 的标量：盖住 ndim=0 的校验，便于及早发现形状错误。"""
    if value.numel() != 1:
        raise ValueError(f"损失项必须是标量，实际形状 {tuple(value.shape)}")
    return _to_float(value.reshape(()))


# --------------------------------------------------------------------------- #
# 主训练函数
# --------------------------------------------------------------------------- #
def train_sgt(
    sgt: torch.nn.Module,
    lm: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]],
    *,
    config: TrainConfig,
    embedding_table: torch.Tensor | None = None,
) -> dict[str, Any]:
    """冻结 ``lm``，只训练 ``sgt``（逐步语义见模块 docstring 与任务单）。

    Args:
        sgt: :class:`sgt.model.SGTModel`（或任何 ``forward(x, attention_mask=...) ->
            (mu, log_sigma2)`` 的模块）。
        lm: 冻结的目标 LLM；必须支持 ``lm(inputs_embeds=..., attention_mask=...,
            use_cache=False).logits``。
        batches: ``dict`` 序列，每项含 ``input_ids [B, T]``（整数）与
            ``attention_mask [B, T]``（True=有效）。步数超过 batch 数时循环复用。
        config: :class:`TrainConfig`。
        embedding_table: 可选 ``[|V|, d]`` embedding 表；默认取
            ``lm.get_input_embeddings().weight.detach()``。

    Returns:
        ``{"history": [...], "config": asdict(config), "median_norm": float,
        "completed_steps": int}``。history 每项含 ``step``、``seed``、``loss``、
        ``loss_u`` 与各 loss term 的值（均为 Python float）。

    Raises:
        ValueError / TypeError: 配置、batch、模块 device/dtype 不一致等前置条件不满足。
    """
    if not isinstance(config, TrainConfig):
        raise TypeError(f"config 必须是 TrainConfig，实际 {type(config).__name__}")
    _validate_config(config)

    # ---- 1. 冻结 lm，sgt 进入训练模式 ------------------------------------ #
    lm.eval()
    for param in lm.parameters():
        param.requires_grad_(False)
    sgt.train()

    # ---- 2. embedding 表 / 中位数范数（只算一次） ------------------------- #
    input_embeddings = lm.get_input_embeddings()
    if input_embeddings is None or getattr(input_embeddings, "weight", None) is None:
        raise ValueError("lm.get_input_embeddings() 必须返回带 weight 的 embedding 层")
    if embedding_table is None:
        table = input_embeddings.weight.detach()
    else:
        if not torch.is_tensor(embedding_table):
            raise TypeError(
                f"embedding_table 必须是 torch.Tensor，实际 {type(embedding_table).__name__}"
            )
        table = embedding_table.detach()
    if table.dim() != 2:
        raise ValueError(f"embedding_table 必须为 [|V|, d]，实际 {tuple(table.shape)}")
    if table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError(f"embedding_table 不能为空，实际 {tuple(table.shape)}")
    if not table.is_floating_point():
        raise TypeError(f"embedding_table 必须是浮点张量，实际 dtype={table.dtype}")

    sgt_param = _first_parameter(sgt, "sgt")
    if table.device != sgt_param.device:
        raise ValueError(
            f"embedding 表 device {table.device} 与 sgt device {sgt_param.device} 不一致；"
            "训练器不搬模型，请先手动对齐"
        )
    if table.dtype != sgt_param.dtype:
        raise ValueError(
            f"embedding 表 dtype {table.dtype} 与 sgt 参数 dtype {sgt_param.dtype} 不一致；"
            "训练器不做隐式 dtype 转换（lm 与 sgt 必须同精度）"
        )
    if input_embeddings.weight.device != table.device:
        raise ValueError(
            f"lm 输入 embedding device {input_embeddings.weight.device} 与传入的 "
            f"embedding 表 device {table.device} 不一致"
        )
    if input_embeddings.weight.dtype != table.dtype:
        raise ValueError(
            f"lm 输入 embedding dtype {input_embeddings.weight.dtype} 与传入的 "
            f"embedding 表 dtype {table.dtype} 不一致"
        )

    median_norm_value = _embedding_median_norm(table)
    target_norm = torch.as_tensor(
        median_norm_value, dtype=table.dtype, device=table.device
    )

    # ---- 3. batch 列表与 optimizer --------------------------------------- #
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("batches 不能为空")
    n_batches = len(batch_list)

    optimizer = torch.optim.AdamW(
        sgt.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    config_dict = asdict(config)

    checkpoint_dir = None
    if config.checkpoint_dir is not None:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []

    # ---- 4. 训练循环 ------------------------------------------------------ #
    for step in range(config.steps):
        batch = batch_list[step % n_batches]
        prime_batch = batch_list[(step + 1) % n_batches]
        ids, mask = _batch_tensors(batch, device=table.device, index=step % n_batches)
        prime_ids, prime_mask = _batch_tensors(
            prime_batch, device=table.device, index=(step + 1) % n_batches
        )

        x = table[ids].detach()  # [B, T, d]；embedding 表冻结
        mu, log_sigma2 = sgt(x, attention_mask=mask)
        sigma = torch.exp(0.5 * log_sigma2).clamp_min(config.sigma_min)
        generator = torch.Generator(device=x.device).manual_seed(
            config.seed * 100003 + step
        )
        x_tilde = sample_sgt(x, mu, sigma, generator=generator)

        with torch.no_grad():
            clean_logits = lm(
                inputs_embeds=x, attention_mask=mask, use_cache=False
            ).logits
        sgt_logits = lm(
            inputs_embeds=x_tilde, attention_mask=mask, use_cache=False
        ).logits
        loss_u = utility_loss(
            clean_logits,
            sgt_logits,
            attention_mask=mask,
            temperature=config.temperature,
        )

        # 独立批（A19）：下一 batch，全部 detach / no_grad。
        with torch.no_grad():
            x_prime = table[prime_ids].detach()
            mu_prime, log_sigma2_prime = sgt(x_prime, attention_mask=prime_mask)

        term_values: dict[str, torch.Tensor] = {}
        for term in config.loss_terms:
            if term == "mi":
                term_values[term] = mi_loss(
                    x_tilde,
                    x_prime,
                    mu_prime,
                    log_sigma2_prime,
                    log_sigma2,
                    mask=mask,
                )
            elif term == "acs":
                term_values[term] = abs_cos_loss(x, x_tilde, mask=mask)
            elif term == "mnp":
                term_values[term] = median_norm_loss(
                    x + mu, target_norm, mask=mask
                )
            elif term == "cge":
                term_values[term] = comp_entropy_loss(log_sigma2, mask=mask)
            elif term == "bcs":
                term_values[term] = cos_sim_loss(x, x_tilde, mask=mask)
            else:  # pragma: no cover - _validate_config 已拦截
                raise ValueError(f"未知 loss term: {term}")

        # 反传用的总损失按各 term 的自然 dtype 累加（MPS 无 float64）。
        loss = loss_u
        for term in config.loss_terms:
            loss = loss + float(config.weights.get(term, 1.0)) * term_values[term]

        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(sgt.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # history 值：float64 重建，使 loss == loss_u + Σ w·term 在 double 下成立。
        loss_u_value = _history_float(loss_u)
        term_float = {
            term: _history_float(value) for term, value in term_values.items()
        }
        loss_value = loss_u_value
        for term in config.loss_terms:
            loss_value += float(config.weights.get(term, 1.0)) * term_float[term]

        record: dict[str, Any] = {
            "step": step,
            "seed": config.seed,
            "loss": loss_value,
            "loss_u": loss_u_value,
        }
        record.update(term_float)
        history.append(record)

        # ---- 5. checkpoint ------------------------------------------------ #
        if checkpoint_dir is not None and (step + 1) % config.log_every == 0:
            torch.save(
                {
                    "sgt": sgt.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config_dict,
                },
                checkpoint_dir / f"step_{step}.pt",
            )

    return {
        "history": history,
        "config": config_dict,
        "median_norm": median_norm_value,
        "completed_steps": len(history),
    }
