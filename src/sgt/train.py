"""M6 训练回路：冻结目标 LLM，只训练 SGT（SPEC §3.5, §6；A12/A17/A19/A22）。

训练目标（逐 step）：

    x         = E[input_ids]                         # embedding 查表，detach
    x_sgt     = x.to(sgt_dtype)                      # 显式：LM 精度 -> SGT 精度
    mu, lv    = SGT(x_sgt, mask)
    sigma     = exp(0.5·lv).clamp_min(sigma_min)
    x̃         = x_sgt + mu + sigma ⊙ u,  u ~ N(0,I)  # Eq (1)，显式 Generator
    x̃_lm      = x̃.to(lm_dtype)                       # 显式：SGT 精度 -> LM 精度
    L_U       = CE(softmax(z_clean/τ) ‖ softmax(z̃/τ))  # 只对有效 token 平均
    L         = L_U + Σ_t w_t · L_t                   # Eq (3)/(11)
    L.backward() -> clip -> AdamW.step()              # 梯度只到 SGT / x̃
    lr_s      = lr * min(1, (s+1)/warmup_steps)       # 线性 warmup（s 从 0 开始），LambdaLR

约定与实现要点：

- 冻结：``lm.eval()`` + 所有参数 ``requires_grad_(False)``；训练器从不调用
  ``lm.to()/train()/zero_grad()``，也不读写 lm 的参数或 buffer（因此 BN/dropout 均关闭，
  buffer 逐位不变；见 ``tests/test_train.py``）。
- 独立批（A19）：第 ``s`` 步用 ``batches[s % n]``，其下一批 ``batches[(s+1) % n]`` 作为
  ``X_b'``；``x'``、``mu'``、``lv'`` 全部在 ``torch.no_grad()`` 下计算（不反传）。
- 随机性：采样 u 只由 ``torch.Generator(device=x_sgt.device).manual_seed(seed*100003+step)``
  决定，不使用全局 RNG。SGTModel 默认 ``dropout=0``，因此 ``sgt.train()`` 不引入额外随机性。
- 设备：训练器不搬模型，按传入 ``lm``/``sgt`` 的设备运行，并校验 ``sgt`` 与 embedding 表
  device 一致（不一致直接报错）。
- 精度（A22，混合精度）：论文未规定训练精度。允许 LM 与 SGT 不同精度（如 LM bf16 +
  SGT fp32，理由是 MPS 内存），但**必须显式往返转换**：``x -> x_sgt``、``x̃ -> x̃_lm``、
  干净侧 ``x -> x_lm``，并记录到返回值的 ``dtypes`` 中（``lm``/``sgt``/``embedding``）。
  训练器只按调用方给定的 ``sgt`` 参数 dtype 计算 SGT（绝不把 SGT 参数或激活静默降到 LM
  的精度）；dtype 相同时这些转换为 no-op，数值与前版逐位一致。``TrainConfig.device`` /
  ``dtype`` 是记录字段，不作为强制转换指令（冻结的 lm 不允许被本模块搬动或换精度）。
- 中位数范数 T（Eq (10)）：对 embedding 表**全部行**的 L2 范数取 ``torch.median``
  （PAPER_ANALYSIS P26 口径）。范数在 CPU 上以 float64 分块计算（MPS 不支持 float64；
  分块避免 128256×2048 的 float64 全表约 2.1 GB 临时内存），只算一次。
- 日志精度：``loss`` 由 float64 的 ``loss_u`` 与各 term 值按同一公式在 Python 中重建，
  因此 ``history[i]["loss"] == loss_u + Σ w_t·term_t`` 在 double 下逐项成立（<1e-9），
  不受参数 dtype（float32/bf16）的求和舍入影响；反传用的总损失仍按各 term 的自然
  dtype 累加（MPS 不支持 float64）。
- §5.3 核对量（GAP-2，全部 ``no_grad``/``detach``）：history 额外记录
  ``cos_mean``（有效 token 上 ``cos(x, x̃)`` 的**有符号**均值）、``sigma_mean`` /
  ``sigma_p50`` / ``sigma_p95``（有效 token 的 σ 统计）、``mu_norm_mean``（逐 token
  ``‖µ‖₂`` 均值）。归约在 CPU float64 上完成（MPS 不支持 float64/quantile），
  因此同 seed 可逐位复现，且不消耗全局 RNG。
- LR warmup（A12，论文只给最终 lr、未给 schedule）：第 s 步（0-based）的实际学习率为
  ``lr * min(1, (s+1)/warmup_steps)``，由 ``torch.optim.lr_scheduler.LambdaLR`` 实现，
  ``scheduler.step()`` 在 ``optimizer.step()`` 之后调用（PyTorch 推荐的顺序，也避免
  "step() before optimizer.step()" 警告）。``warmup_steps=0`` 时 λ ≡ 1（乘 1.0 逐位
  等价于不设 scheduler），即旧行为。每步实际 lr 写入 ``history[i]["lr"]``。
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
    """M6 训练超参（A12：论文未给出的复现选择，全部写入结果 JSON）。

    精度（A22）：论文未规定训练精度。本实现支持 **LM bf16 + SGT fp32 的混合精度**
    （理由：MPS 上的显存/内存约束），具体由传入的 ``lm`` / ``sgt`` 参数 dtype 决定；
    ``train_sgt`` 在两者之间做显式往返转换并记录到返回值的 ``dtypes``。下面的 ``device`` /
    ``dtype`` 字段只用于**记录调用方声明**，不驱动任何 ``.to()`` 转换（冻结的 lm 不允许被
    本模块搬动或换精度）。

    history 相关（GAP-2）：``history`` 每条除 ``loss``/``loss_u``/各 term 外，还记录
    ``cos_mean``（有符号 cos(x, x̃) 均值）、``sigma_mean``/``sigma_p50``/``sigma_p95``、
    ``mu_norm_mean``，用于逐行核对论文 §5.3 的观察。

    ``warmup_steps``（A12）：论文只给最终学习率、未给 schedule。前 ``warmup_steps`` 步
    线性 warmup，第 s 步（0-based）的 lr 为 ``lr * min(1, (s+1)/warmup_steps)``；
    ``0`` 表示不做 warmup（恒定 ``lr``，旧行为）。每步实际 lr 记录在
    ``history[i]["lr"]``。
    """

    steps: int = 100
    lr: float = 1e-4
    warmup_steps: int = 0
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
    if (
        isinstance(config.warmup_steps, bool)
        or int(config.warmup_steps) != config.warmup_steps
        or config.warmup_steps < 0
    ):
        raise ValueError(
            f"warmup_steps 必须为 ≥ 0 的整数，实际 {config.warmup_steps!r}"
        )
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


def _cpu_float64(value: torch.Tensor) -> torch.Tensor:
    """detach 后搬 CPU 再升 float64（MPS 不支持 float64/quantile；CPU 归约可复现）。"""
    return value.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)


def _valid_token_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``[B, T, ...]`` + bool 掩码 -> 有效 token 的 CPU float64 张量 ``[N, ...]``。"""
    return _cpu_float64(values[mask])


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
        "completed_steps": int, "dtypes": {"lm": str, "sgt": str, "embedding": str}}``。
        history 每项含 ``step``、``seed``、``lr``（本步实际学习率）、``loss``、
        ``loss_u``、各 loss term 的值（均为 Python float），以及 §5.3 核对量
        ``cos_mean``、``sigma_mean``、``sigma_p50``、``sigma_p95``、``mu_norm_mean``。
        ``dtypes`` 记录 A22 的实际混合精度（LM / SGT / embedding 表各自 dtype 的字符串形式）。

    Raises:
        ValueError / TypeError: 配置、batch、模块 device 不一致等前置条件不满足。
            dtype 不要求一致（A22 混合精度），但 device 必须一致。
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

    # A22（GAP-8）：允许 LM 与 SGT 不同精度（如 LM bf16 + SGT fp32），但训练器必须显式
    # 往返转换，并把实际精度写进返回值。SGT 的算术精度只由 sgt 参数 dtype 决定——绝不
    # 把 SGT 参数或激活静默降到 LM 的精度；LM 侧的输入则显式转成 embedding 的 dtype。
    lm_dtype = input_embeddings.weight.dtype
    sgt_dtype = sgt_param.dtype

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

    # LR warmup（A12）：第 s 步（0-based）的 lr = lr * min(1, (s+1)/warmup_steps)。
    # LambdaLR 构造时即设置 lr = lambda(0)，正好是 step 0 的目标值；之后每步在
    # optimizer.step() 之后调用 scheduler.step()（PyTorch 推荐顺序）。warmup_steps=0
    # 时 lambda ≡ 1.0，lr 逐位等于 config.lr（旧行为）。
    warmup_steps = int(config.warmup_steps)

    def _warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_warmup_lambda
    )
    config_dict = asdict(config)

    checkpoint_dir = None
    if config.checkpoint_dir is not None:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []

    # ---- 4. 训练循环 ------------------------------------------------------ #
    for step in range(config.steps):
        # 本步实际使用的 lr（scheduler 在 optimizer.step() 之后才推进到下一步的值）。
        step_lr = float(optimizer.param_groups[0]["lr"])
        batch = batch_list[step % n_batches]
        prime_batch = batch_list[(step + 1) % n_batches]
        ids, mask = _batch_tensors(batch, device=table.device, index=step % n_batches)
        prime_ids, prime_mask = _batch_tensors(
            prime_batch, device=table.device, index=(step + 1) % n_batches
        )

        x = table[ids].detach()  # [B, T, d]；embedding 表冻结，dtype = LM 精度
        # A22：显式升/降到 SGT 精度（dtype 相同时为 no-op，数值与同精度训练逐位一致）。
        x_sgt = x.to(dtype=sgt_dtype)
        mu, log_sigma2 = sgt(x_sgt, attention_mask=mask)
        sigma = torch.exp(0.5 * log_sigma2).clamp_min(config.sigma_min)
        generator = torch.Generator(device=x_sgt.device).manual_seed(
            config.seed * 100003 + step
        )
        x_tilde = sample_sgt(x_sgt, mu, sigma, generator=generator)
        # A22：显式转回 LM 精度后再喂 LM；cast 可微，梯度经 x̃_lm -> x̃ 回到 SGT。
        x_tilde_lm = x_tilde.to(dtype=lm_dtype)

        with torch.no_grad():
            clean_logits = lm(
                inputs_embeds=x.to(dtype=lm_dtype), attention_mask=mask, use_cache=False
            ).logits
        sgt_logits = lm(
            inputs_embeds=x_tilde_lm, attention_mask=mask, use_cache=False
        ).logits
        loss_u = utility_loss(
            clean_logits,
            sgt_logits,
            attention_mask=mask,
            temperature=config.temperature,
        )

        # 独立批（A19）：下一 batch，全部 detach / no_grad。
        with torch.no_grad():
            x_prime = table[prime_ids].detach().to(dtype=sgt_dtype)
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
        lr_scheduler.step()  # warmup：在 optimizer.step() 之后推进（见模块 docstring）
        optimizer.zero_grad(set_to_none=True)

        # §5.3 核对量（GAP-2）：全部 no_grad / detach，归约在 CPU float64（可复现、不耗 RNG）。
        # cos 用 SGT 侧输入 x_sgt（混合精度时它与 LM 侧的 x 只差一次显式转换；同精度时即 x）。
        with torch.no_grad():
            cos_mean_value = _history_float(cos_sim_loss(x_sgt, x_tilde, mask=mask))
        sigma_valid = _valid_token_values(sigma, mask).reshape(-1)
        mu_valid = _valid_token_values(mu, mask)
        sigma_mean_value = float(sigma_valid.mean())
        sigma_p50_value = float(torch.quantile(sigma_valid, 0.5))
        sigma_p95_value = float(torch.quantile(sigma_valid, 0.95))
        mu_norm_mean_value = float(torch.linalg.vector_norm(mu_valid, dim=-1).mean())

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
            "lr": step_lr,
            "loss": loss_value,
            "loss_u": loss_u_value,
        }
        record.update(term_float)
        record.update(
            {
                "cos_mean": cos_mean_value,
                "sigma_mean": sigma_mean_value,
                "sigma_p50": sigma_p50_value,
                "sigma_p95": sigma_p95_value,
                "mu_norm_mean": mu_norm_mean_value,
            }
        )
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
        # A22：本次训练实际使用的精度（embedding 表 dtype == 喂给 LM 的精度）。
        "dtypes": {
            "lm": str(lm_dtype),
            "sgt": str(sgt_dtype),
            "embedding": str(table.dtype),
        },
    }
