"""M8a Utility 评测：SGT 注入 LLM 输入 embedding 层 + 跨 benchmark 聚合。

论文 §7 / SPEC §5（D3）的 Utility 协议：在相同 harness、相同 ``limit`` 下分别测
基线模型与 SGT 变换后的模型，逐 benchmark 取准确率差（百分点），再对 benchmark
集合取算术平均（SPEC A11）。本模块只依赖 torch，**不 import lm_eval**；harness
调用与结果落盘见 ``experiments/run_utility.py``。

注入语义（论文 §3 / Eq (1)）：

- 客户端只变换自己构造的 prompt embedding：
  ``x̃ = x + µθ(x) + Σθ(x)^{1/2} u``（:class:`SGTEmbedding`）；
- 自回归逐 token 解码时模型每次只送 1 个新 token（``T == 1``），该 token 的
  embedding 来自服务端 embedding 表，按论文语义**不施加** SGT，直接返回
  ``base(ids)``。

确定性：``SGTEmbedding`` 在每次前向开始时用构造时的 ``seed`` 重置噪声生成器，
因此相同输入的输出逐位可复现（便于门禁/复跑）；这不会改变采样分布，只是固定
了每次前向所抽的 ``u``。
"""

from __future__ import annotations

import json
import math
import numbers
import warnings
from pathlib import Path

import torch
from torch import nn

from .model import SGTModel
from .sampling import sample_sgt

__all__ = [
    "SGTEmbedding",
    "install_sgt_embedding",
    "load_sgt_for_lm",
    "utility_scores",
]


# --------------------------------------------------------------------------- #
# Embedding 层注入
# --------------------------------------------------------------------------- #
class SGTEmbedding(nn.Module):
    """把 base embedding 的输出过一遍 SGT（Eq (1)）。

    输入 ids ``[B, T]`` -> ``x = base(ids)`` -> ``(mu, log_sigma2) = sgt(x)``
    -> ``x_tilde = sample_sgt(x, mu, exp(0.5 * log_sigma2), generator=...)``。

    当 ``T == 1``（自回归逐 token 解码时）直接返回 ``base(ids)``，不施加 SGT
    （论文语义：客户端只变换 prompt，后续 token 由服务端 embedding 表查表）。

    Args:
        base_embedding: 原始输入 embedding（``nn.Embedding`` 或同接口模块）。
        sgt: µθ / Σθ 估计器（:class:`sgt.model.SGTModel`）。
        seed: 每次前向重置的噪声生成器种子。
        sigma_min: ``sigma`` 的下限（``sigma = exp(0.5*log_sigma2)`` 后 clamp）。
    """

    def __init__(
        self,
        base_embedding: nn.Module,
        sgt: SGTModel,
        *,
        seed: int = 0,
        sigma_min: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(base_embedding, nn.Module):
            raise TypeError(f"base_embedding 必须是 nn.Module，实际 {type(base_embedding)!r}")
        if not isinstance(sgt, nn.Module):
            raise TypeError(f"sgt 必须是 nn.Module，实际 {type(sgt)!r}")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError(f"seed 必须是整数，实际 {seed!r}")
        seed = int(seed)
        if seed < 0:
            raise ValueError(f"seed 必须非负，实际 {seed}")
        if not (math.isfinite(sigma_min) and sigma_min > 0.0):
            raise ValueError(f"sigma_min 必须为正的有限值，实际 {sigma_min!r}")

        self.base_embedding = base_embedding
        self.sgt = sgt
        self.seed = seed
        self.sigma_min = float(sigma_min)

    # HF 内部（如 ``tie_weights`` / 量化 / resize）会访问 input embedding 的
    # ``weight``；转发到 base 以保持兼容（Llama-3.2 的 tie_word_embeddings=True）。
    @property
    def weight(self) -> torch.Tensor:
        weight = getattr(self.base_embedding, "weight", None)
        if weight is None:
            raise AttributeError(
                f"{type(self.base_embedding).__name__} 没有 weight 属性，无法代理"
            )
        return weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        if not hasattr(self.base_embedding, "weight"):
            raise AttributeError(
                f"{type(self.base_embedding).__name__} 没有 weight 属性，无法代理"
            )
        self.base_embedding.weight = value

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, T] ids -> [B, T, d] embedding（T > 1 时过 SGT，T <= 1 时原样）。"""
        ids = torch.as_tensor(ids)
        if ids.dim() != 2:
            raise ValueError(f"ids 必须为 [B, T]，实际 {tuple(ids.shape)}")
        x = self.base_embedding(ids)
        if not x.is_floating_point():
            raise TypeError(f"base embedding 输出必须为浮点，实际 dtype={x.dtype}")

        # T == 1：逐 token 解码，按论文语义查服务端 embedding 表，不做 SGT。
        if ids.shape[1] <= 1:
            return x

        param = next(self.sgt.parameters(), None)
        if param is None:
            raise ValueError("sgt 没有任何参数，无法施加变换")
        if param.device != x.device:
            raise RuntimeError(
                f"SGT 参数位于 {param.device}，输入位于 {x.device}；"
                "请先安装到同一设备（install_sgt_embedding 会跟随 base embedding，"
                "之后 model.to(device) 也会移动整个 wrapper）"
            )

        # SGT 通常以 fp32 保存；base 可能是 bf16/fp16。变换在 SGT 自身 dtype 下
        # 计算，再转回 base dtype，避免把 fp32 噪声参数降精度。
        x_sgt = x if x.dtype == param.dtype else x.to(param.dtype)
        mu, log_sigma2 = self.sgt(x_sgt)  # mask=None：不做 padding mask（见模块 docstring）
        sigma = torch.exp(0.5 * log_sigma2).clamp(min=self.sigma_min)

        # 每次前向重置种子，保证相同输入逐位可复现；device 必须与张量一致
        # （MPS 上 CPU generator 会直接报错）。
        generator = torch.Generator(device=x_sgt.device)
        generator.manual_seed(self.seed)
        x_tilde = sample_sgt(x_sgt, mu, sigma, generator=generator)
        return x_tilde.to(dtype=x.dtype)


def _replace_submodule(root: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """兜底：在 ``root`` 中定位 ``old`` 并替换为 ``new``（不依赖 setter 实现）。"""
    for module in list(root.modules()):
        for name, child in module.named_children():
            if child is old:
                setattr(module, name, new)
                return
    raise RuntimeError(
        f"无法在 {type(root).__name__} 中定位输入 embedding {type(old).__name__}，"
        "不能安装 SGTEmbedding"
    )


def install_sgt_embedding(
    lm: nn.Module, sgt: SGTModel, *, seed: int = 0
) -> SGTEmbedding:
    """用 :class:`SGTEmbedding` 替换 ``lm`` 的输入 embedding。

    优先走 ``lm.set_input_embeddings``（HF 的标准入口，模型 forward 实际读取的
    子模块会被替换；Llama 上即 ``model.embed_tokens``）；若模型未实现 setter，
    则在模块树中定位并替换。安装后用 ``lm.get_input_embeddings()`` 校验必须是
    wrapper，否则显式报错。

    ``sgt`` 会被移到与 base embedding 相同的 device（dtype 保持自身，通常 fp32），
    并冻结参数（推理期不需要梯度）。

    Returns:
        安装好的 :class:`SGTEmbedding`。
    """
    if not isinstance(lm, nn.Module):
        raise TypeError(f"lm 必须是 nn.Module，实际 {type(lm)!r}")
    if not isinstance(sgt, nn.Module):
        raise TypeError(f"sgt 必须是 nn.Module，实际 {type(sgt)!r}")

    getter = getattr(lm, "get_input_embeddings", None)
    if not callable(getter):
        raise TypeError(f"{type(lm).__name__} 没有 get_input_embeddings()，无法注入")
    base = getter()
    if not isinstance(base, nn.Module):
        raise ValueError(
            f"get_input_embeddings() 必须返回 nn.Module，实际 {type(base)!r}"
        )
    if isinstance(base, SGTEmbedding):
        raise RuntimeError("该模型已经安装过 SGTEmbedding，拒绝重复注入")

    wrapper = SGTEmbedding(base, sgt, seed=seed)

    sgt.eval()
    for param in sgt.parameters():
        param.requires_grad_(False)
    base_weight = getattr(base, "weight", None)
    sgt_param = next(sgt.parameters(), None)
    if (
        isinstance(base_weight, torch.Tensor)
        and sgt_param is not None
        and base_weight.device.type != "meta"
        and sgt_param.device != base_weight.device
    ):
        sgt.to(base_weight.device)

    installed = False
    setter = getattr(lm, "set_input_embeddings", None)
    if callable(setter):
        try:
            setter(wrapper)
            installed = True
        except NotImplementedError:
            installed = False
    if not installed:
        _replace_submodule(lm, base, wrapper)

    if getter() is not wrapper:
        raise RuntimeError(
            "set_input_embeddings 后 get_input_embeddings() 未返回 SGTEmbedding；"
            "该模型结构不受支持，拒绝静默回退到基线"
        )
    return wrapper


# --------------------------------------------------------------------------- #
# 从训练产物加载 SGT
# --------------------------------------------------------------------------- #
def _read_hidden_size(model_path: Path) -> int:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    d = config.get("hidden_size")
    if d is None and isinstance(config.get("text_config"), dict):
        d = config["text_config"].get("hidden_size")
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise ValueError(f"{config_path} 中未找到合法的 hidden_size，实际 {d!r}")
    return d


def _read_state_dict(ckpt_path: Path) -> dict:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SGT checkpoint 不存在: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # 训练脚本可能把 state_dict 包在 {"state_dict": ...} 里；两种都接受。
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{ckpt_path} 不是非空 state_dict，实际 {type(obj)!r}")
    if not isinstance(obj.get("input_proj.weight"), torch.Tensor):
        raise ValueError(f"{ckpt_path} 缺少 input_proj.weight，不是 SGTModel state_dict")
    return obj


def _count_encoder_layers(state: dict, default: int) -> int:
    indices = set()
    for key in state:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layers":
            if parts[2].isdigit():
                indices.add(int(parts[2]))
    return max(indices) + 1 if indices else default


def load_sgt_for_lm(
    model_path: str | Path,
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
    hidden: int = 512,
    n_layers: int = 2,
    n_heads: int = 4,
    sigma_init: float = 1.0,
) -> SGTModel:
    """读取目标模型 config.json 的 ``hidden_size``，加载 SGT checkpoint。

    维度校验：checkpoint 的 ``input_proj.weight`` 第 1 维（d）必须等于目标模型的
    ``hidden_size``，否则抛 :class:`ValueError`（调用方据此显式失败，不得回退基线）。

    ``hidden`` / ``n_layers`` 在 checkpoint 能推断出真实形状时以 checkpoint 为准
    （训练产物可能不是默认宽度；不一致时发 ``RuntimeWarning``）——否则
    ``load_state_dict`` 会因形状不符失败。``n_heads`` 无法由 state_dict 推断
    （参数形状与头数无关），必须由调用方给出；``max_len`` 取自 ``pos_emb`` 行数，
    保证与训练时的位置编码长度一致。

    Returns:
        放到 ``device`` 并切到 eval 模式的 :class:`SGTModel`。
    """
    model_path = Path(model_path)
    ckpt_path = Path(ckpt_path)
    d = _read_hidden_size(model_path)
    state = _read_state_dict(ckpt_path)

    input_proj = state["input_proj.weight"]
    if input_proj.dim() != 2:
        raise ValueError(
            f"{ckpt_path} 的 input_proj.weight 必须为 2 维，实际 {tuple(input_proj.shape)}"
        )
    ckpt_d = int(input_proj.shape[1])
    if ckpt_d != d:
        raise ValueError(
            f"SGT checkpoint 的 d={ckpt_d} 与模型 {model_path} 的 hidden_size={d} 不一致"
        )
    head = state.get("head.weight")
    if isinstance(head, torch.Tensor) and int(head.shape[0]) != 2 * d:
        raise ValueError(
            f"SGT checkpoint 的 head.weight 形状 {tuple(head.shape)} 与 d={d} 不一致"
        )

    ckpt_hidden = int(input_proj.shape[0])
    ckpt_layers = _count_encoder_layers(state, n_layers)
    if (ckpt_hidden, ckpt_layers) != (hidden, n_layers):
        warnings.warn(
            f"SGT checkpoint 的实际架构为 hidden={ckpt_hidden}, n_layers={ckpt_layers}，"
            f"与请求的 hidden={hidden}, n_layers={n_layers} 不一致；按 checkpoint 构造",
            RuntimeWarning,
            stacklevel=2,
        )
        hidden, n_layers = ckpt_hidden, ckpt_layers

    pos_emb = state.get("pos_emb.weight")
    max_len = int(pos_emb.shape[0]) if isinstance(pos_emb, torch.Tensor) else 512

    sgt = SGTModel(
        d,
        hidden=hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        sigma_init=sigma_init,
    )
    try:
        sgt.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(
            f"无法把 {ckpt_path} 加载到 SGTModel(d={d}, hidden={hidden}, "
            f"n_layers={n_layers}, n_heads={n_heads}, max_len={max_len}): {exc}"
        ) from exc
    sgt.to(device)
    sgt.eval()
    return sgt


# --------------------------------------------------------------------------- #
# Utility 聚合（SPEC A11）
# --------------------------------------------------------------------------- #
def _unwrap_results(results: dict, tasks: list[str]) -> dict:
    """兼容 ``simple_evaluate`` 的完整输出：优先取其中的 ``results`` 子映射。"""
    if isinstance(results, dict) and isinstance(results.get("results"), dict):
        inner = results["results"]
        if not any(task in results for task in tasks):
            return inner
    return results


def _task_metric(task_result) -> float | None:
    """取 ``acc``，无则 ``acc_norm``；同时兼容 lm-eval 0.4.13 的 ``acc,none`` 键名。

    lm-eval 0.4.13 的 ``results`` 把 filter 写进键名（``acc,none`` /
    ``acc_norm,none``），与任务单写的 ``results[task]["acc"]`` 不同；这里两种
    键名都识别，语义仍是「accuracy，缺失时用 normalized accuracy」。
    """
    if not isinstance(task_result, dict):
        return None
    for metric in ("acc", "acc_norm"):
        for key in (metric, f"{metric},none"):
            if key not in task_result:
                continue
            value = task_result[key]
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                continue
            value = float(value)
            if math.isfinite(value):
                return value
    return None


def utility_scores(base_results: dict, sgt_results: dict, tasks: list[str]) -> dict:
    """逐任务算 ``sgt - base``（百分点）并对可用任务取算术平均。

    Args:
        base_results: 基线 ``simple_evaluate(...)["results"]``（也可直接传完整
            返回值，本函数会自动取 ``results`` 子映射）。
        sgt_results: SGT 变换后的同结构结果。
        tasks: benchmark 名列表（顺序即输出顺序）。

    Returns:
        ``{"per_task": {task: {"base", "sgt", "diff_pp"}}, "utility_pp": float | None,
        "tasks_missing": [task, ...]}``。任一结果缺失该任务的 acc/acc_norm 时，
        该任务进入 ``tasks_missing`` 且不参与平均；全部缺失时 ``utility_pp=None``
        （不写 NaN）。
    """
    if not isinstance(tasks, (list, tuple)):
        raise TypeError(f"tasks 必须是列表，实际 {type(tasks)!r}")
    base = _unwrap_results(base_results, list(tasks))
    sgt = _unwrap_results(sgt_results, list(tasks))

    per_task: dict[str, dict] = {}
    tasks_missing: list[str] = []
    diffs: list[float] = []
    for task in tasks:
        base_value = _task_metric(base.get(task))
        sgt_value = _task_metric(sgt.get(task))
        if base_value is None or sgt_value is None:
            tasks_missing.append(task)
            continue
        diff_pp = (sgt_value - base_value) * 100.0
        per_task[task] = {
            "base": base_value,
            "sgt": sgt_value,
            "diff_pp": diff_pp,
        }
        diffs.append(diff_pp)

    utility_pp = sum(diffs) / len(diffs) if diffs else None
    return {
        "per_task": per_task,
        "utility_pp": utility_pp,
        "tasks_missing": tasks_missing,
    }
