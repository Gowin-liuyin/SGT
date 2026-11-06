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

随机性（论文的 stochastic SGT 语义）：SGT 是随机变换，**同一输入两次采样必须不同**，
且不同输入之间的噪声必须独立。``SGTEmbedding(fresh_noise=True)``（默认）在首次前向时
按输入 device 创建 ``torch.Generator`` 并用构造时的 ``seed`` 播种，之后每次前向都
**不重置**、持续推进该生成器：同种子的两个 wrapper 走相同调用序列仍可复现，但同一
wrapper 对同一输入的连续两次调用会得到不同噪声。``fresh_noise=False`` 保留旧行为
（每次前向重置到同一 seed），仅供消融对照——它让同形状的所有输入共享同一噪声场 ``u``，
违反论文的 stochastic 定义，也破坏安全模型（噪声复用使不同输入的噪声相关）。
"""

from __future__ import annotations

import inspect
import json
import math
import numbers
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import SGTModel
from .sampling import sample_sgt

__all__ = [
    "SGTEmbedding",
    "build_sgt_from_state",
    "install_sgt_embedding",
    "load_sgt_for_lm",
    "sgt_config_from_payload",
    "utility_scores",
]

#: ``SGTModel.__init__`` 中带默认值的参数（checkpoint 缺键时回落到这些默认值）。
_SGT_DEFAULTS: dict[str, Any] = {
    name: parameter.default
    for name, parameter in inspect.signature(SGTModel.__init__).parameters.items()
    if name != "self" and parameter.default is not inspect.Parameter.empty
}


# --------------------------------------------------------------------------- #
# Embedding 层注入
# --------------------------------------------------------------------------- #
class SGTEmbedding(nn.Module):
    """把 base embedding 的输出过一遍 SGT（Eq (1)）。

    输入 ids ``[B, T]`` -> ``x = base(ids)`` -> ``(mu, log_sigma2) = sgt(x)``
    -> ``x_tilde = sample_sgt(x, mu, exp(0.5 * log_sigma2), generator=...)``。

    当 ``T == 1``（自回归逐 token 解码时）直接返回 ``base(ids)``，不施加 SGT
    （论文语义：客户端只变换 prompt，后续 token 由服务端 embedding 表查表）。

    噪声生成器（论文的 stochastic SGT 语义）：

    - ``fresh_noise=True``（默认）：首次前向时按输入 device 创建 ``torch.Generator``
      并用 ``seed`` 播种，之后每次前向**不重置**——同一输入两次调用得到不同采样，
      不同输入共享同一条独立随机流（不相关）；同 ``seed`` 的两个 wrapper 走相同调用
      序列时逐位一致。
    - ``fresh_noise=False``：每次前向把生成器重置到 ``seed`` 再采样（旧行为）。
      这固定了每次前向所抽的 ``u``，使同形状的所有输入共享同一噪声场，**不满足**
      论文的 stochastic 定义，且会让不同输入的噪声相关；只用于消融对照。

    Args:
        base_embedding: 原始输入 embedding（``nn.Embedding`` 或同接口模块）。
        sgt: µθ / Σθ 估计器（:class:`sgt.model.SGTModel`）。
        seed: 噪声生成器的种子（首次前向创建生成器时使用）。
        sigma_min: ``sigma`` 的下限（``sigma = exp(0.5*log_sigma2)`` 后 clamp）。
        fresh_noise: 是否每次前向采样新噪声（论文语义）；``False`` 仅作消融对照。
    """

    def __init__(
        self,
        base_embedding: nn.Module,
        sgt: SGTModel,
        *,
        seed: int = 0,
        sigma_min: float = 1e-6,
        fresh_noise: bool = True,
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
        if not isinstance(fresh_noise, bool):
            raise TypeError(f"fresh_noise 必须是 bool，实际 {type(fresh_noise)!r}")

        self.base_embedding = base_embedding
        self.sgt = sgt
        self.seed = seed
        self.sigma_min = float(sigma_min)
        self.fresh_noise = fresh_noise
        # 首次前向时按输入 device 惰性创建（MPS 上 CPU generator 会直接报错）；
        # 不注册为 buffer —— torch.Generator 不是张量/模块。
        self._generator: torch.Generator | None = None

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

        # 噪声生成器：首次前向按输入 device 惰性创建（MPS 上 CPU generator 会报错）。
        # fresh_noise=True 时持续推进（不重置），同一输入两次调用不同；
        # fresh_noise=False 时每次重置到 seed（旧行为，仅消融对照）。
        generator = self._generator
        if generator is None:
            generator = torch.Generator(device=x_sgt.device)
            generator.manual_seed(self.seed)
            self._generator = generator
        elif generator.device.type != x_sgt.device.type:
            raise RuntimeError(
                f"噪声生成器位于 {generator.device}，输入位于 {x_sgt.device}；"
                "请勿在前向之间把 wrapper 移到别的 device"
            )
        if not self.fresh_noise:
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


#: checkpoint 里可能包裹 state_dict 的顶层键，按此顺序尝试解包。
_WRAPPER_KEYS = ("state_dict", "sgt")

#: ``config["sgt"]`` 中会用于重建 :class:`SGTModel` 的键（其余键一律忽略，不报错）。
_SGT_CONFIG_KEYS = frozenset(
    {
        "hidden",
        "n_layers",
        "n_heads",
        "ffn_mult",
        "max_len",
        "pos_encoding",
        "sigma_init",
        "sigma_min",
        "sigma_max",
        "sigma_floor",
        "mu_scale",
        "dropout",
        "pos_emb_init_std",
        "final_norm",
    }
)

#: 无法从 state_dict 推断的“无参数但改变前向行为”选项；config 缺失时只能取默认值。
#: ``pos_encoding`` 只能区分 learned / 非 learned（"sinusoidal" 与 "none" 的 state 相同），
#: ``sigma_max`` 与 ``mu_scale`` / ``sigma_floor`` 一样不改变参数形状。
_UNPARAMETERIZED_OPTION_KEYS = ("mu_scale", "sigma_floor", "sigma_max", "pos_encoding")


def sgt_config_from_payload(payload: Any) -> dict[str, Any]:
    """从 checkpoint 顶层对象里取出 ``config["sgt"]``（或旧式顶层 ``sgt_config``）。

    ``experiments/run_ablation.py`` 保存 ``{"sgt": state_dict, "config": {...}}``，其中
    ``config["sgt"]`` 是 yaml 的 ``sgt:`` 段原样（含 ``final_norm`` / ``mu_scale`` /
    ``sigma_floor`` / ``pos_encoding`` 等**无参数但改变前向行为**的选项，以及 hidden /
    n_layers / n_heads / ffn_mult / max_len）。本函数只取出这个子映射，其余 config 键
    （``train_config`` / ``eval`` / ...）忽略。

    兼容旧式顶层 ``sgt_config``；两者都没有（裸 state_dict，或只含 ``{"sgt": ...}`` 的旧
    checkpoint）时返回 ``{}``，调用方只能按 state_dict 形状推断（见
    :func:`build_sgt_from_state`）。
    """
    if not isinstance(payload, Mapping):
        return {}
    config = payload.get("config")
    if isinstance(config, Mapping):
        nested = config.get("sgt")
        if isinstance(nested, Mapping):
            return dict(nested)
    legacy = payload.get("sgt_config")
    if isinstance(legacy, Mapping):
        return dict(legacy)
    return {}


def _read_checkpoint(ckpt_path: Path) -> tuple[dict, dict[str, Any]]:
    """读取 SGT checkpoint，返回 ``(state_dict, config["sgt"])``。

    state_dict 依次接受三种封装：

    1. 裸 state_dict（顶层直接含 ``input_proj.weight``）；
    2. ``{"state_dict": <state_dict>}``；
    3. ``{"sgt": <state_dict>}``（``sgt.train.train_sgt`` 与
       ``experiments/run_ablation.py`` 的训练器格式；``optimizer`` / ``step`` /
       ``config`` 等额外键被忽略）。

    config 由 :func:`sgt_config_from_payload` 取出（无 config 的旧 checkpoint 返回空
    dict）。其余结构一律 :class:`ValueError`，错误信息列出已尝试的封装形式。
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SGT checkpoint 不存在: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sgt_config = sgt_config_from_payload(obj)
    # 顺序：裸 state_dict 优先，其次 {"state_dict": ...}，最后 {"sgt": ...}。
    if isinstance(obj, dict) and not isinstance(
        obj.get("input_proj.weight"), torch.Tensor
    ):
        for key in _WRAPPER_KEYS:
            inner = obj.get(key)
            if isinstance(inner, dict):
                obj = inner
                break
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{ckpt_path} 不是非空 state_dict，实际 {type(obj)!r}")
    if not isinstance(obj.get("input_proj.weight"), torch.Tensor):
        tried = " / ".join(["裸 state_dict"] + [f"{{{k}: ...}}" for k in _WRAPPER_KEYS])
        raise ValueError(
            f"{ckpt_path} 缺少 input_proj.weight，不是 SGTModel state_dict；"
            f"已尝试的封装：{tried}"
        )
    return obj, sgt_config


def _read_state_dict(ckpt_path: Path) -> dict:
    """读取 SGT state_dict（封装兼容性见 :func:`_read_checkpoint`）。"""
    return _read_checkpoint(ckpt_path)[0]


def _count_encoder_layers(
    state: Mapping[str, torch.Tensor], default: int | None
) -> int | None:
    indices = set()
    for key in state:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layers":
            if parts[2].isdigit():
                indices.add(int(parts[2]))
    return max(indices) + 1 if indices else default


def _infer_sgt_architecture(
    state: Mapping[str, torch.Tensor],
) -> dict[str, int | None]:
    """从 state_dict 的键/形状反推 SGT 架构（``n_heads`` 不可反推）。

    Returns:
        ``{"d", "hidden", "n_layers", "ffn_mult", "max_len"}``。``ffn_mult`` 在缺少
        ``encoder.layers.0.linear1.weight`` 时为 ``None``；``max_len`` 只在
        ``pos_emb.weight`` 存在（``pos_encoding="learned"``）时可得，否则为 ``None``
        （由 config 或 ``SGTModel`` 默认值决定）。
    """
    input_proj = state.get("input_proj.weight")
    if not isinstance(input_proj, torch.Tensor):
        raise ValueError(
            f"SGT state_dict 缺少 input_proj.weight；收到的前 3 个键：{list(state)[:3]}"
        )
    if input_proj.dim() != 2:
        raise ValueError(
            f"input_proj.weight 必须为 2 维，实际 {tuple(input_proj.shape)}"
        )
    hidden, d = (int(value) for value in input_proj.shape)
    head = state.get("head.weight")
    if isinstance(head, torch.Tensor) and int(head.shape[0]) != 2 * d:
        raise ValueError(
            f"head.weight 形状 {tuple(head.shape)} 与 input_proj 的 d={d} "
            f"不一致（应为 [2d, hidden]）"
        )
    linear1 = state.get("encoder.layers.0.linear1.weight")
    ffn_mult = None
    if isinstance(linear1, torch.Tensor):
        rows = int(linear1.shape[0])
        if rows % hidden:
            raise ValueError(
                f"encoder.layers.0.linear1.weight 的 {rows} 行不能被 hidden={hidden} "
                "整除，无法推断 ffn_mult"
            )
        ffn_mult = rows // hidden
    pos_emb = state.get("pos_emb.weight")
    return {
        "d": d,
        "hidden": hidden,
        "n_layers": _count_encoder_layers(state, None),
        "ffn_mult": ffn_mult,
        "max_len": int(pos_emb.shape[0]) if isinstance(pos_emb, torch.Tensor) else None,
    }


def _sgt_load_failure_hint(
    model: SGTModel, state: Mapping[str, torch.Tensor], message: str = ""
) -> str:
    """strict load 失败时按缺失/多余的键指出最可能不匹配的选项。"""
    expected = set(model.state_dict())
    actual = set(state)
    missing = expected - actual
    unexpected = actual - expected
    hints: list[str] = []
    if (
        "pos_emb.weight" in missing
        or "pos_emb.weight" in unexpected
        or "pos_emb" in message
    ):
        hints.append(
            "pos_encoding 不匹配（'learned' 需要 pos_emb.weight，"
            "'sinusoidal'/'none' 不需要）"
        )
    if (
        any(key.startswith("final_norm.") for key in missing | unexpected)
        or "final_norm" in message
    ):
        hints.append("final_norm 不匹配（True 需要 final_norm.weight / final_norm.bias）")
    if not hints:
        hints.append(
            "检查 n_heads（无法由形状推断）/ hidden / n_layers / ffn_mult / max_len "
            "是否与训练时一致"
        )
    return "；".join(hints)


def build_sgt_from_state(
    state: Mapping[str, torch.Tensor],
    sgt_config: Mapping[str, Any] | None = None,
    *,
    n_heads: int | None = None,
    n_heads_precedence: str = "config",
    fallbacks: Mapping[str, Any] | None = None,
) -> tuple[SGTModel, dict[str, Any]]:
    """按 checkpoint 的 ``config["sgt"]`` 重建 :class:`SGTModel`，形状推断只作校验。

    重建规则（config 指 :func:`sgt_config_from_payload` 取出的 ``config["sgt"]``）：

    1. 架构键 ``hidden`` / ``n_layers`` / ``ffn_mult`` / ``max_len``：config 有的用
       config，并与 state_dict 形状推断值**交叉校验**（不一致报 :class:`ValueError`，
       不静默）；config 缺失的用形状推断；形状也推不出（``ffn_mult``、非 ``learned``
       位置编码的 ``max_len``）时用 ``fallbacks`` 或 ``SGTModel`` 默认值。
    2. ``pos_encoding`` / ``final_norm``：config 有的用 config，并与 state_dict 键
       （``pos_emb.weight`` / ``final_norm.weight``）交叉校验；config 缺失时按键推断
       （有 ``pos_emb.weight`` → ``"learned"``，否则 ``"sinusoidal"``；``final_norm``
       按键是否存在）。
    3. 标量选项 ``sigma_init`` / ``sigma_min`` / ``sigma_max`` / ``sigma_floor`` /
       ``mu_scale`` / ``dropout`` / ``pos_emb_init_std``：config 有的用 config（显式
       ``null`` 表示无界，只允许 ``sigma_max`` / ``sigma_floor`` / ``mu_scale``），缺失
       的用 ``fallbacks`` 或 ``SGTModel`` 默认值。``mu_scale`` / ``sigma_floor`` /
       ``sigma_max`` 无法从 state_dict 推断，无 config 时退化为模型默认（前两者
       ``None``）——调用方必须检查 ``meta["inferred_unparameterized_options"]``。
    4. ``n_heads``：不改变参数形状，strict load 发现不了头数不符，因此 config 提供时
       优先用 config（``n_heads_precedence="config"``）；``"caller"`` 时调用方值优先
       （``run_pupa`` 的 ``--sgt-heads`` 覆盖语义）。二者冲突且 config 生效时发
       :class:`RuntimeWarning`。

    加载用 ``strict=True``；失败时错误信息按缺失/多余的键指出可能不匹配的选项
    （``pos_encoding`` / ``final_norm`` 等）。

    Returns:
        ``(model, meta)``：``model`` 在 CPU、fp32、eval 模式；``meta`` 记录实际使用的
        架构、上述选项及其来源（``config`` / ``shape`` / ``inferred`` / ``caller`` /
        ``default``）。
    """
    if n_heads_precedence not in ("config", "caller"):
        raise ValueError(
            f"n_heads_precedence 必须为 'config' 或 'caller'，实际 {n_heads_precedence!r}"
        )
    config = dict(sgt_config or {})
    fallback_values = dict(fallbacks or {})
    inferred = _infer_sgt_architecture(state)
    unknown = sorted(str(key) for key in set(config) - _SGT_CONFIG_KEYS)
    kwargs: dict[str, Any] = {}
    sources: dict[str, str] = {}

    # 1) 架构键：config 优先，state_dict 形状只作交叉校验。
    for key in ("hidden", "n_layers", "ffn_mult", "max_len"):
        if key in config and config[key] is not None:
            value = int(config[key])
            inferred_value = inferred[key]
            if inferred_value is not None and value != int(inferred_value):
                raise ValueError(
                    f"checkpoint config['sgt'][{key!r}]={value} 与 state_dict 形状推断出的 "
                    f"{key}={inferred_value} 不一致；config 与权重不是同一次训练的产物？"
                )
            kwargs[key] = value
            sources[key] = "config"
        elif inferred[key] is not None:
            kwargs[key] = int(inferred[key])
            sources[key] = "shape"
        else:
            kwargs[key] = fallback_values.get(key, _SGT_DEFAULTS[key])
            sources[key] = "caller" if key in fallback_values else "default"

    # 2) 无参数但改变前向行为的选项：config 优先，并与 state_dict 键交叉校验。
    has_pos_emb = isinstance(state.get("pos_emb.weight"), torch.Tensor)
    if config.get("pos_encoding") is not None:
        pos_encoding = config["pos_encoding"]
        if bool(pos_encoding == "learned") != has_pos_emb:
            raise ValueError(
                f"checkpoint config['sgt']['pos_encoding']={pos_encoding!r} 与 state_dict "
                f"{'含' if has_pos_emb else '不含'} pos_emb.weight 不一致"
            )
        kwargs["pos_encoding"] = pos_encoding
        sources["pos_encoding"] = "config"
    else:
        kwargs["pos_encoding"] = "learned" if has_pos_emb else "sinusoidal"
        sources["pos_encoding"] = "inferred"

    has_final_norm = "final_norm.weight" in state
    if config.get("final_norm") is not None:
        final_norm = config["final_norm"]
        if not isinstance(final_norm, bool):
            raise ValueError(
                f"checkpoint config['sgt']['final_norm'] 必须是 bool，实际 {final_norm!r}"
            )
        if final_norm != has_final_norm:
            raise ValueError(
                f"checkpoint config['sgt']['final_norm']={final_norm} 与 state_dict "
                f"{'含' if has_final_norm else '不含'} final_norm.weight 不一致"
            )
        kwargs["final_norm"] = final_norm
        sources["final_norm"] = "config"
    else:
        kwargs["final_norm"] = has_final_norm
        sources["final_norm"] = "inferred"

    # 3) 标量选项：config 原值透传；显式 null（= 无界）只允许在三个有界参数上。
    for key in (
        "sigma_init",
        "sigma_min",
        "sigma_max",
        "sigma_floor",
        "mu_scale",
        "dropout",
        "pos_emb_init_std",
    ):
        if key not in config:
            kwargs[key] = fallback_values.get(key, _SGT_DEFAULTS[key])
            sources[key] = "caller" if key in fallback_values else "default"
            continue
        value = config[key]
        if value is None:
            if key not in ("sigma_max", "sigma_floor", "mu_scale"):
                raise ValueError(
                    f"checkpoint config['sgt'][{key!r}] 不能为 null（只有 sigma_max / "
                    "sigma_floor / mu_scale 允许用 null 表示无界）"
                )
            kwargs[key] = None
            sources[key] = "config"
            continue
        try:
            kwargs[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint config['sgt'][{key!r}]={value!r} 不是数值"
            ) from exc
        sources[key] = "config"

    # 4) n_heads：不改变参数形状，strict load 无法发现头数不符，只能靠 config。
    config_n_heads = config.get("n_heads")
    if config_n_heads is not None and (
        n_heads is None or n_heads_precedence == "config"
    ):
        config_n_heads = int(config_n_heads)
        if n_heads is not None and int(n_heads) != config_n_heads:
            warnings.warn(
                f"调用方请求的 n_heads={int(n_heads)} 与 checkpoint "
                f"config['sgt']['n_heads']={config_n_heads} 不一致；按 checkpoint config 重建",
                RuntimeWarning,
                stacklevel=3,
            )
        kwargs["n_heads"] = config_n_heads
        sources["n_heads"] = "config"
    elif n_heads is not None:
        kwargs["n_heads"] = int(n_heads)
        sources["n_heads"] = "explicit"
    else:
        kwargs["n_heads"] = _SGT_DEFAULTS["n_heads"]
        sources["n_heads"] = "default"

    try:
        model = SGTModel(int(inferred["d"]), **kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"按 checkpoint config 重建 SGTModel 失败（d={inferred['d']}，"
            f"kwargs={kwargs}）：{exc}"
        ) from exc

    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        hint = _sgt_load_failure_hint(model, state, str(exc))
        raise ValueError(
            f"无法把 checkpoint state_dict 加载到重建的 SGTModel（d={model.d}, "
            f"hidden={model.hidden}, n_layers={model.n_layers}, n_heads={model.n_heads}, "
            f"ffn_mult={model.ffn_mult}, max_len={model.max_len}, "
            f"pos_encoding={model.pos_encoding!r}, "
            f"final_norm={model.final_norm is not None}）：{exc}；可能原因：{hint}"
        ) from exc
    model.eval()

    missing_options: list[str] = []
    for key in _UNPARAMETERIZED_OPTION_KEYS:
        # mu_scale / sigma_floor / sigma_max 的显式 null 表示「已知无界」，算 config
        # 提供；pos_encoding 的 null 等于没给（只能按键推断）。
        if key in ("mu_scale", "sigma_floor", "sigma_max"):
            provided = key in config
        else:
            provided = config.get(key) is not None
        if not provided:
            missing_options.append(key)
    meta: dict[str, Any] = {
        "d": model.d,
        "hidden": model.hidden,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "n_heads_source": sources["n_heads"],
        "n_heads_requested": None if n_heads is None else int(n_heads),
        "ffn_mult": model.ffn_mult,
        "max_len": model.max_len,
        "pos_encoding": model.pos_encoding,
        "final_norm": model.final_norm is not None,
        "mu_scale": model.mu_scale,
        "sigma_floor": model.sigma_floor,
        "sigma_init": model.sigma_init,
        "sigma_min": model.sigma_min,
        "sigma_max": model.sigma_max,
        "pos_emb_init_std": model.pos_emb_init_std,
        "pos_encoding_source": sources["pos_encoding"],
        "architecture_sources": {
            key: sources[key] for key in ("hidden", "n_layers", "ffn_mult", "max_len")
        },
        "option_sources": sources,
        "inferred_unparameterized_options": bool(missing_options),
        "missing_unparameterized_options": missing_options,
        "config_sgt": dict(config),
        "config_keys_ignored": unknown,
    }
    return model, meta


def load_sgt_for_lm(
    model_path: str | Path,
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
    hidden: int = 512,
    n_layers: int = 2,
    n_heads: int | None = None,
    sigma_init: float = 1.0,
    return_meta: bool = False,
) -> SGTModel | tuple[SGTModel, dict[str, Any]]:
    """读取目标模型 config.json 的 ``hidden_size``，加载 SGT checkpoint。

    checkpoint 兼容三种封装：裸 state_dict、``{"state_dict": ...}``，以及
    ``sgt.train.train_sgt`` / ``experiments/run_ablation.py`` 保存的
    ``{"sgt": ...}`` 训练器格式（``run_ablation`` 还带 ``config["sgt"]``，见下）。

    重建规则（见 :func:`build_sgt_from_state`）：**优先**用 checkpoint 的
    ``config["sgt"]`` 重建 :class:`SGTModel`（未知键忽略、缺失键用模型默认），state_dict
    形状推断只作一致性校验（不一致报 :class:`ValueError`，不静默）。没有 config 的旧
    checkpoint 按形状推断：``final_norm`` / ``pos_encoding`` 可由键推断，``mu_scale`` /
    ``sigma_floor`` **无法**推断 → 取默认 ``None``，此时 meta 的
    ``inferred_unparameterized_options`` 为 ``True``（调用方应优先使用带 config 的
    checkpoint）。

    维度校验：checkpoint 的 ``input_proj.weight`` 第 1 维（d）必须等于目标模型的
    ``hidden_size``，否则抛 :class:`ValueError`（调用方据此显式失败，不得回退基线）。

    ``hidden`` / ``n_layers`` / ``sigma_init`` 只在 checkpoint 没有对应 config 时使用；
    checkpoint 的实际架构与之不一致时发 ``RuntimeWarning`` 并按 checkpoint 构造。
    ``n_heads`` 无法由 state_dict 推断（参数形状与头数无关）：checkpoint config 有则用
    config（与调用方给的 ``n_heads`` 不一致时发 ``RuntimeWarning``），否则用调用方值，
    再默认 4。

    Args:
        return_meta: 为 ``True`` 时返回 ``(model, meta)``；``meta`` 记录实际使用的
            ``pos_encoding`` / ``final_norm`` / ``mu_scale`` / ``sigma_floor``、架构与各项
            来源，供结果 JSON 追溯。

    Returns:
        放到 ``device`` 并切到 eval 模式的 :class:`SGTModel`；``return_meta=True`` 时返回
        ``(model, meta)``。
    """
    model_path = Path(model_path)
    ckpt_path = Path(ckpt_path)
    d = _read_hidden_size(model_path)
    state, sgt_config = _read_checkpoint(ckpt_path)

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

    sgt, meta = build_sgt_from_state(
        state,
        sgt_config,
        n_heads=n_heads,
        fallbacks={"sigma_init": sigma_init},
    )
    for key, requested in (("hidden", hidden), ("n_layers", n_layers)):
        if int(requested) != int(meta[key]):
            warnings.warn(
                f"SGT checkpoint 的实际架构为 {key}={meta[key]}，与请求的 {requested} "
                "不一致；按 checkpoint 构造",
                RuntimeWarning,
                stacklevel=2,
            )

    sgt.to(device)
    sgt.eval()
    meta = {"ckpt": str(ckpt_path), "model_path": str(model_path), **meta}
    if return_meta:
        return sgt, meta
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
