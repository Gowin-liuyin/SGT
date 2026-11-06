"""M3 `SGTModel` 的验证：API/形状、初始化、确定性、序列与位置依赖、padding 不变性、
梯度、序列化往返、Eq (1) 采样、`sigma_min` 下限（M3 任务单 1–10），以及审计 GAP-1/GAP-3：
宽度 2048 可前向（A1）、`sigma_max` 有界参数化（A21）；以及数值稳定 / 长度泛化的三个
声明性选项：`mu_scale`（A21，µ 有界）、`sigma_floor`（A21，Σθ 下界）、
`pos_encoding="sinusoidal"/"none"`（A1，长度泛化），及其与 `final_norm`/`sigma_max`/
零初始化头的组合（test_16–test_20）。

说明（为何部分测试要先改写输出头）：
任务单要求输出头**零初始化**，因此模型刚构造时 `mu ≡ 0`、`sigma ≡ sigma_init`，前向
输出与输入完全无关（这是初始化契约，不是被测架构性质）。凡是检验“输入/位置如何影响
输出”的测试，都先用固定随机权重填充输出头来模拟训练后状态（`_trained_like`）；零初始化
契约本身由 `test_02` 单独逐项断言。
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgt.model import SGTModel  # noqa: E402
from sgt.sampling import sample_sgt  # noqa: E402

D = 8
HIDDEN = 16
N_LAYERS = 2
N_HEADS = 4
FFN_MULT = 4
MAX_LEN = 16
T = 6
B = 3


def _model(**overrides) -> SGTModel:
    kwargs = dict(
        d=D,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        max_len=MAX_LEN,
        sigma_init=1.0,
        sigma_min=1e-6,
        dropout=0.0,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)  # 固定默认初始化（encoder / pos_emb / input_proj）
    return SGTModel(**kwargs)


def _rand(shape, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=dtype)


def _trained_like(seed: int = 1234, **overrides) -> SGTModel:
    """构造模型并注入固定随机输出头权重（模拟“训练后”状态），eval 模式。"""
    model = _model(**overrides)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.head.weight.copy_(
            torch.randn(
                model.head.weight.shape,
                generator=g,
                dtype=model.head.weight.dtype,
            )
            * 0.2
        )
    model.eval()
    return model


def _sigma_from_log_sigma2(model: SGTModel, log_sigma2: torch.Tensor) -> torch.Tensor:
    """任务单规定的 sigma：exp(0.5·clamp(log_sigma2, min=log(sigma_min²)))。"""
    return torch.exp(0.5 * log_sigma2.clamp(min=model.log_sigma_min2))


# --------------------------------------------------------------------------- #
# 1. 形状
# --------------------------------------------------------------------------- #
def test_01_shapes():
    model = _model()
    x = _rand((B, T, D), 0)
    mu, log_sigma2 = model(x)
    assert mu.shape == (B, T, D)
    assert log_sigma2.shape == (B, T, D)
    assert mu.dtype == torch.float32 and log_sigma2.dtype == torch.float32

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[-1, -2:] = False
    mu_m, log_sigma2_m = model(x, attention_mask=mask)
    assert mu_m.shape == (B, T, D) and log_sigma2_m.shape == (B, T, D)

    # 结构：d -> hidden 投影、n_layers 层双向 pre-LN GELU、hidden -> 2d 输出头。
    assert model.input_proj.weight.shape == (HIDDEN, D)
    assert model.head.weight.shape == (2 * D, HIDDEN)
    assert len(model.encoder.layers) == N_LAYERS
    layer = model.encoder.layers[0]
    assert layer.norm_first is True
    assert layer.self_attn.batch_first is True
    assert layer.activation is F.gelu
    assert layer.linear1.out_features == FFN_MULT * HIDDEN
    assert layer.linear2.out_features == HIDDEN
    assert model.pos_emb.weight.shape == (MAX_LEN, HIDDEN)
    # 双向（非因果）由 test_04 的行为断言验证（改 t=3 的 token 会改变 t=0 的输出）。


# --------------------------------------------------------------------------- #
# 2. 初始化契约
# --------------------------------------------------------------------------- #
def test_02_initialization():
    model = _model(sigma_init=1.0)
    x = _rand((B, T, D), 1)
    mu, log_sigma2 = model(x)
    assert mu.abs().max().item() < 1e-12
    sigma = torch.exp(0.5 * log_sigma2)
    assert torch.allclose(sigma, torch.ones_like(sigma), atol=1e-6)

    # 头参数本身：mu 头权重/偏置为 0；log_sigma2 头权重为 0、偏置为 log(sigma_init²)。
    assert model.head.weight.abs().max().item() == 0.0
    assert model.head.bias[:D].abs().max().item() == 0.0
    assert torch.allclose(
        model.head.bias[D:], torch.full((D,), math.log(1.0**2)), atol=1e-7
    )

    # sigma_init 参数被真正使用。
    for sigma_init in (0.3, 2.5):
        model_s = _model(sigma_init=sigma_init)
        mu_s, log_sigma2_s = model_s(x)
        assert mu_s.abs().max().item() < 1e-12
        sigma_s = torch.exp(0.5 * log_sigma2_s)
        assert torch.allclose(
            sigma_s, torch.full_like(sigma_s, sigma_init), atol=1e-6
        ), f"sigma_init={sigma_init} 时初始 sigma 不是常数"
        assert torch.allclose(
            model_s.head.bias[D:],
            torch.full((D,), math.log(sigma_init**2)),
            atol=1e-7,
        )


# --------------------------------------------------------------------------- #
# 3. 确定性 + 不修改输入
# --------------------------------------------------------------------------- #
def test_03_deterministic_eval_and_no_input_mutation():
    model = _trained_like(seed=3, dropout=0.3)  # eval 下 dropout 必须被关闭
    model.eval()
    x = _rand((B, T, D), 2)
    x_before = x.clone()
    with torch.no_grad():
        mu_a, lv_a = model(x)
        mu_b, lv_b = model(x)
    assert torch.equal(mu_a, mu_b)
    assert torch.equal(lv_a, lv_b)
    assert torch.equal(x, x_before), "forward 修改了输入张量"


# --------------------------------------------------------------------------- #
# 4. 序列依赖（双向）：改位置 3 的 token，所有位置（含 t=0）都变化
# --------------------------------------------------------------------------- #
def test_04_sequence_dependence_all_positions():
    model = _trained_like(seed=11)
    x = _rand((2, T, D), 4)
    x2 = x.clone()
    x2[:, 3] = _rand((2, D), 5) + 1.0  # 明显不同的 token
    with torch.no_grad():
        mu, log_sigma2 = model(x)
        mu2, log_sigma2_2 = model(x2)

    for name, a, b in (
        ("mu", mu, mu2),
        ("log_sigma2", log_sigma2, log_sigma2_2),
    ):
        delta = (b - a).abs().amax(dim=-1)  # [B, T]
        min_delta = delta.min().item()
        assert min_delta > 1e-6, f"{name}: 最小改变量 {min_delta:.3e} ≤ 1e-6"
        # t=0 在因果模型里看不到 t=3，此处必须改变 → 证明双向注意力。
        assert (delta[:, 0] > 1e-6).all(), f"{name}: t=0 未变化，疑似因果掩码"


# --------------------------------------------------------------------------- #
# 5. 位置依赖：token 置换后输出不同；且不是原输出的同置换
# --------------------------------------------------------------------------- #
def test_05_position_dependence():
    model = _trained_like(seed=12)
    x = _rand((2, T, D), 6)
    perm = torch.tensor([2, 0, 5, 1, 4, 3])
    inv = torch.argsort(perm)
    with torch.no_grad():
        mu, log_sigma2 = model(x)
        mu_p, log_sigma2_p = model(x[:, perm])

    for name, a, b in (
        ("mu", mu, mu_p),
        ("log_sigma2", log_sigma2, log_sigma2_p),
    ):
        assert not torch.allclose(a, b, atol=1e-6), f"{name}: 置换前后输出相同"
        # 若没有位置编码，双向 Transformer 对置换等变：b[:, inv] 会逐元素等于 a。
        aligned_diff = (b[:, inv] - a).abs().max().item()
        assert aligned_diff > 1e-6, (
            f"{name}: 置换后的输出等于原输出的同置换（{aligned_diff:.3e}），"
            "位置编码未生效"
        )

    # 直接证据：同一 token 复制到所有位置，各位置输出仍不同 → 位置信息进入了网络。
    token = _rand((1, 1, D), 7).expand(1, T, D).contiguous()
    with torch.no_grad():
        mu_rep, log_sigma2_rep = model(token)
    assert (mu_rep - mu_rep[:, :1]).abs().max().item() > 1e-6
    assert (log_sigma2_rep - log_sigma2_rep[:, :1]).abs().max().item() > 1e-6


# --------------------------------------------------------------------------- #
# 6. padding 不变性：有效位置输出不依赖 padding 内容
# --------------------------------------------------------------------------- #
def test_06_padding_invariance():
    model = _trained_like(seed=13)
    x1 = _rand((B, 4, D), 8)
    padding = _rand((B, 2, D), 9)
    x2 = torch.cat([x1, padding], dim=1)
    mask2 = torch.tensor([[True, True, True, True, False, False]] * B)

    with torch.no_grad():
        mu1, lv1 = model(x1)
        mu2, lv2 = model(x2, attention_mask=mask2)
    assert torch.allclose(mu1, mu2[:, :4], atol=1e-6)
    assert torch.allclose(lv1, lv2[:, :4], atol=1e-6)

    # 换掉 padding 内容，有效位置必须逐位不变。
    x3 = x2.clone()
    x3[:, 4:] = _rand((B, 2, D), 10)
    with torch.no_grad():
        mu3, lv3 = model(x3, attention_mask=mask2)
    assert torch.allclose(mu1, mu3[:, :4], atol=1e-6)
    assert torch.allclose(lv1, lv3[:, :4], atol=1e-6)

    # int64 掩码（HF 风格）与 bool 掩码等价。
    with torch.no_grad():
        mu4, lv4 = model(x2, attention_mask=mask2.long())
    assert torch.equal(mu2, mu4) and torch.equal(lv2, lv4)


# --------------------------------------------------------------------------- #
# 7. 梯度：所有参数梯度有限且非零（mu 头、位置编码非零）
# --------------------------------------------------------------------------- #
def test_07_gradients_finite_and_nonzero():
    model = _trained_like(seed=14)
    model.train()
    x = _rand((B, T, D), 11)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -1] = False
    mu, log_sigma2 = model(x, attention_mask=mask)
    (mu.sum() + log_sigma2.sum()).backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name}: 无梯度"
        assert torch.isfinite(param.grad).all(), f"{name}: 梯度含非有限值"
        assert param.grad.abs().max().item() > 0.0, f"{name}: 梯度全为 0"

    assert model.head.weight.grad.abs().max().item() > 0.0
    grad_pos = model.pos_emb.weight.grad
    assert grad_pos[:T].abs().max().item() > 0.0, "位置编码梯度为 0"
    # 未被任何序列使用的位置行不应有梯度。
    assert grad_pos[T:].abs().max().item() == 0.0


# --------------------------------------------------------------------------- #
# 8. state_dict 保存/加载往返
# --------------------------------------------------------------------------- #
def test_08_state_dict_roundtrip():
    model = _trained_like(seed=15)
    x = _rand((2, T, D), 12)
    with torch.no_grad():
        mu_ref, lv_ref = model(x)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=True)

    fresh = _model()  # 另一组默认初始化
    fresh.load_state_dict(state)
    fresh.eval()
    with torch.no_grad():
        mu_new, lv_new = fresh(x)
    assert torch.equal(mu_ref, mu_new)
    assert torch.equal(lv_ref, lv_new)
    assert {"input_proj.weight", "pos_emb.weight", "head.weight"} <= set(state)


# --------------------------------------------------------------------------- #
# 9. sample()：给定 u 时严格等于 Eq (1)；generator 决定随机性
# --------------------------------------------------------------------------- #
def test_09_sample_formula_and_generator():
    model = _trained_like(seed=16)
    model.eval()
    x = _rand((2, T, D), 13)
    x_before = x.clone()
    u = _rand((2, T, D), 14)

    with torch.no_grad():
        mu, log_sigma2 = model(x)
        sigma = _sigma_from_log_sigma2(model, log_sigma2)
        out = model.sample(x, u=u)
    # Σ 的对角必须严格正（exp 结构保证；MODULE_PLAN M3/c-posdef）。
    assert torch.isfinite(sigma).all() and (sigma > 0).all()
    assert torch.allclose(out, x + mu + sigma * u, atol=1e-12)
    assert torch.equal(x, x_before)

    with torch.no_grad():
        a = model.sample(x, generator=torch.Generator().manual_seed(7))
        b = model.sample(x, generator=torch.Generator().manual_seed(7))
        expected = sample_sgt(
            x, mu, sigma, generator=torch.Generator().manual_seed(7)
        )
        c = model.sample(x, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b), "同 seed 两次采样不逐位一致"
    assert torch.equal(a, expected), "sample() 与 sample_sgt 的组合不一致"
    assert not torch.equal(a, c), "不同 seed 采样结果相同"


# --------------------------------------------------------------------------- #
# 10. sigma_min 下限
# --------------------------------------------------------------------------- #
def test_10_sigma_min_clamp():
    # (a) 样本层面：mu≡0（输出头零初始化）且 x=0 时 out = sigma·u，可直接反解 sigma，
    #     避免 x+mu ~ O(1) 与 sigma·u ~ 1e-6 相减的灾难性抵消。
    plain = _model()
    with torch.no_grad():
        plain.head.bias[D:] = -80.0  # 原始 sigma = exp(-40) ≪ sigma_min
    x0 = torch.zeros(2, T, D)
    u = _rand(x0.shape, 16).abs() + 0.5  # 避免用接近 0 的 u 反解除噪
    with torch.no_grad():
        mu0, lv0 = plain(x0)
        raw0 = torch.exp(0.5 * lv0)
        out0 = plain.sample(x0, u=u)
    assert (raw0 < plain.sigma_min).all(), "构造失败：clamp 未触发"
    sigma_eff = out0 / u
    # 任务单公式在 float32 下与 sigma_min 相差 ~1 ulp（相对 2.3e-7），故用 rtol=1e-6
    # 断言“等于下限”，并单独确认没有低于 sigma_min 超过该浮点误差。
    assert torch.allclose(
        sigma_eff, torch.full_like(sigma_eff, plain.sigma_min), rtol=1e-6
    )
    assert (sigma_eff >= plain.sigma_min * (1 - 1e-6)).all()

    # (b) 含非零 mu 的训练后状态：sample 使用的 sigma 正是 clamp 后的公式值。
    model = _trained_like(seed=17)
    with torch.no_grad():
        model.head.bias[D:] = -80.0
    x = _rand((2, T, D), 15)
    with torch.no_grad():
        mu, log_sigma2 = model(x)
        raw_sigma = torch.exp(0.5 * log_sigma2)
        sigma_used = _sigma_from_log_sigma2(model, log_sigma2)
        out = model.sample(x, u=u)
    assert (raw_sigma < model.sigma_min).all()
    assert torch.allclose(
        sigma_used, torch.full_like(sigma_used, model.sigma_min), rtol=1e-6
    )
    assert (sigma_used >= model.sigma_min * (1 - 1e-6)).all()
    assert torch.allclose(out, x + mu + sigma_used * u, atol=1e-12)

    # (c) 自定义 sigma_min 同样生效。
    plain2 = _model(sigma_min=1e-2)
    with torch.no_grad():
        plain2.head.bias[D:] = -80.0
        out2 = plain2.sample(x0, u=u)
    sigma_eff2 = out2 / u
    assert torch.allclose(sigma_eff2, torch.full_like(sigma_eff2, 1e-2), rtol=1e-6)
    assert (sigma_eff2 >= 1e-2 * (1 - 1e-6)).all()


# --------------------------------------------------------------------------- #
# 结构契约补充：max_len / 形状校验 / dtype
# --------------------------------------------------------------------------- #
def test_11_max_len_and_shape_validation():
    model = _model(max_len=T)
    mu, _ = model(_rand((1, T, D), 17))  # T == max_len 合法
    assert mu.shape == (1, T, D)

    with pytest.raises(ValueError, match="max_len"):
        model(_rand((1, T + 1, D), 18))

    with pytest.raises(ValueError, match="最后一维"):
        model(_rand((1, T, D + 1), 19))

    with pytest.raises(ValueError, match="attention_mask"):
        model(_rand((2, T, D), 20), attention_mask=torch.ones(2, T + 1, dtype=torch.bool))

    with pytest.raises(ValueError, match="至少需要一个 True"):
        model(
            _rand((2, T, D), 21),
            attention_mask=torch.zeros(2, T, dtype=torch.bool),
        )


def test_12_dtype_contract():
    model32 = _model()
    assert next(model32.parameters()).dtype == torch.float32
    mu32, lv32 = model32(_rand((2, 5, D), 22))
    assert mu32.dtype == torch.float32 and lv32.dtype == torch.float32

    model64 = _model().double()
    x64 = _rand((2, 5, D), 23, dtype=torch.float64)
    x64_before = x64.clone()
    mu64, lv64 = model64(x64)
    assert mu64.dtype == torch.float64 and lv64.dtype == torch.float64
    assert torch.equal(x64, x64_before)

    # 不做隐式降精度：fp64 输入配 fp32 参数直接报错，提示先 model.double()。
    with pytest.raises(ValueError, match="dtype"):
        model32(x64)


# --------------------------------------------------------------------------- #
# 13. GAP-3 / A21：sigma_max 有界参数化（Eq (12) 行所必需）
# --------------------------------------------------------------------------- #
def test_13_sigma_max_clamps_forward_and_sample():
    """人为把 log_sigma2 头 bias 设为 log(1.0)（原始 σ=1），sigma_max=0.05 必须生效。

    同时验证 `sample()` 使用的 σ 正是 forward 返回（夹过）的 σ —— 否则损失 / 评测看
    到的 σ 与采样不一致。
    """
    model = _model(sigma_max=0.05)
    with torch.no_grad():
        model.head.bias[D:].fill_(math.log(1.0))  # 原始 σ ≡ 1 ≫ sigma_max
    # 前提：未夹时的原始 σ 确实超过上界（否则该测试不能证明 clamp 生效）。
    raw_sigma = torch.exp(0.5 * model.head.bias[D:])
    assert (raw_sigma > 0.05 + 1e-3).all()

    x = _rand((2, T, D), 30)
    u = _rand((2, T, D), 31)
    with torch.no_grad():
        mu, log_sigma2 = model(x)
        sigma = torch.exp(0.5 * log_sigma2)
        out = model.sample(x, u=u)

    # forward 返回的 log_sigma2 已被夹到 log(sigma_max²)，σ ≤ sigma_max（1e-6 容差）。
    assert (log_sigma2 <= model.log_sigma_max2 + 1e-6).all()
    assert torch.allclose(sigma, torch.full_like(sigma, 0.05), atol=1e-6)
    assert (sigma <= 0.05 + 1e-6).all()

    # sample() 与 x + mu + σ·u 逐位一致，且 σ 用的是返回的（夹过的）值。
    assert torch.allclose(out, x + mu + sigma * u, atol=1e-12)
    # 用输出反解 σ（u 取非零）：必须 ≤ sigma_max + 1e-6，而非原始 σ=1。
    # 容差依据：float32 下 (x + mu + σu) - x - mu 的相消误差 ~1e-7/|u|，取 |u|>0.1 ⇒ ~1e-6，
    # 故用 atol=1e-5 判“等于上界”。
    mask_u = u.abs() > 0.1
    sigma_eff = (out - x - mu)[mask_u] / u[mask_u]
    assert (sigma_eff <= 0.05 + 1e-6).all()
    assert torch.allclose(sigma_eff, torch.full_like(sigma_eff, 0.05), atol=1e-5)

    # 上界只在超出时生效：sigma_init=1 且 sigma_max=2 时 σ 仍为 1。
    loose = _model(sigma_max=2.0)
    with torch.no_grad():
        _, lv_loose = loose(x)
    assert torch.allclose(torch.exp(0.5 * lv_loose), torch.ones(2, T, D), atol=1e-6)


# --------------------------------------------------------------------------- #
# 14. sigma_max=None：保持论文 Def 3.1 的无界定义
# --------------------------------------------------------------------------- #
def test_14_sigma_max_none_is_unbounded():
    model = _model(sigma_max=None)
    assert model.sigma_max is None and model.log_sigma_max2 is None
    with torch.no_grad():
        model.head.bias[D:].fill_(math.log(1.0))
    x = _rand((2, T, D), 32)
    with torch.no_grad():
        _mu, log_sigma2 = model(x)
    sigma = torch.exp(0.5 * log_sigma2)
    assert torch.allclose(sigma, torch.ones_like(sigma), atol=1e-6)
    # 同一构造在 sigma_max=0.05 下会被夹到 0.05 → 默认行为确实是无界的。
    bounded = _model(sigma_max=0.05)
    with torch.no_grad():
        bounded.head.bias[D:].fill_(math.log(1.0))
        _, lv_bounded = bounded(x)
    assert (torch.exp(0.5 * lv_bounded) <= 0.05 + 1e-6).all()

    # 参数非法必须快速失败（sigma_max 必须 > sigma_min 且有限）。
    for bad in (1e-6, 1e-9, 0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="sigma_max"):
            _model(sigma_max=bad)


# --------------------------------------------------------------------------- #
# 15. GAP-1 / A1：宽度可配（hidden=2048, n_heads=8 必须可前向）
# --------------------------------------------------------------------------- #
def test_15_hidden_2048_forward():
    d, t, b = 2048, 4, 1
    torch.manual_seed(0)
    model = SGTModel(d, hidden=2048, n_layers=2, n_heads=8, max_len=8)
    x = torch.randn(b, t, d)
    mask = torch.ones(b, t, dtype=torch.bool)
    with torch.no_grad():
        mu, log_sigma2 = model(x, attention_mask=mask)
    assert mu.shape == (b, t, d) and log_sigma2.shape == (b, t, d)
    sigma = torch.exp(0.5 * log_sigma2)
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all() and (sigma > 0).all()
    # 结构：宽度 2048 真的被用上（不是静默回退到默认 512）。
    assert model.input_proj.weight.shape == (2048, d)
    assert model.head.weight.shape == (2 * d, 2048)

    # n_heads 必须整除 hidden（A1 的可配置前提）。
    with pytest.raises(ValueError, match="整除"):
        SGTModel(8, hidden=16, n_heads=3)


# --------------------------------------------------------------------------- #
# A21（µθ 有界）/ A21（Σθ 下界）/ A1（位置编码与长度泛化）辅助
# --------------------------------------------------------------------------- #
def _fill_head_weights(model: SGTModel, seed: int = 1234, scale: float = 0.2) -> None:
    """给零初始化的输出头注入固定随机权重（模拟训练后状态；见模块 docstring）。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.head.weight.copy_(
            torch.randn(model.head.weight.shape, generator=g) * scale
        )
    model.eval()


# --------------------------------------------------------------------------- #
# 16. A21：mu_scale 有界参数化（Eq (9) 的 |cos| 对 µ 尺度不变 ⇒ 无界头发散）
# --------------------------------------------------------------------------- #
def test_16_mu_scale_bounds_mu():
    x_seeds = (0, 1, 2, 3, 4, 5)
    # (a) mu_scale=1.0：随机权重 + 多次前向，|mu| 严格 < 1。
    #     同一组权重在 mu_scale=None 下的原始 |mu| 确实 > 1（否则 clamp 未真正生效，
    #     测试就没有意义）。
    for head_seed in (1234, 4321, 999):
        raw = _model()  # mu_scale 默认 None
        _fill_head_weights(raw, seed=head_seed, scale=0.2)
        bounded = _model(mu_scale=1.0)
        bounded.load_state_dict(raw.state_dict())
        bounded.eval()

        raw_max = 0.0
        bounded_max = 0.0
        with torch.no_grad():
            for seed in x_seeds:
                x = _rand((B, T, D), seed)
                raw_max = max(raw_max, raw(x)[0].abs().max().item())
                bounded_max = max(bounded_max, bounded(x)[0].abs().max().item())
        assert raw_max > 1.0, f"head_seed={head_seed}: 构造失败，原始 |µ| 未超过 1"
        assert bounded_max < 1.0, (
            f"head_seed={head_seed}: mu_scale=1.0 下 |µ|={bounded_max:.6f} 未严格 < 1"
        )

    # (b) 有界值就是 M·tanh(raw/M)（逐元素）：同权重两份模型逐值比对。
    unbounded = _model()
    _fill_head_weights(unbounded, seed=1234, scale=0.2)
    phi = _model(mu_scale=1.0)
    phi.load_state_dict(unbounded.state_dict())
    phi.eval()
    x = _rand((B, T, D), 40)
    with torch.no_grad():
        mu_raw, _ = unbounded(x)
        mu_phi, _ = phi(x)
    assert torch.allclose(mu_phi, torch.tanh(mu_raw), atol=1e-7)
    # 大 M 时 tanh 近似恒等：mu_scale=1e3 时与原始值几乎一致（不改变 Eq (1) 的形式）。
    # 容差 2e-5：M·tanh(raw/M) = raw - raw³/(3M²) + O(raw⁵/M⁴)，|raw|≤3 时截断误差
    # ~9e-6，加上 float32 舍入（实测最大 7.6e-6）。
    loose = _model(mu_scale=1e3)
    loose.load_state_dict(unbounded.state_dict())
    with torch.no_grad():
        mu_loose, _ = loose(x)
    assert torch.allclose(mu_loose, mu_raw, atol=2e-5)

    # (c) mu_scale=None 时确实无此限制：头权重放大 10× 后 |µ| 远超 1。
    scaled = _model()
    _fill_head_weights(scaled, seed=1234, scale=0.2)
    with torch.no_grad():
        scaled.head.weight.mul_(10.0)
    with torch.no_grad():
        mu_scaled, _ = scaled(x)
    assert mu_scaled.abs().max().item() > 1.0, "mu_scale=None 时不应有上界"
    # 同权重 + mu_scale=1.0：仍不超过 1。用 <=（非严格）：float32 下 tanh 在
    # |raw/mu_scale| ≳ 9 时饱和到 ±1，返回值可能恰为 ±M（浮点极限，见模块 docstring）。
    clamped = _model(mu_scale=1.0)
    clamped.load_state_dict(scaled.state_dict())
    with torch.no_grad():
        mu_clamped, _ = clamped(x)
    assert (mu_clamped.abs() <= 1.0).all()

    # (d) 校验：mu_scale 必须为正的有限值。
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="mu_scale"):
            _model(mu_scale=bad)


# --------------------------------------------------------------------------- #
# 17. A21：sigma_floor 下界（Eq (12) 的 -log|Σ| 与 Eq (9) 的 |cos| 都会把 σ 推向 0）
# --------------------------------------------------------------------------- #
def test_17_sigma_floor_clamps_forward_and_sample():
    model = _model(sigma_floor=0.005, sigma_max=0.02)
    with torch.no_grad():
        model.head.bias[D:].fill_(-50.0)  # 原始 σ = exp(-25) ≈ 1.4e-11 ≪ 0.005
    # 前提：未夹时原始 σ 确实低于下限（否则该测试不能证明 clamp 生效）。
    raw_sigma = torch.exp(0.5 * model.head.bias[D:])
    assert (raw_sigma < 0.005).all()

    x = _rand((2, T, D), 41)
    u = _rand((2, T, D), 42)
    with torch.no_grad():
        mu, log_sigma2 = model(x)
        sigma = torch.exp(0.5 * log_sigma2)
        out = model.sample(x, u=u)
    # 任务单容差：σ ∈ [floor - 1e-9, sigma_max + 1e-9]（float32 下 exp(log(f²)) 有
    # ~1e-10 的舍入误差，实测 0.004999999888）。
    assert (sigma >= 0.005 - 1e-9).all(), "σ 低于 sigma_floor"
    assert (sigma <= 0.02 + 1e-9).all(), "σ 高于 sigma_max"
    # σ 正好停在下限（不是仅仅落进区间），且 forward 返回的就是夹过的值。
    assert torch.allclose(sigma, torch.full_like(sigma, 0.005), atol=1e-9)
    assert (log_sigma2 >= model.log_sigma_floor2 - 1e-9).all()
    # sample() 与损失/评测共用同一 σ：x̃ = x + mu + σ·u 逐位一致。
    assert torch.allclose(out, x + mu + sigma * u, atol=1e-12)

    # 默认 sigma_floor=None：不施加下界，返回原始（可塌缩的）σ。
    plain = _model()
    with torch.no_grad():
        plain.head.bias[D:].fill_(-50.0)
    with torch.no_grad():
        _, lv_plain = plain(x)
    assert (torch.exp(0.5 * lv_plain) < 0.005).all(), "默认行为不应有下界"

    # 区间内不夹：sigma_init=0.01 ∈ [0.005, 0.02] 时 σ 保持 0.01。
    in_range = _model(sigma_init=0.01, sigma_floor=0.005, sigma_max=0.02)
    with torch.no_grad():
        _, lv_in = in_range(x)
    assert torch.allclose(
        torch.exp(0.5 * lv_in), torch.full((2, T, D), 0.01), atol=1e-6
    )

    # 上界仍生效（与下界对称）：bias 设大时 σ 停在 sigma_max。
    high = _model(sigma_floor=0.005, sigma_max=0.02)
    with torch.no_grad():
        high.head.bias[D:].fill_(50.0)
    with torch.no_grad():
        _, lv_high = high(x)
    sigma_high = torch.exp(0.5 * lv_high)
    assert (sigma_high <= 0.02 + 1e-9).all()
    assert torch.allclose(sigma_high, torch.full_like(sigma_high, 0.02), atol=1e-9)

    # 校验：正有限值，且 sigma_min <= sigma_floor <= sigma_max。
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="sigma_floor"):
            _model(sigma_floor=bad)
    with pytest.raises(ValueError, match="sigma_min"):
        _model(sigma_floor=1e-9)  # 低于 sigma_min=1e-6
    with pytest.raises(ValueError, match="sigma_floor"):
        _model(sigma_floor=0.05, sigma_max=0.02)  # 高于 sigma_max
    # 边界合法：== sigma_min、== sigma_max、以及 sigma_max=None 时只校验 >= sigma_min。
    _model(sigma_floor=1e-6)
    _model(sigma_floor=0.02, sigma_max=0.02)
    _model(sigma_floor=0.02)


# --------------------------------------------------------------------------- #
# 18. A1：sinusoidal 位置编码（无参数、按 T 动态生成、长度泛化）
# --------------------------------------------------------------------------- #
def test_18_pos_encoding_sinusoidal_length_generalization():
    max_len = 8
    model = _trained_like(seed=21, max_len=max_len, pos_encoding="sinusoidal")
    learned = _model(max_len=max_len, pos_encoding="learned")
    # 无参数：pos_emb 不存在，参数集合恰为 learned 去掉 pos_emb.weight。
    assert model.pos_emb is None
    assert set(dict(model.named_parameters())) == (
        set(dict(learned.named_parameters())) - {"pos_emb.weight"}
    )
    assert sum(p.numel() for p in learned.parameters()) - sum(
        p.numel() for p in model.parameters()
    ) == max_len * HIDDEN

    # 公式：pe[pos, 2i] = sin(pos / 10000^(2i/hidden))，pe[pos, 2i+1] = cos(...)。
    # 容差 1e-5：float32 下 sin/cos 的参数最大到 63 rad，实测最大误差 8.8e-7。
    pe = model._sinusoidal_pos_encoding(
        64, device=torch.device("cpu"), dtype=torch.float32
    )
    assert pe.shape == (64, HIDDEN)
    for pos in (0, 1, 17, 63):
        for i in range(HIDDEN // 2):
            angle = pos / (10000 ** (2 * i / HIDDEN))
            assert abs(pe[pos, 2 * i].item() - math.sin(angle)) < 1e-5
            assert abs(pe[pos, 2 * i + 1].item() - math.cos(angle)) < 1e-5
    # pos=0 行固定为 [0, 1, 0, 1, ...]。
    assert torch.equal(pe[0, 0::2], torch.zeros(HIDDEN // 2))
    assert torch.equal(pe[0, 1::2], torch.ones(HIDDEN // 2))

    # T=64 > max_len=8 可前向；同一输入两次前向逐位一致（无随机性）。
    x64 = _rand((2, 64, D), 40)
    with torch.no_grad():
        mu_a, lv_a = model(x64)
        mu_b, lv_b = model(x64)
    assert mu_a.shape == (2, 64, D) and lv_a.shape == (2, 64, D)
    assert torch.isfinite(mu_a).all() and torch.isfinite(lv_a).all()
    assert torch.equal(mu_a, mu_b) and torch.equal(lv_a, lv_b)
    # T=64 与 T=65 两种长度都能跑通（长度泛化；sinusoidal 按 T 动态生成）。
    with torch.no_grad():
        mu65, lv65 = model(_rand((2, 65, D), 41))
    assert mu65.shape == (2, 65, D) and lv65.shape == (2, 65, D)

    # max_len 上限校验只对 learned 生效：同样 T=9 在 learned 下仍抛 ValueError。
    learned_trained = _trained_like(seed=21, max_len=max_len)
    with pytest.raises(ValueError, match="max_len"):
        learned_trained(_rand((1, max_len + 1, D), 42))

    # 位置信息确实进入网络：置换 token 后，输出不等于原输出的同置换。
    x = _rand((2, max_len, D), 43)
    perm = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])
    inv = torch.argsort(perm)
    with torch.no_grad():
        mu_p0, _ = model(x)
        mu_p1, _ = model(x[:, perm])
    assert (mu_p1[:, inv] - mu_p0).abs().max().item() > 1e-6

    # hidden 为奇数：最后一维置 0（不报错），前向正常。
    torch.manual_seed(0)
    odd = SGTModel(
        4, hidden=17, n_layers=1, n_heads=1, max_len=4, pos_encoding="sinusoidal"
    )
    pe_odd = odd._sinusoidal_pos_encoding(5)
    assert pe_odd.shape == (5, 17)
    assert (pe_odd[:, -1] == 0).all(), "hidden 为奇数时最后一维应置 0"
    with torch.no_grad():
        mu_odd, _ = odd(_rand((1, 5, 4), 44))
    assert mu_odd.shape == (1, 5, 4)

    # 非法 pos_encoding 快速失败。
    for bad in ("rope", "Learned", ""):
        with pytest.raises(ValueError, match="pos_encoding"):
            _model(pos_encoding=bad)


# --------------------------------------------------------------------------- #
# 19. A1：pos_encoding="none"（无位置编码，同样不受 max_len 限制）
# --------------------------------------------------------------------------- #
def test_19_pos_encoding_none_length_generalization():
    max_len = 8
    model = _trained_like(seed=22, max_len=max_len, pos_encoding="none")
    assert model.pos_emb is None
    assert all("pos_emb" not in name for name, _ in model.named_parameters())

    x64 = _rand((2, 64, D), 50)
    with torch.no_grad():
        mu_a, lv_a = model(x64)
        mu_b, lv_b = model(x64)
    assert mu_a.shape == (2, 64, D) and lv_a.shape == (2, 64, D)
    assert torch.equal(mu_a, mu_b) and torch.equal(lv_a, lv_b)
    # 刚超过 max_len 的 T=9 也可前向。
    with torch.no_grad():
        mu9, _ = model(_rand((1, max_len + 1, D), 51))
    assert mu9.shape == (1, max_len + 1, D)

    # 无位置信息的直接证据：双向 Transformer 对 token 置换等变，即把输出按同一置换
    # 排回后与原输出一致。容差 1e-5：置换只改变注意力求和的顺序，float32 非结合性
    # 造成 ~3.6e-7 的差异。
    x = _rand((2, max_len, D), 52)
    perm = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])
    inv = torch.argsort(perm)
    with torch.no_grad():
        mu_p0, lv_p0 = model(x)
        mu_p1, lv_p1 = model(x[:, perm])
    assert torch.allclose(mu_p1[:, inv], mu_p0, atol=1e-5)
    assert torch.allclose(lv_p1[:, inv], lv_p0, atol=1e-5)


# --------------------------------------------------------------------------- #
# 20. 三个新选项与既有 final_norm / sigma_max / 零初始化头的组合
# --------------------------------------------------------------------------- #
def test_20_options_combined_with_final_norm_and_clamps():
    torch.manual_seed(0)  # 固定默认初始化，使本测试与文件内其它测试的执行顺序无关
    model = SGTModel(
        16,
        hidden=16,
        n_layers=1,
        n_heads=2,
        max_len=8,
        sigma_init=0.01,
        sigma_max=0.02,
        sigma_floor=0.005,
        mu_scale=1.0,
        pos_encoding="sinusoidal",
    )
    assert model.final_norm is not None  # 默认 final_norm=True
    assert model.pos_emb is None
    assert model.mu_scale == 1.0
    assert model.sigma_floor == 0.005
    assert model.log_sigma_floor2 == math.log(0.005**2)
    assert model.log_sigma_max2 == math.log(0.02**2)

    # 随机头权重 + 极端 sigma 头 bias：同时触发 mu_scale 与 sigma_floor。
    _fill_head_weights(model, seed=7, scale=0.2)
    with torch.no_grad():
        model.head.bias[model.d :].fill_(-50.0)
    x = _rand((2, 64, 16), 60)  # T=64 > max_len=8
    mask = torch.ones(2, 64, dtype=torch.bool)
    mask[0, 50:] = False
    with torch.no_grad():
        mu, log_sigma2 = model(x, attention_mask=mask)
        sigma = torch.exp(0.5 * log_sigma2)
    assert mu.shape == (2, 64, 16) and log_sigma2.shape == (2, 64, 16)
    assert mu.abs().max().item() < 1.0
    assert (sigma >= 0.005 - 1e-9).all() and (sigma <= 0.02 + 1e-9).all()
    assert torch.isfinite(mu).all() and torch.isfinite(log_sigma2).all()

    # 该 mu_scale 断言非平凡：同权重、mu_scale=None 时原始 |µ| 超过 1。
    raw = SGTModel(
        16, hidden=16, n_layers=1, n_heads=2, max_len=8, sigma_init=0.01,
        sigma_max=0.02, sigma_floor=0.005, pos_encoding="sinusoidal",
    )
    raw.load_state_dict(model.state_dict())
    with torch.no_grad():
        mu_raw, _ = raw(x, attention_mask=mask)
    assert mu_raw.abs().max().item() > 1.0

    # 上界与下界同时生效：bias 设大时 σ 停在 sigma_max（仍有限、仍 <1 的 µ）。
    with torch.no_grad():
        model.head.bias[model.d :].fill_(50.0)
        mu_high, lv_high = model(x, attention_mask=mask)
    sigma_high = torch.exp(0.5 * lv_high)
    assert (sigma_high <= 0.02 + 1e-9).all()
    assert torch.isfinite(mu_high).all() and torch.isfinite(lv_high).all()
    assert mu_high.abs().max().item() < 1.0
