"""M3 `SGTModel` 的验证：API/形状、初始化、确定性、序列与位置依赖、padding 不变性、
梯度、序列化往返、Eq (1) 采样、`sigma_min` 下限（M3 任务单 1–10）。

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
