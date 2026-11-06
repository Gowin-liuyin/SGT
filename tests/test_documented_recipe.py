"""文档配方对照实验（``experiments/run_documented_recipe.py``）的验证。

**不加载 1B 模型、不联网、不读 parquet**：全部用合成小张量 / 微型 d=8 SGT。

覆盖任务单 1–4：

1. 总损失组合：``(1-α)(w1·L_M1 + w2·L_M2) + α(β·L_N + γ·L_S)``，含 ``w1=0`` 时
   ``L_M1`` 不改变总损失；
2. ``L_N``（std-log-ratio）：对两个 batch 的**反对称性**（交换变号）、同 batch 为 0、
   有限性、mask 生效；
3. warmup+cosine 计划：``lr_multiplier(0)=0``、warmup 末步 ``=1``、
   最后一步/``total`` ``=lr_min_factor``；
4. checkpoint 含 ``config["sgt"]``（全部构造参数）且能被
   ``sgt.utility.build_sgt_from_state`` 重建、权重逐位一致。

额外（不依赖模型）：``configs/documented_recipe.yaml`` 的配方值与任务单一致。

说明：``L_N`` 按任务单公式是**有符号**量（论文 Eq (8) 第一项的符号），因此第 2 条
验证的是「交换两个 batch 变号」而不是「非负距离」。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_documented_recipe as R  # noqa: E402
from sgt.model import SGTModel  # noqa: E402
from sgt.utility import build_sgt_from_state, sgt_config_from_payload  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "documented_recipe.yaml"


# --------------------------------------------------------------------------- #
# 1. 总损失组合
# --------------------------------------------------------------------------- #
def test_recipe_total_loss_matches_documented_formula():
    """手算：``(1-α)(w1·L_M1+w2·L_M2)+α(β·L_N+γ·L_S)``（w1=0，L_M1 不实现）。"""
    loss_m2 = torch.tensor(0.25, dtype=torch.float64)
    loss_n = torch.tensor(-0.5, dtype=torch.float64)
    loss_s = torch.tensor(0.75, dtype=torch.float64)
    alpha, beta, gamma, w1, w2 = 0.7, 0.01, 1.0, 0.0, 1.4

    total = R.recipe_total_loss(
        loss_m2, loss_n, loss_s, alpha=alpha, beta=beta, gamma=gamma, w1=w1, w2=w2
    )
    expected = (1.0 - alpha) * (w1 * 0.0 + w2 * loss_m2) + alpha * (
        beta * loss_n + gamma * loss_s
    )
    assert total.dtype == torch.float64
    assert torch.allclose(total, expected, rtol=1e-12, atol=1e-12)
    # 位置参数调用等价（便于下游独立重算时不受 keyword-only 限制）。
    positional = R.recipe_total_loss(loss_m2, loss_n, loss_s, alpha, beta, gamma, w1, w2)
    assert torch.equal(positional, total)
    # 数值手算（独立于上面的表达式，防止两边同错）：
    # 0.3*1.4*0.25 = 0.105；0.7*(0.01*(-0.5)+0.75) = 0.7*0.745 = 0.5215
    assert abs(float(total) - 0.6265) < 1e-12


def test_recipe_total_loss_w1_zero_ignores_loss_m1():
    """``w1=0`` 时传入任意 ``L_M1`` 不改变总损失；``w1>0`` 时按公式线性出现。"""
    m2 = torch.tensor(0.4, dtype=torch.float64)
    ln = torch.tensor(0.2, dtype=torch.float64)
    ls = torch.tensor(0.6, dtype=torch.float64)
    kwargs = dict(alpha=0.7, beta=0.01, gamma=1.0)

    without = R.recipe_total_loss(m2, ln, ls, w1=0.0, w2=1.4, **kwargs)
    with_m1 = R.recipe_total_loss(m2, ln, ls, w1=0.0, w2=1.4, loss_m1=2.0, **kwargs)
    assert torch.allclose(without, with_m1, rtol=0.0, atol=0.0)

    with_w1 = R.recipe_total_loss(m2, ln, ls, w1=1.0, w2=1.4, loss_m1=2.0, **kwargs)
    expected = (1.0 - 0.7) * (1.0 * 2.0 + 1.4 * m2) + 0.7 * (0.01 * ln + 1.0 * ls)
    assert torch.allclose(with_w1, expected, rtol=1e-12, atol=1e-12)


def test_recipe_total_loss_gradients_flow():
    """总损失对 ``L_M2``/``L_N``/``L_S`` 的梯度即文档系数（不经过任何 detach）。"""
    values = [
        torch.tensor(0.3, dtype=torch.float64, requires_grad=True) for _ in range(3)
    ]
    total = R.recipe_total_loss(
        *values, alpha=0.7, beta=0.01, gamma=1.0, w1=0.0, w2=1.4
    )
    total.backward()
    assert values[0].grad.item() == pytest.approx((1.0 - 0.7) * 1.4, rel=0, abs=1e-12)
    assert values[1].grad.item() == pytest.approx(0.7 * 0.01, rel=0, abs=1e-12)
    assert values[2].grad.item() == pytest.approx(0.7 * 1.0, rel=0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 2. L_N（std-log-ratio）对称性 / 有限性
# --------------------------------------------------------------------------- #
def test_std_log_ratio_is_antisymmetric_and_finite():
    """``L_N(b,a) = -L_N(a,b)``；两批相同 → 0；数值 = 逐特征均值的差。"""
    generator = torch.Generator().manual_seed(11)
    a = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64)
    b = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64)

    loss_ab = R.std_log_ratio_loss(a, b)
    loss_ba = R.std_log_ratio_loss(b, a)
    assert torch.isfinite(loss_ab)
    assert float(loss_ab) == pytest.approx(-float(loss_ba), rel=0, abs=1e-12)

    # 逐特征均值之差（独立手算；损失内部按 fp32 计算，这里同样先转 fp32）。
    expected = (b.float().mean(dim=(0, 1)) - a.float().mean(dim=(0, 1))).mean()
    assert float(loss_ab) == pytest.approx(float(expected), rel=0, abs=1e-12)

    assert float(R.std_log_ratio_loss(a, a)) == pytest.approx(0.0, abs=1e-12)


def test_std_log_ratio_mask_and_shape_guards():
    """mask 只统计有效 token；d 不一致时报错。"""
    a = torch.tensor([[[0.0, 2.0], [10.0, 10.0]]], dtype=torch.float64)  # [1,2,2]
    b = torch.tensor([[[1.0, 4.0], [99.0, 99.0]]], dtype=torch.float64)
    mask = torch.tensor([[True, False]])

    # mask 后 a 的逐特征均值 = [0,2]，b = [1,4]：差 = [1,2]，均值 = 1.5。
    loss = R.std_log_ratio_loss(a, b, mask_a=mask, mask_b=mask)
    assert float(loss) == pytest.approx(1.5, rel=0, abs=1e-12)

    # 不 mask 时被 padding token 拉偏：a 均值 = [5,6]，b = [50,51.5]，差均值 = 45.25。
    assert float(R.std_log_ratio_loss(a, b)) == pytest.approx(45.25, rel=0, abs=1e-12)

    with pytest.raises(ValueError, match="d 必须一致"):
        R.std_log_ratio_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))
    with pytest.raises(ValueError, match="mask 未选中"):
        R.std_log_ratio_loss(a, b, mask_a=torch.zeros(1, 2, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# 3. warmup + cosine 学习率计划
# --------------------------------------------------------------------------- #
def test_lr_multiplier_warmup_cosine():
    """0 步为 0、5% 处为 1、最后一步与 ``total`` 处为 ``lr_min_factor``。"""
    total, warmup_fraction, min_factor = 100, 0.05, 0.1
    assert R.lr_multiplier(0, total, warmup_fraction, min_factor) == 0.0
    assert R.lr_multiplier(5, total, warmup_fraction, min_factor) == pytest.approx(
        1.0, rel=0, abs=1e-12
    )
    assert R.lr_multiplier(total - 1, total, warmup_fraction, min_factor) == pytest.approx(
        min_factor, rel=0, abs=1e-12
    )
    assert R.lr_multiplier(total, total, warmup_fraction, min_factor) == pytest.approx(
        min_factor, rel=0, abs=1e-12
    )
    # warmup 内线性递增；warmup 后单调不增，且始终落在 [min_factor, 1]。
    values = [R.lr_multiplier(step, total, warmup_fraction, min_factor) for step in range(total)]
    assert all(
        values[i] <= values[i + 1] + 1e-12 for i in range(5)
    )
    assert all(values[i] >= values[i + 1] - 1e-12 for i in range(5, total - 1))
    assert min(values[5:]) == pytest.approx(min_factor, rel=0, abs=1e-12)
    assert max(values) == pytest.approx(1.0, rel=0, abs=1e-12)

    # 正式 800 步：warmup = 40 步。
    assert R.lr_multiplier(40, 800, 0.05, 0.1) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert R.lr_multiplier(799, 800, 0.05, 0.1) == pytest.approx(0.1, rel=0, abs=1e-12)


def test_lr_multiplier_rejects_bad_inputs():
    with pytest.raises(ValueError, match="total"):
        R.lr_multiplier(0, 0)
    with pytest.raises(ValueError, match="warmup_fraction"):
        R.lr_multiplier(0, 10, warmup_fraction=1.0)
    with pytest.raises(ValueError, match="min_factor"):
        R.lr_multiplier(0, 10, min_factor=1.5)


# --------------------------------------------------------------------------- #
# 4. checkpoint：config["sgt"] + build_sgt_from_state 重建
# --------------------------------------------------------------------------- #
def _micro_sgt_config() -> dict:
    return R.build_sgt_config(
        {
            "d": 8,
            "hidden": 8,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_mult": 2,
            "max_len": 16,
            "sigma_init": 0.01,
            "sigma_min": 1e-3,
            "sigma_max": 1.0,
            "pos_encoding": "sinusoidal",
            "final_norm": True,
            "mu_scale": None,
        }
    )


def test_checkpoint_roundtrip_via_build_sgt_from_state(tmp_path: Path):
    """checkpoint 的 ``config["sgt"]`` 含全部构造参数，可重建且权重逐位一致。"""
    torch.manual_seed(0)
    sgt_config = _micro_sgt_config()
    sgt = R.build_sgt(sgt_config)
    with torch.no_grad():  # 让权重非零，确认重建的是同一组参数而不是初始值
        for parameter in sgt.parameters():
            parameter.add_(0.01 * torch.randn(parameter.shape))

    recipe = {"alpha": 0.7, "beta": 0.01, "gamma": 1.0, "w1": 0.0, "w2": 1.4}
    train = {
        "lr": 3.0e-4,
        "betas": [0.9, 0.95],
        "eps": 1.0e-5,
        "weight_decay": 0.0,
        "steps": 3,
        "batch_size": 1,
        "max_len": 16,
        "seed": 0,
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "lr_min_factor": 0.1,
        "lm_dtype": "bfloat16",
    }
    payload = R.build_checkpoint_payload(
        sgt, sgt_config=sgt_config, recipe=recipe, train=train, seed=0
    )

    assert payload["variant"] == R.VARIANT == "documented_recipe"
    assert payload["seed"] == 0
    assert payload["config"]["sgt"] == sgt_config
    assert payload["config"]["recipe"]["alpha"] == 0.7
    assert payload["config"]["recipe"]["w2"] == 1.4
    assert payload["config"]["recipe"]["lr"] == 3.0e-4
    assert payload["config"]["recipe"]["steps"] == 3
    assert payload["config"]["paper_semantics"] is False

    ckpt = tmp_path / "ckpt.pt"
    torch.save(payload, ckpt)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)

    loaded_config = sgt_config_from_payload(loaded)
    assert loaded_config == sgt_config
    # build_sgt_from_state 会交叉校验 config 与 state_dict 形状，并把 n_heads 从 config 取。
    rebuilt, meta = build_sgt_from_state(loaded["sgt"], loaded_config)
    assert rebuilt.d == 8 and rebuilt.hidden == 8 and rebuilt.n_heads == 2
    assert meta["n_heads_source"] == "config"
    assert meta["pos_encoding"] == "sinusoidal"
    assert meta["option_sources"]["sigma_max"] == "config"
    assert rebuilt.sigma_max == pytest.approx(1.0)
    assert rebuilt.sigma_min == pytest.approx(1e-3)
    for key, value in sgt.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], value), key


def test_checkpoint_config_contains_every_constructor_argument():
    """``config["sgt"]`` 覆盖 ``SGTModel.__init__`` 的每个参数（含默认值补全）。"""
    import inspect

    sgt_config = _micro_sgt_config()
    expected = {
        name
        for name in inspect.signature(SGTModel.__init__).parameters
        if name != "self"
    }
    assert expected <= set(sgt_config)
    assert sgt_config["sigma_floor"] is None
    assert sgt_config["dropout"] == 0.0


# --------------------------------------------------------------------------- #
# 5. 配置文件 + 数据批（不加载模型）
# --------------------------------------------------------------------------- #
def test_documented_recipe_yaml_matches_task_values():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    run = R.resolve_run_config(raw)

    assert run["recipe"] == {
        "alpha": 0.7,
        "beta": 0.01,
        "gamma": 1.0,
        "w1": 0.0,
        "w2": 1.4,
    }
    assert run["train"]["lr"] == 3.0e-4
    assert run["train"]["betas"] == [0.9, 0.95]
    assert run["train"]["eps"] == 1.0e-5
    assert run["train"]["weight_decay"] == 0.0
    assert run["train"]["steps"] == 800
    assert run["train"]["batch_size"] == 1
    assert run["train"]["max_len"] == 256
    assert run["train"]["seed"] == 0
    assert run["train"]["grad_clip"] == 1.0
    assert run["train"]["warmup_fraction"] == 0.05
    assert run["train"]["lr_min_factor"] == 0.1
    assert run["train"]["lm_dtype"] == "bfloat16"
    assert run["data"]["max_examples"] == 20000
    assert run["sgt"]["d"] == 2048
    assert run["sgt"]["hidden"] == 2048
    assert run["sgt"]["n_layers"] == 2
    assert run["sgt"]["n_heads"] == 8
    assert run["sgt"]["ffn_mult"] == 4
    assert run["sgt"]["max_len"] == 256
    assert run["sgt"]["sigma_init"] == 0.01
    assert run["sgt"]["sigma_min"] == 1e-3
    assert run["sgt"]["sigma_max"] == 1.0
    assert run["sgt"]["pos_encoding"] == "sinusoidal"
    assert run["sgt"]["final_norm"] is True
    assert run["sgt"]["mu_scale"] is None


def test_step_batches_are_deterministic_and_disjoint():
    """每步的主/辅 batch 按固定顺序取、互不重叠；同参数两次迭代逐位一致。"""
    sequences = [[i] for i in range(10)]
    first = list(R.step_batches(sequences, steps=4, batch_size=2))
    second = list(R.step_batches(sequences, steps=4, batch_size=2))
    assert first == second
    assert first[0] == ([[0], [1]], [[2], [3]])
    assert first[1] == ([[4], [5]], [[6], [7]])
    # 越界后回到开头（循环取数）。
    assert first[2] == ([[8], [9]], [[0], [1]])

    ids, mask = R.collate([[1, 2, 3], [4]], pad_id=0)
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_format_example_uses_three_fields_with_newlines():
    text = R.format_example(
        {"system_prompt": "SYS", "question": "Q", "response": "R", "id": "x"}
    )
    assert text == "SYS\nQ\nR"
    # 空 system_prompt 保留分隔符（任务单要求逐字拼接三字段）。
    assert R.format_example({"system_prompt": "", "question": "Q", "response": "R"}) == "\nQ\nR"


def test_resolve_run_config_cli_overrides():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    resolved = R.resolve_run_config(
        copy.deepcopy(raw), steps=20, max_examples=200, model_path="/tmp/tiny"
    )
    assert resolved["train"]["steps"] == 20
    assert resolved["data"]["max_examples"] == 200
    assert resolved["model"] == "/tmp/tiny"
    assert resolved["sgt"]["d"] == 2048
