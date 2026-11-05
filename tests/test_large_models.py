"""M9 大模型脚本（``experiments/run_large_models.py`` + ``configs/large_models.yaml``）的验证。

覆盖任务单 M9 的验收判据：

1. yaml 三个模型 key 与 repo_id 逐字正确（Table 2），loss 组合为 MI+AbsCos+Norm；
2. ``--model unknown`` → 非零退出 + stderr 列出可选值；
3. ``--dry-run --dry-run-model .models/tiny-llama`` → exit 0，JSON ``mode="dry_run"``、
   ``history`` 长度 1、loss 有限（tiny 不存在时用 transformers 现场构造到 tmp）；
4. 非 dry-run 且无 ≥8 GPU → exit 3、JSON ``status="blocked"``、``llm_loaded=False``
   （没有尝试加载 70B）；
5. ``launch_command`` 含 ``torchrun`` / ``--nnodes`` / ``--nproc-per-node`` / ``--fsdp2``。

全部用 subprocess 跑真实 CLI；不联网、不加载 70B、不读 data_cache。
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "run_large_models.py"
CONFIG = REPO_ROOT / "configs" / "large_models.yaml"
TINY_MODEL = REPO_ROOT / ".models" / "tiny-llama"

#: 论文 Table 2（p.10）的三个目标模型，repo_id 逐字转录。
TABLE2_MODELS = {
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "deepseek-r1-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

REQUIRED_KEYS = (
    "model_key",
    "repo_id",
    "mode",
    "status",
    "launch_command",
    "world_size",
    "config_echo",
    "timestamp",
)


def run_cli(*args: str, timeout: float = 900.0) -> subprocess.CompletedProcess:
    """以仓库为 cwd 跑真实 CLI（与交付自测命令同口径）。"""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def gpu_cluster_available() -> bool:
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return count >= 8 and shutil.which("torchrun") is not None


def read_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Table 2 配置：三个模型 key 与 repo_id 逐字一致
# --------------------------------------------------------------------------- #
def test_config_table2_models_and_loss_combo():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    models = raw["models"]
    assert set(models) == set(TABLE2_MODELS)
    for key, repo_id in TABLE2_MODELS.items():
        assert models[key]["repo_id"] == repo_id

    # hidden_size：Llama-70B / DeepSeek-R1-Distill-Llama-70B = 8192，Qwen3-32B = 5120。
    assert models["llama-3.3-70b"]["d"] == 8192
    assert models["deepseek-r1-70b"]["d"] == 8192
    assert models["qwen3-32b"]["d"] == 5120

    train = raw["train"]
    assert train["epochs"] == 2
    assert train["dataset"] == "Open-Orca/OpenOrca"
    assert train["frac"] == 1.0
    assert train["validation_fraction"] == 0.005
    assert train["loss_terms"] == ["mi", "acs", "mnp"]  # MI+AbsCos+Norm
    assert train["alphas"] == {"mi": 4.8828125e-4, "acs": 1.0, "mnp": 1.0}

    parallel = raw["parallel"]
    assert parallel == {"fsdp2": True, "tp_size": 8, "nodes": 64, "gpus_per_node": 8}


# --------------------------------------------------------------------------- #
# 2. 未知 --model：非零退出 + stderr 列出可选值
# --------------------------------------------------------------------------- #
def test_unknown_model_lists_choices(tmp_path):
    out = tmp_path / "unknown.json"
    proc = run_cli(
        "--config", str(CONFIG), "--model", "no-such-model", "--out", str(out)
    )
    assert proc.returncode != 0
    assert "no-such-model" in proc.stderr
    assert "可选值" in proc.stderr
    for key in TABLE2_MODELS:
        assert key in proc.stderr
    assert not out.exists()


# --------------------------------------------------------------------------- #
# 3. dry-run：tiny 模型真实跑 1 步 train_sgt
# --------------------------------------------------------------------------- #
def _make_tiny_model(tmp_path: Path) -> Path:
    """仓库没有 .models/tiny-llama 时，用 transformers 现场构造一个小 Llama 到 tmp。"""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=512, special_tokens=["[UNK]", "[PAD]", "[EOS]", "[BOS]"]
    )
    tokenizer.train_from_iterator(
        [
            "the stained glass transform obfuscates embeddings before they reach the model",
            "privacy is measured with nearest neighbour reconstruction failure rates",
            "mutual information between clean and obfuscated embeddings stays bounded",
        ],
        trainer,
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
    )
    model_dir = tmp_path / "tiny-llama"
    fast.save_pretrained(str(model_dir))
    config = LlamaConfig(
        vocab_size=fast.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    LlamaForCausalLM(config).save_pretrained(str(model_dir))
    return model_dir


def test_dry_run_tiny_model(tmp_path):
    if (TINY_MODEL / "config.json").is_file():
        model_dir = TINY_MODEL
    else:  # pragma: no cover - 本仓库存在 .models/tiny-llama
        model_dir = _make_tiny_model(tmp_path)
    out = tmp_path / "dry_run.json"
    proc = run_cli(
        "--config", str(CONFIG),
        "--model", "llama-3.3-70b",
        "--dry-run",
        "--dry-run-model", str(model_dir),
        "--out", str(out),
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    payload = read_payload(out)

    assert payload["mode"] == "dry_run"
    assert payload["status"] == "ok"
    assert payload["model_key"] == "llama-3.3-70b"
    assert payload["repo_id"] == TABLE2_MODELS["llama-3.3-70b"]
    assert payload["world_size"] == 64 * 8
    assert payload["config_echo"]["model"]["d"] == 8192  # 回显的是 Table 2 配置，不是 tiny
    assert payload["d_model"] == 64  # 实际加载的是 tiny 模型
    assert payload["data_source"] == "synthetic"
    assert payload["loss_terms"] == ["mi", "acs", "mnp"]
    assert payload["batch_size"] == 1
    assert payload["max_len"] == 16

    history = payload["history"]
    assert len(history) == 1
    assert history[0]["step"] == 0
    for name in ("loss", "loss_u", "mi", "acs", "mnp"):
        assert name in history[0]
    assert history[0]["loss"] is not None
    assert math.isfinite(history[0]["loss"])
    assert math.isfinite(history[0]["loss_u"])
    assert payload["elapsed_seconds"] >= 0.0
    for field in REQUIRED_KEYS:
        assert field in payload


# --------------------------------------------------------------------------- #
# 4. 无多卡集群：exit 3 + status=blocked，且不加载 70B
# --------------------------------------------------------------------------- #
def test_blocked_without_multi_gpu_cluster(tmp_path):
    if gpu_cluster_available():
        pytest.skip("本机有 ≥8 GPU + torchrun，会走 ready 分支（集群上另测）")
    out = tmp_path / "blocked.json"
    proc = run_cli("--config", str(CONFIG), "--model", "llama-3.3-70b", "--out", str(out))
    assert proc.returncode == 3, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    payload = read_payload(out)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "no multi-GPU cluster"
    assert payload["mode"] == "launch"
    assert payload["model_key"] == "llama-3.3-70b"
    assert payload["repo_id"] == TABLE2_MODELS["llama-3.3-70b"]
    assert payload["world_size"] == 512
    # 没有尝试加载 70B：脚本在加载前就退出。
    assert payload["llm_loaded"] is False
    assert payload["env"]["cuda_device_count"] < 8
    for field in REQUIRED_KEYS:
        assert field in payload
    # 未降级到单卡训练：JSON 不含任何训练历史。
    assert "history" not in payload


# --------------------------------------------------------------------------- #
# 5. launch_command 含分布式启动要素
# --------------------------------------------------------------------------- #
def test_launch_command_contains_distributed_flags(tmp_path):
    out = tmp_path / "launch.json"
    proc = run_cli(
        "--config", str(CONFIG),
        "--model", "qwen3-32b",
        "--nodes", "32",
        "--gpus-per-node", "8",
        "--out", str(out),
    )
    # 无集群时 exit 3；有集群时 exit 0。两种情况下都只生成命令、不执行。
    assert proc.returncode in (0, 3), f"stdout={proc.stdout}\nstderr={proc.stderr}"
    payload = read_payload(out)
    command = payload["launch_command"]

    assert "torchrun" in command
    assert "--nnodes=32" in command
    assert "--nproc-per-node=8" in command
    assert "--fsdp2" in command
    assert "--tp-size 8" in command
    assert "--worker" in command
    assert "--model qwen3-32b" in command
    assert "experiments/run_large_models.py" in command
    assert payload["world_size"] == 32 * 8
    assert payload["config_echo"]["resolved"]["dp_size"] == 32
    assert payload["config_echo"]["resolved"]["tp_size"] == 8
