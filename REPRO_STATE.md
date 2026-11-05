# 复现状态（会话恢复用）

更新时间: 2026-11-05 19:35 (Asia/Shanghai)
论文: Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform (arXiv:2506.09452v1)
论文源: `2506.09452v1.pdf`（本地，文本抽取 `paper.txt`，不入库）
代码源: 无（`github.com/Gowin-liuyin/SGT` 为空仓库；本仓库为交付仓库）
当前阶段: Phase 1（M7 消融 + M8 Utility/PII）
当前目标: M3 模型、M4 指标、M5 数据
本轮门禁: M0/M1/M2 PASS
证据: `results/gate_m0_skeleton.json`、`gate_m1_theory.json`、`gate_m2_losses.json`

| 模块 | 依赖 | 状态 | 门禁证据 | 备注 |
|---|---|---|---|---|
| M0 骨架 | — | PASS | `results/gate_m0_skeleton.json` | 运行器 self-test 通过 |
| M1 理论核心 | M0 | PASS | `results/gate_m1_theory.json` | PAC 复现 Table1/2；Eq(6) 反例（F29） |
| M2 损失 | M1 | PASS | `results/gate_m2_losses.json` | Eq8 pair/full 双实现；L_U=CE 非 KL |
| M3 模型 | M1 | PASS | `results/gate_m3_model.json` | 双向 Transformer + 零初始化头 |
| M4 指标与攻击 | M1 | PASS | `results/gate_m4_metrics.json` | rank/TTR/SymTTR/BeamClean |
| M5 数据 | M0 | PASS | `results/gate_m5_data.json` | 4000 OpenOrca + 1000 Alpaca 缓存；PUPA=user_query |
| M6 训练 | M2,M3,M4,M5 | PASS | `results/gate_m6_train.json` | 8 tests；LM 逐位冻结 |
| M7 消融实验 | M6 | PENDING | — | Table 1 降规模、Fig 3/4 |
| M8 Utility/PII | M6,M7 | IN_PROGRESS | gate_m8a 6/7 | lm-eval 注入跑通；PII 修复中 |
| M9 大模型脚本 | M8 | PENDING | — | 本地 BLOCKED |
| M10 报告与回归 | M1..M9 | PENDING | — | claims 全部有结论 |

**环境已就绪**
- `.venv`（Python 3.12.13）：torch 2.15.0（MPS 可用）、transformers 5.22.1、datasets 5.1.0、numpy 2.5.4。
- `unsloth/Llama-3.2-1B-Instruct` 正在下载到 `~/.cache/huggingface`（meta-llama 官方仓库 403 gated，见 D1）。
- 网络：HF/PyPI/GitHub 可达（SOCKS 代理，`httpx[socks]` 已装）。

**下一动作**
1. 等 M2（losses）与 M5（data）交付 → 跑 `gates/gate_m2_losses.py` / `gates/gate_m5_data.py` → commit。
2. 派工 M3（SGT 模型）与 M4（指标/攻击）。
3. Llama-3.2-1B 权重并行下载中（`.models/llama-3.2-1b-instruct/`，8 线程 range 下载，约 26%）。

**阻塞项**
- 无（硬件限制已由用户决策：本地降规模 + 如实声明，D2–D4）。
