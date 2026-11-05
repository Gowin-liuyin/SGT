# 复现状态（会话恢复用）

更新时间: 2026-11-05 19:05 (Asia/Shanghai)
论文: Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform (arXiv:2506.09452v1)
论文源: `2506.09452v1.pdf`（本地，文本抽取 `paper.txt`，不入库）
代码源: 无（`github.com/Gowin-liuyin/SGT` 为空仓库；本仓库为交付仓库）
当前阶段: Phase 0（骨架）
当前目标: 骨架门禁 M0 PASS
本轮门禁: M0 — 进行中
证据: `results/gate_m0_skeleton.json`

| 模块 | 依赖 | 状态 | 门禁证据 | 备注 |
|---|---|---|---|---|
| M0 骨架 | — | IN_PROGRESS | — | 文档 + 运行器 + 自检 |
| M1 理论核心 | M0 | PENDING | — | Eq(1)(2)(4)(5)(6)(7), Prop3.3, Cor3.4, PAC |
| M2 损失 | M1 | PENDING | — | Alg.1, Eq(8)–(13) |
| M3 模型 | M2 | PENDING | — | µθ/Σθ transformer |
| M4 指标与攻击 | M1,M3 | PENDING | — | rank/TTR/SymTTR/MI/PAC/BeamClean |
| M5 数据 | M0 | PENDING | — | OpenOrca/Alpaca/PUPA |
| M6 训练 | M2,M3,M5 | PENDING | — | 冻结 LM + SGT |
| M7 消融实验 | M6 | PENDING | — | Table 1 降规模、Fig 3/4 |
| M8 Utility/PII | M6,M7 | PENDING | — | lm-eval, PAPILLON |
| M9 大模型脚本 | M8 | PENDING | — | 本地 BLOCKED |
| M10 报告与回归 | M1..M9 | PENDING | — | claims 全部有结论 |

**环境已就绪**
- `.venv`（Python 3.12.13）：torch 2.15.0（MPS 可用）、transformers 5.22.1、datasets 5.1.0、numpy 2.5.4。
- `unsloth/Llama-3.2-1B-Instruct` 正在下载到 `~/.cache/huggingface`（meta-llama 官方仓库 403 gated，见 D1）。
- 网络：HF/PyPI/GitHub 可达（SOCKS 代理，`httpx[socks]` 已装）。

**下一动作**
1. 跑 `gates/gate_m0_skeleton.py`，M0 PASS 后 commit。
2. 派工 M1（`src/sgt/sampling.py` + `src/sgt/theory.py` + tests），验收后 commit。
3. 并行派工 M5（`src/sgt/data.py`）。

**阻塞项**
- 无（硬件限制已由用户决策：本地降规模 + 如实声明，D2–D4）。
