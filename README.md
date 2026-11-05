# SGT 论文复现（arXiv:2506.09452）

本仓库是论文 **Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform**
（Roberts, Mylonakis, Roy, Kale；Protopia AI；arXiv:2506.09452v1）的独立复现。

- 论文是唯一权威：`2506.09452v1.pdf`（本地）。
- 官方代码不可得（原仓库为空），实现全部由 `SPEC.md` 推导。
- 每个模块有机器可读门禁，证据落在 `results/gate_*.json`；进度见 `REPRO_STATE.md`。

## 目录

| 路径 | 作用 |
|---|---|
| `SPEC.md` | 实例化规范：公式、接口、歧义决策 A1..A17、偏差 D1..D6 |
| `docs/PAPER_ANALYSIS.md` | 论文逐行对照（原文 → 应然实现 → 验收判据） |
| `docs/MODULE_PLAN.md` | 模块 M0..M10 与门禁判据 |
| `docs/CLAIMS_CHECKLIST.md` | 论文声明核对表 C1..C34 |
| `docs/FINDINGS.md` | 论文不一致/缺失清单 F0..F28 |
| `src/sgt/` | 实现（sampling / model / losses / metrics / attacks / data / train） |
| `experiments/` | 实验入口 |
| `gates/` | 门禁脚本与 fail-fast 运行器 |
| `results/` | 门禁与实验结果 JSON（入库） |
| `REPRODUCTION_REPORT.md` | 最终报告 |

## 快速开始

```bash
./repro.sh setup            # 创建 .venv 并安装依赖
./repro.sh gates            # 按 M0→M10 跑全部门禁（第一个失败即停）
./repro.sh gate m1          # 只跑某模块
./repro.sh ablation         # Table 1 降规模消融
./repro.sh constant-noise   # Fig. 4
./repro.sh utility          # lm-eval Utility
./repro.sh report           # 生成对照表
```

## 复现范围（重要）

论文实验从 1×A100 6 小时（Llama-3.2-1B 消融）到 64 节点×8×A100（70B 模型）。本复现在
Apple M5 / 16GB / MPS 上运行：

- **公式、算法、指标、攻击、协议**按论文逐条实现并做数值级验证（M1–M4）。
- **Llama-3.2-1B 实验**以缩小规模复跑，量化偏差并如实声明（M6–M8）。
- **70B 级实验**只提供可复跑脚本，判定为 BLOCKED（M9）。

完整声明见 `SPEC.md` §0 的 D1–D6 与 `REPRODUCTION_REPORT.md` §1。

## 环境

Python 3.12、torch 2.15（MPS）、transformers 5.22、datasets 5.1、numpy 2.5。
模型权重使用 `unsloth/Llama-3.2-1B-Instruct`（官方 `meta-llama` 仓库为 gated，403；权重同源）。
