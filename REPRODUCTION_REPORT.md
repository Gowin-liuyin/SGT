# SGT 复现报告（骨架，随模块推进填写）

论文：Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform（arXiv:2506.09452v1）

## 0. 结论摘要

（待填：公式级验证结论、可复现声明、BLOCKED 范围）

## 1. 复现范围

- 做了什么：见 `docs/MODULE_PLAN.md`；每模块的门禁证据在 `results/`。
- 没做什么：70B 级训练（D4）、论文规模消融（D2）、官方外部实现（D5）。

## 2. 环境与硬件差异

| 项 | 论文 | 本复现 |
|---|---|---|
| 硬件 | 1–64 节点 ×8×A100 80GB | Apple M5 / 16GB / MPS |
| 精度 | bf16 | 待记录 |
| 模型来源 | meta-llama gated | unsloth 镜像（D1） |

## 3. 方法复现（逐模块）

| 模块 | 内容 | 门禁 | 证据 |
|---|---|---|---|
| M0 | 骨架 | | |
| M1 | 理论核心 | | |
| … | | | |

## 4. 实验数值对照

（Table 1/2/3 论文值 vs 复现值 vs 偏差原因；图 3/4）

## 5. 论文问题清单

见 `docs/FINDINGS.md`（F0..F28）。

## 6. 歧义与实例化决策

见 `SPEC.md` §7（A1..A17）。

## 7. 复现结论统计

| 判定 | 数量 | 说明 |
|---|---|---|
| VERIFIED | | |
| FALSIFIED | | |
| BLOCKED | | |

## 8. 复跑方式

```bash
./repro.sh setup      # venv + 依赖
./repro.sh gates      # 全部门禁，fail-fast
./repro.sh ablation   # Table 1 降规模
./repro.sh utility    # lm-eval
```

## 9. 局限

（待填）
