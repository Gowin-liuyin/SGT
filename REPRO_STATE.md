# 复现状态（会话恢复用）

更新时间: 2026-11-05 21:30 (Asia/Shanghai)
论文: Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform (arXiv:2506.09452v1)
论文源: `2506.09452v1.pdf`；代码源: 无官方实现（原仓库为空），本仓库即复现实现
当前阶段: Phase R（报告完成，门禁回归）
当前目标: M10 全绿 + commit/push

| 模块 | 依赖 | 状态 | 门禁证据 | 备注 |
|---|---|---|---|---|
| M0 骨架 | — | PASS | `results/gate_m0_skeleton.json` | 运行器 self-test（fail-fast + pass:false） |
| M1 理论核心 | M0 | PASS | `results/gate_m1_theory.json` | Eq(1)/Prop3.3/Cor3.4/Eq5-7；PAC 全列复现 |
| M2 损失 | M1 | PASS | `results/gate_m2_losses.json` | Alg.1、Eq(8) pair/full、Eq(9)-(13) |
| M3 模型 | M1 | PASS | `results/gate_m3_model.json` | 双向 Transformer + 零初始化头 |
| M4 指标与攻击 | M1 | PASS | `results/gate_m4_metrics.json` | 秩/TTR/SymTTR/BeamClean |
| M5 数据 | M0 | PASS | `results/gate_m5_data.json` | OpenOrca 4000 条缓存、PUPA=user_query |
| M6 训练 | M2,M3,M4,M5 | PASS | `results/gate_m6_train.json` | LM 逐位冻结、可复现 |
| M7 消融实验 | M6 | PASS | `results/gate_m7_ablation.json` | 5 变体 + comp_entropy 发散证据（F31） |
| M8a Utility | M6,M7 | PASS | `results/gate_m8a_utility.json` | lm-eval 注入 + Utility 聚合 |
| M8b PII | M6,M7 | PASS | `results/gate_m8b_pii.json` | PUPA NN + PII-Ratio |
| M9 大模型脚本 | M8 | PASS | `results/gate_m9_large.json` | 本机执行 BLOCKED（D4） |
| M10 报告与回归 | M1..M9 | PASS | `results/gate_m10_report.json` | 35 条 claim 全部有结论 |

**关键结果**
- PAC-Adv 反解逐位复现 Table 1/2（worst 0.11pp）：`results/gate_m1_theory.json`。
- Table 1 降规模：5 变体见 `results/ablation_*.json` + `results/ablation_summary.json`；Utility 见 `results/utility_*.json`。
- Fig.3：`results/fig3_ranksh.json`；Fig.4：`results/constnoise_low.json` + `results/constnoise_util_*.json`。
- Table 3（PUPA）：`results/pupa_*.json`（NN only）。
- 论文缺陷：F29（Eq 6 假上界）、F30（MI 非 MI）、F31（Eq 12 发散）、F32（降规模 Utility 未复现）。

**下一动作**
1. 跑 `./repro.sh gates` 全量回归（M0→M10），确认无回退。
2. `git commit && git push`（每节点已提交，见 git log）。
3. 若要继续提高复现度：按论文级预算重训（更多步数/数据）→ Table 1 的 Utility 列（F32）；有 A100 集群时跑 M9 脚本。

**阻塞项**
- 70B 实验：无多卡集群（D4，用户已确认本地降规模策略）。
- 论文级训练预算：本机 16GB/无 CUDA（D2/D3）。
