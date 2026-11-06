# 外部证据核查（2026-11-06）

目的：确认论文是否有更新版本、官方实现/权重是否可得，以及社区复现是否与我们的发现一致。

## 1. 论文版本

- arXiv: **2506.09452v1 only**（Submitted 11 Jun 2025 06:56:12 UTC；540 KB）。arXiv abs 页的 Submission history 只有 `[v1]`，没有 v2/更新。
- Comments: “Submitted to IEEE S&P 2026”。未检索到公开的 camera-ready / 会议版本（IEEE S&P 2026 论文集尚未公开可查）。
- 结论：**没有更新版本的论文**；我们复现所依据的 v1 即最新公开版本。

## 2. 官方实现/权重

| 资源 | 状态 |
|---|---|
| PyPI `stainedglass-core` 0.0.1 | **空壳**：wheel 仅 1.4 KB（METADATA/WHEEL/top_level.txt，无代码），2023 年发布；要求 Python <3.12 |
| `protopia-ai/safeclaw`（GitHub） | 官方 SGT 集成示例（OpenClaw + SGT Proxy docker），**不含训练代码**；`verify_sgt_protection.py` 调用本地代理 `/stainedglass` |
| HF 模型 `Protopia/SGT-for-Qwen3-32B-swept-water-bfloat16` | 存在但**需申请授权**（safeclaw README 提到），公共检索不到 |
| 官方文档 | `docs.protopia.ai/engine/1.2.1/recommendations_for_training_llms`：**含训练配方**（见 F36） |

## 3. 第三方复现（katcinskiy/stained-glass-transform-pytorch）

非官方 PyTorch 实现，`train_paper/` 按论文实现、`train/` 为作者自用版本。其 `training_notes.md` 的关键结论与我们的实测一致：

1. **MI（Mahalanobis）项不稳定**：“produced extremely large values (on the order of ~50,000), and scaling α1 did not resolve the issue” —— 本仓库实测初始化梯度范数 7.3e4、MI 值 1.8e4（F21/我们的 α 平衡诊断）。
2. 厂商文档只提 `std_log_ratio_loss_weight`，暗示 **Mahalanobis 项可能已从产品中移除**（与 F36 一致）。
3. 他们最终**去掉 MI 与 MNP**，只用 KL + cosine（无私密性理论保证）。
4. 架构细节：`nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=2d)` × **1 层**、**无位置编码**、`mu_head`/`logvar_head` 两个 `nn.Linear(d,d)`、`logvar = clamp(logvar, -70, 0)`、`mu` 零初始化、`logvar_bias=-5`（σ≈0.082）。
5. 优化：lr 3e-4（paper 配置）/5e-5（其实验）、batch 8、grad accum 2、grad clipping、50 epochs；AdamW weight decay 0。
6. 数据：AG News 10k（其实验）、Qwen2.5-1.5B 为目标模型。

## 4. 对本复现的影响

- 我们的 A21（有界 σ、µ）、A22（bf16 LM + fp32 SGT）、sinusoidal/无位置编码、SymTTR 选模等选择，与厂商文档/第三方实现的方向一致，**不是随意偏离**。
- 论文的 Eq (8) 与厂商文档的损失**不是同一个东西**：这为“按论文逐行实现却复现不出 Table 1”提供了外部解释（F36）。
- 若要把 C42 从 BLOCKED 变成结论，下一步是**按厂商文档配方**（α=0.7、β=0.01、γ=1.0、w2=1.4、lr 3e-4、warmup+cosine、σ scale (1e-3,1)）训练一个**明确标注“非论文语义”**的对照变体，比较其效用-隐私曲线。
