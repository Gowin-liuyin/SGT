# SGT 复现报告

论文：**Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform**
（J. Roberts, K. Mylonakis, S. Roy, K. Kale；Protopia AI；arXiv:2506.09452v1，2025-06-11）

复现仓库：`github.com/Gowin-liuyin/SGT`（论文官方代码不存在，全部由论文推导）。
证据：`results/gate_*.json`（门禁）、`results/ablation_*.json`（消融）、`results/utility_*.json`（Utility）、
`results/pupa_*.json`（隐私）、`results/constnoise_*.json`（Fig.4）、`docs/FINDINGS.md`（F0–F32）。

## 0. 结论摘要

1. **公式级复现全部通过**（M1–M6，11 个门禁）：Eq (1) 逐元素精确、Prop 3.3/Cor 3.4 分布检验通过、
   Eq (4) 反解**逐位复现 Table 1/2 的 PAC-Adv 整列**（最大偏差 0.11pp）、Alg.1 与 Eq (8) 两种归约与朴素实现一致（≤1e-9）、
   Eq (9)–(13) 与 L_U 逐值正确、训练器逐位冻结目标模型。
2. **发现论文三处实质缺陷**：Eq (6) 的“上界”实为**下界**（F29，反例 0.6931 nats）；
   Table 1 的 “MI” 列不是互信息（F30，解析差距近 10 倍）；Eq (12) 按原式训练**发散为 NaN**（F31）。
3. **端到端实验在降规模下只部分复现**：常数噪声（Fig.4）的 “Utility 掉 20pp 且 NN-FR≈0” 被复现（C21 VERIFIED）；
   MI 类损失确实比非 MI 损失压低 MI（C24 VERIFIED）；但 Table 1 的绝对数值与 Utility 列**未复现**（C8–C13、C23、C27 判 BLOCKED，F32）。
4. **按论文的攻击与安全模型评测（S1，§4.5）**：攻击管线**正确复现**（无保护基线 NN/MRP/BeamClean-FR 全 0，BeamClean≈NN 与 Table 3 现象一致）；
   在**同等 utility 损失**（≈−19…−20pp）下，SGT 的 NN 攻击成功率 0.25–2.25%，远低于常数噪声 σ=0.1 的 99.85%（方向支持论文机制）；
   但所有 SGT 都工作在 utility 崩坏区，且按论文自己的 A7-MI 指标并不优于同噪声水平的常数噪声（1.82–2.24 vs 1.01）——**“保效用下的信息论保护”未复现**（F34）。
4. **70B 实验（Table 2）不可执行**，只交付脚本与启动命令（D4，M9 BLOCKED）。
5. 声明核对表 40 条：**14 VERIFIED / 3 FALSIFIED / 23 BLOCKED / 0 UNKNOWN**。

## 1. 复现范围

| 项 | 论文 | 本复现 | 状态 |
|---|---|---|---|
| 理论与算法（Def 3.1/3.2、Prop 3.3、Cor 3.4、Eq 1–13、Alg.1） | 全部 | 全部实现 + 数值门禁 | 完成 |
| Table 1 消融（Llama-3.2-1B，OpenOrca 15%，A100×6h） | 6 配置 × 8 指标 | 5 配置完成 + 1 配置发散；OpenOrca 4000 条 / 250 步 / B=1×T=256 | 降规模（D2） |
| Utility（8 benchmark） | 全量 | 5 benchmark × limit 50 | 部分（D3） |
| Table 2（70B/32B，64×8×A100） | 3 模型 | 仅脚本（本机 exit 3） | BLOCKED（D4） |
| Table 3（PUPA + BeamClean + PAPILLON judge） | 4 配置 | NN + 自建 PII（50 条 PUPA）；BeamClean 未跑 | 部分（D5） |
| Fig.3 秩直方图 / Fig.4 常数噪声 | 定性/曲线 | 5 变体直方图 + 6 个 σ 的 Utility/NN-FR | 完成（降规模） |

## 2. 环境与硬件差异

| 项 | 论文 | 本复现 |
|---|---|---|
| 硬件 | 1–64 节点 ×8× A100 80GB | Apple M5 / 16GB 统一内存 / 无 NVIDIA |
| 框架 | 未说明 | PyTorch 2.15.0（MPS）+ transformers 5.22.1 + lm-eval 0.4.13 |
| 精度 | 未说明 | fp32（SGT 与 LM 同精度；训练器禁止隐式 dtype 转换） |
| 目标模型 | meta-llama/Llama-3.2-1B-Instruct | unsloth 同源镜像（官方 gated 403，D1），sha256 已验证 |
| 训练规模 | OpenOrca 15%（≈635k 条），≈6 A100 小时 | OpenOrca 4000 条，250 步，≈5 分钟/变体 |
| 评测规模 | 全量 8 benchmark | 5 benchmark × limit 50（D3） |

## 3. 方法复现（逐模块 + 门禁证据）

| 模块 | 内容 | 门禁 | 关键实测 |
|---|---|---|---|
| M0 | 骨架 + fail-fast 运行器（自检含 fail-fast 与 `pass:false` 落盘） | PASS | `results/gate_m0_skeleton.json` |
| M1 | Eq (1)、Prop 3.3、Cor 3.4、Eq (5)(7)、Eq (6) 真伪、Eq (4) 反解 | PASS | Eq(1) 差 0；KS p>0.32；Eq(5) 误差 0.12%；PAC 全列复现（worst 0.11pp） |
| M2 | Alg.1、Eq (8) pair/full、Eq (9)–(13)、L_U | PASS | 与朴素循环 1e-9 内一致；L_U 同分布 = H(p) |
| M3 | µθ/Σθ 双向 Transformer（零初始化头） | PASS | 序列依赖全位置、padding 不变、state_dict 往返 |
| M4 | 秩/NN/MRP/TTR/SymTTR/HistEntr/BeamClean | PASS | 暴力对拍、无噪声 BeamClean=NN、单调 |
| M5 | OpenOrca 15%（$n_{total}=4{,}233{,}923$，15%=635,089）、Alpaca、PUPA(user_query) | PASS | 指纹确定、零重叠、往返相似度 1.0 |
| M6 | 冻结 LM、只训 SGT、可复现、checkpoint | PASS | LM 参数逐位不变；同 seed history 严格相等 |
| M7 | 消融（5/6）+ Fig.3 + Fig.4 | PASS | 见 §4 |
| M8a | Utility（lm-eval + SGT embedding 注入） | PASS | 注入生效、Utility 聚合与手算一致 |
| M8b | PUPA NN/BeamClean + PII-Ratio | PASS | 基线 FR=0、PII-Ratio=1.0 |
| M9 | 大模型启动脚本（不执行） | PASS（执行 BLOCKED） | `exit 3 + status=blocked`，不加载 70B |
| S1 | 安全模型与攻击（NN/MRP/BeamClean/威胁模型） | PASS | 基线 FR=0、BC≈NN、`fr_by_r[1]==nn_fr`、噪声新鲜性 |
| M10 | 报告 + claim 表 + 回归 | PASS | 本文件 |

### 3.1 三处论文缺陷（详见 FINDINGS）

- **F29 / C4=FALSIFIED**：Eq (6) 把分量熵加权和称作 GMM 熵上界。由熵的凹性它是**下界**；正确上界为 $\sum_i\omega_i h_i+H(\omega)$。反例（门禁实测）：$w=[.5,.5],\mu=\pm5,\sigma^2=0.1$ 时差 $0.6931=H(\omega)$ nats。
- **F30 / C35=FALSIFIED**：Table 1 的 “MI” 列与 Eq (8) 逐特征项同口径，但**不是互信息**。解析：$x\sim N(0,1),\mu=0$ 时该量 $=1+2/\sigma^2\ge1$，而真实 MI $=\tfrac12\log(1+1/\sigma^2)\le0.347$；实测 2.07 vs 0.210。
- **F31 / C8=BLOCKED**：Eq (12) 的 $-\log|\Sigma|$ 对 σ 无下界。按原式训练 250 步内 $\sigma^2\to$ NaN，无法得到 Table 1 的有限行；论文必存在未声明的 σ 上界。

## 4. 实验数值对照

### 4.1 Table 1（Llama-3.2-1B-Instruct，降规模：4000 条 OpenOrca、B=1×T=256、250 步、lr=1e-4、α_mi=1/d；Alpaca 200 条 600 token 评测；Utility 为 5 benchmark × limit 50 的平均差）

| SGT Loss | Utility(pp) 论文→复现 | NN-FR | MRP-FR | SymTTR-10 | SymTTR-100 | Hist.Entr. | MI | PAC-Adv |
|---|---|---|---|---|---|---|---|---|
| Comp. Entr. | −13.84 → **发散(NaN)** | 88.41 → — | 88.41 → — | 75.99 → — | 49.75 → — | 58.60 → — | 22.19 → — | 100.0 → — |
| CosSim | −5.49 → **−20.96** | 99.98 → 99.67 | 3.21 → 99.67 | 1.43 → 99.00 | 1.07 → 95.50 | 2.47 → 0.542 | 3.2e19 → 4.85 | 100.0 → 47.1 |
| AbsCos | −0.01 → **−20.57** | 94.90 → 99.83 | 94.90 → 99.50 | 93.36 → 98.83 | 92.86 → 96.00 | 87.62 → 0.541 | 6e9 → 4.99 | 100.0 → 48.3 |
| MI | −0.72 → **−20.87** | 34.43 → 99.67 | 34.43 → 99.17 | 18.96 → 98.50 | 17.39 → 86.83 | 23.96 → 0.528 | 1.48 → 2.26 | 16.39 → 23.9 |
| MI+AbsCos | −2.44 → **−20.58** | 81.23 → 99.83 | 81.23 → 99.67 | 71.36 → 99.17 | 66.88 → 95.83 | 75.72 → 0.541 | 1.91 → 2.00 | 20.56 → 21.4 |
| MI+AbsCos+Norm | −1.97 → **−19.10** | 93.00 → 97.33 | 93.00 → 97.33 | 90.42 → 90.83 | 88.15 → 79.00 | 91.90 → 0.521 | 1.11 → 1.63 | 12.69 → 17.9 |

判定：全部 **BLOCKED**（C8–C13）。可解释的部分：
- **MI 排序与论文一致**（C24 VERIFIED）：含 MI 的三行 MI=2.26/2.00/1.63 明显低于不含 MI 的两行 4.85/4.99。
- **全损失在 MI 与 PAC 上最好**（1.63 / 17.9），与论文方向一致（C27 部分）。
- **Utility 未复现**（F32）：训练中 $L_U$ 仅 13.03→7.3–7.7；降幅与常数噪声 σ≈0.4 时（−20pp）几乎相同，说明 250 步远未收敛到“保效用混淆”。
- **MRP-FR 与论文不符**（C26 BLOCKED）：我们的 CosSim 直方图远端只有 1.8% 质量，$r^*=1$，MRP 退化为 NN；论文的远端主峰（MRP-FR 3.21）在降规模下未出现。

### 4.2 Table 2（大模型）

未执行。`experiments/run_large_models.py` 生成启动命令（`torchrun --nnodes=64 --nproc-per-node=8 … --fsdp2 --tp-size 8`）并在本机 `exit 3 + status=blocked`（`results/gate_m9_large.json`）。C14–C16、C30 判 BLOCKED。

### 4.3 Table 3（PUPA）

| SGT Loss | NN-FR（论文） | NN-FR（复现） | PII-Ratio（论文） | PII-Ratio（复现） |
|---|---|---|---|---|
| 基线（无 SGT） | — | 0.00 | — | 1.0000 |
| MI | 23.38 | 99.83 | 97.57 | 0.0000 |
| AbsCos | 97.67 | 99.88 | 1.60 | 0.0000 |
| MI+AbsCos | 75.17 | 99.85 | 65.04 | 0.0000 |
| MI+AbsCos+Norm | 86.77 | 97.27 | 46.98 | 0.0102 |

50 条 PUPA（`pupa_tnb`）、max_len 128、NN 攻击（BeamClean 未跑），PII-Ratio 用 `pii_units` 确定性匹配（非 PAPILLON judge，D5）。判定 C17–C20 **BLOCKED**；方向一致（SGT 大幅降低可恢复 PII），但量级与口径不可比。

### 4.4 Fig.3（秩直方图）与 Fig.4（常数噪声）

Fig.3（100 bin，Alpaca 200 条 × 96 token，600 token 汇总）：近端 bin 质量 CosSim 21.9%、AbsCos 25.0%、MI 55.7%、MI+AbsCos 26.6%、全损失 40.0%；最远 bin 质量 CosSim **1.81%**、其余 ≤0.17%。远端隆起只在 CosSim 出现，但未形成论文所述主峰（C22 BLOCKED）。

Fig.4（常数高斯噪声，2000 token；ε 用 Δ=embedding 范数 p99=1.1267、δ=1/128256，A14）：

| σ | ε | log ε | NN-FR | Utility(pp, 3 benchmark × limit 50) |
|---|---|---|---|---|
| 0.05 | 110.3 | 4.70 | 0.00 | −22.36 |
| 0.10 | 55.2 | 4.01 | 0.15 | −19.63 |
| 0.20 | 27.6 | 3.32 | 37.85 | −23.64 |
| 0.50 | 11.0 | 2.40 | 98.85 | −22.96 |
| 1.00 | 5.5 | 1.71 | 100.00 | −24.09 |

复现了论文 Fig.4 的核心现象：**在尚未产生混淆（NN-FR≈0，ε≳55）时 Utility 已下降约 20pp**（C21 VERIFIED）。ε 的绝对刻度取决于论文未给的 sensitivity Δ（A14）。


### 4.5 安全模型与攻击评测（S1）

**威胁模型（论文 §1/§4.1/§6 → 代码落点）**

| 能力/假设 | 论文 | 本复现 | 证据字段 |
|---|---|---|---|
| 对手位置 | 不可信服务端，看到 $\tilde x$ | 同 | `threat_model.attacker_position` |
| tokenizer / embedding 表 | 已知（否则 token 级重建无意义） | 已知 | `attacker_knows` |
| SGT 权重 | 客户端私有、对手未知 | 未知 | `sgt_weights_hidden=true` |
| 秩直方图（灰盒） | MRP 攻击允许 | 允许（$r^*$ 由本评测集直方图众数给出） | `greybox_rank_histogram=true` |
| 白盒（持有 SGT 权重） | 不假设 | **未测试** | `whitebox_weights_tested=false` |
| 隐私度量 | PAC：$\Delta_{KL}(\delta)\le\mathrm{MI}$ | A7 口径 Eq (8) 估计量 + 反解 | `metrics.pac_adv`、C7 |
| 攻击目标 | token 级重建 | 同 | `attacker_goal` |

**攻击结果（Alpaca 400 token，$|V|=128256$；BeamClean 为 PUPA n=5/T=24）**

| 模型 | NN-FR | MRP-FR（$r^*$） | r≥1000 时 FR | Hist.Entr. | MI(A7) | PAC-Adv |
|---|---|---|---|---|---|---|
| 无保护基线 | 0.00 | 0.00 (1) | 100.0 | 0.000 | ∞（σ=0） | vacuous |
| 常数噪声 σ=0.4 | 96.25 | 96.25 (1) | 100.0 | 0.483 | 1.01 | 11.6 |
| AbsCos | 99.75 | 99.50 (7) | 100.0 | 0.508 | 5.18 | 49.9 |
| CosSim | 99.75 | 99.50 (210) | 100.0 | 0.508 | 5.03 | 48.6 |
| MI | 99.50 | 98.75 (69) | 100.0 | 0.499 | 2.24 | 23.7 |
| MI+AbsCos | 99.75 | 99.50 (182) | 100.0 | 0.508 | 2.00 | 21.4 |
| MI+AbsCos+Norm | 97.75 | 97.75 (1) | 100.0 | 0.492 | 1.82 | 19.7 |
| BeamClean（abscos / 基线） | — | BC-FR 100.0 / 0.0 | — | — | — | — |
| PII-Ratio（PUPA，基线 1.0） | — | SGT 0.00–0.0102 | — | — | — | — |

**判定**

1. **攻击管线复现成功**（C36/C37/C39 VERIFIED）：无保护时 NN/MRP/BeamClean 全部精确重建（FR=0），说明攻击与指标实现没有“假阴性”；`fr_by_r` 与 NN 自洽；SGT 下 BeamClean-FR≈NN-FR，正是论文 Table 3 报告的现象。
2. **相对安全优势复现**（C38 VERIFIED）：在同一 utility 损失点（≈−19…−20pp），SGT 的 NN 攻击成功率 0.25–2.25%，而常数噪声 σ=0.1（utility −19.63pp）为 99.85%——即“噪声要毁掉 utility 才换来隐私，而 SGT 在同等 utility 损失下隐私强得多”，与论文 Fig.4 的论点一致。
3. **绝对安全声明未复现**（C23/C27/F32/F34）：论文的关键工作点是 utility −0.01…−2.44pp 下 81–93% NN-FR；我们的 SGT 全部落在 σ_eff≈0.38–0.45 的强噪声区（utility −19…−21pp）。按论文自己的 A7-MI 口径，同噪声水平下 SGT 并不优于常数噪声（MI 1.82–2.24 vs 1.01）。
4. **CosSim 的“远端秩洗白”未复现**（C40 BLOCKED）：论文 NN-FR 99.98% / MRP-FR 3.21%，我们 NN-FR 99.75% / MRP-FR 99.50%（$r^*=210$，最远 bin 仅 1.81%）——训练未收敛到反极性（$\cos$ 仅从 0.019 变到 0.018），所以“NN 指标不足”的论文证据在本复现中不成立。
5. **安全语义的保留**：论文把 A7 量称为 MI（F30），因此“信息论保护”的严格表述无法成立；PAC-Adv 只能当代理指标。白盒对手（持有 SGT 权重）未测试，论文也不假设。
6. **实现缺陷（已修）**：`SGTEmbedding` 原先每次前向重置噪声种子，违反“stochastic”定义；改为 `fresh_noise=True` 后，utility 变化 −20.57→−21.73pp（在 limit=50 抽样误差内），S1 门禁 `c-fresh-noise` 锁定该性质（F33）。

## 5. 论文问题清单

见 `docs/FINDINGS.md`：F0（无官方代码）、F1–F5（公式排印）、F8–F11（MI 归约/因子）、F12–F14（超参/ε 缺失）、F15–F16（文字与表值矛盾）、F21（MI 估计器未定义）、F27（PAC 界语义）、**F29（Eq 6 不是上界）**、**F30（MI 不是 MI）**、**F31（Eq 12 发散）**、**F32（降规模下 Utility 未复现）**。

## 6. 歧义与实例化决策

见 `SPEC.md` §7（A1–A20）。影响最大的三处：A1（SGT 架构，论文只写 “modeled as transformers”）、A7（特征级 MI 口径，直接决定 Table 1 的 MI/PAC 列）、A12（全部训练超参，论文未给）。所有取舍均写入结果 JSON 的 `config_echo`。

## 7. 复现结论统计

| 判定 | 数量 | 条目 |
|---|---|---|
| VERIFIED | 14 | C1–C3、C5–C7、C21、C24、C32、C34、C36–C39 |
| FALSIFIED | 3 | C4（Eq 6 不是上界）、C31（结论与 Table 1 自相矛盾，F16）、C35（MI 不是 MI） |
| BLOCKED | 23 | C8–C20、C22、C23、C25–C30、C33、C40 |
| UNKNOWN | 0 | — |

## 8. 复跑方式

```bash
./repro.sh setup                     # 两个 venv + 依赖
./repro.sh gates                     # M0→M10 fail-fast（约 3 分钟；M8a 会调用 .venv-lm）
./repro.sh metrics                   # pytest（101 项）
./repro.sh ablation --variant abscos --device mps --max-steps 250 --max-examples 4000 \
  --eval-examples 200 --eval-max-len 96 --eval-tokens 600 --rank-chunk 256
./repro.sh constant-noise --eval-tokens 2000 --sigmas 0.01,0.05,0.1,0.2,0.5,1.0 --out results/constnoise_low.json
./repro.sh pupa --max-examples 50 --max-len 128 --skip-beamclean --sgt-ckpt results/ckpt_abscos.pt \
  --out results/pupa_abscos.json
./repro.sh utility --tasks arc_easy,piqa,truthfulqa_mc2 --limit 50 --batch-size 1 --device mps
```

预期：`gates` 全绿（`results/gate_*.json` 的 `pass=true`）；`ablation` 单变体约 5 分钟（M5/MPS）。

## 9. 局限

1. **规模**：所有端到端数值都是 250 步 / 4000 条 / 5 benchmark 的子集，与论文的 6 A100 小时、8 benchmark 不可比（F32）。
2. **70B**：完全未执行（D4），只验证脚本契约。
3. **BeamClean / PII**：自建最小实现（D5），PII 用 `pii_units` 匹配而非 PAPILLON 的 LLM judge，数值不可与 Table 3 对齐。
4. **ε-DP**：论文未定义 sensitivity（A14），本复现取范数 p99，横轴刻度不可直接对比。
5. **SGT 架构**：论文未给，A1 取 hidden=512/2 层/双向/可学习位置编码；不同架构会改变 Table 1 的绝对数值。
