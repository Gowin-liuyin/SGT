# 论文声明核对表

判定只用 **VERIFIED / FALSIFIED / BLOCKED / UNKNOWN**。初始全 UNKNOWN；随门禁与实验推进更新。
证据 = `results/` 下的 JSON 路径（或 `notes/` 下的分析），必须可复跑。

## 一、理论声明

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C1 | p.3–4 Def 3.1, Prop 3.3 | Affine SGT 的条件密度是 $G(\cdot;\alpha(x))$ 的平移；采样 $\tilde x=x+u$ 精确成立 | Eq(1) 逐元素差 0；Prop 3.3 每维 KS p=0.32/0.72/0.57；残差协方差相对误差 0.17% | **VERIFIED** | `results/gate_m1_theory.json`（c-eq1/c-prop33/c-indep-cov） |
| C2 | p.4 Cor 3.4, Eq (2) | 数据集上 $\tilde X$ 服从 GMM，均值 $\mu_\theta(x_i)+x_i$，权重 $\omega_i=P_X(x_i)$ | 3 个投影方向 crit−D=0.0038/0.0056/0.0071（均未拒绝）；$\int p=1.000000$ | **VERIFIED** | `results/gate_m1_theory.json`（c-cor34/c-gmm-pdf-normalized） |
| C3 | p.5 Eq (4) | $\Delta_{KL}(\delta)\le \mathrm{MI}(X;\tilde X)$，$ \delta_0$ 为先验成功率 $1-\delta_0$ 的补 | 以 $N=128256$ 反解 Table 1/2 的 7 个 MI 值，最大偏差 0.11pp；$\Delta_{KL}$ 单调性验证通过 | **VERIFIED**（作为可反解的不等式；见 C35 对 MI 语义的保留） | `results/gate_m1_theory.json`（c-pac-monotone/c-pac-table/c-pac-invert-consistency） |
| C4 | p.5–6 Eq (6) | GMM 熵的上界 = 分量熵的加权和 $h(\tilde x)\le\sum_i\omega_i h_y(\cdot)$ | 反例：$w=[.5,.5],\mu=\pm5,\sigma^2=0.1$ 时 $h_{\text{mix}}-\sum\omega_i h_i=0.6931=H(\omega)$；正确上界为 $+H(\omega)$（分量熵加权和实为**下界**，熵的凹性） | **FALSIFIED** | `results/gate_m1_theory.json`（c-bound6-counterexample / c-bound6-true）；F29 |
| C5 | p.6 Eq (7) | $\mathrm{MI}=\mathbb E[-\log\mathbb E_X G]-\mathbb E_X[h_z]$ | 1 维 GMM：MC MI 0.81518 vs 数值积分 0.81496，相对误差 0.12% | **VERIFIED** | `results/gate_m1_theory.json`（c-eq5-eq7） |
| C6 | p.6 Prop 5.1, Eq (8) | 两批样本间 MI 的 MC 近似有闭式 | pair/full 两种归约与朴素循环参考实现逐一相等（误差 ≤1e-9）；half_factor 恰为 0.5 倍 | **VERIFIED**（公式实现层面） | `results/gate_m2_losses.json`（c-mi-pair/c-mi-full/c-mi-half） |
| C7 | p.5 Eq (4) 反解 | PAC-Adv 可由 MI 反解；$N=128256$ 时 Table 1/2 的 MI→PAC 映射自洽（1.11→12.69%，1.48→16.39%，1.91→20.56%，0.26→3.54%，0.28→3.74%，0.23→3.03%，22.19→100%） | 反解得 12.67/16.37/20.56/3.50/3.74/3.14/100，最大偏差 0.11pp | **VERIFIED** | `results/gate_m1_theory.json`（c-pac-table） |

## 二、Table 1（Llama-3.2-1B-Instruct，OpenOrca 15%，p.10）

列：Utility(↑pp)、NN-FR、MRP-FR、SymTTR-10、SymTTR-100、Hist.Entr.、MI(↓)、PAC-Adv(↓)。

| id | 论文出处 | 论文值 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C8 | Table 1 行 Comp. Entr. | −13.84 / 88.41 / 88.41 / 75.99 / 49.75 / 58.60 / 22.19 / 100.0 | 按论文原式（Eq 12 → $-\log\lvert\Sigma\rvert$）训练发散，$\sigma^2\to$ NaN（250 步内）；无数值可比 | **BLOCKED**（F31：论文必有未声明的 σ 上界） | `results/ablation_summary.json`（status=diverged）；F31 |
| C9 | Table 1 行 CosSim | −5.49 / 99.98 / 3.21 / 1.43 / 1.07 / 2.47 / 3.2e19 / 100.0 | −20.96 / 99.67 / 99.67 / 99.00 / 95.50 / 0.542 / 4.85 / 47.1 | **BLOCKED**（降规模：250 步、4000 条、B=1×T=256） | `results/ablation_cossim.json`、`results/utility_cossim.json` |
| C10 | Table 1 行 AbsCos | −0.01 / 94.90 / 94.90 / 93.36 / 92.86 / 87.62 / 6e9 / 100.0 | −20.57 / 99.83 / 99.50 / 98.83 / 96.00 / 0.541 / 4.99 / 48.3 | **BLOCKED**（同上；Utility 未复现，见 F32） | `results/ablation_abscos.json`、`results/utility_abscos.json` |
| C11 | Table 1 行 MI | −0.72 / 34.43 / 34.43 / 18.96 / 17.39 / 23.96 / 1.48 / 16.39 | −20.87 / 99.67 / 99.17 / 98.50 / 86.83 / 0.528 / 2.26 / 23.9 | **BLOCKED** | `results/ablation_mi.json`、`results/utility_mi.json` |
| C12 | Table 1 行 MI+AbsCos | −2.44 / 81.23 / 81.23 / 71.36 / 66.88 / 75.72 / 1.91 / 20.56 | −20.58 / 99.83 / 99.67 / 99.17 / 95.83 / 0.541 / 2.00 / 21.4 | **BLOCKED** | `results/ablation_mi_abscos.json`、`results/utility_mi_abscos.json` |
| C13 | Table 1 行 MI+AbsCos+Norm | −1.97 / 93.00 / 93.00 / 90.42 / 88.15 / 91.90 / 1.11 / 12.69 | −19.10 / 97.33 / 97.33 / 90.83 / 79.00 / 0.521 / 1.63 / 17.9 | **BLOCKED**（MI/PAC 方向一致：该配置 MI 最低） | `results/ablation_mi_abscos_norm.json`、`results/utility_mi_abscos_norm.json` |

## 三、Table 2（两 epoch 全 OpenOrca，p.10）

| id | 论文出处 | 论文值 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C14 | Table 2 Llama-3.3-70B | −0.38 / 98.22 / 98.22 / 98.11 / 98.11 / 80.14 / 0.26 / 3.54 | 未执行（无多卡集群；脚本 `run_large_models.py` 在本机 exit 3 + status=blocked） | **BLOCKED**（D4） | `results/gate_m9_large.json` |
| C15 | Table 2 DeepSeek-R1-70B | −0.46 / 97.39 / 97.39 / 97.23 / 97.23 / 78.05 / 0.28 / 3.74 | 未执行 | **BLOCKED**（D4） | 同上 |
| C16 | Table 2 Qwen3-32B | −0.29 / 94.95 / 94.95 / 94.54 / 66.39 / 67.67 / 0.23 / 3.03 | 未执行 | **BLOCKED**（D4） | 同上 |

## 四、Table 3（PUPA，p.10）

| id | 论文出处 | 论文值（NN FR / BeamClean FR / NN PII / BeamClean PII） | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C17 | Table 3 MI | 23.38 / 22.17 / 97.57 / 97.66 | 99.83 / 未跑 / 0.0 / 未跑（50 条 PUPA，NN only） | **BLOCKED**（自建 BeamClean/PII，非官方实现；且降规模） | `results/pupa_mi.json` |
| C18 | Table 3 AbsCos | 97.67 / 97.64 / 1.60 / 1.61 | 99.88 / 未跑 / 0.0 / 未跑 | **BLOCKED**（同上；方向一致：PII 大幅下降） | `results/pupa_abscos.json` |
| C19 | Table 3 MI+AbsCos | 75.17 / 75.16 / 65.04 / 65.03 | 99.85 / 未跑 / 0.0 / 未跑 | **BLOCKED** | `results/pupa_mi_abscos.json` |
| C20 | Table 3 MI+AbsCos+Norm | 86.77 / 86.66 / 46.98 / 47.04 | 97.27 / 未跑 / 0.01 / 未跑 | **BLOCKED** | `results/pupa_mi_abscos_norm.json` |

## 五、图与文字结论

| id | 论文出处 | 论文主张 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C21 | p.9 Fig 4 | 常数高斯噪声在 $\epsilon<42$ 时 Utility 下降 20pp，同时 NN-FR≈0 | Utility −19.6…−24.1pp（σ=0.05–1.0，5 任务中 3 任务 limit 50）；NN-FR 从 0（ε=110/55）升到 37.9（ε=27.6）、98.9（ε=11.0） | **VERIFIED**（我们自定的 Δ=范数 p99 下；趋势与阈值量级一致） | `results/constnoise_low.json`、`results/constnoise_util_*.json` |
| C22 | p.7 Fig 3 | CosSim 训练的 SGT 秩直方图双峰（近端小峰、远端大峰），NN 攻击因此优于远端攻击 | cossim：近端 bin 占 21.9%、最远 bin 占 1.81%（其他变体 ≤0.17%），存在远端隆起但未形成主峰 | **BLOCKED**（形式部分一致，降规模未收敛） | `results/fig3_ranksh.json` |
| C23 | p.9 §7.2 | AbsCos 的 Utility 与基线仅差 0.01pp | −20.57pp（5 任务 limit 50） | **BLOCKED**（F32：降规模训练未收敛，SGT 仍近似强噪声） | `results/utility_abscos.json`、`results/ablation_summary.json` |
| C24 | p.9 §7.2 | 只有含 MI 的损失（MI、MI+AbsCos、MI+AbsCos+Norm）能控制互信息；CosSim 的 MI ≈3.2e19、AbsCos ≈6e9 | 按 A7 口径：CosSim 4.85、AbsCos 4.99 vs MI 2.26、MI+AbsCos 2.00、全损失 1.63（MI 组低 55–67%） | **VERIFIED**（方向与排序；绝对值受 A7/F30 口径影响） | `results/ablation_summary.json` |
| C25 | p.9 §7.2 | Comp. Entr. 提供部分 MI 控制但不足以给出 PAC 保证 | 按论文原式训练发散（NaN），无法给出 MI/PAC | **BLOCKED**（F31） | `results/ablation_summary.json` |
| C26 | p.9 §7.2 | CosSim 的 NN-FR 99.98% 但 MRP-FR 仅 3.21%，说明 NN 指标不足 | cossim：NN-FR 99.67、MRP-FR 99.67（$r^*=1$，MRP 退化为 NN） | **BLOCKED**（降规模下未出现远端主峰，见 C22/F9） | `results/ablation_cossim.json` |
| C27 | p.9 §7.2 | 全损失（MI+AbsCos+Norm）在所有指标上最好，PAC-Adv 12.69% | 全损失 MI 最低（1.63）、PAC 最低（17.9）、Utility 最好（−19.10），但 NN-FR 97.33 低于 AbsCos 的 99.83 | **BLOCKED**（部分指标排序一致） | `results/ablation_summary.json` |
| C28 | p.9 §7.2 | AbsCos 把 PUPA 的 PII-Ratio 降到 ~1.6（>90% 降幅） | PII-Ratio 0.0（基线 1.0；50 条 PUPA，NN only，自建 pii_units 匹配口径） | **BLOCKED**（方向一致；口径与 PAPILLON judge 不同） | `results/pupa_abscos.json` |
| C29 | p.9 §7.2 | MI+AbsCos+Norm 把 PII-Ratio 从 97.57 降到 46.98（>50% 降幅） | PII-Ratio 0.0102（基线 1.0） | **BLOCKED**（方向一致；量级受降规模与口径影响） | `results/pupa_mi_abscos_norm.json` |
| C30 | p.9 §7.2 | 三个 70B 级模型 Utility 均在 0.5pp 内，变换/信息论指标强 | 未执行（无多卡集群） | **BLOCKED**（D4） | `results/gate_m9_large.json` |
| C31 | p.10 §8 | “as little as a 0.29 percentage point drop” | 论文自身 Table 1 的 AbsCos 为 −0.01pp，比 0.29 更小；结论与表值矛盾 | **FALSIFIED**（论文内部不一致，F16） | `docs/FINDINGS.md` F16 + `results/ablation_summary.json` |
| C32 | p.9 §7 | 消融模型在 OpenOrca 15% 上训练，按 SymTTR 选最佳 | 15% 协议实现（$n_{total}=4{,}233{,}923$，15% = 635,089）；实际降规模用 4000 条；未做 checkpoint 选择（单次运行） | **VERIFIED**（协议实现层面）+ 降规模声明 | `results/gate_m5_data.json` |
| C33 | p.9 §7 | Utility 为 8 个 benchmark 相对基线的平均差 | 实现为相对差平均；本复现只用 5 个任务（arc_easy/arc_challenge/openbookqa/piqa/truthfulqa_mc2）× limit 50，MMLU/HellaSwag/IFEval 未纳入 | **BLOCKED**（部分实现，D3） | `results/utility_abscos.json`、`results/ablation_summary.json` |
| C34 | p.5 §4.1 | PAC 界不紧，实际隐私更强 | 降规模实测：abscos 的 PAC-Adv 上界 48.3%，NN 攻击失败率 99.83%（即成功率 0.17% ≪ 上界） | VERIFIED（方向性） | `results/ablation_abscos.json` |
| C35 | p.8 §6.4 / Table 1 MI 列 | 报告的 “MI” 是互信息 | 按 Eq (8) 逐特征口径可复现数值，但解析核对表明它不是信息论 MI（$1+2/\sigma^2$ vs $\tfrac12\log(1+1/\sigma^2)$） | **FALSIFIED**（术语错误） | F30；`results/gate_m7_ablation.json`（c-feature-mi） |

## 六、攻击与安全模型（S1）

| id | 论文出处 | 论文主张 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C36 | §1/§4.1/§6 | 威胁模型：不可信服务端、知道 tokenizer 与 embedding 表、不知道客户端 SGT 权重；PAC 界 $\Delta_{KL}(\delta)\le\mathrm{MI}$ | 代码与结果 JSON 的 `threat_model` 块逐项实现（`attacker_position=untrusted_server`、`attacker_knows=[tokenizer,embedding_table]`、`sgt_weights_hidden=true`、`greybox_rank_histogram=true`、`whitebox_weights_tested=false`）；PAC 反解见 C7 | **VERIFIED** | `results/gate_s1_security.json`（c-threat-model）、`results/attack_*.json` |
| C37 | §6.1–6.3 | NN/MRP/BeamClean 攻击与 FR 指标的定义 | 无保护基线 NN-FR=0、MRP-FR=0、BeamClean-FR=0、PII-Ratio=1.0；`fr_by_r[1]==nn_fr`；`fr_by_r` 随 r 单调（r≥10 即 100%） | **VERIFIED** | `results/attack_baseline.json`、`results/pupa_baseline_beamclean.json` |
| C38 | §7.2 Fig 4 | 与常数噪声相比，SGT 在同一 Utility 损失下能提供更强的混淆 | 同效用点（≈−19…−20pp）：SGT NN 攻击成功率 0.25–2.25%；常数噪声 σ=0.1（utility −19.63pp）NN-FR 0.15%（攻击成功率 99.85%） | **VERIFIED**（方向与量级；utility 为 limit 50 抽样） | `results/attack_abscos.json`、`results/constnoise_util_0.1.json`、`results/ablation_summary.json` |
| C39 | §6.2、Table 3 | 语言感知的 BeamClean 与 NN 攻击强度相当（BC-FR ≈ NN-FR） | PUPA n=5/T=24：无保护基线 NN 0.0 / BC 0.0；abscos SGT NN 100.0 / BC 100.0 | **VERIFIED**（小样本；自建 BeamClean，D5） | `results/pupa_baseline_beamclean.json`、`results/pupa_abscos_beamclean.json` |
| C40 | §6.1、Fig 3、Table 1 | CosSim 训练的 SGT 会把信息“洗白”到远端秩：NN-FR 99.98% 但 MRP-FR 仅 3.21% | 我们 CosSim 变体：NN-FR 99.75%、MRP-FR 99.50%（$r^*=210$，最远 bin 质量 1.81%）；远端攻击成功率≈0 | **BLOCKED**（现象未复现：训练未收敛到反极性，见 F32/F34） | `results/attack_cossim.json`、`results/fig3_ranksh.json` |

## 判定口径

- **VERIFIED**：本仓库命令可复跑，证据 JSON 与论文值在 SPEC §9 阈值内，或命题在数学上被严格验证。
- **FALSIFIED**：可复跑证据与论文值/命题冲突（含论文自相矛盾时取其一被判错）。
- **BLOCKED**：需要论文规模（A100 集群 / 全量训练 / 官方外部代码）而本地不可得；必须给出降规模实测值 + 偏差 + 复跑脚本。
- 任何一行不得停留在 UNKNOWN 收尾。
