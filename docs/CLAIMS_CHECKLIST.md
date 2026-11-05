# 论文声明核对表

判定只用 **VERIFIED / FALSIFIED / BLOCKED / UNKNOWN**。初始全 UNKNOWN；随门禁与实验推进更新。
证据 = `results/` 下的 JSON 路径（或 `notes/` 下的分析），必须可复跑。

## 一、理论声明

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C1 | p.3–4 Def 3.1, Prop 3.3 | Affine SGT 的条件密度是 $G(\cdot;\alpha(x))$ 的平移；采样 $\tilde x=x+u$ 精确成立 | — | UNKNOWN | 计划 `results/gate_m1_theory.json` |
| C2 | p.4 Cor 3.4, Eq (2) | 数据集上 $\tilde X$ 服从 GMM，均值 $\mu_\theta(x_i)+x_i$，权重 $\omega_i=P_X(x_i)$ | — | UNKNOWN | 同上（c-cor34） |
| C3 | p.5 Eq (4) | $\Delta_{KL}(\delta)\le \mathrm{MI}(X;\tilde X)$，$ \delta_0$ 为先验成功率 $1-\delta_0$ 的补 | — | UNKNOWN | `gate_m1_theory.json`（c-pac-*），并用于反解 Table 1/2 的 PAC-Adv |
| C4 | p.5–6 Eq (6) | GMM 熵的上界 = 分量熵的加权和 $h(\tilde x)\le\sum_i\omega_i h_y(\cdot)$ | 反例：$w=[.5,.5],\mu=\pm5,\sigma^2=0.1$ 时 $h_{\text{mix}}-\sum\omega_i h_i=0.6931=H(\omega)$；正确上界为 $+H(\omega)$（分量熵加权和实为**下界**，熵的凹性） | **FALSIFIED** | `results/gate_m1_theory.json`（c-bound6-counterexample / c-bound6-true）；F29 |
| C5 | p.6 Eq (7) | $\mathrm{MI}=\mathbb E[-\log\mathbb E_X G]-\mathbb E_X[h_z]$ | — | UNKNOWN | `gate_m2_losses.json`（c-eq7） |
| C6 | p.6 Prop 5.1, Eq (8) | 两批样本间 MI 的 MC 近似有闭式 | — | UNKNOWN | `gate_m2_losses.json`（c-eq8） |
| C7 | p.5 Eq (4) 反解 | PAC-Adv 可由 MI 反解；$N=128256$ 时 Table 1/2 的 MI→PAC 映射自洽（1.11→12.69%，1.48→16.39%，1.91→20.56%，0.26→3.54%，0.28→3.74%，0.23→3.03%，22.19→100%） | — | UNKNOWN | `gate_m1_theory.json`（c-pac-table） |

## 二、Table 1（Llama-3.2-1B-Instruct，OpenOrca 15%，p.10）

列：Utility(↑pp)、NN-FR、MRP-FR、SymTTR-10、SymTTR-100、Hist.Entr.、MI(↓)、PAC-Adv(↓)。

| id | 论文出处 | 论文值 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C8 | Table 1 行 Comp. Entr. | −13.84 / 88.41 / 88.41 / 75.99 / 49.75 / 58.60 / 22.19 / 100.0 | — | UNKNOWN | `results/gate_m7_ablation.json` |
| C9 | Table 1 行 CosSim | −5.49 / 99.98 / 3.21 / 1.43 / 1.07 / 2.47 / 3.2e19 / 100.0 | — | UNKNOWN | 同上 |
| C10 | Table 1 行 AbsCos | −0.01 / 94.90 / 94.90 / 93.36 / 92.86 / 87.62 / 6e9 / 100.0 | — | UNKNOWN | 同上 |
| C11 | Table 1 行 MI | −0.72 / 34.43 / 34.43 / 18.96 / 17.39 / 23.96 / 1.48 / 16.39 | — | UNKNOWN | 同上 |
| C12 | Table 1 行 MI+AbsCos | −2.44 / 81.23 / 81.23 / 71.36 / 66.88 / 75.72 / 1.91 / 20.56 | — | UNKNOWN | 同上 |
| C13 | Table 1 行 MI+AbsCos+Norm | −1.97 / 93.00 / 93.00 / 90.42 / 88.15 / 91.90 / 1.11 / 12.69 | — | UNKNOWN | 同上 |

## 三、Table 2（两 epoch 全 OpenOrca，p.10）

| id | 论文出处 | 论文值 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C14 | Table 2 Llama-3.3-70B | −0.38 / 98.22 / 98.22 / 98.11 / 98.11 / 80.14 / 0.26 / 3.54 | — | UNKNOWN | 计划 `results/gate_m9_large.json`（本地 BLOCKED） |
| C15 | Table 2 DeepSeek-R1-70B | −0.46 / 97.39 / 97.39 / 97.23 / 97.23 / 78.05 / 0.28 / 3.74 | — | UNKNOWN | 同上 |
| C16 | Table 2 Qwen3-32B | −0.29 / 94.95 / 94.95 / 94.54 / 66.39 / 67.67 / 0.23 / 3.03 | — | UNKNOWN | 同上 |

## 四、Table 3（PUPA，p.10）

| id | 论文出处 | 论文值（NN FR / BeamClean FR / NN PII / BeamClean PII） | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C17 | Table 3 MI | 23.38 / 22.17 / 97.57 / 97.66 | — | UNKNOWN | `results/gate_m8_utility.json` |
| C18 | Table 3 AbsCos | 97.67 / 97.64 / 1.60 / 1.61 | — | UNKNOWN | 同上 |
| C19 | Table 3 MI+AbsCos | 75.17 / 75.16 / 65.04 / 65.03 | — | UNKNOWN | 同上 |
| C20 | Table 3 MI+AbsCos+Norm | 86.77 / 86.66 / 46.98 / 47.04 | — | UNKNOWN | 同上 |

## 五、图与文字结论

| id | 论文出处 | 论文主张 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C21 | p.9 Fig 4 | 常数高斯噪声在 $\epsilon<42$ 时 Utility 下降 20pp，同时 NN-FR≈0 | — | UNKNOWN | `results/constnoise_*.json` |
| C22 | p.7 Fig 3 | CosSim 训练的 SGT 秩直方图双峰（近端小峰、远端大峰），NN 攻击因此优于远端攻击 | — | UNKNOWN | `results/fig3_*.json` |
| C23 | p.9 §7.2 | AbsCos 的 Utility 与基线仅差 0.01pp | — | UNKNOWN | `gate_m7_ablation.json` |
| C24 | p.9 §7.2 | 只有含 MI 的损失（MI、MI+AbsCos、MI+AbsCos+Norm）能控制互信息；CosSim 的 MI ≈3.2e19、AbsCos ≈6e9 | — | UNKNOWN | 同上 |
| C25 | p.9 §7.2 | Comp. Entr. 提供部分 MI 控制但不足以给出 PAC 保证 | — | UNKNOWN | 同上 |
| C26 | p.9 §7.2 | CosSim 的 NN-FR 99.98% 但 MRP-FR 仅 3.21%，说明 NN 指标不足 | — | UNKNOWN | 同上 |
| C27 | p.9 §7.2 | 全损失（MI+AbsCos+Norm）在所有指标上最好，PAC-Adv 12.69% | — | UNKNOWN | 同上 |
| C28 | p.9 §7.2 | AbsCos 把 PUPA 的 PII-Ratio 降到 ~1.6（>90% 降幅） | — | UNKNOWN | `gate_m8_utility.json` |
| C29 | p.9 §7.2 | MI+AbsCos+Norm 把 PII-Ratio 从 97.57 降到 46.98（>50% 降幅） | — | UNKNOWN | 同上 |
| C30 | p.9 §7.2 | 三个 70B 级模型 Utility 均在 0.5pp 内，变换/信息论指标强 | — | UNKNOWN | `gate_m9_large.json`（BLOCKED） |
| C31 | p.10 §8 | “as little as a 0.29 percentage point drop” | — | UNKNOWN | 引用 C16；与 C23（0.01pp）矛盾，见 F16 |
| C32 | p.9 §7 | 消融模型在 OpenOrca 15% 上训练，按 SymTTR 选最佳 | — | UNKNOWN | `gate_m5_data.json`、`gate_m7_ablation.json` |
| C33 | p.9 §7 | Utility 为 8 个 benchmark 相对基线的平均差 | — | UNKNOWN | `gate_m8_utility.json` |
| C34 | p.5 §4.1 | PAC 界不紧，实际隐私更强 | — | UNKNOWN | 报告分析（不可证伪于本复现规模） |

## 判定口径

- **VERIFIED**：本仓库命令可复跑，证据 JSON 与论文值在 SPEC §9 阈值内，或命题在数学上被严格验证。
- **FALSIFIED**：可复跑证据与论文值/命题冲突（含论文自相矛盾时取其一被判错）。
- **BLOCKED**：需要论文规模（A100 集群 / 全量训练 / 官方外部代码）而本地不可得；必须给出降规模实测值 + 偏差 + 复跑脚本。
- 任何一行不得停留在 UNKNOWN 收尾。
