# 论文问题清单（不一致 / 缺失 / 可实现性）

编号稳定，报告与 claim 表引用同一编号。严重度：**H** 影响复现结论，**M** 影响数值可比性，**L** 排版/表述。

| id | 位置 | 问题 | 严重度 | 影响 | 处理 |
|---|---|---|---|---|---|
| F0 | 仓库链接 | `github.com/Gowin-liuyin/SGT` 为空仓库（0 commit），无任何参考实现 | H | 全部实现只能从论文推导；歧义无法用官方代码裁决 | 报告声明；SPEC 记录每个歧义的决策 |
| F1 | p.7 Eq (12) | 印作 $L^{CGE}_O=\log(-|\Sigma_\theta(X_b)|^{-1})$：负数取对数无定义 | M | Comp.Entr 损失无法按字面实现 | 按 Bound (6) 语义实现为 $-\log|\Sigma|$（A5），M2 门禁记录 |
| F2 | p.6 Eq (9) 与 p.7 正文 | 公式 $\sum\frac{X_b\cdot\tilde X_b}{\lVert X_b\rVert\lVert\tilde X_b\rVert}$ 无绝对值；正文说 “computes the absolute value of the cosine similarity” | M | ACS 损失符号/归约不明 | A4：实现逐 token $\lvert\cos\rvert$ 平均，保留 `signed` 开关 |
| F3 | p.9 §7.2（两处） | “Comp. Entr. +Cat” 出现在正文，但 `Cat` 从未定义，Table 1 也没有该行 | M | 无法判断是否存在第七种配置 | 报告列出；不实现未定义损失；claim 只覆盖 Table 1 的 6 行 |
| F4 | p.4 §3 | $\Sigma_\theta:\mathbb R^{d}\to(\Delta_d^+)^T$ 定义域印错（应为 $\mathbb R^{d\times T}$），同段后文自证 | L | 无 | 按 $d\times T$ 实现 |
| F5 | p.4 Cor 3.4 | “then $Y$ is distributed as a Gaussian Mixture Model”，$Y$ 未定义（应为 $\tilde X$） | L | 无 | 按 $\tilde X$ 实现 |
| F6 | p.5 Eq (5) | 排版 “$\mathrm{M\,I}(X;\tilde X)$” 断字 | L | 无 | — |
| F7 | p.5 Eq (4) 附近 | $\Delta_{KL}$ 的下标 $\delta$ 与符号 $\delta_0$（先验成功率 $1-\delta_0$）排印含混 | L | 反解时需谨慎 | M1/c-pac 用定义式，反解结果与 Table 1/2 对齐后才算通过 |
| F8 | p.8 §6.4 vs Table 1 | Table 1 的 MI=22.19（Comp.Entr）**大于** $\log N=\log 128256\approx 11.76$，而正文说 “any MI above $\log(N)$ ... falls out of the scope”；论文用 “feature level” 解释，但未给出特征级 MI 的精确定义 | H | Table 1 的 MI 列不可复算；PAC-Adv 与 MI 的映射依赖口径 | A7 登记；M4 输出 mean/sum/median 三种口径；报告并列 |
| F9 | p.10 Table 1 | 5 行的 NN-FR 与 MRP-FR **完全相同**（88.41、94.90、34.43、81.23、93.00），只有 CosSim 不同（99.98 vs 3.21） | H | “NN 指标不足”的核心证据只来自 CosSim 一行；其余行的 MRP 可能是退化实现（$r=1$）或报告错误 | M7 独立计算两指标；若本地结果系统性不同，报告为 F9 的证据 |
| F10 | p.6 Eq (8) vs Alg.1 L6 | Eq (8) 是 $\frac1{|B'|}\sum_{\ell',i}$（双下标）却除以单批大小；Alg.1 明确按同下标 $i$ 配对 | M | MI 损失数值差 | A2：默认 Alg.1 配对，另报全交叉 |
| F11 | p.6 Eq (8) | 由 Eq (7) 展开应带 $\tfrac12$ 因子（$\log|\Sigma|$ 与 Mahalanobis 各 $\tfrac12$），Eq (8) 两项都无 | M | 两项相对权重差 2 倍（$\alpha_1$ 不能同时吸收） | A3：按字面实现 + 可选 `half_factor` |
| F12 | p.9 §7 | 未给任何训练超参：优化器、lr、batch、步数/epoch（消融）、$B$、序列长度、$\alpha_1..\alpha_3$、warmup、是否共享 SGT 层 | H | 论文数值不可精确复现 | A12：写入 `configs/ablation.yaml` 并声明为复现选择 |
| F13 | p.9 §7 | “trained on a single Nvidia A100s 80GB for roughly six hours” 但未给 token/步数，无法换算规模 | M | 无法估计训练量 | 记录到报告的偏差声明 |
| F14 | p.9 Fig 4 | 横轴 $\log(\epsilon)$ 范围 0.5–3.5（$\epsilon\lesssim 33$），正文却说 “for all $\epsilon$ smaller than 42”；且高斯机制的 $\epsilon$ 需要 sensitivity $\Delta$，论文未定义 | H | Fig 4 横轴映射不可复现 | A14：自定 $\Delta$（99 分位范数）并同时报 $\sigma$；claim C21 判 BLOCKED 或带此声明 |
| F15 | p.9 §7.2 | 正文 “AbsCos having MI of over 32 quintillion” 与 Table 1（AbsCos MI=$6\times10^9$）矛盾；32 quintillion = $3.2\times10^{19}$ 是 **CosSim** 的值 | M | 文字与表冲突 | 以表为准（可复跑证据优先），报告 F15 |
| F16 | p.10 §8 vs Table 1 | 结论 “as little as a 0.29 percentage point drop” 与 Table 1 的 AbsCos（−0.01pp）矛盾；0.29 来自 Table 2 的 Qwen3-32B | M | 摘要/结论的最优值归属错误 | 报告 F16；claim C31 按 Table 1/2 分别判定 |
| F17 | p.9 §7 | “Obfuscation quality is evaluated on the Alpaca dataset” 未说明哪些指标在 Alpaca、哪些在 OpenOrca 验证子集上算 | M | 指标口径不明 | A13/报告：统一在 Alpaca 上算，并记录 |
| F18 | p.9 §7 | “select the best performer based on SymTTR” 未说明 SymTTR-10 还是 -100、验证集大小 | M | 模型选择不可复现 | M7 固定用 SymTTR-100（Table 1 主列），报告声明 |
| F19 | p.7 §6.1 | MRP 的 $r$ 定义为 “maximum rank probability $r$ over a dataset $D$”，未给估计方式与校准集 | M | MRP-FR 不可复现（且与 F9 相关） | A8 |
| F20 | p.8 §6.3 | TTR-1 与 NN-FR 在定义上等价，但 Table 1 只列 NN-FR，无 TTR-1 交叉验证 | L | 无法交叉核对 | M4 同时输出，报告一致性 |
| F21 | p.8 §6.4 | MI 的估计器未说明：是否复用 Alg.1/Eq (8)、在多少样本上、是否包含条件熵解析项 | H | MI 列不可复算 | M4 实现两种估计器（Eq 8 式 + 独立 MC），报告差异 |
| F22 | p.8 §6.4 | Hist. Entropy 用 $\log N$（$N$=词表大小），但不同模型 $N$ 不同（128256/128256/151936？），表间数值不可直接比 | L | 跨表比较需注明 | 报告记录每个模型的 $N$ |
| F23 | p.10 Table 1 vs Table 3 | 同为 MI 损失 + Llama-1B，NN-FR 在 Alpaca 为 34.43、在 PUPA 为 23.38，论文未解释数据集差异 | M | 需说明指标随数据集变化 | 报告记录实测两数据集差异 |
| F24 | p.4 “dataset ... random variable in $X\in L^2$ ... (differential) entropy” | 离散 token 的熵与微分熵混用；$X$ 是离散点集却有密度的表述 | L | 理论表述不严谨 | 报告注明 |
| F25 | p.4 Eq (1) | $\Sigma^{1/2}$ 未说明对角矩阵开方；由 $\Sigma\in(\Delta_d^+)^T$ 可推 | L | 无 | 实现为逐维开方 |
| F26 | p.10 Table 2 | 列是 TTR-50 而非 Table 1 的 SymTTR-10/100，跨表不可直接比 | L | 比较需注意 | 报告注明 |
| F27 | p.1 abstract / p.10 §8 | “information theoretically provides privacy” 依赖 Eq (4)，而作者自己说该界不紧（p.5）；且 MI 为特征级、重建为整嵌入级 | H | 核心声明的强度需谨慎 | 报告 F27 + C34 的判定 |
| F28 | p.9 §7.2 | “The top performer in terms of NN-FR was CosSim, achieving an impressive 99.98%” 与 “AbsCos provided the best protection in terms of Utility and reconstructions” 并列，未处理“哪一个是最好”的口径冲突 | L | 结论表述含混 | 报告按指标分别陈述 |
| **F29** | p.5–6 §5.1 Eq (6) | 论文把“分量熵的加权和” $\sum_i\omega_i h_y(\mathcal N(y;\mu(x_i),\Sigma(x_i)))$ 称为 GMM 熵的**上界**。该式不成立：由熵对分布是**凹函数**，$\sum_i\omega_i h_i$ 是混合熵的**下界**（不是上界）；正确上界是 $\sum_i\omega_i h_i + H(\omega)$，分离分量时两者相差 $H(\omega)$。数值反例（本仓库 M1 门禁用 `scipy.integrate.quad` 独立验证）：$w=[.5,.5]$、$\mu=\pm5$、$\sigma=0.1$ 时 $h_{	ext{mix}}=-0.1905$、$\sum\omega_i h_i=-0.8836$，差 $+0.6931=H(\omega)$；$w=[.2,.5,.3]$、$\mu=[-10,0,10]$、$\sigma=0.2$ 时差 $+1.0297$ nats | H | ① 以 Eq (6) 为出发点的 $L^{CGE}$（Eq 12）“上界”语义失效；② §5.1 对 Bound (6) 的批评（“不含均值项”）反而更强；③ 但主方法 Eq (8) 不依赖 Eq (6)，只是用 MC 直接估 $h(	ilde X)$，故 SGT 主体不受影响 | 报告 F29；claim C4 判 FALSIFIED；M1 门禁同时验证真上界与反例；不修改论文语义去“凑”通过 |
