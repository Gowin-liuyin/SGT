# 模块与门禁计划

依赖顺序即执行顺序（`gates/run_gates.py` 中的 `MODULES`）。**上游门禁未 PASS，下游不得开工**。
门禁脚本命名：`gates/gate_<id>_<name>.py`，结果落盘 `results/gate_<id>_<name>.json`。

| 模块 | 名称 | 依赖 | 产物 | 门禁 | 通过判据（可执行） |
|---|---|---|---|---|---|
| **M0** | 骨架 | — | 目录、SPEC、分析、计划、claim 表、运行器、状态表、报告骨架 | `gate_m0_skeleton.py` | 必需文件存在且非空；`run_gates --self-test` 返回 0（含 fail-fast 与 `pass:false` 落盘）；MODULE_PLAN 可解析出 M0..M10；`results/` 可写；状态表初始全 PENDING |
| **M1** | 理论核心 | M0 | `src/sgt/sampling.py`、`src/sgt/theory.py`、`tests/test_theory.py` | `gate_m1_theory.py` | ① Eq(1) 逐元素复算误差 0；② 残差协方差 vs $\Sigma$ 相对误差 ≤2%；③ Prop 3.3 KS $p>0.01$；④ Cor 3.4 能量距离在 95% 临界内；⑤ Eq(5)(7) 与数值积分误差 ≤1e-6；⑥ Eq(6) 上界 100/100 组成立；⑦ Eq(4) 反解复现 Table 1/2 PAC-Adv（±0.15pp） |
| **M2** | 损失函数 | M1 | `src/sgt/losses.py`、`tests/test_losses.py` | `gate_m2_losses.py` | Eq(8)(9)(10)(11)(12)(13) 逐条数值对照（1e-6）；Alg.1 与直译参考实现一致；$L^{CGE}$ 对 $\mu$ 梯度为 0；padding mask 正确；梯度非零/有限 |
| **M3** | SGT 模型 | M2 | `src/sgt/model.py`、`tests/test_model.py` | `gate_m3_model.py` | 形状/正定性；stochastic（异 seed 异输出）；sequence-dependent（改一个 token 使输出变化，且变化幅度 > 0 阈值）；可训练参数梯度非零；序列化往返；确定性（同 seed 同输出位一致） |
| **M4** | 指标与攻击 | M1、M3 | `src/sgt/metrics.py`、`src/sgt/attacks.py`、`tests/test_metrics.py` | `gate_m4_metrics.py` | 秩/NN/MRP/TTR/SymTTR/Histent 在解析构造上的精确值；SymTTR≤TTR；恒等变换 FR=0、TTR=0；MI-MC 与解析 1 维 GMM 误差 ≤2%；PAC 单调性；BeamClean 无噪声≈NN、随 $\sigma$ 单调 |
| **M5** | 数据管线 | M0 | `src/sgt/data.py`、`configs/data.yaml`、`tests/test_data.py` | `gate_m5_data.py` | OpenOrca 15% 条数可复算且哈希记录；Alpaca 评估集与训练集零重叠；PUPA 可解析；tokenize→decode 往返一致；batch 形状/mask 正确；同 seed 顺序一致 |
| **M6** | 训练回路 | M2、M3、M5 | `src/sgt/train.py`、`experiments/train_sgt.py`、`configs/ablation.yaml`、`tests/test_train.py` | `gate_m6_train.py` | 冻结 LM 参数在训练前后逐位相同（哈希）；SGT 参数变化；$L_U$ 在 $\tilde x=x$ 时为 0 且与手写 CE 一致；10 步 smoke 损失下降；同 seed 两次运行 checkpoint 位一致；超参全部写入结果 JSON |
| **M7** | 消融实验 | M6 | `experiments/run_ablation.py`、`run_constant_noise.py`、`results/*.json` | `gate_m7_ablation.py` | 6 种 loss 配置各产出完整 8 指标；Fig.3/Fig.4 数据可生成；常数噪声扫描单调性断言；结果与 Table 1 对照表生成（含偏差与 BLOCKED 标注）；不得缺列 |
| **M8** | Utility / PII | M6、M7 | `src/sgt/utility.py`、`src/sgt/pii.py`、`experiments/run_pupa.py` | `gate_m8_utility.py` | lm-eval 基线可复跑且 8 任务名齐全；Utility 聚合公式与手算一致；PII-Ratio 恒等文本≈1；PUPA 指标 schema 完整；失败任务记 BLOCKED 而非跳过 |
| **M9** | 大模型脚本 | M8 | `experiments/run_large_models.py`、`configs/large_models.yaml` | `gate_m9_large.py` | 三模型配置与 loss 组合静态断言；`--dry-run` 用 tiny 模型跑通 1 步；FSDP2/TP 参数存在但本地不启用；执行状态记 BLOCKED |
| **M10** | 报告与回归 | M1..M9 | `REPRODUCTION_REPORT.md`、`docs/CLAIMS_CHECKLIST.md` 终版、`REPRO_STATE.md` | `gate_m10_report.py` | 核对表无 UNKNOWN；每条 VERIFIED/FALSIFIED/BLOCKED 有证据路径；报告中每个数字可在 `results/` 找到；重跑全部上游门禁无回退 |

## 并行与写域

- M1 与 M5 无共享文件，可并行（M5 只写 `src/sgt/data.py`、`configs/data.yaml`、`tests/test_data.py`）。
- M2 依赖 M1；M3 依赖 M2；M4 依赖 M1、M3。
- M6 之后串行（同一训练产物）。
- 写域冲突纪律：一个任务单只允许改白名单文件；`gates/`、`docs/`、`SPEC.md` 由 Lead 独占。

## 门禁结果格式

```json
{"gate":"m1","name":"theory","module_entry":"Eq.(1),(2),(4),(5),(6),(7)","checks":[{"id":"c-eq1","expected":"...","actual":"...","pass":true,"detail":"..."}],"pass":true,"status":"PASS","timestamp":"...","env":{...}}
```

## 当前状态

见 `REPRO_STATE.md`。门禁状态以 `results/` 中的 JSON 为准，文档只是索引。
