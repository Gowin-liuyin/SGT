#!/usr/bin/env bash
# SGT 复现统一入口。所有命令都从仓库根目录执行，保证路径一致。
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
PYLM=.venv-lm/bin/python
MODEL=${SGT_MODEL:-.models/llama-3.2-1b-instruct}

setup() {
  if [ ! -d .venv ]; then
    /Users/yin/.local/bin/python3.12 -m venv .venv || python3 -m venv .venv
  fi
  "$PY" -m pip -q install --upgrade pip
  "$PY" -m pip -q install torch transformers datasets accelerate numpy scipy pytest "httpx[socks]"
  if [ ! -d .venv-lm ]; then
    /Users/yin/.local/bin/python3.12 -m venv .venv-lm
  fi
  .venv-lm/bin/pip -q install --upgrade pip
  .venv-lm/bin/pip -q install lm-eval torch transformers accelerate "httpx[socks]"
  echo "setup done: $("$PY" -V) / lm-eval $("$PYLM" -c 'import lm_eval;print(lm_eval.__version__)')"
}

gates() { "$PY" -m gates.run_gates --all; }

case "${1:-}" in
  setup) setup ;;
  gates) gates ;;
  gate) "$PY" "gates/gate_${2}_${3:-$(ls gates/ | sed -n "s/^gate_${2}_\(.*\)\.py$/\1/p" | head -1)}.py" ;;
  self-test) "$PY" -m gates.run_gates --self-test ;;
  metrics) "$PY" -m pytest tests/ -q ;;
  train) shift; "$PY" experiments/train_sgt.py "$@" ;;
  ablation) shift; "$PY" experiments/run_ablation.py --config configs/ablation.yaml "$@" ;;
  constant-noise) shift; "$PY" experiments/run_constant_noise.py --model "$MODEL" "$@" ;;
  pupa) shift; "$PY" experiments/run_pupa.py --model "$MODEL" "$@" ;;
  utility) shift; "$PYLM" experiments/run_utility.py --model "$MODEL" "$@" ;;
  *)
    cat >&2 <<'USAGE'
用法: ./repro.sh <command> [args]
  setup                     创建 .venv / .venv-lm 并安装依赖
  gates                     按 M0→M10 跑全部门禁（fail-fast）
  gate <mN> [name]          只跑某个门禁
  self-test                 门禁运行器自检（fail-fast + 结果落盘）
  metrics                   跑 pytest 全量单元测试
  train ...                 训练单个 SGT（experiments/train_sgt.py）
  ablation --variant <name> 训练+评测一个 Table 1 变体（args 透传）
  constant-noise            Fig.4 常数噪声扫描（args 透传）
  pupa                      PUPA 上的 NN/BeamClean/PII-Ratio（args 透传）
  utility                   lm-eval 8 任务 Utility（.venv-lm，args 透传）
USAGE
    exit 2 ;;
esac
