#!/usr/bin/env bash
# SGT 复现统一入口。所有命令都从仓库根目录执行，保证路径一致。
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python

setup() {
  if [ ! -d .venv ]; then
    /Users/yin/.local/bin/python3.12 -m venv .venv || python3 -m venv .venv
  fi
  "$PY" -m pip -q install --upgrade pip
  "$PY" -m pip -q install torch transformers datasets accelerate numpy scipy "httpx[socks]"
  echo "setup done: $("$PY" -V)"
}

gates() {
  "$PY" -m gates.run_gates --all
}

case "${1:-}" in
  setup) setup ;;
  gates) gates ;;
  gate) "$PY" "gates/gate_${2}_${3:-$(ls gates/ | sed -n "s/^gate_${2}_\(.*\)\.py$/\1/p" | head -1)}.py" ;;
  self-test) "$PY" -m gates.run_gates --self-test ;;
  ablation) "$PY" experiments/run_ablation.py --config configs/ablation.yaml ;;
  constant-noise) "$PY" experiments/run_constant_noise.py --config configs/constant_noise.yaml ;;
  pupa) "$PY" experiments/run_pupa.py --config configs/pupa.yaml ;;
  utility) "$PY" experiments/run_utility.py --config configs/utility.yaml ;;
  report) "$PY" experiments/make_report.py ;;
  *)
    echo "用法: ./repro.sh {setup|gates|gate <mN>|self-test|ablation|constant-noise|pupa|utility|report}" >&2
    exit 2
    ;;
esac
