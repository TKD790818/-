#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "找不到 .venv，請先依 README 安裝套件。"
  read -n 1 -s -r -p "按任意鍵結束"
  echo
  exit 1
fi

open "http://127.0.0.1:8010"
PYTHONPATH=src .venv/bin/python -m tw_ai_quant.cli \
  --config configs/example.yml \
  web \
  --host 127.0.0.1 \
  --port 8010
