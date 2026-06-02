#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "找不到 .venv，請先依 README 安裝套件。"
  read -n 1 -s -r -p "按任意鍵結束"
  echo
  exit 1
fi

PYTHONPATH=src .venv/bin/python -m tw_ai_quant.cli \
  --config configs/example.yml \
  --output-dir artifacts_real \
  --report-dir reports_real \
  run

open reports_real/dashboard.html

echo "已更新並開啟真實資料儀表板：reports_real/dashboard.html"
read -n 1 -s -r -p "按任意鍵關閉此視窗"
echo
