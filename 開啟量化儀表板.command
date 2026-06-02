#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "找不到 .venv，請先依 README 安裝套件。"
  read -n 1 -s -r -p "按任意鍵結束"
  echo
  exit 1
fi

PYTHONPATH=src .venv/bin/python -m tw_ai_quant.cli --output-dir artifacts --report-dir reports report
open reports/dashboard.html

echo "已開啟 reports/dashboard.html"
echo "如果瀏覽器沒有自動開啟，請手動打開：$(pwd)/reports/dashboard.html"
read -n 1 -s -r -p "按任意鍵關閉此視窗"
echo
