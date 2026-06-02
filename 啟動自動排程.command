#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "找不到 .venv，請先依 README 安裝套件。"
  read -n 1 -s -r -p "按任意鍵結束"
  echo
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "找不到 .env，請先設定 Telegram Token 與 Chat ID。"
  read -n 1 -s -r -p "按任意鍵結束"
  echo
  exit 1
fi

mkdir -p artifacts_real

echo "股智雷達 AI 自動排程已啟動"
echo "盤中：09:05～13:25，每 30 分鐘更新行情與當沖推薦"
echo "收盤後：14:30 跑完整 AI 分析、回測、報告"
echo "晚上：20:30 自動 Telegram 推播"
echo
echo "請保持這個終端機視窗開著；關掉視窗，排程就會停止。"
echo "排程中心：http://127.0.0.1:8010/schedule"
echo "紀錄檔：$(pwd)/artifacts_real/scheduler.log"
echo

open "http://127.0.0.1:8010/schedule" || true

PYTHONPATH=src .venv/bin/python -m tw_ai_quant.cli \
  --config configs/example.yml \
  schedule \
  --mode real \
  --poll-seconds 60 2>&1 | tee -a artifacts_real/scheduler.log
