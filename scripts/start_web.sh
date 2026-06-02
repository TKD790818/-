#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts_real reports_real

if [ ! -f "artifacts_real/latest_risk_plan.csv" ]; then
  echo "找不到雲端訊號資料，先嘗試建立第一份分析結果..."
  python -m tw_ai_quant.cli \
    --config configs/example.yml \
    --output-dir artifacts_real \
    --report-dir reports_real \
    run || echo "初次分析失敗，網站仍會啟動；請稍後從排程中心或雲端 Job 更新資料。"
fi

exec python -m tw_ai_quant.cli \
  --config configs/example.yml \
  web \
  --host 0.0.0.0 \
  --port "${PORT:-8010}"
