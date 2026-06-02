#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p artifacts_real reports_real

if [ "${RUN_STARTUP_ANALYSIS:-false}" = "true" ]; then
  echo "RUN_STARTUP_ANALYSIS=true，啟動前先嘗試建立分析結果..."
  "$PYTHON_BIN" -m tw_ai_quant.cli \
    --config configs/example.yml \
    --output-dir artifacts_real \
    --report-dir reports_real \
    run || echo "初次分析失敗，網站仍會啟動；請稍後從排程中心或雲端 Job 更新資料。"
else
  echo "略過啟動前完整分析，先啟動網站。"
fi

exec "$PYTHON_BIN" -m tw_ai_quant.cli \
  --config configs/example.yml \
  web \
  --host 0.0.0.0 \
  --port "${PORT:-8010}"
