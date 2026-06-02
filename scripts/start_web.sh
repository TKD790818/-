#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p artifacts_real reports_real

if [ ! -f "artifacts_real/latest_risk_plan.csv" ] && [ -d "cloud_seed/artifacts_real" ]; then
  echo "找不到雲端訊號資料，從 cloud_seed 還原預載分析結果..."
  find cloud_seed/artifacts_real -maxdepth 1 -type f | while read -r seed_file; do
    filename="$(basename "$seed_file")"
    if [[ "$filename" == *.gz ]]; then
      target="artifacts_real/${filename%.gz}"
      [ -f "$target" ] || gzip -dc "$seed_file" > "$target"
    else
      target="artifacts_real/$filename"
      [ -f "$target" ] || cp "$seed_file" "$target"
    fi
  done
fi

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
