#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-artifacts_real}"
SEED_DIR="${2:-cloud_seed/artifacts_real}"

mkdir -p "$SEED_DIR"

copy_plain() {
  local filename="$1"
  if [ -f "$SOURCE_DIR/$filename" ]; then
    cp "$SOURCE_DIR/$filename" "$SEED_DIR/$filename"
  fi
}

copy_gzip() {
  local filename="$1"
  if [ -f "$SOURCE_DIR/$filename" ]; then
    gzip -c "$SOURCE_DIR/$filename" > "$SEED_DIR/$filename.gz"
  fi
}

copy_plain "daily_sentiment.csv"
copy_plain "data_coverage.csv"
copy_plain "daytrade_latest.csv"
copy_plain "daytrade_latest.json"
copy_plain "equity_curve.csv"
copy_plain "feature_importance.csv"
copy_plain "latest_risk_plan.csv"
copy_plain "metrics.csv"
copy_plain "model_comparison.csv"
copy_plain "news_items.csv"

copy_gzip "features.csv"
copy_gzip "signal_history.csv"

echo "cloud_seed 已更新：$SEED_DIR"
