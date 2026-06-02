from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"timezone": "Asia/Taipei"},
    "data": {
        "tickers": ["2330.TW", "2317.TW", "2454.TW"],
        "start": "2020-01-01",
        "end": None,
        "benchmark": "^TWII",
        "vix": "^VIX",
        "us10y": "^TNX",
        "prices_csv": None,
        "macro_csv": None,
        "news_csv": None,
        "news_rss_sources": [
            {"name": "Yahoo股市-台股動態", "url": "https://tw.stock.yahoo.com/rss?category=tw-market"},
            {"name": "Yahoo股市-最新新聞", "url": "https://tw.stock.yahoo.com/rss?category=news"},
            {"name": "中央社-產經證券", "url": "https://feeds.feedburner.com/rsscna/finance"},
        ],
    },
    "sentiment": {
        "max_items_per_source": 50,
        "request_timeout": 15,
    },
    "features": {
        "horizon_days": 1,
        "benchmark_column": "benchmark_close",
    },
    "model": {
        "type": "random_forest",
        "candidates": ["random_forest", "extra_trees", "gradient_boosting"],
        "selection_metric": "roc_auc",
        "test_size": 0.2,
        "model_path": "artifacts/model.joblib",
        "random_state": 42,
        "buy_threshold": 0.58,
        "sell_threshold": 0.42,
    },
    "risk": {
        "capital": 1_000_000,
        "risk_per_trade": 0.01,
        "max_position_pct": 0.2,
        "lot_size": 1_000,
        "atr_stop_multiplier": 2.0,
        "take_profit_r_multiple": 2.0,
    },
    "backtest": {
        "initial_capital": 1_000_000,
        "annual_trading_days": 252,
        "cost_rate": 0.0015,
    },
    "telegram": {
        "enabled": False,
        "bot_token_env": "TELEGRAM_BOT_TOKEN",
        "chat_id_env": "TELEGRAM_CHAT_ID",
    },
    "schedule": {
        "intraday": {
            "enabled": True,
            "start": "09:05",
            "end": "13:25",
            "interval_minutes": 30,
            "daytrade_limit": 50,
        },
        "after_close": {
            "enabled": True,
            "time": "14:30",
        },
        "evening_notify": {
            "enabled": True,
            "time": "20:30",
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    _load_env_file(Path(".env"))
    if path is None:
        return deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    _load_env_file(config_path.parent / ".env")
    _load_env_file(config_path.parent.parent / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping")

    return deep_merge(DEFAULT_CONFIG, loaded)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
