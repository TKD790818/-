from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .automation import automation_status, run_automation_job
from .config import load_config
from .dashboard import generate_dashboard
from .features import feature_label
from .market_data import load_market_profile
from .notify import send_telegram_message
from .pipeline import run_pipeline
from .universe import ticker_group_map


APP_TITLE = "台股 AI 量化分析系統"
DAYTRADE_CACHE_SECONDS = 25
AI_BUY_THRESHOLD = 0.58
AI_SELL_THRESHOLD = 0.42
_DAYTRADE_CACHE: dict[tuple[str, str, int], tuple[float, dict[str, object]]] = {}
_INTRADAY_SNAPSHOT_CACHE: dict[tuple[str, tuple[str, ...]], tuple[float, dict[str, dict[str, object]], str, str]] = {}
TECHNICAL_FIELDS = [
    "sma_5",
    "sma_20",
    "sma_60",
    "sma_120",
    "ema_12",
    "ema_26",
    "rsi_14",
    "kd_k",
    "kd_d",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_percent",
    "atr_14",
    "momentum_10",
    "obv",
    "vwap_20",
    "volume_ratio_20",
    "beta_60",
    "daily_return",
    "benchmark_return",
    "sentiment_score",
]


class RunRequest(BaseModel):
    mode: Literal["demo", "real"] = "real"


class NotificationRequest(BaseModel):
    mode: Literal["demo", "real"] = "real"


class JobRequest(BaseModel):
    mode: Literal["demo", "real"] = "real"


def create_app(config_path: str = "configs/example.yml") -> FastAPI:
    app = FastAPI(title=APP_TITLE, version="0.1.0")
    app.state.config_path = config_path

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _index_html()

    @app.get("/stock/{ticker}", response_class=HTMLResponse)
    def stock_detail(ticker: str, mode: Literal["demo", "real"] = "real") -> str:
        return _stock_html(ticker, mode)

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest_page() -> str:
        return _backtest_html()

    @app.get("/notify", response_class=HTMLResponse)
    def notify_page() -> str:
        return _notify_html()

    @app.get("/schedule", response_class=HTMLResponse)
    def schedule_page() -> str:
        return _schedule_html()

    @app.get("/api/status")
    def status(mode: Literal["demo", "real"] = "real") -> dict[str, object]:
        output_dir, report_dir = _paths(mode)
        return {
            "mode": mode,
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "dashboard_exists": (report_dir / "dashboard.html").exists(),
            "signals_exists": (output_dir / "latest_risk_plan.csv").exists(),
            "metrics_exists": (output_dir / "metrics.csv").exists(),
        }

    @app.get("/api/signals")
    def signals(mode: Literal["demo", "real"] = "real") -> list[dict[str, object]]:
        output_dir, _ = _paths(mode)
        records = _records(output_dir / "latest_risk_plan.csv")
        config = load_config(app.state.config_path)
        feature_lookup = _latest_score_feature_lookup(output_dir)
        group_lookup = ticker_group_map(config)
        tickers = [str(row.get("ticker")) for row in records if row.get("ticker")]
        snapshots, source, warning = _intraday_snapshots(tickers, feature_lookup, mode)
        risk_config = config.get("risk", {}) if isinstance(config.get("risk"), dict) else {}
        updated_at = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S")
        enriched = [
            _apply_intraday_signal_snapshot(
                _enrich_signal_record(record, feature_lookup, group_lookup),
                snapshots.get(str(record.get("ticker")), {}),
                risk_config,
                source,
                warning,
                updated_at,
            )
            for record in records
        ]
        return _sort_signal_records(enriched)

    @app.get("/api/daytrade")
    def daytrade(mode: Literal["demo", "real"] = "real", limit: int = 10) -> dict[str, object]:
        return _daytrade_payload(mode, app.state.config_path, limit=max(1, min(limit, 50)))

    @app.get("/api/groups")
    def groups() -> list[dict[str, object]]:
        return _group_options(app.state.config_path)

    @app.get("/api/metrics")
    def metrics(mode: Literal["demo", "real"] = "real") -> dict[str, object]:
        output_dir, _ = _paths(mode)
        records = _records(output_dir / "metrics.csv")
        return records[0] if records else {}

    @app.get("/api/coverage")
    def coverage(mode: Literal["demo", "real"] = "real") -> list[dict[str, object]]:
        output_dir, _ = _paths(mode)
        records = _records(output_dir / "data_coverage.csv")
        group_lookup = ticker_group_map(load_config(app.state.config_path))
        for record in records:
            record["stock_group"] = record.get("stock_group") or group_lookup.get(str(record.get("ticker")), "未分類")
        return records

    @app.get("/api/news")
    def news(mode: Literal["demo", "real"] = "real") -> list[dict[str, object]]:
        output_dir, _ = _paths(mode)
        return _records(output_dir / "news_items.csv")

    @app.get("/api/sentiment")
    def sentiment(mode: Literal["demo", "real"] = "real") -> list[dict[str, object]]:
        output_dir, _ = _paths(mode)
        return _records(output_dir / "daily_sentiment.csv")

    @app.get("/api/model-comparison")
    def model_comparison(mode: Literal["demo", "real"] = "real") -> list[dict[str, object]]:
        output_dir, _ = _paths(mode)
        return _records(output_dir / "model_comparison.csv")

    @app.get("/api/feature-importance")
    def feature_importance(mode: Literal["demo", "real"] = "real") -> list[dict[str, object]]:
        output_dir, _ = _paths(mode)
        return _records(output_dir / "feature_importance.csv")

    @app.get("/api/feature-importance/meta")
    def feature_importance_meta(mode: Literal["demo", "real"] = "real") -> dict[str, object]:
        output_dir, _ = _paths(mode)
        return _file_meta(output_dir / "feature_importance.csv")

    @app.get("/api/technical/{ticker}")
    def technical(ticker: str, mode: Literal["demo", "real"] = "real") -> dict[str, object]:
        return _technical_payload(ticker, mode)

    @app.get("/api/chart/{ticker}")
    def chart(ticker: str, mode: Literal["demo", "real"] = "real", interval: str = "1d", period: str = "6mo") -> dict[str, object]:
        return _chart_payload(ticker, mode, interval, period)

    @app.get("/api/backtest")
    def backtest(mode: Literal["demo", "real"] = "real") -> dict[str, object]:
        return _backtest_payload(mode, app.state.config_path)

    @app.get("/api/notification")
    def notification(mode: Literal["demo", "real"] = "real") -> dict[str, object]:
        return _notification_payload(mode, app.state.config_path)

    @app.get("/api/schedule")
    def schedule_status(mode: Literal["demo", "real"] = "real") -> dict[str, object]:
        return automation_status(app.state.config_path, mode)

    @app.post("/api/jobs/{job_name}")
    def run_job(job_name: Literal["intraday", "after_close", "evening_notify"], request: JobRequest) -> dict[str, object]:
        result = run_automation_job(job_name, app.state.config_path, request.mode)
        return {
            "job": result.job,
            "status": result.status,
            "message": result.message,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "output": result.output,
        }

    @app.post("/api/notification/send")
    def send_notification(request: NotificationRequest) -> dict[str, object]:
        config = load_config(app.state.config_path)
        status = _telegram_status(config, include_secret=True)
        if not status["ready"]:
            raise HTTPException(status_code=400, detail=status["reason"])
        payload = _notification_payload(request.mode, app.state.config_path)
        send_telegram_message(str(status["bot_token"]), str(status["chat_id"]), str(payload["message"]))
        return {"ok": True, "sent": True, "message_length": len(str(payload["message"]))}

    @app.post("/api/run")
    def run(request: RunRequest) -> dict[str, object]:
        config = load_config(app.state.config_path)
        output_dir, report_dir = _paths(request.mode)
        result = run_pipeline(config, demo=request.mode == "demo", output_dir=output_dir)
        dashboard_path = generate_dashboard(output_dir, report_dir)
        return {
            "ok": True,
            "mode": request.mode,
            "dashboard": str(dashboard_path),
            "training_metrics": result["training_metrics"],
            "backtest_metrics": result["backtest_metrics"],
            "signals": len(result["signals"]),
        }

    return app


def run_web_app(
    config_path: str = "configs/example.yml",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    import uvicorn

    uvicorn.run(create_app(config_path), host=host, port=port)


def _paths(mode: str) -> tuple[Path, Path]:
    if mode == "demo":
        return Path("artifacts"), Path("reports")
    return Path("artifacts_real"), Path("reports_real")


def _group_options(config_path: str) -> list[dict[str, object]]:
    config = load_config(config_path)
    tickers = list(dict.fromkeys(config.get("data", {}).get("tickers", [])))
    groups = ticker_group_map(config)
    counts = Counter(groups.get(str(ticker), "未分類") for ticker in tickers)
    return [
        {
            "stock_group": group,
            "configured_count": int(count),
        }
        for group, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _file_meta(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "path": str(path), "modified_at": "", "rows": 0}
    rows = 0
    try:
        rows = int(pd.read_csv(path, usecols=[0]).shape[0])
    except Exception:
        rows = 0
    modified_at = pd.Timestamp.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {"exists": True, "path": str(path), "modified_at": modified_at, "rows": rows}


def _backtest_payload(mode: str, config_path: str) -> dict[str, object]:
    output_dir, _ = _paths(mode)
    curve_path = output_dir / "equity_curve.csv"
    metrics_path = output_dir / "metrics.csv"
    signal_path = output_dir / "signal_history.csv"
    latest_path = output_dir / "latest_risk_plan.csv"

    if not curve_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {curve_path}")

    curve = pd.read_csv(curve_path)
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve["strategy_return"] = pd.to_numeric(curve["strategy_return"], errors="coerce").fillna(0)
    curve["turnover"] = pd.to_numeric(curve.get("turnover", 0), errors="coerce").fillna(0)
    curve["equity"] = pd.to_numeric(curve["equity"], errors="coerce")
    curve = curve.dropna(subset=["date", "equity"]).sort_values("date")
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1

    metrics = _records(metrics_path)[0] if metrics_path.exists() and _records(metrics_path) else {}
    monthly = (
        curve.set_index("date")["strategy_return"]
        .resample("ME")
        .apply(lambda returns: (1 + returns).prod() - 1)
        .dropna()
        .tail(12)
        .reset_index(name="monthly_return")
    )
    monthly["month"] = monthly["date"].dt.strftime("%Y-%m")

    signal_summary = _signal_summary(signal_path, latest_path)
    latest_signal_date = ""
    if latest_path.exists():
        latest = pd.read_csv(latest_path, usecols=lambda column: column == "date")
        latest_signal_date = str(latest["date"].max()) if not latest.empty and "date" in latest else ""
    config = load_config(config_path)
    backtest_config = config.get("backtest", {})

    return {
        "metrics": metrics,
        "summary": {
            "start_date": _date_text(curve["date"].iloc[0]),
            "end_date": _date_text(curve["date"].iloc[-1]),
            "days": int(len(curve)),
            "best_day": _json_value(curve["strategy_return"].max()),
            "worst_day": _json_value(curve["strategy_return"].min()),
            "win_rate": _json_value(curve["strategy_return"].gt(0).mean()),
            "avg_turnover": _json_value(curve["turnover"].mean()),
            "verified_end_date": _date_text(curve["date"].iloc[-1]),
            "latest_signal_date": latest_signal_date,
            "initial_capital": _json_value(backtest_config.get("initial_capital")),
            "cost_rate": _json_value(backtest_config.get("cost_rate")),
        },
        "curve": _records_from_frame(curve[["date", "strategy_return", "turnover", "equity", "drawdown"]]),
        "monthly_returns": _records_from_frame(monthly[["month", "monthly_return"]]),
        "signal_summary": signal_summary,
        "trade_log_status": "交易紀錄表會放在框架最後，等未來有實際操作成交資料後再啟用。",
    }


def _signal_summary(signal_path: Path, latest_path: Path) -> dict[str, object]:
    summary: dict[str, object] = {"history": {}, "latest": {}, "latest_composite": {}, "buy_days": 0}
    if signal_path.exists():
        history = pd.read_csv(signal_path)
        if not history.empty and "ml_signal" in history.columns:
            prob_up = pd.to_numeric(history.get("prob_up"), errors="coerce")
            history["ml_signal"] = "HOLD"
            history.loc[prob_up >= AI_BUY_THRESHOLD, "ml_signal"] = "BUY"
            history.loc[prob_up <= AI_SELL_THRESHOLD, "ml_signal"] = "SELL"
            counts = history["ml_signal"].value_counts().to_dict()
            summary["history"] = {str(key): int(value) for key, value in counts.items()}
            daily_buy = history.assign(is_buy=history["ml_signal"].eq("BUY")).groupby("date")["is_buy"].sum()
            summary["buy_days"] = int(daily_buy.gt(0).sum())
    if latest_path.exists():
        latest = pd.read_csv(latest_path)
        if not latest.empty and "ml_signal" in latest.columns:
            latest_records = latest.to_dict("records")
            for record in latest_records:
                _normalize_probability_fields(record)
            latest = pd.DataFrame(latest_records)
            counts = latest["ml_signal"].value_counts().to_dict()
            summary["latest"] = {str(key): int(value) for key, value in counts.items()}
            feature_lookup = _latest_score_feature_lookup(latest_path.parent)
            composite_counts = pd.Series(
                [
                    _enrich_signal_record(record, feature_lookup)["composite_signal"]["code"]
                    for record in latest_records
                ]
            ).value_counts().to_dict()
            summary["latest_composite"] = {str(key): int(value) for key, value in composite_counts.items()}
    return summary


def _notification_payload(mode: str, config_path: str) -> dict[str, object]:
    config = load_config(config_path)
    output_dir, _ = _paths(mode)
    metrics = _records(output_dir / "metrics.csv")[0] if (output_dir / "metrics.csv").exists() else {}
    signals = _records(output_dir / "latest_risk_plan.csv")
    feature_lookup = _latest_score_feature_lookup(output_dir)
    group_lookup = ticker_group_map(config)
    signals = [_enrich_signal_record(record, feature_lookup, group_lookup) for record in signals]

    sorted_signals = sorted(
        signals,
        key=lambda row: (
            _number(row.get("recommendation_score")) or 0,
            _number(row.get("prob_up")) or 0,
        ),
        reverse=True,
    )
    top_signals = sorted_signals[:10]
    latest_date = max((str(row.get("date", "")) for row in signals), default="")
    signal_counts = pd.Series([row["composite_signal"]["code"] for row in signals]).value_counts().to_dict()
    ai_signal_counts = pd.Series([row.get("ml_signal", "HOLD") for row in signals]).value_counts().to_dict()
    status = _telegram_status(config, include_secret=False)
    message = _format_daily_notification_message(top_signals, metrics, latest_date, signal_counts, ai_signal_counts)
    return {
        "status": status,
        "summary": {
            "latest_date": latest_date,
            "signal_count": len(signals),
            "green": int(signal_counts.get("GREEN", 0)),
            "yellow": int(signal_counts.get("YELLOW", 0)),
            "red": int(signal_counts.get("RED", 0)),
            "ai_buy": int(ai_signal_counts.get("BUY", 0)),
            "ai_hold": int(ai_signal_counts.get("HOLD", 0)),
            "ai_sell": int(ai_signal_counts.get("SELL", 0)),
            "message_length": len(message),
        },
        "metrics": metrics,
        "top_signals": top_signals,
        "message": message,
        "schedule_note": "排程中心已提供盤中更新、收盤分析與晚間 Telegram 推播；此頁保留訊息預覽與手動推播。",
    }


def _daytrade_payload(mode: str, config_path: str, limit: int = 10) -> dict[str, object]:
    cache_key = (mode, config_path, limit)
    cached = _DAYTRADE_CACHE.get(cache_key)
    now = monotonic()
    if cached and now - cached[0] <= DAYTRADE_CACHE_SECONDS:
        payload = cached[1].copy()
        payload["cached"] = True
        return payload

    output_dir, _ = _paths(mode)
    signals = _records(output_dir / "latest_risk_plan.csv")
    feature_lookup = _latest_score_feature_lookup(output_dir)
    group_lookup = ticker_group_map(load_config(config_path))
    tickers = [str(row.get("ticker")) for row in signals if row.get("ticker")]

    snapshots, source, warning = _intraday_snapshots(tickers, feature_lookup, mode)
    records: list[dict[str, object]] = []
    for record in signals:
        ticker = str(record.get("ticker"))
        record = _enrich_signal_record(record, feature_lookup, group_lookup)
        candidate = _daytrade_candidate(record, snapshots.get(ticker, {}))
        if candidate:
            records.append(candidate)

    records.sort(
        key=lambda row: (
            _number(row.get("daytrade_score")) or 0,
            _number(row.get("volume_ratio")) or 0,
            _number(row.get("intraday_return")) or 0,
        ),
        reverse=True,
    )
    top = records[:limit]
    payload = {
        "updated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "cached": False,
        "warning": warning,
        "universe_count": len(tickers),
        "scored_count": len(records),
        "items": top,
        "method": "當沖分數 = AI機率、VWAP、量能、均線趨勢、MACD、RSI、突破與風報比綜合排序。",
    }
    _DAYTRADE_CACHE[cache_key] = (now, payload)
    return payload


def _intraday_snapshots(
    tickers: list[str],
    feature_lookup: dict[str, dict[str, object]],
    mode: str,
) -> tuple[dict[str, dict[str, object]], str, str]:
    if not tickers:
        return {}, "no tickers", "目前沒有股票清單可計算。"
    cache_key = (mode, tuple(sorted(dict.fromkeys(tickers))))
    cached = _INTRADAY_SNAPSHOT_CACHE.get(cache_key)
    now = monotonic()
    if cached and now - cached[0] <= DAYTRADE_CACHE_SECONDS:
        snapshots, source, warning = cached[1], cached[2], cached[3]
        return {ticker: snapshot.copy() for ticker, snapshot in snapshots.items()}, source, warning
    if mode == "demo":
        return _cache_intraday_snapshots(cache_key, _daily_fallback_snapshots(feature_lookup), "Demo daily fallback", "")

    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            period="1d",
            interval="1m",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        snapshots = _snapshots_from_yfinance(raw, tickers, feature_lookup)
        if snapshots:
            return _cache_intraday_snapshots(cache_key, snapshots, "Yahoo Finance 1m", "")
    except Exception as exc:
        warning = f"盤中資料暫時無法取得，已改用日線備援：{exc}"
        return _cache_intraday_snapshots(cache_key, _daily_fallback_snapshots(feature_lookup), "features.csv fallback", warning)

    return _cache_intraday_snapshots(
        cache_key,
        _daily_fallback_snapshots(feature_lookup),
        "features.csv fallback",
        "盤中資料暫無有效資料，已改用日線備援。",
    )


def _cache_intraday_snapshots(
    cache_key: tuple[str, tuple[str, ...]],
    snapshots: dict[str, dict[str, object]],
    source: str,
    warning: str,
) -> tuple[dict[str, dict[str, object]], str, str]:
    _INTRADAY_SNAPSHOT_CACHE[cache_key] = (
        monotonic(),
        {ticker: snapshot.copy() for ticker, snapshot in snapshots.items()},
        source,
        warning,
    )
    return snapshots, source, warning


def _snapshots_from_yfinance(
    raw: pd.DataFrame,
    tickers: list[str],
    feature_lookup: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    if raw.empty:
        return {}

    snapshots: dict[str, dict[str, object]] = {}
    for ticker in tickers:
        part = _yfinance_part(raw, ticker, tickers)
        if part is None or part.empty:
            continue
        part = part.reset_index().rename(
            columns={
                "Date": "date",
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        required = ["open", "high", "low", "close", "volume"]
        if any(column not in part.columns for column in required):
            continue
        for column in required:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
        part = part[part["close"].gt(0)]
        if part.empty:
            continue

        close = _number(part["close"].iloc[-1])
        open_price = _number(part["open"].dropna().iloc[0]) if not part["open"].dropna().empty else close
        high = _number(part["high"].max())
        low = _number(part["low"].min())
        volume = _number(part["volume"].fillna(0).sum()) or 0
        typical = (part["high"] + part["low"] + part["close"]) / 3
        volume_series = part["volume"].fillna(0)
        vwap = None
        if volume_series.sum() > 0:
            vwap = _number((typical * volume_series).sum() / volume_series.sum())
        momentum_15m = None
        if len(part) > 15:
            base = _number(part["close"].iloc[-16])
            momentum_15m = None if base in {None, 0} or close is None else close / base - 1

        daily_volume_ma20 = _number(feature_lookup.get(ticker, {}).get("volume_ma_20"))
        expected_volume = None if daily_volume_ma20 in {None, 0} else daily_volume_ma20 * _market_elapsed_fraction()
        volume_ratio = None if expected_volume in {None, 0} else volume / expected_volume
        snapshots[ticker] = {
            "current_price": close,
            "open_price": open_price,
            "intraday_high": high,
            "intraday_low": low,
            "intraday_volume": volume,
            "intraday_return": None if open_price in {None, 0} or close is None else close / open_price - 1,
            "vwap": vwap,
            "vwap_gap": None if vwap in {None, 0} or close is None else close / vwap - 1,
            "momentum_15m": momentum_15m,
            "volume_ratio": volume_ratio,
        }
    return snapshots


def _yfinance_part(raw: pd.DataFrame, ticker: str, tickers: list[str]) -> pd.DataFrame | None:
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            return raw[ticker].copy()
        if ticker in raw.columns.get_level_values(-1):
            return raw.xs(ticker, axis=1, level=-1).copy()
        return None
    if len(tickers) == 1:
        return raw.copy()
    return None


def _daily_fallback_snapshots(feature_lookup: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    snapshots = {}
    for ticker, row in feature_lookup.items():
        close = _number(row.get("close"))
        vwap = _number(row.get("vwap_20"))
        snapshots[ticker] = {
            "current_price": close,
            "open_price": close,
            "intraday_high": _number(row.get("high")),
            "intraday_low": _number(row.get("low")),
            "intraday_volume": _number(row.get("volume")),
            "intraday_return": _number(row.get("daily_return")),
            "vwap": vwap,
            "vwap_gap": None if vwap in {None, 0} or close is None else close / vwap - 1,
            "momentum_15m": _number(row.get("momentum_10")),
            "volume_ratio": _number(row.get("volume_ratio_20")),
        }
    return snapshots


def _market_elapsed_fraction() -> float:
    now = pd.Timestamp.now(tz="Asia/Taipei")
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=30, second=0, microsecond=0)
    if now < market_open or now > market_close:
        return 1.0
    total_seconds = (market_close - market_open).total_seconds()
    elapsed_seconds = (now - market_open).total_seconds()
    return _clamp(elapsed_seconds / total_seconds, 0.08, 1.0)


def _daytrade_candidate(record: dict[str, object], snapshot: dict[str, object]) -> dict[str, object] | None:
    current = _number(snapshot.get("current_price")) or _number(record.get("close"))
    if current is None:
        return None

    intraday_return = _number(snapshot.get("intraday_return"))
    volume_ratio = _number(snapshot.get("volume_ratio")) or _number(record.get("volume_ratio_20"))
    vwap_gap = _number(snapshot.get("vwap_gap"))
    momentum_15m = _number(snapshot.get("momentum_15m"))
    high = _number(snapshot.get("intraday_high"))
    low = _number(snapshot.get("intraday_low"))
    range_pct = None if high is None or low is None or current == 0 else (high - low) / current
    recommendation = _number(record.get("recommendation_score")) or 0
    prob_up = _number(record.get("prob_up")) or 0
    atr = _number(record.get("atr_14"))
    risk_gap = max(current * 0.008, (atr or 0) * 0.35, current * 0.003)
    entry_price = current
    stop_loss = max(0, current - risk_gap)
    take_profit = current + risk_gap * 1.5
    reward_risk = None if current == stop_loss else (take_profit - entry_price) / (entry_price - stop_loss)
    conditions = _daytrade_conditions(record, current, volume_ratio, vwap_gap)
    score, basis = _daytrade_score(
        recommendation,
        prob_up,
        intraday_return,
        volume_ratio,
        vwap_gap,
        momentum_15m,
        range_pct,
        reward_risk,
        conditions,
    )
    signal = _daytrade_signal(score, conditions)
    return {
        "ticker": record.get("ticker"),
        "stock_name": record.get("stock_name") or record.get("ticker"),
        "stock_group": record.get("stock_group") or "未分類",
        "current_price": round(current, 2),
        "intraday_return": intraday_return,
        "volume_ratio": volume_ratio,
        "vwap_gap": vwap_gap,
        "momentum_15m": momentum_15m,
        "range_pct": range_pct,
        "recommendation_score": int(recommendation),
        "prob_up": prob_up,
        "daytrade_score": int(round(score)),
        "signal": signal,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "basis": "、".join(basis) if basis else "等待盤中量價確認",
    }


def _daytrade_conditions(
    record: dict[str, object],
    current: float,
    volume_ratio: float | None,
    vwap_gap: float | None,
) -> dict[str, bool]:
    sma5 = _number(record.get("sma_5"))
    sma20 = _number(record.get("sma_20"))
    sma60 = _number(record.get("sma_60"))
    macd = _number(record.get("macd"))
    macd_signal = _number(record.get("macd_signal"))
    prev_high20 = _number(record.get("prev_high_20"))
    prev_high60 = _number(record.get("prev_high_60"))
    rsi = _number(record.get("rsi_14"))
    return {
        "trend_up": bool(_all_numbers(current, sma20, sma60) and current > sma20 > sma60),
        "short_up": bool(_all_numbers(sma5, sma20) and sma5 > sma20),
        "macd_up": bool(_all_numbers(macd, macd_signal) and macd > macd_signal),
        "vol_up": bool(volume_ratio is not None and volume_ratio >= 1.5),
        "breakout20": bool(_all_numbers(current, prev_high20) and current >= prev_high20),
        "breakout60": bool(_all_numbers(current, prev_high60) and current >= prev_high60),
        "rsi_good": bool(rsi is not None and 50 <= rsi <= 75),
        "rsi_hot": bool(rsi is not None and rsi > 82),
        "weak": bool(_all_numbers(current, sma20) and current < sma20),
        "vwap_up": bool(vwap_gap is not None and vwap_gap >= 0),
    }


def _daytrade_score(
    recommendation: float,
    prob_up: float,
    intraday_return: float | None,
    volume_ratio: float | None,
    vwap_gap: float | None,
    momentum_15m: float | None,
    range_pct: float | None,
    reward_risk: float | None,
    conditions: dict[str, bool],
) -> tuple[float, list[str]]:
    basis: list[str] = []
    score = _clamp(recommendation, 0, 100) * 0.12
    score += _clamp((prob_up - 0.42) / 0.18, 0, 1) * 8

    if conditions["trend_up"]:
        score += 10
        basis.append("多頭排列")
    if conditions["short_up"]:
        score += 4
        basis.append("短均向上")
    if conditions["macd_up"]:
        score += 7
        basis.append("MACD翻正")
    if conditions["rsi_good"]:
        score += 7
        basis.append("RSI健康")
    if conditions["breakout20"]:
        score += 5
        basis.append("突破20日高")
    if conditions["breakout60"]:
        score += 3
        basis.append("突破60日高")

    if intraday_return is not None:
        if intraday_return > 0:
            score += _clamp(intraday_return / 0.035, 0, 1) * 10
            basis.append("盤中翻紅")
        else:
            score -= 8
    if volume_ratio is not None:
        score += _clamp(volume_ratio / 2.5, 0, 1) * 14
        if conditions["vol_up"]:
            score += 4
            basis.append("量能放大")
        elif volume_ratio < 0.8:
            score -= 5
    if vwap_gap is not None:
        if conditions["vwap_up"]:
            score += 10
            basis.append("站上VWAP")
        else:
            score -= 10
            basis.append("未站上VWAP")
    if momentum_15m is not None:
        if momentum_15m > 0:
            score += _clamp(momentum_15m / 0.018, 0, 1) * 8
            basis.append("短線動能")
        else:
            score -= 4
    if range_pct is not None:
        if 0.008 <= range_pct <= 0.07:
            score += 5
            basis.append("波動適中")
        elif range_pct > 0.09:
            score -= 8
            basis.append("波動過大")
    if reward_risk is not None:
        score += _clamp(reward_risk / 2, 0, 1) * 5
        if reward_risk >= 1.5:
            basis.append("風報比達標")

    if conditions["weak"]:
        score -= 12
        basis.append("跌破MA20")
    if conditions["rsi_hot"]:
        score -= 12
        basis.append("RSI過熱")

    return _clamp(score, 0, 100), basis


def _daytrade_signal(score: float, conditions: dict[str, bool]) -> str:
    strong_gate = (
        conditions["vwap_up"]
        and conditions["vol_up"]
        and not conditions["weak"]
        and not conditions["rsi_hot"]
        and (conditions["macd_up"] or conditions["trend_up"])
    )
    if score >= 78 and strong_gate:
        return "🟢 強勢當沖"
    if score < 45 or conditions["weak"] or conditions["rsi_hot"]:
        return "🔴 避開"
    if score >= 60 and conditions["vwap_up"]:
        return "🟡 觀察"
    return "⚪ 等待"


def _telegram_status(config: dict[str, object], include_secret: bool = False) -> dict[str, object]:
    telegram_config = config.get("telegram", {}) if isinstance(config.get("telegram"), dict) else {}
    enabled = bool(telegram_config.get("enabled"))
    token_env = str(telegram_config.get("bot_token_env", "TELEGRAM_BOT_TOKEN"))
    chat_env = str(telegram_config.get("chat_id_env", "TELEGRAM_CHAT_ID"))
    bot_token = os.getenv(token_env, "")
    chat_id = os.getenv(chat_env, "")
    ready = enabled and bool(bot_token) and bool(chat_id)
    if not enabled:
        reason = "Telegram 尚未啟用，請把 configs/example.yml 的 telegram.enabled 改成 true。"
    elif not bot_token or not chat_id:
        reason = f"缺少環境變數：{token_env if not bot_token else ''} {chat_env if not chat_id else ''}".strip()
    else:
        reason = "Telegram 設定完成，可手動推播。"

    status: dict[str, object] = {
        "enabled": enabled,
        "ready": ready,
        "reason": reason,
        "bot_token_env": token_env,
        "chat_id_env": chat_env,
        "bot_token_present": bool(bot_token),
        "chat_id_present": bool(chat_id),
    }
    if include_secret:
        status["bot_token"] = bot_token
        status["chat_id"] = chat_id
    return status


def _format_daily_notification_message(
    top_signals: list[dict[str, object]],
    metrics: dict[str, object],
    latest_date: str,
    signal_counts: dict[object, int],
    ai_signal_counts: dict[object, int],
) -> str:
    lines = [
        "📡 股智雷達 AI｜每日台股訊號",
        f"資料日期：{latest_date or '—'}",
        (
            "回測摘要："
            f"CAGR {_display_percent(_number(metrics.get('cagr')))}｜"
            f"Sharpe {_display(_number(metrics.get('sharpe')), 2)}｜"
            f"MDD {_display_percent(_number(metrics.get('max_drawdown')))}"
        ),
        (
            "綜合燈號："
            f"🟢 {signal_counts.get('GREEN', 0)} 強勢｜"
            f"🟡 {signal_counts.get('YELLOW', 0)} 觀察｜"
            f"🔴 {signal_counts.get('RED', 0)} 弱勢"
        ),
        "",
        "🏆 推薦評分 Top 10",
    ]
    for index, row in enumerate(top_signals, start=1):
        name = str(row.get("stock_name") or row.get("ticker") or "")
        score = _number(row.get("recommendation_score"))
        composite = row.get("composite_signal", {})
        composite_text = composite.get("text", "🟡 觀察") if isinstance(composite, dict) else "🟡 觀察"
        lines.append(
            f"{index}. {name}｜{_display(score, 0)}分｜{composite_text}｜"
            f"現價 {_display(_number(row.get('close')), 2)}｜"
            f"停損 {_display(_number(row.get('stop_loss')), 2)}｜"
            f"停利 {_display(_number(row.get('take_profit')), 2)}"
        )
    lines.extend(
        [
            "",
            "提醒：此為量化研究訊號，不是投資建議；請搭配自身風險控管。",
        ]
    )
    return "\n".join(lines)


def _enrich_signal_record(
    record: dict[str, object],
    feature_lookup: dict[str, dict[str, object]],
    group_lookup: dict[str, str] | None = None,
) -> dict[str, object]:
    enriched = record.copy()
    ticker = str(enriched.get("ticker"))
    enriched.update(feature_lookup.get(ticker, {}))
    enriched["stock_group"] = enriched.get("stock_group") or (group_lookup or {}).get(ticker, "未分類")
    _normalize_probability_fields(enriched)
    enriched["recommendation_score"] = _recommendation_score(enriched)
    enriched["recommendation_score_detail"] = _recommendation_score_detail(enriched)
    enriched["composite_signal"] = _composite_signal(enriched)
    return enriched


def _apply_intraday_signal_snapshot(
    record: dict[str, object],
    snapshot: dict[str, object],
    risk_config: dict[str, object],
    source: str,
    warning: str,
    updated_at: str,
) -> dict[str, object]:
    enriched = record.copy()
    analysis_close = _number(enriched.get("close"))
    current = _number(snapshot.get("current_price")) or analysis_close
    if current is None:
        return enriched

    intraday_volume = _number(snapshot.get("intraday_volume"))
    volume_ratio = _number(snapshot.get("volume_ratio"))
    intraday_return = _number(snapshot.get("intraday_return"))
    vwap = _number(snapshot.get("vwap"))
    vwap_gap = _number(snapshot.get("vwap_gap"))

    enriched["analysis_close"] = analysis_close
    enriched["current_price"] = round(current, 2)
    enriched["close"] = round(current, 2)
    enriched["entry_price"] = round(current, 2)
    enriched["intraday_source"] = source
    enriched["intraday_updated_at"] = updated_at
    enriched["intraday_warning"] = warning
    enriched["intraday_synced"] = source == "Yahoo Finance 1m"
    if intraday_return is not None:
        enriched["intraday_return"] = intraday_return
        enriched["daily_return"] = intraday_return
    if intraday_volume is not None:
        enriched["intraday_volume"] = intraday_volume
        enriched["volume"] = intraday_volume
    if volume_ratio is not None:
        enriched["volume_ratio"] = volume_ratio
        enriched["volume_ratio_20"] = volume_ratio
    if vwap is not None:
        enriched["intraday_vwap"] = vwap
    if vwap_gap is not None:
        enriched["vwap_gap"] = vwap_gap

    _refresh_intraday_risk_prices(enriched, risk_config)
    enriched["recommendation_score"] = _recommendation_score(enriched)
    enriched["recommendation_score_detail"] = _recommendation_score_detail(enriched)
    enriched["composite_signal"] = _composite_signal(enriched)
    return enriched


def _refresh_intraday_risk_prices(row: dict[str, object], risk_config: dict[str, object]) -> None:
    current = _number(row.get("close"))
    if current is None:
        return

    atr_multiplier = _number(risk_config.get("atr_stop_multiplier")) or 2.0
    take_profit_r_multiple = _number(risk_config.get("take_profit_r_multiple")) or 2.0
    capital = _number(risk_config.get("capital")) or 0
    risk_per_trade = _number(risk_config.get("risk_per_trade")) or 0
    max_position_pct = _number(risk_config.get("max_position_pct")) or 0
    lot_size = int(_number(risk_config.get("lot_size")) or 1000)
    atr = _number(row.get("atr_14"))
    risk_per_share = None if atr is None else atr * atr_multiplier
    if risk_per_share is None or risk_per_share <= 0:
        original_risk = _number(row.get("risk_per_share"))
        original_entry = _number(row.get("entry_price")) or _number(row.get("analysis_close"))
        if original_risk is not None and original_entry not in {None, 0}:
            risk_per_share = current * (original_risk / original_entry)
    if risk_per_share is None or risk_per_share <= 0:
        risk_per_share = current * 0.03

    stop_loss = max(0, current - risk_per_share)
    take_profit = current + risk_per_share * take_profit_r_multiple
    row["risk_per_share"] = round(risk_per_share, 2)
    row["stop_loss"] = round(stop_loss, 2)
    row["take_profit"] = round(take_profit, 2)

    if str(row.get("ml_signal", "HOLD")) != "BUY" or risk_per_share <= 0 or current <= 0:
        row["position_size"] = 0
        row["position_value"] = 0
        row["portfolio_weight"] = 0
        return

    risk_budget = capital * risk_per_trade
    risk_shares = risk_budget // risk_per_share if risk_budget > 0 else 0
    capital_shares = (capital * max_position_pct) // current if capital > 0 and max_position_pct > 0 else 0
    raw_size = int(min(risk_shares, capital_shares))
    position_size = raw_size // lot_size * lot_size if lot_size > 0 else raw_size
    row["position_size"] = int(max(position_size, 0))
    row["position_value"] = round(row["position_size"] * current, 2)
    row["portfolio_weight"] = None if capital <= 0 else row["position_value"] / capital


def _normalize_probability_fields(row: dict[str, object]) -> None:
    prob_up = _number(row.get("prob_up"))
    prob_down = _number(row.get("prob_down"))
    if prob_up is None and prob_down is not None:
        prob_up = 1 - _clamp(prob_down, 0, 1)
    if prob_up is None:
        row["prob_up"] = None
        row["prob_down"] = None
        row["ai_edge"] = None
        row["ai_confidence"] = None
        row["ai_signal_text"] = "⚪ 無資料"
        return

    prob_up = _clamp(prob_up, 0, 1)
    prob_down = 1 - prob_up
    edge = prob_up - prob_down
    row["prob_up"] = prob_up
    row["prob_down"] = prob_down
    row["ai_edge"] = edge
    row["ai_confidence"] = abs(edge)
    row["ai_signal_text"] = _ai_signal_text(prob_up)
    row["ml_signal"] = _ai_signal_code(prob_up)


def _ai_signal_code(prob_up: float | None) -> str:
    if prob_up is None:
        return "HOLD"
    if prob_up >= AI_BUY_THRESHOLD:
        return "BUY"
    if prob_up <= AI_SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _ai_signal_text(prob_up: float | None) -> str:
    if prob_up is None:
        return "⚪ 無資料"
    if prob_up >= AI_BUY_THRESHOLD:
        return "🟢 AI買進"
    if prob_up <= AI_SELL_THRESHOLD:
        return "🔴 AI賣出"
    if prob_up >= 0.52:
        return "🟡 AI偏多"
    if prob_up <= 0.48:
        return "🟡 AI偏空"
    return "🟡 AI中性"


def _sort_signal_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    signal_priority = {"GREEN": 2, "YELLOW": 1, "RED": 0}
    return sorted(
        records,
        key=lambda row: (
            signal_priority.get(str((row.get("composite_signal") or {}).get("code")), 0)
            if isinstance(row.get("composite_signal"), dict)
            else 0,
            _number(row.get("recommendation_score")) or 0,
            _number(row.get("prob_up")) or 0,
        ),
        reverse=True,
    )


def _composite_signal(row: dict[str, object]) -> dict[str, str]:
    score = _number(row.get("recommendation_score"))
    if score is None:
        score = _recommendation_score(row)
    prob_up = _number(row.get("prob_up")) or 0
    detail = row.get("recommendation_score_detail")
    if not isinstance(detail, dict):
        detail = _recommendation_score_detail(row)
    conditions = detail.get("conditions", {}) if isinstance(detail, dict) else {}
    weak = bool(conditions.get("weak"))
    rsi_hot = bool(conditions.get("rsi_hot"))
    ml_signal = str(row.get("ml_signal", "HOLD"))

    if ml_signal == "SELL" or score < 45 or prob_up <= AI_SELL_THRESHOLD or (weak and score < 60):
        code = "RED"
        text = "🔴 弱勢"
        tone = "negative"
    elif score >= 75 and prob_up >= 0.50 and not weak and not rsi_hot:
        code = "GREEN"
        text = "🟢 強勢"
        tone = "positive"
    else:
        code = "YELLOW"
        text = "🟡 觀察"
        tone = "neutral"

    basis = (
        f"綜合評分 {score:.0f}；"
        "已整合模型機率、趨勢、量能與風報比，不單看原始機率。"
    )
    return {"code": code, "text": text, "tone": tone, "basis": basis}


def _technical_payload(ticker: str, mode: str) -> dict[str, object]:
    output_dir, _ = _paths(mode)
    features_path = output_dir / "features.csv"
    if not features_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {features_path}")

    features = pd.read_csv(features_path)
    stock = features[features["ticker"].eq(ticker)].copy()
    if stock.empty:
        raise HTTPException(status_code=404, detail=f"Ticker not found: {ticker}")

    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")
    stock = stock.sort_values("date")
    close = pd.to_numeric(stock.get("close"), errors="coerce")
    if "sma_120" not in stock.columns:
        stock["sma_120"] = close.rolling(120).mean()
    stock["prev_high_20"] = pd.to_numeric(stock.get("high"), errors="coerce").rolling(20).max().shift(1)
    stock["prev_high_60"] = pd.to_numeric(stock.get("high"), errors="coerce").rolling(60).max().shift(1)
    stock["volume_ma_20"] = pd.to_numeric(stock.get("volume"), errors="coerce").rolling(20).mean()
    latest = stock.iloc[-1]
    risk = _latest_row(output_dir / "latest_risk_plan.csv", ticker)
    history_fields = ["date", "close", "sma_20", "sma_60", "sma_120", "vwap_20"]
    history = _records_from_frame(stock[history_fields].tail(120))

    market_profile = load_market_profile(ticker, output_dir / "market_profile_cache")

    return {
        "ticker": ticker,
        "stock_name": str(latest.get("stock_name", ticker)),
        "latest_date": _date_text(latest.get("date")),
        "summary_cards": _technical_cards(latest, risk),
        "indicators": _technical_indicators(latest),
        "market_profile": _market_profile(latest, risk, market_profile),
        "risk": _risk_summary(risk, latest),
        "history": history,
    }


def _chart_payload(ticker: str, mode: str, interval: str, period: str) -> dict[str, object]:
    normalized_interval = interval if interval in {"1d", "1m", "5m", "30m", "60m"} else "1d"
    normalized_period = _normalize_chart_period(normalized_interval, period)
    output_dir, _ = _paths(mode)

    if normalized_interval == "1d":
        frame = _daily_chart_frame(output_dir, ticker, normalized_period)
        source = "features.csv"
    else:
        frame = _intraday_chart_frame(ticker, normalized_interval, normalized_period)
        source = "Yahoo Finance"

    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No chart data: {ticker}")

    return {
        "ticker": ticker,
        "interval": normalized_interval,
        "period": normalized_period,
        "source": source,
        "candles": _chart_records(frame, normalized_interval),
    }


def _daily_chart_frame(output_dir: Path, ticker: str, period: str) -> pd.DataFrame:
    features_path = output_dir / "features.csv"
    if not features_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {features_path}")

    features = pd.read_csv(features_path)
    frame = features[features["ticker"].eq(ticker)].copy()
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"Ticker not found: {ticker}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date")
    start = _period_start(frame["date"].max(), period)
    if start is not None:
        frame = frame[frame["date"] >= start]
    return _add_chart_indicators(frame)


def _intraday_chart_frame(ticker: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise HTTPException(status_code=404, detail=f"No intraday data returned for {ticker}")

    raw = _flatten_yfinance_columns(raw)
    frame = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HTTPException(status_code=502, detail=f"Intraday data missing columns: {missing}")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame[frame["close"].gt(0)].sort_values("date")
    return _add_chart_indicators(frame)


def _flatten_yfinance_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    price_columns = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if price_columns.intersection(set(raw.columns.get_level_values(0))):
        return raw.droplevel(1, axis=1) if raw.columns.nlevels > 1 else raw
    if price_columns.intersection(set(raw.columns.get_level_values(-1))):
        return raw.droplevel(0, axis=1)
    return raw


def _add_chart_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    chart = frame.copy()
    close = pd.to_numeric(chart["close"], errors="coerce")
    high = pd.to_numeric(chart["high"], errors="coerce")
    low = pd.to_numeric(chart["low"], errors="coerce")
    volume = pd.to_numeric(chart.get("volume", 0), errors="coerce")
    chart["sma_20"] = chart.get("sma_20", close.rolling(20).mean())
    chart["sma_60"] = chart.get("sma_60", close.rolling(60).mean())
    chart["sma_120"] = chart.get("sma_120", close.rolling(120).mean())
    chart["ema_12"] = close.ewm(span=12, adjust=False).mean()
    chart["ema_26"] = close.ewm(span=26, adjust=False).mean()
    chart["macd"] = chart.get("macd", chart["ema_12"] - chart["ema_26"])
    chart["macd_signal"] = chart.get("macd_signal", pd.to_numeric(chart["macd"], errors="coerce").ewm(span=9, adjust=False).mean())
    chart["macd_hist"] = chart.get("macd_hist", chart["macd"] - chart["macd_signal"])
    chart["volume"] = volume.fillna(0)
    chart["high"] = high
    chart["low"] = low
    return chart


def _chart_records(frame: pd.DataFrame, interval: str) -> list[dict[str, object]]:
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_20",
        "sma_60",
        "sma_120",
        "macd",
        "macd_signal",
        "macd_hist",
    ]
    rows = []
    for _, row in frame[columns].tail(600).iterrows():
        rows.append(
            {
                "date": _chart_date(row["date"], interval),
                "open": _number(row["open"]),
                "high": _number(row["high"]),
                "low": _number(row["low"]),
                "close": _number(row["close"]),
                "volume": _number(row["volume"]),
                "sma20": _number(row["sma_20"]),
                "sma60": _number(row["sma_60"]),
                "sma120": _number(row["sma_120"]),
                "macd": _number(row["macd"]),
                "macdSignal": _number(row["macd_signal"]),
                "macdHist": _number(row["macd_hist"]),
            }
        )
    return rows


def _normalize_chart_period(interval: str, period: str) -> str:
    allowed = {
        "1d": {"3mo", "6mo", "1y", "2y", "5y", "all"},
        "1m": {"1d", "5d", "7d"},
        "5m": {"1d", "5d", "1mo", "60d"},
        "30m": {"5d", "1mo", "60d"},
        "60m": {"1mo", "3mo", "6mo", "1y", "2y"},
    }
    defaults = {"1d": "6mo", "1m": "1d", "5m": "5d", "30m": "1mo", "60m": "3mo"}
    return period if period in allowed[interval] else defaults[interval]


def _period_start(latest: pd.Timestamp, period: str) -> pd.Timestamp | None:
    if period == "all":
        return None
    offsets = {
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "5y": pd.DateOffset(years=5),
    }
    offset = offsets.get(period)
    return None if offset is None else latest - offset


def _chart_date(value: object, interval: str) -> str:
    timestamp = pd.Timestamp(value)
    if interval == "1d":
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.strftime("%m-%d %H:%M")


def _technical_cards(row: pd.Series, risk: pd.Series | None) -> list[dict[str, object]]:
    close = _number(row.get("close"))
    sma20 = _number(row.get("sma_20"))
    sma60 = _number(row.get("sma_60"))
    sma120 = _number(row.get("sma_120"))
    ema12 = _number(row.get("ema_12"))
    ema26 = _number(row.get("ema_26"))
    rsi = _number(row.get("rsi_14"))
    kd_k = _number(row.get("kd_k"))
    kd_d = _number(row.get("kd_d"))
    macd = _number(row.get("macd"))
    macd_signal = _number(row.get("macd_signal"))
    macd_hist = _number(row.get("macd_hist"))
    bb_percent = _number(row.get("bb_percent"))
    atr = _number(row.get("atr_14"))
    volume_ratio = _number(row.get("volume_ratio_20"))
    vwap = _number(row.get("vwap_20"))
    beta = _number(row.get("beta_60"))
    composite_row = row.to_dict()
    if risk is not None:
        composite_row.update(risk.to_dict())
    _normalize_probability_fields(composite_row)
    composite_row["recommendation_score"] = _recommendation_score(composite_row)
    composite_row["recommendation_score_detail"] = _recommendation_score_detail(composite_row)
    composite_signal = _composite_signal(composite_row)

    cards = [
        _card("綜合燈號", composite_signal["text"], "五構面評分 + AI 機率綜合判斷", composite_signal["tone"]),
        _card("趨勢結構", *_trend_status(close, sma20, sma60)),
        _card("長線均線", _display(sma120, 2), *_long_ma_status(close, sma120)),
        _card("短線均線", *_ema_status(ema12, ema26)),
        _card("RSI", _display(rsi, 1), *_rsi_status(rsi)),
        _card("KD", f"K {_display(kd_k, 1)} / D {_display(kd_d, 1)}", *_kd_status(kd_k, kd_d)),
        _card("MACD", _display(macd_hist, 2), *_macd_status(macd, macd_signal)),
        _card("布林通道", _display_percent(bb_percent), *_bollinger_status(bb_percent)),
        _card("ATR 波動", _display_percent(None if close in {None, 0} or atr is None else atr / close), "波動占股價比例", "neutral"),
        _card("量能", f"{_display(volume_ratio, 2)} 倍", *_volume_status(volume_ratio)),
        _card("VWAP", _display(vwap, 2), *_vwap_status(close, vwap)),
        _card("Beta", _display(beta, 2), *_beta_status(beta)),
    ]
    return cards


def _technical_indicators(row: pd.Series) -> list[dict[str, object]]:
    indicators = []
    for field in TECHNICAL_FIELDS:
        value = _number(row.get(field))
        indicators.append(
            {
                "field": field,
                "label": feature_label(field),
                "value": value,
                "display": _indicator_display(field, value),
            }
        )
    return indicators


def _market_profile(
    row: pd.Series,
    risk: pd.Series | None,
    market_data: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    data = row.to_dict()
    if risk is not None:
        data.update(risk.to_dict())
    _normalize_probability_fields(data)
    data["recommendation_score"] = _recommendation_score(data)
    data["recommendation_score_detail"] = _recommendation_score_detail(data)
    scorecard = _recommendation_score_detail(data)
    composite_signal = _composite_signal(data)

    close = _number(data.get("close"))
    sma5 = _number(data.get("sma_5"))
    sma20 = _number(data.get("sma_20"))
    sma60 = _number(data.get("sma_60"))
    sma120 = _number(data.get("sma_120"))
    macd = _number(data.get("macd"))
    macd_signal = _number(data.get("macd_signal"))
    rsi = _number(data.get("rsi_14"))
    atr = _number(data.get("atr_14"))
    volume = _number(data.get("volume"))
    volume_ma20 = _number(data.get("volume_ma_20"))
    volume_ratio = _number(data.get("volume_ratio_20"))
    beta = _number(data.get("beta_60"))
    prev_high20 = _number(data.get("prev_high_20"))
    prev_high60 = _number(data.get("prev_high_60"))
    entry = _number(data.get("entry_price"))
    stop_loss = _number(data.get("stop_loss"))
    take_profit = _number(data.get("take_profit"))

    trend_up = _all_numbers(close, sma20, sma60) and close > sma20 > sma60
    short_up = _all_numbers(sma5, sma20) and sma5 > sma20
    long_up = _all_numbers(close, sma120) and close > sma120
    weak = _all_numbers(close, sma20) and close < sma20
    macd_up = _all_numbers(macd, macd_signal) and macd > macd_signal
    vol_up = (
        _all_numbers(volume, volume_ma20)
        and volume > volume_ma20 * 1.5
    ) or (volume_ratio is not None and volume_ratio > 1.5)
    breakout20 = _all_numbers(close, prev_high20) and close >= prev_high20
    breakout60 = _all_numbers(close, prev_high60) and close >= prev_high60
    rsi_good = rsi is not None and 50 <= rsi <= 75
    rsi_hot = rsi is not None and rsi > 82
    external = market_data if isinstance(market_data, dict) else {}
    fundamental = external.get("fundamental") if isinstance(external.get("fundamental"), dict) else {}
    institutional = external.get("institutional") if isinstance(external.get("institutional"), dict) else {}
    margin_data = external.get("margin") if isinstance(external.get("margin"), dict) else {}
    chips = external.get("chips") if isinstance(external.get("chips"), dict) else {}
    institutional_history = _institutional_history(institutional)
    margin_history = _margin_history(margin_data)

    return [
        _profile_card(
            "技術面總覽",
            "均線、動能、突破條件",
            [
                _profile_row("收盤價", _display(close, 2), "最新行情基準", "neutral"),
                _profile_row("MA20 / MA60 / MA120", f"{_display(sma20, 2)} / {_display(sma60, 2)} / {_display(sma120, 2)}", "20日 / 60日 / 120日均線", "neutral"),
                _profile_row("距 MA20", _display_signed_percent(_gap(close, sma20)), "判斷是否跌破短中線", "negative" if weak else "positive"),
                _profile_row("距 MA60", _display_signed_percent(_gap(close, sma60)), "觀察中期趨勢強弱", "positive" if _gap(close, sma60) is not None and _gap(close, sma60) >= 0 else "neutral"),
                _profile_signal_row("多頭排列", trend_up, "Close > MA20 > MA60", "尚未形成多頭排列"),
                _profile_signal_row("短線轉強", short_up, "MA5 > MA20", "短線均線仍需觀察"),
                _profile_signal_row("長線站上", long_up, "Close > MA120", "尚未站上 120 日均線"),
            ],
        ),
        _profile_card(
            "動能與量能",
            "MACD、RSI、成交量",
            [
                _profile_signal_row("MACD 偏多", macd_up, "MACD > Signal", "MACD 尚未站上訊號線", "🟢 偏多", "🔴 未偏多"),
                _profile_row("RSI", _display(rsi, 1), "50～75 屬相對健康；82 以上視為過熱", "negative" if rsi_hot else "positive" if rsi_good else "neutral"),
                _profile_signal_row("RSI 健康區", rsi_good, "RSI 介於 50～75", "RSI 不在健康動能區", "🟢 健康", "🔴 非健康"),
                _profile_signal_row("RSI 過熱", not rsi_hot, "未進入過熱扣分區", "RSI > 82，視為過熱", "🟢 未過熱", "🔴 過熱"),
                _profile_row("量能比", f"{_display(volume_ratio, 2)} 倍", "Volume / VOL20", "positive" if vol_up else "neutral"),
                _profile_signal_row("爆量條件", vol_up, "Volume > VOL20 × 1.5", "量能未明顯放大", "🟢 放量", "🔴 未放量"),
                _profile_row("ATR 波動", _display_percent(None if close in {None, 0} or atr is None else atr / close), "用於停損與風險距離", "neutral"),
                _profile_row("Beta", _display(beta, 2), "相對大盤敏感度", "negative" if beta is not None and beta >= 1.2 else "neutral"),
            ],
        ),
        _profile_card(
            "AI 與風報比",
            "機率、評分、入場風控",
            [
                _profile_row("綜合燈號", composite_signal["text"], composite_signal["basis"], composite_signal["tone"]),
                _profile_row("推薦評分", f"{scorecard['total']} 分", scorecard["label"], "positive" if scorecard["total"] >= 75 else "negative" if scorecard["total"] < 45 else "neutral"),
                _profile_row("風報比", _display(scorecard["reward_risk_ratio"], 2), "停利距離 / 停損距離", "positive" if (scorecard["reward_risk_ratio"] or 0) >= 1.5 else "neutral"),
                _profile_row("入場 / 停損 / 停利", f"{_display(entry, 2)} / {_display(stop_loss, 2)} / {_display(take_profit, 2)}", "由風控模組依 ATR 推估", "neutral"),
                _profile_breakout_row("20日突破", close, prev_high20, breakout20),
                _profile_breakout_row("60日突破", close, prev_high60, breakout60),
            ],
        ),
        _profile_card(
            "基本面",
            "估值、殖利率與財報季",
            [
                _metric_profile_row("P/E 本益比", _display(_number(fundamental.get("pe")), 2), _source_note(fundamental, "TWSE/TPEx 本益比資料")),
                _metric_profile_row("P/B 股價淨值比", _display(_number(fundamental.get("pb")), 2), _source_note(fundamental, "TWSE/TPEx 股淨比資料")),
                _metric_profile_row("殖利率", _display_percent_points(_number(fundamental.get("dividend_yield"))), _source_note(fundamental, "TWSE/TPEx 殖利率資料"), "positive" if (_number(fundamental.get("dividend_yield")) or 0) >= 4 else "neutral"),
                _metric_profile_row("每股股利", _display(_number(fundamental.get("dividend_per_share")), 2), _source_note(fundamental, "TPEx 每股股利；TWSE 下一步接財報/股利表")),
                _metric_profile_row("財報季", str(fundamental.get("report_period") or "—"), _source_note(fundamental, "財報年季資料")),
            ],
        ),
        _profile_card(
            "法人買賣超",
            "外資、投信、自營商",
            [
                _metric_profile_row("外資買賣超", _display_signed_share_lots(_number(institutional.get("foreign_net"))), _source_note(institutional, "TWSE/TPEx 三大法人"), _signed_tone(_number(institutional.get("foreign_net")))),
                _metric_profile_row("投信買賣超", _display_signed_share_lots(_number(institutional.get("investment_trust_net"))), _source_note(institutional, "TWSE/TPEx 三大法人"), _signed_tone(_number(institutional.get("investment_trust_net")))),
                _metric_profile_row("自營商買賣超", _display_signed_share_lots(_number(institutional.get("dealer_net"))), _source_note(institutional, "TWSE/TPEx 三大法人"), _signed_tone(_number(institutional.get("dealer_net")))),
                _metric_profile_row("三大法人合計", _display_signed_share_lots(_number(institutional.get("total_net"))), f"統計日：{institutional.get('date') or '—'}｜單位換算為張", _signed_tone(_number(institutional.get("total_net")))),
                _metric_profile_row("累積資料", f"{len(institutional_history)} 日", f"目前已累積 {len(institutional_history)} 日，可切換 1/3/5/10 日檢視"),
            ],
            key="institutional",
            controls={"periods": [1, 3, 5, 10], "default": 1, "type": "institutional"},
            history=institutional_history,
        ),
        _profile_card(
            "融資融券",
            "信用交易與券資比",
            [
                _metric_profile_row("融資餘額", _display_lot_units(_number(margin_data.get("margin_balance"))), _source_note(margin_data, "TWSE/TPEx 融資融券餘額")),
                _metric_profile_row("融資日增減", _display_signed_lot_units(_number(margin_data.get("margin_change"))), "今日餘額 - 前日餘額", _signed_tone(_number(margin_data.get("margin_change")))),
                _metric_profile_row("融券餘額", _display_lot_units(_number(margin_data.get("short_balance"))), _source_note(margin_data, "TWSE/TPEx 融資融券餘額")),
                _metric_profile_row("融券日增減", _display_signed_lot_units(_number(margin_data.get("short_change"))), "今日餘額 - 前日餘額", _signed_tone(_number(margin_data.get("short_change")), positive_is_good=False)),
                _metric_profile_row("券資比", _display_percent(_number(margin_data.get("short_margin_ratio"))), "融券餘額 / 融資餘額", "negative" if (_number(margin_data.get("short_margin_ratio")) or 0) >= 0.3 else "neutral"),
                _metric_profile_row("融資使用率", _display_percent_points(_number(margin_data.get("margin_usage_rate"))), f"統計日：{margin_data.get('date') or '—'}｜TPEx 提供；TWSE 暫以餘額為主"),
                _metric_profile_row("累積資料", f"{len(margin_history)} 日", f"目前已累積 {len(margin_history)} 日，可切換 1/3/5/10 日檢視"),
            ],
            key="margin",
            controls={"periods": [1, 3, 5, 10], "default": 1, "type": "margin"},
            history=margin_history,
        ),
        _profile_card(
            "集保籌碼",
            "零股、小額與大戶持股",
            [
                _metric_profile_row("零股股東", _display_people(_number(chips.get("odd_lot_holders"))), _source_note(chips, "TDCC 集保股權分散表")),
                _metric_profile_row("零股占比", _display_percent_points(_number(chips.get("odd_lot_ratio"))), "持股分級 1"),
                _metric_profile_row("20張以下", _display_percent_points(_number(chips.get("small_holder_ratio"))), "持股分級 1～5 合計"),
                _metric_profile_row("400張以上", _display_percent_points(_number(chips.get("large_holder_ratio"))), "持股分級 12～16 合計", "positive" if (_number(chips.get("large_holder_ratio")) or 0) >= 40 else "neutral"),
                _metric_profile_row("千張以上", f"{_display_people(_number(chips.get('thousand_lot_holders')))} / {_display_percent_points(_number(chips.get('thousand_lot_ratio')))}", "持股分級 15"),
                _metric_profile_row("總股東人數", _display_people(_number(chips.get("total_holders"))), _source_note(chips, "TDCC 集保股權分散表")),
            ],
        ),
    ]


def _profile_card(
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    key: str | None = None,
    controls: dict[str, object] | None = None,
    history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    card = {"title": title, "subtitle": subtitle, "rows": rows}
    if key:
        card["key"] = key
    if controls:
        card["controls"] = controls
    if history is not None:
        card["history"] = history
    return card


def _profile_row(label: str, value: str, note: str, tone: str) -> dict[str, object]:
    return {"label": label, "value": value, "note": note, "tone": tone}


def _profile_signal_row(
    label: str,
    active: bool,
    true_note: str,
    false_note: str,
    true_value: str = "🟢 成立",
    false_value: str = "🔴 未成立",
) -> dict[str, object]:
    return _profile_row(label, true_value if active else false_value, true_note if active else false_note, "positive" if active else "negative")


def _profile_bool_row(label: str, active: bool, true_note: str, false_note: str) -> dict[str, object]:
    return _profile_row(label, "成立" if active else "未成立", true_note if active else false_note, "positive" if active else "neutral")


def _profile_breakout_row(label: str, close: float | None, previous_high: float | None, active: bool) -> dict[str, object]:
    gap = _gap(close, previous_high)
    note = f"目前 {_display(close, 2)} / 前高 {_display(previous_high, 2)} / 差距 {_display_signed_percent(gap)}"
    return _profile_row(label, "🟢 已突破" if active else "🔴 未突破", note, "positive" if active else "negative")


def _institutional_history(institutional: dict[str, object]) -> list[dict[str, object]]:
    history = institutional.get("history")
    rows = history if isinstance(history, list) else []
    if not rows and institutional.get("date"):
        rows = [institutional]
    return [
        {
            "date": row.get("date"),
            "foreign_net": _json_value(_number(row.get("foreign_net"))),
            "investment_trust_net": _json_value(_number(row.get("investment_trust_net"))),
            "dealer_net": _json_value(_number(row.get("dealer_net"))),
            "total_net": _json_value(_number(row.get("total_net"))),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def _margin_history(margin_data: dict[str, object]) -> list[dict[str, object]]:
    history = margin_data.get("history")
    rows = history if isinstance(history, list) else []
    if not rows and margin_data.get("date"):
        rows = [margin_data]
    return [
        {
            "date": row.get("date"),
            "margin_balance": _json_value(_number(row.get("margin_balance"))),
            "margin_change": _json_value(_number(row.get("margin_change"))),
            "short_balance": _json_value(_number(row.get("short_balance"))),
            "short_change": _json_value(_number(row.get("short_change"))),
            "short_margin_ratio": _json_value(_number(row.get("short_margin_ratio"))),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def _pending_profile_row(label: str, source: str) -> dict[str, object]:
    return _profile_row(label, "待接資料", f"下一階段串接：{source}", "pending")


def _metric_profile_row(label: str, value: str, note: str, tone: str = "neutral") -> dict[str, object]:
    if value in {"", "—", "None"}:
        return _profile_row(label, "待接資料", note, "pending")
    return _profile_row(label, value, note, tone)


def _source_note(data: dict[str, object], fallback: str) -> str:
    source = data.get("source")
    date = data.get("date")
    if source and date:
        return f"{source}｜{date}"
    if source:
        return str(source)
    return f"待接資料：{fallback}"


def _display_percent_points(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}%"


def _display_people(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} 人"


def _display_lot_units(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} 張"


def _display_signed_lot_units(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f} 張"


def _display_signed_share_lots(value: float | None) -> str:
    if value is None:
        return "—"
    lots = value / 1000
    sign = "+" if lots > 0 else ""
    return f"{sign}{lots:,.0f} 張"


def _signed_tone(value: float | None, positive_is_good: bool = True) -> str:
    if value is None or value == 0:
        return "neutral"
    is_positive = value > 0
    if positive_is_good:
        return "positive" if is_positive else "negative"
    return "negative" if is_positive else "positive"


def _risk_summary(risk: pd.Series | None, feature_row: pd.Series | None = None) -> dict[str, object]:
    if risk is None:
        return {}
    keys = ["entry_price", "stop_loss", "take_profit", "prob_up", "prob_down", "position_size", "ml_signal"]
    summary = {key: _json_value(risk.get(key)) for key in keys if key in risk.index}
    if feature_row is not None:
        feature_keys = [
            "close",
            "sma_5",
            "sma_20",
            "sma_60",
            "sma_120",
            "high",
            "low",
            "daily_return",
            "momentum_10",
            "atr_14",
            "vwap_20",
            "macd",
            "macd_signal",
            "volume",
            "volume_ratio_20",
            "volume_ma_20",
            "prev_high_20",
            "prev_high_60",
            "rsi_14",
        ]
        summary.update({key: _json_value(feature_row.get(key)) for key in feature_keys if key in feature_row.index})
    _normalize_probability_fields(summary)
    summary["recommendation_score"] = _recommendation_score(summary)
    summary["recommendation_score_detail"] = _recommendation_score_detail(summary)
    return summary


def _recommendation_score(row: dict[str, object]) -> int | None:
    return _recommendation_scorecard(row)["total"]


def _recommendation_score_detail(row: dict[str, object]) -> dict[str, object]:
    return _recommendation_scorecard(row)


def _recommendation_scorecard(row: dict[str, object]) -> dict[str, object]:
    prob_up = _number(row.get("prob_up"))
    ai_score = 0.0 if prob_up is None else _clamp((prob_up - 0.42) / (0.58 - 0.42), 0, 1) * 30

    close = _number(row.get("close"))
    sma5 = _number(row.get("sma_5"))
    sma20 = _number(row.get("sma_20"))
    sma60 = _number(row.get("sma_60"))
    macd = _number(row.get("macd"))
    macd_signal = _number(row.get("macd_signal"))
    volume = _number(row.get("volume"))
    volume_ma20 = _number(row.get("volume_ma_20"))
    volume_ratio = _number(row.get("volume_ratio_20"))
    prev_high20 = _number(row.get("prev_high_20"))
    prev_high60 = _number(row.get("prev_high_60"))
    rsi = _number(row.get("rsi_14"))

    trend_up = _all_numbers(close, sma20, sma60) and close > sma20 > sma60
    short_up = _all_numbers(sma5, sma20) and sma5 > sma20
    weak = _all_numbers(close, sma20) and close < sma20
    macd_up = _all_numbers(macd, macd_signal) and macd > macd_signal
    vol_up = (
        _all_numbers(volume, volume_ma20)
        and volume > volume_ma20 * 1.5
    ) or (volume_ratio is not None and volume_ratio > 1.5)
    breakout20 = _all_numbers(close, prev_high20) and close >= prev_high20
    breakout60 = _all_numbers(close, prev_high60) and close >= prev_high60
    rsi_good = rsi is not None and 50 <= rsi <= 75
    rsi_hot = rsi is not None and rsi > 82

    trend_score = min(25, (15 if trend_up else 0) + (6 if short_up else 0))
    technical_score = min(
        20,
        (8 if macd_up else 0)
        + (5 if breakout20 else 0)
        + (5 if breakout60 else 0)
        + (7 if rsi_good else 0),
    )
    volume_score = 10 if vol_up else 4 if volume_ratio is not None and volume_ratio >= 1 else 0

    entry = _number(row.get("entry_price"))
    stop_loss = _number(row.get("stop_loss"))
    take_profit = _number(row.get("take_profit"))
    reward_risk = None
    risk_reward_score = 0.0
    if entry not in {None, 0} and stop_loss is not None and take_profit is not None:
        risk = max(entry - stop_loss, 0)
        reward = max(take_profit - entry, 0)
        if risk > 0:
            reward_risk = reward / risk
            risk_reward_score = min(reward_risk, 2) / 2 * 15

    penalty = (10 if weak else 0) + (8 if rsi_hot else 0)
    total = ai_score + trend_score + technical_score + volume_score + risk_reward_score - penalty
    total = int(round(_clamp(total, 0, 100)))
    return {
        "total": total,
        "label": _recommendation_label(total),
        "weights": {
            "ai_probability": round(ai_score, 1),
            "trend": round(trend_score, 1),
            "technical": round(technical_score, 1),
            "volume": round(volume_score, 1),
            "reward_risk": round(risk_reward_score, 1),
            "penalty": -penalty,
        },
        "conditions": {
            "trend_up": bool(trend_up),
            "short_up": bool(short_up),
            "macd_up": bool(macd_up),
            "vol_up": bool(vol_up),
            "breakout20": bool(breakout20),
            "breakout60": bool(breakout60),
            "rsi_good": bool(rsi_good),
            "rsi_hot": bool(rsi_hot),
            "weak": bool(weak),
        },
        "reward_risk_ratio": None if reward_risk is None else round(reward_risk, 2),
    }


def _latest_score_feature_lookup(output_dir: Path) -> dict[str, dict[str, object]]:
    features_path = output_dir / "features.csv"
    if not features_path.exists():
        return {}

    features = pd.read_csv(features_path)
    if features.empty or "ticker" not in features.columns:
        return {}

    rows: dict[str, dict[str, object]] = {}
    for ticker, part in features.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        enriched = part.copy()
        enriched["prev_high_20"] = pd.to_numeric(enriched.get("high"), errors="coerce").rolling(20).max().shift(1)
        enriched["prev_high_60"] = pd.to_numeric(enriched.get("high"), errors="coerce").rolling(60).max().shift(1)
        enriched["volume_ma_20"] = pd.to_numeric(enriched.get("volume"), errors="coerce").rolling(20).mean()
        latest = enriched.iloc[-1]
        keys = [
            "sma_5",
            "sma_20",
            "sma_60",
            "sma_120",
            "close",
            "high",
            "low",
            "daily_return",
            "momentum_10",
            "atr_14",
            "vwap_20",
            "macd",
            "macd_signal",
            "volume",
            "volume_ratio_20",
            "volume_ma_20",
            "prev_high_20",
            "prev_high_60",
            "rsi_14",
        ]
        rows[str(ticker)] = {key: _json_value(latest.get(key)) for key in keys if key in latest.index}
    return rows


def _all_numbers(*values: float | None) -> bool:
    return all(value is not None for value in values)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _recommendation_label(score: int) -> str:
    if score >= 75:
        return "強勢"
    if score >= 60:
        return "偏強"
    if score >= 45:
        return "觀察"
    return "偏弱"


def _latest_row(path: Path, ticker: str) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    rows = frame[frame["ticker"].eq(ticker)].copy()
    if rows.empty:
        return None
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    return rows.sort_values("date").iloc[-1]


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return json.loads(output.to_json(orient="records", date_format="iso"))


def _card(title: str, value: str, status: str, tone: str) -> dict[str, str]:
    return {"title": title, "value": value, "status": status, "tone": tone}


def _trend_status(close: float | None, sma20: float | None, sma60: float | None) -> tuple[str, str, str]:
    if None in {close, sma20, sma60}:
        return "資料不足", "需至少 60 日資料", "neutral"
    if close > sma20 > sma60:
        return "多頭排列", "股價站上 20 日與 60 日均線", "positive"
    if close < sma20 < sma60:
        return "空頭排列", "股價低於 20 日與 60 日均線", "negative"
    return "盤整觀察", "均線尚未形成明確排列", "neutral"


def _long_ma_status(close: float | None, sma120: float | None) -> tuple[str, str]:
    if None in {close, sma120}:
        return "需至少 120 日資料", "neutral"
    if close > sma120:
        return "股價站上 120 日均線", "positive"
    return "股價跌破 120 日均線", "negative"


def _ema_status(ema12: float | None, ema26: float | None) -> tuple[str, str, str]:
    if None in {ema12, ema26}:
        return "資料不足", "短線 EMA 無法判斷", "neutral"
    if ema12 > ema26:
        return "短線偏多", "12 日 EMA 高於 26 日 EMA", "positive"
    return "短線偏弱", "12 日 EMA 低於 26 日 EMA", "negative"


def _rsi_status(rsi: float | None) -> tuple[str, str]:
    if rsi is None:
        return "RSI 資料不足", "neutral"
    if rsi >= 70:
        return "偏熱，留意拉回", "negative"
    if rsi <= 30:
        return "偏冷，可能超賣", "positive"
    return "動能中性", "neutral"


def _kd_status(kd_k: float | None, kd_d: float | None) -> tuple[str, str]:
    if None in {kd_k, kd_d}:
        return "KD 資料不足", "neutral"
    if kd_k > kd_d and kd_k < 80:
        return "黃金交叉偏多", "positive"
    if kd_k < kd_d and kd_k > 20:
        return "死亡交叉偏弱", "negative"
    if kd_k >= 80:
        return "高檔鈍化/過熱", "neutral"
    return "低檔整理/超賣", "neutral"


def _macd_status(macd: float | None, macd_signal: float | None) -> tuple[str, str]:
    if None in {macd, macd_signal}:
        return "MACD 資料不足", "neutral"
    if macd > macd_signal:
        return "MACD 偏多", "positive"
    return "MACD 偏空", "negative"


def _bollinger_status(bb_percent: float | None) -> tuple[str, str]:
    if bb_percent is None:
        return "布林資料不足", "neutral"
    if bb_percent > 1:
        return "突破上緣，偏熱", "negative"
    if bb_percent < 0:
        return "跌破下緣，偏弱", "negative"
    if bb_percent >= 0.8:
        return "接近上緣", "positive"
    if bb_percent <= 0.2:
        return "接近下緣", "neutral"
    return "通道中段", "neutral"


def _volume_status(volume_ratio: float | None) -> tuple[str, str]:
    if volume_ratio is None:
        return "量能資料不足", "neutral"
    if volume_ratio >= 1.5:
        return "量能放大", "positive"
    if volume_ratio <= 0.7:
        return "量能收縮", "neutral"
    return "量能正常", "neutral"


def _vwap_status(close: float | None, vwap: float | None) -> tuple[str, str]:
    if None in {close, vwap}:
        return "VWAP 資料不足", "neutral"
    if close > vwap:
        return "站上 VWAP", "positive"
    return "低於 VWAP", "negative"


def _beta_status(beta: float | None) -> tuple[str, str]:
    if beta is None:
        return "Beta 資料不足", "neutral"
    if beta >= 1.2:
        return "高波動，較敏感", "negative"
    if beta <= 0.8:
        return "低波動，較防守", "positive"
    return "接近大盤波動", "neutral"


def _signal_text(signal: str | None) -> str:
    return {"BUY": "🟢 買進", "HOLD": "🟡 觀望", "SELL": "🔴 賣出"}.get(signal or "HOLD", "🟡 觀望")


def _signal_tone(signal: str | None) -> str:
    return {"BUY": "positive", "SELL": "negative", "HOLD": "neutral"}.get(signal or "HOLD", "neutral")


def _indicator_display(field: str, value: float | None) -> str:
    if field in {"daily_return", "benchmark_return", "sentiment_score", "bb_width", "bb_percent", "momentum_10"}:
        return _display_percent(value)
    if field in {"obv", "volume"}:
        return _display(value, 0)
    return _display(value, 2)


def _display(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def _display_percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _display_signed_percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1%}"


def _gap(value: float | None, base: float | None) -> float | None:
    if value is None or base in {None, 0}:
        return None
    return value / base - 1


def _number(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _json_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, int | float | str | bool):
        return value
    return str(value)


def _date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _index_html() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台股 AI 量化分析系統</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f8ff;
      --panel: rgba(255, 255, 255, .95);
      --line: rgba(37, 99, 235, .13);
      --text: #0f1f35;
      --muted: #64748b;
      --cyan: #0284c7;
      --blue: #2563eb;
      --navy: #061b3a;
      --green: #16a34a;
      --yellow: #d97706;
      --red: #dc2626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 6%, rgba(56,189,248,.26), transparent 34%),
        radial-gradient(circle at 88% 0%, rgba(37,99,235,.15), transparent 30%),
        linear-gradient(135deg, #ffffff, #eef6ff 58%, #f8fbff);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
    }
    .app-shell { width: min(1480px, calc(100% - 36px)); margin: 0 auto; padding: 28px 0; display: grid; grid-template-columns: 210px 1fr; gap: 18px; }
    .sidebar {
      position: sticky;
      top: 24px;
      height: calc(100vh - 56px);
      border-radius: 30px;
      padding: 20px 16px;
      color: #dbeafe;
      background: linear-gradient(180deg, #061b3a, #0b2a58);
      box-shadow: 0 24px 55px rgba(15, 23, 42, .18);
    }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 24px; font-weight: 950; letter-spacing: -.03em; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 14px; background: linear-gradient(135deg, #38bdf8, #2563eb); box-shadow: 0 12px 24px rgba(37,99,235,.34); }
    .brand small { display: block; color: #7dd3fc; font-size: 12px; letter-spacing: .08em; }
    .nav-list { display: grid; gap: 8px; }
    .nav-item { display: flex; align-items: center; gap: 10px; padding: 12px; border-radius: 16px; color: #bfdbfe; text-decoration: none; font-weight: 850; }
    .nav-item.active, .nav-item:hover { color: #ffffff; background: rgba(59,130,246,.36); }
    .nav-icon { width: 26px; height: 26px; display: grid; place-items: center; border-radius: 10px; background: rgba(255,255,255,.12); }
    .sidebar-note { margin-top: 22px; padding: 14px; border-radius: 18px; color: #bfdbfe; background: rgba(255,255,255,.08); font-size: 13px; line-height: 1.6; }
    .shell { min-width: 0; padding: 0; }
    .hero, .panel, .card {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 18px 48px rgba(37, 99, 235, .10);
      backdrop-filter: blur(18px);
    }
    .hero { border-radius: 30px; padding: 30px; display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 24px; align-items: center; overflow: hidden; position: relative; }
    .hero::after { content: ""; position: absolute; inset: auto -80px -110px auto; width: 380px; height: 260px; border-radius: 999px; background: radial-gradient(circle, rgba(14,165,233,.22), transparent 68%); pointer-events: none; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 900; margin: 0 0 8px; }
    h1 { font-size: clamp(38px, 5vw, 64px); letter-spacing: -.06em; line-height: .95; margin: 0 0 14px; color: #071b36; }
    h2 { margin: 0; letter-spacing: -.02em; }
    p { color: var(--muted); line-height: 1.7; margin: 0; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
    button, select, a.button {
      border: 1px solid rgba(37,99,235,.18);
      border-radius: 999px;
      color: var(--text);
      background: #eff6ff;
      padding: 11px 16px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
	    }
	    button.primary { color: #ffffff; border-color: transparent; background: linear-gradient(135deg, #2563eb, #06b6d4); box-shadow: 0 12px 26px rgba(37,99,235,.22); }
	    .auto-refresh-button.active { color: #ffffff; border-color: transparent; background: linear-gradient(135deg, #16a34a, #06b6d4); }
	    .hero-art { position: relative; z-index: 1; display: grid; gap: 12px; }
    .radar-card { border: 1px solid rgba(37,99,235,.12); border-radius: 24px; padding: 18px; background: linear-gradient(135deg, rgba(239,246,255,.9), rgba(255,255,255,.98)); }
    .radar { height: 150px; border-radius: 22px; background:
      radial-gradient(circle at 70% 42%, rgba(14,165,233,.24), transparent 34%),
      repeating-radial-gradient(circle at 70% 42%, rgba(37,99,235,.16) 0 1px, transparent 1px 26px),
      linear-gradient(135deg, #ffffff, #e0f2fe);
      position: relative; overflow: hidden;
    }
    .radar::before { content: ""; position: absolute; inset: 20px 40px 20px auto; width: 130px; transform-origin: 20% 50%; background: linear-gradient(90deg, rgba(14,165,233,.55), transparent); clip-path: polygon(0 50%, 100% 0, 100% 100%); }
    .hero-stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .hero-stat { border-radius: 18px; padding: 12px; background: #f8fbff; border: 1px solid rgba(37,99,235,.10); }
    .hero-stat span { display: block; color: var(--muted); font-size: 12px; }
    .hero-stat strong { display: block; margin-top: 5px; color: #0f1f35; font-size: 20px; }
    .grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin: 16px 0; }
    .card { border-radius: 22px; padding: 18px; }
    .card span { display: block; color: var(--muted); font-size: 13px; }
    .card strong { display: block; margin-top: 8px; font-size: 25px; letter-spacing: -.03em; }
    .panel { border-radius: 28px; padding: 22px; margin-top: 16px; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 16px; }
    .status { color: var(--muted); font-size: 14px; }
    .title-row { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
    .score-note { margin-top: 8px; font-size: 13px; color: var(--muted); }
    .filter-panel { margin-top: 16px; }
    .filter-bar { display: flex; flex-wrap: wrap; gap: 8px; }
    .filter-chip {
      border-color: rgba(37,99,235,.16);
      color: #075985;
      background: #f8fbff;
      box-shadow: none;
      padding: 9px 12px;
    }
    .filter-chip.active {
      color: #ffffff;
      border-color: transparent;
      background: linear-gradient(135deg, #2563eb, #06b6d4);
      box-shadow: 0 12px 24px rgba(37,99,235,.18);
    }
    .filter-chip small { color: inherit; opacity: .72; font-weight: 900; }
    .freshness {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid rgba(53,208,127,.28);
      border-radius: 999px;
      color: #15803d;
      background: #dcfce7;
      padding: 6px 10px;
      font-size: 13px;
      font-weight: 900;
    }
    .stock-link { color: #075985; font-weight: 950; text-decoration: none; border-bottom: 1px solid rgba(14,165,233,.45); }
    .stock-link:hover { color: var(--blue); }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 13px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .table-wrap { overflow-x: auto; }
	    .badge { border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 900; }
	    .BUY { color: #15803d; background: #dcfce7; }
	    .HOLD { color: #a16207; background: #fef3c7; }
	    .SELL { color: #b91c1c; background: #fee2e2; }
	    .GREEN { color: #15803d; background: #dcfce7; }
	    .YELLOW { color: #a16207; background: #fef3c7; }
	    .RED { color: #b91c1c; background: #fee2e2; }
	    .positive { color: #15803d; background: #dcfce7; }
	    .neutral { color: #a16207; background: #fef3c7; }
	    .negative { color: #b91c1c; background: #fee2e2; }
	    .score-pill { border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 900; color: #1d4ed8; background: #dbeafe; }
	    .score-pill.high { color: #15803d; background: #dcfce7; }
	    .score-pill.low { color: #b91c1c; background: #fee2e2; }
	    .split { display: grid; grid-template-columns: 1.25fr .75fr; gap: 16px; }
	    .best { color: #15803d; font-weight: 900; }
	    .importance { display: grid; gap: 12px; }
	    .importance-row { display: grid; gap: 8px; }
	    .importance-meta { display: flex; justify-content: space-between; gap: 12px; color: var(--text); }
	    .bar { height: 9px; overflow: hidden; border-radius: 999px; background: #e2e8f0; }
	    .bar span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #2563eb, #06b6d4); }
	    @media (max-width: 1120px) { .app-shell { grid-template-columns: 1fr; } .sidebar { position: static; height: auto; } .nav-list { grid-template-columns: repeat(3, 1fr); } .hero { grid-template-columns: 1fr; } }
	    @media (max-width: 980px) { .hero { display: block; } .hero-art { margin-top: 18px; } .actions { margin-top: 18px; justify-content: flex-start; } .grid, .split { grid-template-columns: repeat(2, 1fr); } }
	    @media (max-width: 620px) { .app-shell { width: min(100% - 22px, 1480px); } .grid, .split, .nav-list { grid-template-columns: 1fr; } }
	  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">AI</span>
        <div>股智雷達 <small>TAIWAN QUANT</small></div>
      </div>
      <nav class="nav-list" aria-label="主要功能">
	        <a class="nav-item active" href="/"><span class="nav-icon">⌂</span>看盤首頁</a>
	        <a class="nav-item" href="#groupFilterPanel"><span class="nav-icon">▤</span>組群篩選</a>
	        <a class="nav-item" href="#daytradePanel"><span class="nav-icon">⚡</span>當沖雷達</a>
	        <a class="nav-item" href="#signalsPanel"><span class="nav-icon">◎</span>今日訊號</a>
        <a class="nav-item" href="/backtest"><span class="nav-icon">↗</span>回測中心</a>
        <a class="nav-item" href="/notify"><span class="nav-icon">✉</span>推播中心</a>
        <a class="nav-item" href="/schedule"><span class="nav-icon">⏱</span>排程中心</a>
        <a class="nav-item" href="#modelsPanel"><span class="nav-icon">▣</span>AI 模型</a>
        <a class="nav-item" href="#importancePanel"><span class="nav-icon">◇</span>特徵權重</a>
        <a class="nav-item" href="#newsPanel"><span class="nav-icon">✦</span>新聞情緒</a>
        <a class="nav-item" href="#coveragePanel"><span class="nav-icon">✓</span>資料狀態</a>
      </nav>
      <div class="sidebar-note">把行情、AI 燈號、風控價格與模型狀態集中在同一個看盤工作台。</div>
    </aside>

  <main class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">股智雷達 AI</p>
        <h1>看盤首頁</h1>
        <p>一頁掌握市場強弱：AI 訊號、推薦評分、入場價、停損停利與模型健康度。</p>
      </div>
      <div class="actions">
        <select id="mode">
          <option value="real">真實資料</option>
          <option value="demo">Demo 資料</option>
	        </select>
	        <button onclick="loadAll()">重新載入</button>
	        <button id="autoRefreshButton" class="auto-refresh-button active" onclick="toggleAutoRefresh()">⏱ 30秒自動刷新：開</button>
	        <button class="primary" onclick="runPipeline()">更新分析</button>
        <a class="button" href="/docs" target="_blank">API 文件</a>
      </div>
      <div class="hero-art">
        <div class="radar-card">
          <div class="radar"></div>
        </div>
        <div class="hero-stats">
	          <div class="hero-stat"><span>已分析 / 股票池</span><strong id="heroStockCount">—</strong></div>
          <div class="hero-stat"><span>資料狀態</span><strong id="heroFreshness">更新中</strong></div>
        </div>
      </div>
    </section>

	    <section class="panel filter-panel" id="groupFilterPanel">
	      <div class="panel-head">
	        <div>
	          <p class="eyebrow">Stock Groups</p>
	          <h2>組群分類篩選</h2>
	          <p class="score-note">可切換全部股票或單一熱門族群；數字代表「已分析 / 已設定」，0/12 表示已納入股票池但尚未跑更新分析。</p>
	        </div>
	        <div class="status" id="groupFilterStatus">載入中...</div>
	      </div>
	      <div class="filter-bar" id="groupFilters"></div>
	    </section>

	    <section class="panel" id="daytradePanel">
	      <div class="panel-head">
	        <div>
	          <p class="eyebrow">Intraday Radar</p>
	          <h2>盤中當沖推薦 Top 10</h2>
	          <p class="score-note">以 AI 機率、VWAP、量能、MA20/MA60 趨勢、MACD、RSI、突破與風報比排序；綠燈需同時通過量價門檻。</p>
	        </div>
	        <div class="status" id="daytradeStatus">載入中...</div>
	      </div>
	      <div class="table-wrap"><table id="daytrade"></table></div>
	    </section>

		    <section class="panel" id="signalsPanel">
	      <div class="panel-head">
	        <div>
	          <p class="eyebrow">Latest Signals</p>
          <div class="title-row">
            <h2>最新 AI 訊號</h2>
            <span class="freshness" id="freshness">🟢 已更新最新資料</span>
          </div>
          <p class="score-note">只顯示最終綜合燈號：已把 AI 模型、技術面、量能、趨勢與風報比整合成同一套判斷。</p>
        </div>
        <div class="status" id="status">載入中...</div>
      </div>
	      <div class="table-wrap"><table id="signals"></table></div>
	    </section>

	    <section class="split">
	      <section class="panel" id="modelsPanel">
	        <div class="panel-head">
	          <div>
	            <p class="eyebrow">Model Arena</p>
	            <h2>AI 模型比較</h2>
	          </div>
	          <div class="status" id="modelStatus"></div>
	        </div>
	        <div class="table-wrap"><table id="models"></table></div>
	      </section>

	      <section class="panel" id="importancePanel">
	        <div class="panel-head">
	          <div>
	            <p class="eyebrow">Feature Importance</p>
	            <h2>特徵重要度</h2>
	          </div>
            <div class="status" id="importanceStatus">載入中...</div>
	        </div>
	        <div class="importance" id="importance"></div>
	      </section>
	    </section>

	    <section class="panel" id="newsPanel">
	      <div class="panel-head">
	        <div>
          <p class="eyebrow">News Sentiment</p>
          <h2>新聞情緒因子</h2>
        </div>
        <div class="status" id="newsStatus"></div>
      </div>
      <div class="table-wrap"><table id="news"></table></div>
    </section>

    <section class="panel" id="coveragePanel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">Data Coverage</p>
          <h2>資料抓取狀態</h2>
        </div>
        <div class="status" id="coverageStatus"></div>
      </div>
      <div class="table-wrap"><table id="coverage"></table></div>
    </section>
  </main>
  </div>

  <script>
		    const percentKeys = new Set(["cagr", "max_drawdown", "total_return", "accuracy", "roc_auc", "selection_score", "importance_pct", "daily_return"]);
	    const signalLabels = {BUY: "🟢 買進", HOLD: "🟡 觀望", SELL: "🔴 賣出"};
	    const modelLabels = {
	      random_forest: "隨機森林",
	      extra_trees: "極端隨機森林",
	      gradient_boosting: "梯度提升樹",
	      logistic_regression: "邏輯回歸",
	      lightgbm: "LightGBM",
	      xgboost: "XGBoost"
	    };
		    const featureLabels = {
	      sma_5: "5日均線",
	      sma_20: "20日均線",
	      sma_60: "60日均線",
	      ema_12: "12日EMA",
	      ema_26: "26日EMA",
	      rsi_14: "RSI相對強弱",
	      kd_k: "KD-K值",
	      kd_d: "KD-D值",
	      macd: "MACD差離值",
	      macd_signal: "MACD訊號線",
	      macd_hist: "MACD柱狀體",
	      bb_width: "布林通道寬度",
	      bb_percent: "布林通道位置",
	      atr_14: "ATR波動度",
	      momentum_10: "10日動能",
	      obv: "OBV能量潮",
	      vwap_20: "20日VWAP",
	      beta_60: "60日Beta",
	      volume_ratio_20: "20日量能比",
	      daily_return: "個股日報酬",
	      benchmark_return: "大盤日報酬",
	      vix_close: "VIX恐慌指數",
	      us10y_close: "美國10年債殖利率",
	      cpi: "CPI消費者物價",
	      gdp: "GDP經濟成長",
	      pmi: "PMI採購經理人指數",
		      sentiment_score: "新聞情緒分數"
		    };
	    const refreshIntervalMs = 30000;
	    let autoRefreshEnabled = true;
	    let autoRefreshTimer = null;
	    let isRefreshing = false;
	    let activeGroup = "全部";
	    let allSignalRows = [];
	    let groupOptions = [];
	    let lastDaytradePayload = {items: []};

    function mode() {
      return document.getElementById("mode").value;
    }

    async function getJson(path) {
      const separator = path.includes("?") ? "&" : "?";
      const response = await fetch(`${path}${separator}mode=${mode()}`);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function fmt(value, key = "") {
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
	      if (key === "prob_up" || key === "prob_down" || percentKeys.has(key)) return `${(Number(value) * 100).toFixed(1)}%`;
      if (["close", "entry_price", "stop_loss", "take_profit"].includes(key)) return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
      if (key === "position_size" || key === "rows") return Number(value).toLocaleString();
      return value;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    function displayBasis(value) {
      const hiddenBasis = new Set(["盤中翻紅", "短線動能"]);
      return String(value || "")
        .split("、")
        .filter(label => label && !hiddenBasis.has(label))
        .join("、") || "等待量價確認";
    }

    function latestSignalDate(rows) {
      const dates = rows.map(row => String(row.date || "")).filter(Boolean).sort();
      return dates.length ? dates[dates.length - 1] : "—";
    }

    function signalSyncMeta(rows) {
      const sourceRow = rows.find(row => row.intraday_source || row.intraday_updated_at) || {};
      const source = sourceRow.intraday_source || "";
      const updatedAt = sourceRow.intraday_updated_at || "";
      const warning = sourceRow.intraday_warning || "";
      const liveCount = rows.filter(row => row.intraday_synced).length;
      const isLive = liveCount > 0 && source === "Yahoo Finance 1m";
      const sourceText = isLive ? "Yahoo 1分線" : source ? "日線備援" : "尚未同步";
      const freshness = updatedAt
        ? `${isLive ? "🟢" : "🟡"} 盤中同步：${updatedAt}｜${sourceText}`
        : "⚪ 尚無盤中同步";
      const statusSuffix = updatedAt ? `｜盤中同步 ${sourceText}${warning ? "｜備援提醒" : ""}` : "";
      return {
        freshness,
        hero: updatedAt ? (isLive ? "盤中同步" : "日線備援") : "未同步",
        statusSuffix,
      };
    }

    function rowGroup(row) {
      return row.stock_group || "未分類";
    }

    function rowsForActiveGroup(rows) {
      return activeGroup === "全部" ? rows : rows.filter(row => rowGroup(row) === activeGroup);
    }

    function groupConfiguredCount(group) {
      if (group === "全部") {
        return groupOptions.reduce((total, option) => total + Number(option.configured_count || 0), 0);
      }
      const option = groupOptions.find(item => item.stock_group === group);
      return Number(option?.configured_count || 0);
    }

    function renderGroupFilters() {
      const loadedCounts = allSignalRows.reduce((counts, row) => {
        const group = rowGroup(row);
        counts[group] = (counts[group] || 0) + 1;
        return counts;
      }, {});
      const configuredGroups = groupOptions.map(option => option.stock_group).filter(Boolean);
      const loadedGroups = Object.keys(loadedCounts);
      const groups = Array.from(new Set([...configuredGroups, ...loadedGroups])).sort((a, b) => a.localeCompare(b, "zh-Hant"));
      const choices = ["全部", ...groups];
      if (!choices.includes(activeGroup)) activeGroup = "全部";
      const root = document.getElementById("groupFilters");
      root.innerHTML = choices.map(group => {
        const loaded = group === "全部" ? allSignalRows.length : (loadedCounts[group] || 0);
        const configured = groupConfiguredCount(group);
        const countText = configured ? `${loaded}/${configured}` : `${loaded}`;
        return `<button class="filter-chip ${group === activeGroup ? "active" : ""}" data-group="${escapeHtml(group)}">${escapeHtml(group)} <small>${countText}</small></button>`;
      }).join("");
      root.querySelectorAll(".filter-chip").forEach(button => {
        button.addEventListener("click", () => setActiveGroup(button.dataset.group || "全部"));
      });
      const visibleRows = rowsForActiveGroup(allSignalRows);
      const configured = groupConfiguredCount(activeGroup);
      const countText = configured ? `${visibleRows.length}/${configured} 檔` : `${visibleRows.length} 檔`;
      document.getElementById("groupFilterStatus").textContent = activeGroup === "全部" ? `全部 ${countText}` : `${activeGroup} ${countText}`;
    }

    function setActiveGroup(group) {
      activeGroup = group;
      renderFilteredTables();
    }

    function emptyGroupMessage() {
      const configured = groupConfiguredCount(activeGroup);
      if (activeGroup !== "全部" && configured > 0) {
        return `${activeGroup} 已納入 ${configured} 檔股票池，但目前尚未完成資料更新；請按「更新分析」後就會產生訊號。`;
      }
      return "目前沒有可顯示的資料。";
    }

    function renderFilteredTables() {
      renderGroupFilters();
      const filteredSignals = rowsForActiveGroup(allSignalRows);
      const filteredDaytrade = rowsForActiveGroup(lastDaytradePayload.items || []);
      renderDaytrade({
        ...lastDaytradePayload,
        items: filteredDaytrade,
        filtered_item_count: filteredDaytrade.length,
        original_item_count: (lastDaytradePayload.items || []).length,
      });
      renderSignals(filteredSignals, allSignalRows);
    }

    function scoreTooltip(detail) {
      if (!detail || !detail.weights) return "";
      const w = detail.weights;
      const c = detail.conditions || {};
      const active = Object.entries({
        "多頭排列": c.trend_up,
        "短均向上": c.short_up,
        "MACD偏多": c.macd_up,
        "量能放大": c.vol_up,
        "突破20日高": c.breakout20,
        "突破60日高": c.breakout60,
        "RSI健康": c.rsi_good,
        "RSI過熱": c.rsi_hot,
        "跌破20日": c.weak,
      }).filter(([, ok]) => ok).map(([label]) => label).join("、") || "無明確加分條件";
      return `AI ${w.ai_probability}/30｜趨勢 ${w.trend}/25｜技術 ${w.technical}/20｜量能 ${w.volume}/10｜風報比 ${w.reward_risk}/15｜扣分 ${w.penalty}｜條件：${active}`;
    }

	    function renderDaytrade(payload) {
	      const table = document.getElementById("daytrade");
	      const rows = payload.items || [];
	      const displayRows = rows.slice(0, 10);
	      const warning = payload.warning ? `｜${payload.warning}` : "";
	      const filterText = payload.original_item_count !== undefined ? `｜篩選 ${rows.length}/${payload.original_item_count}｜顯示 ${displayRows.length}` : `｜顯示 ${displayRows.length}`;
	      document.getElementById("daytradeStatus").textContent =
	        `${payload.updated_at || "—"}｜${payload.source || "—"}｜${payload.scored_count || 0}/${payload.universe_count || 0} 檔${filterText}${payload.cached ? "｜快取" : ""}${warning}`;
	      if (!rows.length) {
	        table.innerHTML = `<tbody><tr><td>${escapeHtml(emptyGroupMessage())}</td></tr></tbody>`;
	        return;
	      }
	      table.innerHTML = `<thead><tr>
	        <th>排名</th><th>股名</th><th>組群</th><th>現價</th><th>量能倍</th><th>當沖分</th><th>燈號</th><th>參考價/停損/目標</th><th>依據</th>
	      </tr></thead>` +
	        `<tbody>${displayRows.map((row, index) => `<tr>
	          <td>${index + 1}</td>
	          <td><a class="stock-link" href="/stock/${encodeURIComponent(row.ticker)}?mode=${mode()}">${escapeHtml(row.stock_name || row.ticker)}</a></td>
	          <td>${escapeHtml(row.stock_group || "未分類")}</td>
	          <td>${fmt(row.current_price, "close")}</td>
	          <td>${row.volume_ratio === null || row.volume_ratio === undefined ? "—" : `${Number(row.volume_ratio).toLocaleString(undefined, { maximumFractionDigits: 2 })}倍`}</td>
	          <td><span class="score-pill ${row.daytrade_score >= 78 ? "high" : row.daytrade_score < 55 ? "low" : ""}">${row.daytrade_score} 分</span></td>
	          <td>${escapeHtml(row.signal)}</td>
	          <td>${fmt(row.entry_price, "close")} / ${fmt(row.stop_loss, "close")} / ${fmt(row.take_profit, "close")}</td>
	          <td>${escapeHtml(displayBasis(row.basis))}</td>
	        </tr>`).join("")}</tbody>`;
	    }

		    function renderSignals(rows, allRows = rows) {
	      const table = document.getElementById("signals");
	      const latestDate = latestSignalDate(allRows);
	      const syncMeta = signalSyncMeta(allRows);
	      const head = ["股名", "組群", "目前價", "綜合燈號", "推薦評分", "入場推薦價", "停損價", "停利價"];
	      const keys = ["stock_name", "stock_group", "close", "composite_signal", "recommendation_score", "entry_price", "stop_loss", "take_profit"];
	      if (!rows.length) {
	        table.innerHTML = `<thead><tr>${head.map(x => `<th>${x}</th>`).join("")}</tr></thead>` +
	          `<tbody><tr><td colspan="${head.length}">${escapeHtml(emptyGroupMessage())}</td></tr></tbody>`;
	        document.getElementById("freshness").textContent = latestDate === "—" ? "⚪ 尚無 AI 訊號資料" : syncMeta.freshness;
	        const configured = groupConfiguredCount(activeGroup);
	        const countText = configured ? `0/${configured}` : "0";
	        document.getElementById("status").textContent = `${activeGroup} ${countText} 檔（已分析/設定）｜AI資料日 ${latestDate}${syncMeta.statusSuffix}`;
	        document.getElementById("heroStockCount").textContent = `${allRows.length}/${groupConfiguredCount("全部")} 檔`;
	        document.getElementById("heroFreshness").textContent = syncMeta.hero;
	        return;
	      }
	      table.innerHTML = `<thead><tr>${head.map(x => `<th>${x}</th>`).join("")}</tr></thead>` +
	        `<tbody>${rows.map(row => `<tr>${keys.map(key => {
	          if (key === "composite_signal") {
	            const signal = row.composite_signal || {};
	            return `<td><span class="badge ${signal.code || "YELLOW"}" title="${escapeHtml(signal.basis || "")}">${signal.text || "🟡 觀察"}</span></td>`;
	          }
	          if (key === "stock_name") return `<td><a class="stock-link" href="/stock/${encodeURIComponent(row.ticker)}?mode=${mode()}">${fmt(row[key], key)}</a></td>`;
	          if (key === "stock_group") return `<td>${escapeHtml(row[key] || "未分類")}</td>`;
	          if (key === "recommendation_score") {
	            const score = Number(row[key]);
	            const tone = score >= 70 ? "high" : score <= 45 ? "low" : "";
	            const tooltip = escapeHtml(scoreTooltip(row.recommendation_score_detail));
	            return `<td><span class="score-pill ${tone}" title="${tooltip}">${Number.isFinite(score) ? `${score} 分` : "—"}</span></td>`;
	          }
	          return `<td>${fmt(row[key], key)}</td>`;
	        }).join("")}</tr>`).join("")}</tbody>`;
	      document.getElementById("freshness").textContent = latestDate === "—" ? "⚪ 尚無 AI 訊號資料" : syncMeta.freshness;
	      const configured = groupConfiguredCount(activeGroup);
	      const totalConfigured = groupConfiguredCount("全部");
	      const countText = configured ? `${rows.length}/${configured}` : `${rows.length}`;
	      document.getElementById("status").textContent = `${activeGroup} ${countText} 檔（已分析/設定）｜AI資料日 ${latestDate}${syncMeta.statusSuffix}｜五構面評分`;
	      document.getElementById("heroStockCount").textContent = totalConfigured ? `${allRows.length}/${totalConfigured} 檔` : `${allRows.length} 檔`;
	      document.getElementById("heroFreshness").textContent = syncMeta.hero;
	    }

	    function modelStatusText(status) {
	      return {success: "成功", failed: "失敗"}[status] || status;
	    }

	    function renderModelComparison(rows) {
	      const table = document.getElementById("models");
	      const head = ["最佳", "模型", "狀態", "準確率", "ROC AUC", "選模分數", "錯誤"];
	      const keys = ["selected", "model", "status", "accuracy", "roc_auc", "selection_score", "error"];
	      const best = rows.find(row => row.selected === true || String(row.selected).toLowerCase() === "true");
	      document.getElementById("modelStatus").textContent = best ? `最佳模型：${modelLabels[best.model] || best.model}` : `${rows.length} 個模型`;
	      table.innerHTML = `<thead><tr>${head.map(x => `<th>${x}</th>`).join("")}</tr></thead>` +
	        `<tbody>${rows.map(row => `<tr>${keys.map(key => {
	          if (key === "selected") return `<td class="best">${row[key] === true || String(row[key]).toLowerCase() === "true" ? "最佳" : ""}</td>`;
	          if (key === "status") return `<td>${modelStatusText(row[key])}</td>`;
	          if (key === "model") return `<td>${modelLabels[row[key]] || row[key]}</td>`;
	          return `<td>${fmt(row[key], key)}</td>`;
	        }).join("")}</tr>`).join("")}</tbody>`;
	    }

	    function renderFeatureImportance(rows, meta = {}) {
	      const root = document.getElementById("importance");
	      const visible = rows.slice(0, 10);
	      const sourceTime = meta.modified_at || "尚未產生";
	      document.getElementById("importanceStatus").textContent =
	        `${visible.length}/${rows.length} 項｜資料時間 ${sourceTime}`;
	      if (!visible.length) {
	        root.innerHTML = `<p>目前選出的模型沒有可直接讀取的特徵重要度。</p>`;
	        return;
	      }
	      root.innerHTML = visible.map(row => {
	        const pct = Number(row.importance_pct || 0);
	        const width = Math.max(2, Math.min(100, pct * 100));
	        const label = row.feature_label || featureLabels[row.feature] || row.feature;
	        return `<div class="importance-row">
	          <div class="importance-meta"><span>${label}</span><strong>${fmt(pct, "importance_pct")}</strong></div>
	          <div class="bar"><span style="width:${width}%"></span></div>
	        </div>`;
	      }).join("");
	    }

    function renderCoverage(rows) {
      const table = document.getElementById("coverage");
      const failed = rows.filter(row => !row.loaded);
      const loaded = rows.length - failed.length;
      document.getElementById("coverageStatus").textContent = `${loaded}/${rows.length} 檔成功`;
      const visible = failed.length ? failed : rows.slice(0, 12);
      table.innerHTML = `<thead><tr><th>代號</th><th>股名</th><th>組群</th><th>狀態</th><th>資料筆數</th><th>最後日期</th></tr></thead>` +
        `<tbody>${visible.map(row => `<tr>
          <td>${row.ticker}</td><td>${row.stock_name}</td><td>${row.stock_group || "未分類"}</td><td>${row.loaded ? "成功" : "失敗"}</td>
          <td>${fmt(row.rows, "rows")}</td><td>${fmt(row.last_date)}</td>
        </tr>`).join("")}</tbody>`;
    }

    function sentimentText(label) {
      return {positive: "利多", neutral: "中性", negative: "利空"}[label] || label;
    }

    function renderNews(rows) {
      const table = document.getElementById("news");
      const visible = rows.slice(0, 12);
      document.getElementById("newsStatus").textContent = `${rows.length} 則新聞`;
      table.innerHTML = `<thead><tr><th>日期</th><th>來源</th><th>股名</th><th>情緒</th><th>分數</th><th>標題</th></tr></thead>` +
        `<tbody>${visible.map(row => `<tr>
          <td>${fmt(row.date)}</td>
          <td>${fmt(row.source)}</td>
          <td>${fmt(row.stock_name)}</td>
          <td><span class="badge ${row.sentiment_label}">${sentimentText(row.sentiment_label)}</span></td>
          <td>${Number(row.sentiment_score || 0).toFixed(2)}</td>
          <td>${fmt(row.title)}</td>
        </tr>`).join("")}</tbody>`;
    }

	    async function loadAll() {
	      if (isRefreshing) return;
	      isRefreshing = true;
	      document.getElementById("status").textContent = "載入中...";
	      document.getElementById("freshness").textContent = "🟡 更新確認中";
	      document.getElementById("heroFreshness").textContent = "更新中";
      try {
		        const [signals, coverage, groups] = await Promise.all([
		          getJson("/api/signals"),
		          getJson("/api/coverage"),
		          getJson("/api/groups")
		        ]);
		        const daytrade = await getJson("/api/daytrade?limit=50").catch(error => ({items: [], warning: error.message}));
		        const news = await getJson("/api/news").catch(() => []);
		        const models = await getJson("/api/model-comparison").catch(() => []);
		        const importance = await getJson("/api/feature-importance").catch(() => []);
		        const importanceMeta = await getJson("/api/feature-importance/meta").catch(() => ({}));
		        allSignalRows = signals;
		        groupOptions = groups;
		        lastDaytradePayload = daytrade;
		        renderFilteredTables();
	        renderModelComparison(models);
	        renderFeatureImportance(importance, importanceMeta);
	        renderCoverage(coverage);
	        renderNews(news);
	      } catch (error) {
	        document.getElementById("freshness").textContent = "🔴 更新失敗";
	        document.getElementById("heroFreshness").textContent = "失敗";
	        document.getElementById("status").textContent = `錯誤：${error.message}`;
	      } finally {
	        isRefreshing = false;
	      }
	    }

	    function updateAutoRefreshButton() {
	      const button = document.getElementById("autoRefreshButton");
	      button.classList.toggle("active", autoRefreshEnabled);
	      button.textContent = autoRefreshEnabled ? "⏱ 30秒自動刷新：開" : "⏸ 30秒自動刷新：關";
	    }

	    function scheduleAutoRefresh() {
	      if (autoRefreshTimer) clearInterval(autoRefreshTimer);
	      if (autoRefreshEnabled) autoRefreshTimer = setInterval(loadAll, refreshIntervalMs);
	      updateAutoRefreshButton();
	    }

	    function toggleAutoRefresh() {
	      autoRefreshEnabled = !autoRefreshEnabled;
	      scheduleAutoRefresh();
	    }

    async function runPipeline() {
      document.getElementById("status").textContent = "分析更新中，請稍等...";
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({mode: mode()})
        });
        if (!response.ok) throw new Error(await response.text());
        await loadAll();
      } catch (error) {
        document.getElementById("status").textContent = `更新失敗：${error.message}`;
      }
    }

	    loadAll();
	    scheduleAutoRefresh();
  </script>
</body>
</html>"""


def _schedule_html() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>排程中心｜台股 AI 量化分析系統</title>
  <style>
    :root {
      color-scheme: light;
      --panel: rgba(255, 255, 255, .95);
      --line: rgba(37, 99, 235, .13);
      --text: #0f1f35;
      --muted: #64748b;
      --cyan: #0284c7;
      --blue: #2563eb;
      --green: #16a34a;
      --yellow: #d97706;
      --red: #dc2626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 6%, rgba(56,189,248,.26), transparent 34%),
        radial-gradient(circle at 88% 0%, rgba(37,99,235,.15), transparent 30%),
        linear-gradient(135deg, #ffffff, #eef6ff 58%, #f8fbff);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
    }
    .app-shell { width: min(1480px, calc(100% - 36px)); margin: 0 auto; padding: 28px 0; display: grid; grid-template-columns: 210px 1fr; gap: 18px; }
    .sidebar {
      position: sticky;
      top: 24px;
      height: calc(100vh - 56px);
      border-radius: 30px;
      padding: 20px 16px;
      color: #dbeafe;
      background: linear-gradient(180deg, #061b3a, #0b2a58);
      box-shadow: 0 24px 55px rgba(15, 23, 42, .18);
    }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 24px; font-weight: 950; letter-spacing: -.03em; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 14px; background: linear-gradient(135deg, #38bdf8, #2563eb); box-shadow: 0 12px 24px rgba(37,99,235,.34); }
    .brand small { display: block; color: #7dd3fc; font-size: 12px; letter-spacing: .08em; }
    .nav-list { display: grid; gap: 8px; }
    .nav-item { display: flex; align-items: center; gap: 10px; padding: 12px; border-radius: 16px; color: #bfdbfe; text-decoration: none; font-weight: 850; }
    .nav-item.active, .nav-item:hover { color: #ffffff; background: rgba(59,130,246,.36); }
    .nav-icon { width: 26px; height: 26px; display: grid; place-items: center; border-radius: 10px; background: rgba(255,255,255,.12); }
    .sidebar-note { margin-top: 22px; padding: 14px; border-radius: 18px; color: #bfdbfe; background: rgba(255,255,255,.08); font-size: 13px; line-height: 1.6; }
    .shell { min-width: 0; }
    .hero, .panel, .card {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 18px 48px rgba(37, 99, 235, .10);
      backdrop-filter: blur(18px);
    }
    .hero { border-radius: 30px; padding: 30px; display: flex; justify-content: space-between; gap: 24px; align-items: center; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 900; margin: 0 0 8px; }
    h1 { font-size: clamp(38px, 5vw, 64px); letter-spacing: -.06em; line-height: .95; margin: 0 0 14px; color: #071b36; }
    h2 { margin: 0; letter-spacing: -.02em; }
    p { color: var(--muted); line-height: 1.7; margin: 0; }
    button, select, a.button {
      border: 1px solid rgba(37,99,235,.18);
      border-radius: 999px;
      color: var(--text);
      background: #eff6ff;
      padding: 11px 16px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
    }
    button.primary { color: #ffffff; border-color: transparent; background: linear-gradient(135deg, #2563eb, #06b6d4); box-shadow: 0 12px 26px rgba(37,99,235,.22); }
    button:disabled { cursor: wait; opacity: .65; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0; }
    .card { border-radius: 22px; padding: 18px; }
    .card span { display: block; color: var(--muted); font-size: 13px; }
    .card strong { display: block; margin-top: 8px; font-size: 24px; letter-spacing: -.03em; }
    .panel { border-radius: 28px; padding: 22px; margin-top: 16px; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 16px; }
    .status { color: var(--muted); font-size: 14px; }
    .job-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .job-card { border: 1px solid var(--line); border-radius: 22px; padding: 18px; background: #f8fbff; display: grid; gap: 12px; }
    .job-card h3 { margin: 0; font-size: 22px; }
    .job-meta { display: grid; gap: 7px; color: var(--muted); font-size: 14px; }
    .badge { display: inline-flex; width: fit-content; border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 900; }
    .success { color: #15803d; background: #dcfce7; }
    .failed { color: #b91c1c; background: #fee2e2; }
    .pending { color: #a16207; background: #fef3c7; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 13px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .table-wrap { overflow-x: auto; }
    .log-box { white-space: pre-wrap; border-radius: 18px; padding: 16px; background: #0f1f35; color: #dbeafe; min-height: 86px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.6; }
    @media (max-width: 1120px) { .app-shell { grid-template-columns: 1fr; } .sidebar { position: static; height: auto; } .nav-list { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 980px) { .hero { display: block; } .actions { margin-top: 18px; justify-content: flex-start; } .grid, .job-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">AI</span>
        <div>股智雷達 <small>TAIWAN QUANT</small></div>
      </div>
      <nav class="nav-list" aria-label="主要功能">
        <a class="nav-item" href="/"><span class="nav-icon">⌂</span>看盤首頁</a>
        <a class="nav-item" href="/backtest"><span class="nav-icon">↗</span>回測中心</a>
        <a class="nav-item" href="/notify"><span class="nav-icon">✉</span>推播中心</a>
        <a class="nav-item active" href="/schedule"><span class="nav-icon">⏱</span>排程中心</a>
      </nav>
      <div class="sidebar-note">排程中心負責：盤中行情更新、收盤完整分析、晚間 Telegram 推播。</div>
    </aside>

    <main class="shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Automation Center</p>
          <h1>排程中心</h1>
          <p>把盤中更新、收盤 AI 分析、晚上推播變成固定流程；可以先手動測試，再用一鍵腳本開啟長駐排程。</p>
        </div>
        <div class="actions">
          <select id="mode">
            <option value="real">真實資料</option>
            <option value="demo">Demo 資料</option>
          </select>
          <button onclick="loadSchedule()">重新載入</button>
          <button class="primary" onclick="runJob('intraday')">手動跑盤中</button>
        </div>
      </section>

      <section class="grid" id="summary"></section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Scheduled Jobs</p>
            <h2>三段式自動流程</h2>
          </div>
          <div class="status" id="status">載入中...</div>
        </div>
        <div class="job-grid" id="jobs"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Last Runs</p>
            <h2>最近執行紀錄</h2>
          </div>
        </div>
        <div class="table-wrap"><table id="history"></table></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">CLI Command</p>
            <h2>長駐排程啟動指令</h2>
          </div>
        </div>
        <div class="log-box" id="commandBox">方式 1：直接點兩下「啟動自動排程.command」

方式 2：終端機手動啟動
cd "/Users/elvis/Documents/量化系統"
PYTHONPATH=src .venv/bin/python -m tw_ai_quant.cli --config configs/example.yml schedule --mode real</div>
      </section>
    </main>
  </div>

  <script>
    const labels = {
      intraday: "盤中行情與當沖",
      after_close: "收盤完整 AI 分析",
      evening_notify: "晚間 Telegram 推播"
    };
    const descriptions = {
      intraday: "盤中只更新行情、當沖分數與訊號快照。",
      after_close: "收盤後重抓資料、重建特徵、訓練模型、回測並輸出報告。",
      evening_notify: "晚上讀取最新訊號，送出 Telegram 每日摘要。"
    };
    let latestPayload = null;

    function mode() {
      return document.getElementById("mode").value;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    async function getJson(path) {
      const separator = path.includes("?") ? "&" : "?";
      const response = await fetch(`${path}${separator}mode=${mode()}`);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function jobTime(job, settings) {
      if (job === "intraday") return `${settings.start}～${settings.end}｜每 ${settings.interval_minutes} 分鐘`;
      return settings.time || "—";
    }

    function renderSummary(payload) {
      const artifacts = payload.artifacts || {};
      const schedulerLabel = artifacts.scheduler_active
        ? "運行中"
        : artifacts.scheduler_heartbeat_exists
          ? `已停止 / ${artifacts.scheduler_heartbeat_age_seconds || 0} 秒前`
          : "尚未啟動";
      document.getElementById("summary").innerHTML = [
        ["目前時間", payload.now],
        ["長駐排程", schedulerLabel],
        ["最新訊號", artifacts.signals_exists ? "已建立" : "尚未建立"],
        ["盤中快照", artifacts.daytrade_exists ? "已建立" : "尚未建立"],
      ].map(([label, value]) => `<article class="card"><span>${label}</span><strong>${value}</strong></article>`).join("");
    }

    function renderJobs(payload) {
      const schedule = payload.schedule || {};
      const state = payload.state?.jobs || {};
      document.getElementById("jobs").innerHTML = Object.entries(labels).map(([job, label]) => {
        const settings = schedule[job] || {};
        const last = state[job] || {};
        const status = last.status || "pending";
        const statusText = {success: "成功", failed: "失敗", pending: "尚未執行"}[status] || status;
        return `<article class="job-card">
          <span class="badge ${status}">${statusText}</span>
          <h3>${label}</h3>
          <p>${descriptions[job]}</p>
          <div class="job-meta">
            <div>排程：${settings.enabled ? "啟用" : "停用"}｜${jobTime(job, settings)}</div>
            <div>最後完成：${last.finished_at || "—"}</div>
            <div>結果：${escapeHtml(last.message || "尚無紀錄")}</div>
          </div>
          <button class="primary" onclick="runJob('${job}')">立即執行</button>
        </article>`;
      }).join("");
    }

    function renderHistory(payload) {
      const rows = Object.entries(payload.state?.jobs || {}).map(([job, row]) => ({job, ...row}));
      if (!rows.length) {
        document.getElementById("history").innerHTML = `<tbody><tr><td>目前尚無排程執行紀錄。</td></tr></tbody>`;
        return;
      }
      document.getElementById("history").innerHTML = `<thead><tr><th>任務</th><th>狀態</th><th>開始</th><th>完成</th><th>訊息</th></tr></thead>` +
        `<tbody>${rows.map(row => `<tr>
          <td>${labels[row.job] || row.job}</td>
          <td><span class="badge ${row.status || "pending"}">${row.status || "pending"}</span></td>
          <td>${row.started_at || "—"}</td>
          <td>${row.finished_at || "—"}</td>
          <td>${escapeHtml(row.message || "—")}</td>
        </tr>`).join("")}</tbody>`;
    }

    function renderAll(payload) {
      latestPayload = payload;
      document.getElementById("status").textContent = `${payload.mode}｜${payload.timezone}｜${payload.now}`;
      renderSummary(payload);
      renderJobs(payload);
      renderHistory(payload);
    }

    async function loadSchedule() {
      document.getElementById("status").textContent = "載入中...";
      try {
        renderAll(await getJson("/api/schedule"));
      } catch (error) {
        document.getElementById("status").textContent = `錯誤：${error.message}`;
      }
    }

    async function runJob(job) {
      const buttons = [...document.querySelectorAll("button")];
      buttons.forEach(button => button.disabled = true);
      document.getElementById("status").textContent = `${labels[job]} 執行中...`;
      try {
        const response = await fetch(`/api/jobs/${job}`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({mode: mode()})
        });
        if (!response.ok) throw new Error(await response.text());
        await loadSchedule();
      } catch (error) {
        document.getElementById("status").textContent = `執行失敗：${error.message}`;
      } finally {
        buttons.forEach(button => button.disabled = false);
      }
    }

    loadSchedule();
  </script>
</body>
</html>"""


def _backtest_html() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>回測中心｜台股 AI 量化分析系統</title>
  <style>
    :root {
      color-scheme: light;
      --panel: rgba(255, 255, 255, .95);
      --line: rgba(37, 99, 235, .13);
      --text: #0f1f35;
      --muted: #64748b;
      --cyan: #0284c7;
      --blue: #2563eb;
      --green: #16a34a;
      --yellow: #d97706;
      --red: #dc2626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 6%, rgba(56,189,248,.26), transparent 34%),
        radial-gradient(circle at 88% 0%, rgba(37,99,235,.15), transparent 30%),
        linear-gradient(135deg, #ffffff, #eef6ff 58%, #f8fbff);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
    }
    .app-shell { width: min(1480px, calc(100% - 36px)); margin: 0 auto; padding: 28px 0; display: grid; grid-template-columns: 210px 1fr; gap: 18px; }
    .sidebar {
      position: sticky;
      top: 24px;
      height: calc(100vh - 56px);
      border-radius: 30px;
      padding: 20px 16px;
      color: #dbeafe;
      background: linear-gradient(180deg, #061b3a, #0b2a58);
      box-shadow: 0 24px 55px rgba(15, 23, 42, .18);
    }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 24px; font-weight: 950; letter-spacing: -.03em; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 14px; background: linear-gradient(135deg, #38bdf8, #2563eb); box-shadow: 0 12px 24px rgba(37,99,235,.34); }
    .brand small { display: block; color: #7dd3fc; font-size: 12px; letter-spacing: .08em; }
    .nav-list { display: grid; gap: 8px; }
    .nav-item { display: flex; align-items: center; gap: 10px; padding: 12px; border-radius: 16px; color: #bfdbfe; text-decoration: none; font-weight: 850; }
    .nav-item.active, .nav-item:hover { color: #ffffff; background: rgba(59,130,246,.36); }
    .nav-icon { width: 26px; height: 26px; display: grid; place-items: center; border-radius: 10px; background: rgba(255,255,255,.12); }
    .sidebar-note { margin-top: 22px; padding: 14px; border-radius: 18px; color: #bfdbfe; background: rgba(255,255,255,.08); font-size: 13px; line-height: 1.6; }
    .shell { min-width: 0; }
    .hero, .panel, .card {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 18px 48px rgba(37, 99, 235, .10);
      backdrop-filter: blur(18px);
    }
    .hero { border-radius: 30px; padding: 30px; display: flex; justify-content: space-between; gap: 24px; align-items: center; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 900; margin: 0 0 8px; }
    h1 { font-size: clamp(38px, 5vw, 64px); letter-spacing: -.06em; line-height: .95; margin: 0 0 14px; color: #071b36; }
    h2 { margin: 0; letter-spacing: -.02em; }
    p { color: var(--muted); line-height: 1.7; margin: 0; }
    button, select, a.button {
      border: 1px solid rgba(37,99,235,.18);
      border-radius: 999px;
      color: var(--text);
      background: #eff6ff;
      padding: 11px 16px;
      font-weight: 850;
      text-decoration: none;
      cursor: pointer;
    }
    button.primary { color: #ffffff; border-color: transparent; background: linear-gradient(135deg, #2563eb, #06b6d4); box-shadow: 0 12px 26px rgba(37,99,235,.22); }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
    .grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin: 16px 0; }
    .card { border-radius: 22px; padding: 18px; }
    .card span { display: block; color: var(--muted); font-size: 13px; }
    .card strong { display: block; margin-top: 8px; font-size: 25px; letter-spacing: -.03em; }
    .panel { border-radius: 28px; padding: 22px; margin-top: 16px; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 16px; }
    .status { color: var(--muted); font-size: 14px; }
    .split { display: grid; grid-template-columns: 1.3fr .7fr; gap: 16px; }
    .chart { width: 100%; min-height: 480px; border-radius: 22px; background: #ffffff; border: 1px solid rgba(37,99,235,.12); }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 13px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .table-wrap { overflow-x: auto; }
    .pill { display: inline-flex; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 900; color: #1d4ed8; background: #dbeafe; }
    .todo { display: grid; gap: 12px; color: var(--muted); }
    .todo div { padding: 14px; border: 1px dashed rgba(37,99,235,.22); border-radius: 18px; background: #f8fbff; }
    @media (max-width: 1120px) { .app-shell { grid-template-columns: 1fr; } .sidebar { position: static; height: auto; } .nav-list { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 980px) { .hero, .panel-head { display: block; } .actions { margin-top: 18px; justify-content: flex-start; } .grid, .split { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 620px) { .app-shell { width: min(100% - 22px, 1480px); } .grid, .split, .nav-list { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">AI</span>
        <div>股智雷達 <small>BACKTEST</small></div>
      </div>
      <nav class="nav-list" aria-label="回測功能">
        <a class="nav-item" href="/"><span class="nav-icon">⌂</span>看盤首頁</a>
        <a class="nav-item active" href="/backtest"><span class="nav-icon">↗</span>回測中心</a>
        <a class="nav-item" href="/notify"><span class="nav-icon">✉</span>推播中心</a>
        <a class="nav-item" href="/#signalsPanel"><span class="nav-icon">◎</span>今日訊號</a>
        <a class="nav-item" href="/#modelsPanel"><span class="nav-icon">▣</span>AI 模型</a>
        <a class="nav-item" href="/#importancePanel"><span class="nav-icon">◇</span>特徵權重</a>
        <a class="nav-item" href="/#coveragePanel"><span class="nav-icon">✓</span>資料狀態</a>
      </nav>
      <div class="sidebar-note">回測中心用來驗證訊號是否有策略價值；交易紀錄表先保留在最後階段。</div>
    </aside>

    <main class="shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Backtest Center</p>
          <h1>回測中心</h1>
          <p>檢查 AI 訊號在歷史測試區間的資金曲線、回撤、勝率與訊號分布。</p>
        </div>
        <div class="actions">
          <select id="mode">
            <option value="real">真實資料</option>
            <option value="demo">Demo 資料</option>
          </select>
          <button onclick="loadBacktest()">重新載入</button>
          <a class="button" href="/">回看盤首頁</a>
        </div>
      </section>

      <section class="grid" id="metrics"></section>

      <section class="split">
        <article class="panel">
          <div class="panel-head">
            <div>
              <p class="eyebrow">Equity / Drawdown</p>
              <h2>資金曲線與回撤</h2>
            </div>
            <div class="status" id="curveStatus">載入中...</div>
          </div>
          <div id="curve"></div>
        </article>

        <article class="panel">
          <div class="panel-head">
            <div>
              <p class="eyebrow">Signal Summary</p>
              <h2>訊號分布</h2>
            </div>
          </div>
          <div class="table-wrap"><table id="signals"></table></div>
          <div class="todo" style="margin-top:16px">
            <div id="tradeLogStatus">交易紀錄表：待實際操作後啟用。</div>
          </div>
        </article>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Monthly Return</p>
            <h2>近 12 個月策略報酬</h2>
          </div>
          <span class="pill" id="period">—</span>
        </div>
        <div class="table-wrap"><table id="monthly"></table></div>
      </section>
    </main>
  </div>

  <script>
    const metricLabels = {
      cagr: "年化報酬 CAGR",
      sharpe: "夏普比率",
      max_drawdown: "最大回撤",
      total_return: "總報酬",
      win_rate: "日勝率",
      avg_turnover: "平均換手",
      verified_end_date: "回測可驗證到",
      latest_signal_date: "最新訊號日"
    };
    const percentKeys = new Set(["cagr", "max_drawdown", "total_return", "win_rate", "avg_turnover", "monthly_return", "strategy_return", "drawdown"]);

    function mode() {
      return document.getElementById("mode").value;
    }

    async function getJson(path) {
      const response = await fetch(`${path}?mode=${mode()}`);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function fmt(value, key = "") {
      if (key === "verified_end_date" || key === "latest_signal_date") return value;
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
      if (percentKeys.has(key)) return `${(Number(value) * 100).toFixed(1)}%`;
      if (key === "equity" || key === "initial_capital") return Number(value).toLocaleString(undefined, {maximumFractionDigits: 0});
      if (key === "sharpe") return Number(value).toFixed(2);
      return Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
    }

    function renderMetrics(payload) {
      const combined = {...payload.metrics, ...payload.summary};
      document.getElementById("metrics").innerHTML = Object.entries(metricLabels).map(([key, label]) => `
        <article class="card"><span>${label}</span><strong>${fmt(combined[key], key)}</strong></article>
      `).join("");
      document.getElementById("period").textContent = `${payload.summary.start_date} → ${payload.summary.end_date}`;
    }

    function renderSignalSummary(summary) {
      const rows = [
        ["最新綜合強勢", summary.latest_composite?.GREEN || 0],
        ["最新綜合觀察", summary.latest_composite?.YELLOW || 0],
        ["最新綜合弱勢", summary.latest_composite?.RED || 0],
        ["最新AI買進", summary.latest?.BUY || 0],
        ["最新AI觀望", summary.latest?.HOLD || 0],
        ["最新AI賣出", summary.latest?.SELL || 0],
        ["歷史AI買進日數", summary.buy_days || 0],
        ["歷史AI買進訊號", summary.history?.BUY || 0],
        ["歷史AI觀望訊號", summary.history?.HOLD || 0],
        ["歷史AI賣出訊號", summary.history?.SELL || 0],
      ];
      document.getElementById("signals").innerHTML = `<thead><tr><th>項目</th><th>數值</th></tr></thead>` +
        `<tbody>${rows.map(([label, value]) => `<tr><td>${label}</td><td>${Number(value).toLocaleString()}</td></tr>`).join("")}</tbody>`;
    }

    function renderMonthly(rows) {
      document.getElementById("monthly").innerHTML = `<thead><tr><th>月份</th><th>策略報酬</th></tr></thead>` +
        `<tbody>${rows.map(row => `<tr><td>${row.month}</td><td>${fmt(row.monthly_return, "monthly_return")}</td></tr>`).join("")}</tbody>`;
    }

    function renderCurve(curve) {
      if (!curve.length) {
        document.getElementById("curve").innerHTML = `<p>尚無回測曲線資料。</p>`;
        return;
      }
      const width = 980, height = 470, left = 58, right = 28;
      const equityTop = 35, equityHeight = 270;
      const drawTop = 345, drawHeight = 82;
      const plotWidth = width - left - right;
      const equityValues = curve.map(row => Number(row.equity)).filter(Number.isFinite);
      const drawValues = curve.map(row => Number(row.drawdown)).filter(Number.isFinite);
      const minEquity = Math.min(...equityValues);
      const maxEquity = Math.max(...equityValues);
      const minDraw = Math.min(...drawValues, 0);
      const x = index => left + index / Math.max(1, curve.length - 1) * plotWidth;
      const equityY = value => equityTop + (maxEquity - value) / Math.max(1, maxEquity - minEquity) * equityHeight;
      const drawY = value => drawTop + (0 - value) / Math.max(.001, 0 - minDraw) * drawHeight;
      const equityPoints = curve.map((row, index) => `${x(index).toFixed(2)},${equityY(Number(row.equity)).toFixed(2)}`).join(" ");
      const drawPoints = curve.map((row, index) => `${x(index).toFixed(2)},${drawY(Number(row.drawdown)).toFixed(2)}`).join(" ");
      const drawArea = `${left},${drawY(0)} ${drawPoints} ${width - right},${drawY(0)}`;
      document.getElementById("curve").innerHTML = `
        <svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="資金曲線與回撤">
          <rect width="${width}" height="${height}" rx="20" fill="#ffffff"/>
          <text x="${left}" y="24" fill="#0f172a" font-size="13" font-weight="800">資金曲線</text>
          <line x1="${left}" y1="${equityTop + equityHeight}" x2="${width - right}" y2="${equityTop + equityHeight}" stroke="#cbd5e1"/>
          <line x1="${left}" y1="${equityTop}" x2="${left}" y2="${equityTop + equityHeight}" stroke="#cbd5e1"/>
          <polyline points="${equityPoints}" fill="none" stroke="#2563eb" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
          <text x="${left}" y="${equityTop + equityHeight + 20}" fill="#64748b" font-size="12">${curve[0].date}</text>
          <text x="${width - right}" y="${equityTop + equityHeight + 20}" fill="#64748b" font-size="12" text-anchor="end">${curve[curve.length - 1].date}</text>
          <text x="${width - right}" y="${equityY(equityValues[equityValues.length - 1]) - 8}" fill="#1d4ed8" font-size="13" font-weight="800" text-anchor="end">${fmt(equityValues[equityValues.length - 1], "equity")}</text>

          <text x="${left}" y="${drawTop - 16}" fill="#0f172a" font-size="13" font-weight="800">回撤</text>
          <line x1="${left}" y1="${drawY(0)}" x2="${width - right}" y2="${drawY(0)}" stroke="#cbd5e1"/>
          <polygon points="${drawArea}" fill="rgba(220,38,38,.14)"/>
          <polyline points="${drawPoints}" fill="none" stroke="#dc2626" stroke-width="2.4"/>
          <text x="${width - right}" y="${drawTop + drawHeight + 24}" fill="#64748b" font-size="12" text-anchor="end">最大回撤 ${fmt(Math.min(...drawValues), "drawdown")}</text>
        </svg>`;
    }

    async function loadBacktest() {
      document.getElementById("curveStatus").textContent = "載入中...";
      try {
        const payload = await getJson("/api/backtest");
        renderMetrics(payload);
        renderCurve(payload.curve || []);
        renderSignalSummary(payload.signal_summary || {});
        renderMonthly(payload.monthly_returns || []);
        document.getElementById("tradeLogStatus").textContent = payload.trade_log_status;
        document.getElementById("curveStatus").textContent = `${payload.summary.days.toLocaleString()} 個交易日`;
      } catch (error) {
        document.getElementById("curveStatus").textContent = `錯誤：${error.message}`;
      }
    }

    loadBacktest();
  </script>
</body>
</html>"""


def _notify_html() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>推播中心｜台股 AI 量化分析系統</title>
  <style>
    :root {
      color-scheme: light;
      --panel: rgba(255, 255, 255, .95);
      --line: rgba(37, 99, 235, .13);
      --text: #0f1f35;
      --muted: #64748b;
      --cyan: #0284c7;
      --blue: #2563eb;
      --green: #16a34a;
      --yellow: #d97706;
      --red: #dc2626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 6%, rgba(56,189,248,.26), transparent 34%),
        radial-gradient(circle at 88% 0%, rgba(37,99,235,.15), transparent 30%),
        linear-gradient(135deg, #ffffff, #eef6ff 58%, #f8fbff);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
    }
    .app-shell { width: min(1480px, calc(100% - 36px)); margin: 0 auto; padding: 28px 0; display: grid; grid-template-columns: 210px 1fr; gap: 18px; }
    .sidebar {
      position: sticky;
      top: 24px;
      height: calc(100vh - 56px);
      border-radius: 30px;
      padding: 20px 16px;
      color: #dbeafe;
      background: linear-gradient(180deg, #061b3a, #0b2a58);
      box-shadow: 0 24px 55px rgba(15, 23, 42, .18);
    }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 24px; font-weight: 950; letter-spacing: -.03em; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 14px; background: linear-gradient(135deg, #38bdf8, #2563eb); box-shadow: 0 12px 24px rgba(37,99,235,.34); }
    .brand small { display: block; color: #7dd3fc; font-size: 12px; letter-spacing: .08em; }
    .nav-list { display: grid; gap: 8px; }
    .nav-item { display: flex; align-items: center; gap: 10px; padding: 12px; border-radius: 16px; color: #bfdbfe; text-decoration: none; font-weight: 850; }
    .nav-item.active, .nav-item:hover { color: #ffffff; background: rgba(59,130,246,.36); }
    .nav-icon { width: 26px; height: 26px; display: grid; place-items: center; border-radius: 10px; background: rgba(255,255,255,.12); }
    .sidebar-note { margin-top: 22px; padding: 14px; border-radius: 18px; color: #bfdbfe; background: rgba(255,255,255,.08); font-size: 13px; line-height: 1.6; }
    .shell { min-width: 0; }
    .hero, .panel, .card {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 18px 48px rgba(37, 99, 235, .10);
      backdrop-filter: blur(18px);
    }
    .hero { border-radius: 30px; padding: 30px; display: flex; justify-content: space-between; gap: 24px; align-items: center; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 900; margin: 0 0 8px; }
    h1 { font-size: clamp(38px, 5vw, 64px); letter-spacing: -.06em; line-height: .95; margin: 0 0 14px; color: #071b36; }
    h2 { margin: 0; letter-spacing: -.02em; }
    p { color: var(--muted); line-height: 1.7; margin: 0; }
    button, select, a.button {
      border: 1px solid rgba(37,99,235,.18);
      border-radius: 999px;
      color: var(--text);
      background: #eff6ff;
      padding: 11px 16px;
      font-weight: 850;
      text-decoration: none;
      cursor: pointer;
	    }
	    button.primary { color: #ffffff; border-color: transparent; background: linear-gradient(135deg, #2563eb, #06b6d4); box-shadow: 0 12px 26px rgba(37,99,235,.22); }
	    .auto-refresh-button.active { color: #ffffff; border-color: transparent; background: linear-gradient(135deg, #16a34a, #06b6d4); }
	    button:disabled { opacity: .45; cursor: not-allowed; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0; }
    .card { border-radius: 22px; padding: 18px; }
    .card span { display: block; color: var(--muted); font-size: 13px; }
    .card strong { display: block; margin-top: 8px; font-size: 24px; letter-spacing: -.03em; }
    .panel { border-radius: 28px; padding: 22px; margin-top: 16px; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 16px; }
    .status { color: var(--muted); font-size: 14px; }
    .split { display: grid; grid-template-columns: .85fr 1.15fr; gap: 16px; }
    .message {
      width: 100%;
      min-height: 460px;
      resize: vertical;
      border: 1px solid rgba(37,99,235,.16);
      border-radius: 22px;
      padding: 18px;
      color: #0f172a;
      background: #ffffff;
      font: 15px/1.65 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      white-space: pre-wrap;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 13px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .table-wrap { overflow-x: auto; }
    .pill { display: inline-flex; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 900; color: #1d4ed8; background: #dbeafe; }
    .ok { color: #15803d; background: #dcfce7; }
    .warn { color: #a16207; background: #fef3c7; }
    .bad { color: #b91c1c; background: #fee2e2; }
    .setup { display: grid; gap: 12px; }
    .setup div { padding: 14px; border: 1px dashed rgba(37,99,235,.22); border-radius: 18px; background: #f8fbff; color: var(--muted); line-height: 1.65; }
    code { color: #1d4ed8; font-weight: 800; }
    @media (max-width: 1120px) { .app-shell { grid-template-columns: 1fr; } .sidebar { position: static; height: auto; } .nav-list { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 980px) { .hero, .panel-head { display: block; } .actions { margin-top: 18px; justify-content: flex-start; } .grid, .split { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 620px) { .app-shell { width: min(100% - 22px, 1480px); } .grid, .split, .nav-list { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">AI</span>
        <div>股智雷達 <small>NOTIFY</small></div>
      </div>
      <nav class="nav-list" aria-label="推播功能">
        <a class="nav-item" href="/"><span class="nav-icon">⌂</span>看盤首頁</a>
        <a class="nav-item" href="/backtest"><span class="nav-icon">↗</span>回測中心</a>
        <a class="nav-item active" href="/notify"><span class="nav-icon">✉</span>推播中心</a>
        <a class="nav-item" href="/#signalsPanel"><span class="nav-icon">◎</span>今日訊號</a>
        <a class="nav-item" href="/#modelsPanel"><span class="nav-icon">▣</span>AI 模型</a>
        <a class="nav-item" href="/#coveragePanel"><span class="nav-icon">✓</span>資料狀態</a>
      </nav>
      <div class="sidebar-note">先把每日訊號訊息格式固定，之後再接每天自動排程。</div>
    </aside>

    <main class="shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Notification Center</p>
          <h1>推播中心</h1>
          <p>預覽每日 Telegram 訊息，檢查 Bot 設定狀態，準備接上每日自動推播流程。</p>
        </div>
        <div class="actions">
          <select id="mode">
            <option value="real">真實資料</option>
            <option value="demo">Demo 資料</option>
	          </select>
	          <button onclick="loadNotification()">重新載入</button>
	          <button id="autoRefreshButton" class="auto-refresh-button active" onclick="toggleAutoRefresh()">⏱ 30秒自動刷新：開</button>
	          <button class="primary" id="sendButton" onclick="sendNotification()" disabled>手動推播</button>
        </div>
      </section>

      <section class="grid" id="cards"></section>

      <section class="split">
        <article class="panel">
          <div class="panel-head">
            <div>
              <p class="eyebrow">Telegram Setup</p>
              <h2>設定狀態</h2>
            </div>
            <span class="pill" id="readyPill">檢查中</span>
          </div>
          <div class="setup" id="setup"></div>
          <div class="panel-head" style="margin-top:22px">
            <div>
              <p class="eyebrow">Top Signals</p>
              <h2>推播名單</h2>
            </div>
          </div>
          <div class="table-wrap"><table id="topSignals"></table></div>
        </article>

        <article class="panel">
          <div class="panel-head">
            <div>
              <p class="eyebrow">Message Preview</p>
              <h2>每日訊息預覽</h2>
            </div>
            <div class="status" id="messageStatus">載入中...</div>
          </div>
          <textarea class="message" id="message" readonly></textarea>
        </article>
      </section>
    </main>
  </div>

	  <script>
	    let latestPayload = null;
	    const refreshIntervalMs = 30000;
	    let autoRefreshEnabled = true;
	    let autoRefreshTimer = null;
	    let isRefreshing = false;

    function mode() {
      return document.getElementById("mode").value;
    }

    async function getJson(path) {
      const response = await fetch(`${path}?mode=${mode()}`);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function fmt(value, kind = "") {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
      if (kind === "pct") return `${(Number(value) * 100).toFixed(1)}%`;
      if (kind === "money") return Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
      return Number(value).toLocaleString();
    }

    function signalText(row) {
      return row?.composite_signal?.text || "🟡 觀察";
    }

    function renderCards(payload) {
      const status = payload.status;
      const summary = payload.summary;
      const items = [
        ["Telegram", status.ready ? "可推播" : status.enabled ? "待補設定" : "未啟用"],
        ["資料日期", summary.latest_date || "—"],
        ["綜合燈號", `🟢${summary.green} 🟡${summary.yellow} 🔴${summary.red}`],
        ["訊息長度", `${summary.message_length} 字`],
      ];
      document.getElementById("cards").innerHTML = items.map(([label, value]) =>
        `<article class="card"><span>${label}</span><strong>${value}</strong></article>`
      ).join("");
    }

    function renderSetup(status, note) {
      const readyPill = document.getElementById("readyPill");
      readyPill.textContent = status.ready ? "🟢 可推播" : status.enabled ? "🟡 待設定" : "🔴 未啟用";
      readyPill.className = `pill ${status.ready ? "ok" : status.enabled ? "warn" : "bad"}`;
      document.getElementById("sendButton").disabled = !status.ready;
      document.getElementById("setup").innerHTML = `
        <div>狀態：${status.reason}</div>
        <div>Bot Token：<code>${status.bot_token_env}</code>｜${status.bot_token_present ? "已設定" : "未設定"}</div>
        <div>Chat ID：<code>${status.chat_id_env}</code>｜${status.chat_id_present ? "已設定" : "未設定"}</div>
        <div>${note}</div>
      `;
    }

    function renderTopSignals(rows) {
      document.getElementById("topSignals").innerHTML = `<thead><tr><th>股名</th><th>評分</th><th>綜合燈號</th><th>停損</th><th>停利</th></tr></thead>` +
        `<tbody>${rows.map(row => `<tr>
          <td>${row.stock_name || row.ticker}</td>
          <td>${row.recommendation_score} 分</td>
          <td>${signalText(row)}</td>
          <td>${fmt(row.stop_loss, "money")}</td>
          <td>${fmt(row.take_profit, "money")}</td>
        </tr>`).join("")}</tbody>`;
    }

	    async function loadNotification() {
	      if (isRefreshing) return;
	      isRefreshing = true;
	      document.getElementById("messageStatus").textContent = "載入中...";
	      try {
        latestPayload = await getJson("/api/notification");
        renderCards(latestPayload);
        renderSetup(latestPayload.status, latestPayload.schedule_note);
        renderTopSignals(latestPayload.top_signals || []);
        document.getElementById("message").value = latestPayload.message;
        document.getElementById("messageStatus").textContent = `預覽完成｜${latestPayload.summary.message_length} 字`;
	      } catch (error) {
	        document.getElementById("messageStatus").textContent = `錯誤：${error.message}`;
	      } finally {
	        isRefreshing = false;
	      }
	    }

	    function updateAutoRefreshButton() {
	      const button = document.getElementById("autoRefreshButton");
	      button.classList.toggle("active", autoRefreshEnabled);
	      button.textContent = autoRefreshEnabled ? "⏱ 30秒自動刷新：開" : "⏸ 30秒自動刷新：關";
	    }

	    function scheduleAutoRefresh() {
	      if (autoRefreshTimer) clearInterval(autoRefreshTimer);
	      if (autoRefreshEnabled) autoRefreshTimer = setInterval(loadNotification, refreshIntervalMs);
	      updateAutoRefreshButton();
	    }

	    function toggleAutoRefresh() {
	      autoRefreshEnabled = !autoRefreshEnabled;
	      scheduleAutoRefresh();
	    }

    async function sendNotification() {
      document.getElementById("messageStatus").textContent = "推播中...";
      try {
        const response = await fetch("/api/notification/send", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({mode: mode()})
        });
        if (!response.ok) throw new Error(await response.text());
        const result = await response.json();
        document.getElementById("messageStatus").textContent = `已送出｜${result.message_length} 字`;
      } catch (error) {
        document.getElementById("messageStatus").textContent = `推播失敗：${error.message}`;
      }
    }

	    loadNotification();
	    scheduleAutoRefresh();
  </script>
</body>
</html>"""


def _stock_html(ticker: str, mode: str) -> str:
    return (
        """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>個股技術指標分析</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef6ff;
      --panel: rgba(255, 255, 255, .94);
      --line: rgba(37, 99, 235, .16);
      --text: #102033;
      --muted: #64748b;
      --cyan: #0284c7;
      --green: #16a34a;
      --yellow: #d97706;
      --red: #dc2626;
      --blue: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 5%, rgba(96,165,250,.28), transparent 34%),
        radial-gradient(circle at 92% 0%, rgba(14,165,233,.18), transparent 30%),
        linear-gradient(135deg, #f8fbff, #eaf4ff 55%, #f4f8ff);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
    }
    .app-shell { width: min(1480px, calc(100% - 36px)); margin: 0 auto; padding: 28px 0; display: grid; grid-template-columns: 210px 1fr; gap: 18px; }
    .sidebar {
      position: sticky;
      top: 24px;
      height: calc(100vh - 56px);
      border-radius: 30px;
      padding: 20px 16px;
      color: #dbeafe;
      background: linear-gradient(180deg, #061b3a, #0b2a58);
      box-shadow: 0 24px 55px rgba(15, 23, 42, .18);
    }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 24px; font-weight: 950; letter-spacing: -.03em; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 14px; background: linear-gradient(135deg, #38bdf8, #2563eb); box-shadow: 0 12px 24px rgba(37,99,235,.34); }
    .brand small { display: block; color: #7dd3fc; font-size: 12px; letter-spacing: .08em; }
    .nav-list { display: grid; gap: 8px; }
    .nav-item { display: flex; align-items: center; gap: 10px; padding: 12px; border-radius: 16px; color: #bfdbfe; text-decoration: none; font-weight: 850; }
    .nav-item.active, .nav-item:hover { color: #ffffff; background: rgba(59,130,246,.36); }
    .nav-icon { width: 26px; height: 26px; display: grid; place-items: center; border-radius: 10px; background: rgba(255,255,255,.12); }
    .sidebar-note { margin-top: 22px; padding: 14px; border-radius: 18px; color: #bfdbfe; background: rgba(255,255,255,.08); font-size: 13px; line-height: 1.6; }
    .shell { min-width: 0; padding: 0; }
    .hero, .panel, .card {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 24px 70px rgba(37, 99, 235, .12);
      backdrop-filter: blur(18px);
    }
    .hero { border-radius: 30px; padding: 28px; display: flex; justify-content: space-between; gap: 24px; align-items: center; overflow: hidden; position: relative; }
    .hero::after { content: ""; position: absolute; inset: auto -80px -120px auto; width: 360px; height: 260px; border-radius: 999px; background: radial-gradient(circle, rgba(14,165,233,.22), transparent 68%); pointer-events: none; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 900; margin: 0 0 8px; }
    h1 { font-size: clamp(34px, 5vw, 56px); letter-spacing: -.04em; margin: 0 0 12px; }
    h2 { margin: 0; letter-spacing: -.02em; }
    p { color: var(--muted); line-height: 1.7; margin: 0; }
    a.button, button, select {
      border: 1px solid rgba(37,99,235,.22);
      border-radius: 999px;
      color: var(--text);
      background: rgba(219,234,254,.78);
      padding: 11px 16px;
      font-weight: 900;
      text-decoration: none;
      white-space: nowrap;
      cursor: pointer;
	    }
	    button.active { background: linear-gradient(135deg, rgba(37,99,235,.18), rgba(14,165,233,.14)); border-color: rgba(37,99,235,.45); color: #1d4ed8; }
	    .auto-refresh-button.active { color: #ffffff; border-color: transparent; background: linear-gradient(135deg, #16a34a, #06b6d4); }
	    select { appearance: none; }
	    .actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; align-items: center; }
	    .chart-controls { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; align-items: center; }
    .chart-zoom { display: grid; grid-template-columns: auto minmax(180px, 1fr) auto; gap: 12px; align-items: center; margin: 12px 0 10px; }
    .zoom-buttons { display: flex; flex-wrap: wrap; gap: 8px; }
    .zoom-buttons button { padding: 8px 12px; font-size: 13px; }
    .range-label { display: flex; align-items: center; gap: 10px; color: var(--muted); font-weight: 800; }
    #chartRange { width: 100%; accent-color: var(--blue); }
    #chartWindowLabel { color: var(--muted); font-size: 13px; font-weight: 800; text-align: right; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0; }
    .card { border-radius: 22px; padding: 18px; }
    .card span { display: block; color: var(--muted); font-size: 13px; }
    .card strong { display: block; margin-top: 8px; font-size: 25px; letter-spacing: -.03em; }
    .card small { display: block; margin-top: 8px; color: var(--muted); line-height: 1.5; }
    .positive strong, .positive .tone { color: var(--green); }
    .negative strong, .negative .tone { color: var(--red); }
    .neutral strong, .neutral .tone { color: var(--yellow); }
    .split { display: grid; grid-template-columns: 1.35fr .65fr; gap: 16px; }
    .panel { border-radius: 28px; padding: 22px; margin-top: 16px; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 16px; }
    .status { color: var(--muted); font-size: 14px; }
    .chart { width: 100%; min-height: 640px; border-radius: 22px; background: #ffffff; border: 1px solid rgba(37,99,235,.12); }
    .chart-inspector { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0 8px; color: #334155; font-size: 13px; font-weight: 850; }
    .chart-inspector span { display: inline-flex; align-items: center; gap: 5px; padding: 7px 10px; border-radius: 999px; background: rgba(219,234,254,.72); border: 1px solid rgba(37,99,235,.12); }
    .chart-inspector .up { color: var(--red); background: rgba(254,226,226,.8); }
    .chart-inspector .down { color: var(--green); background: rgba(220,252,231,.82); }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 13px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .table-wrap { overflow-x: auto; }
	    .risk-grid { display: grid; gap: 12px; }
	    .risk-item { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 0; border-bottom: 1px solid var(--line); }
	    .risk-item span { color: var(--muted); }
	    .risk-item strong { font-size: 22px; }
	    .profile-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
	    .profile-card { border: 1px solid var(--line); border-radius: 22px; padding: 16px; background: rgba(248,251,255,.92); }
	    .profile-card h3 { margin: 0 0 4px; font-size: 19px; letter-spacing: -.02em; }
	    .profile-card p { font-size: 13px; margin-bottom: 10px; }
	    .profile-row { display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 10px; padding: 10px 0; border-top: 1px solid rgba(37,99,235,.10); }
	    .profile-row span { color: var(--muted); font-size: 13px; font-weight: 850; }
	    .profile-row strong { display: block; color: var(--text); font-size: 16px; }
	    .profile-row small { display: block; margin-top: 3px; color: var(--muted); line-height: 1.45; }
	    .profile-row.positive strong { color: var(--green); }
	    .profile-row.negative strong { color: var(--red); }
	    .profile-row.pending strong { color: #7c3aed; }
	    .profile-row.pending { background: rgba(124,58,237,.04); margin: 0 -8px; padding-left: 8px; padding-right: 8px; border-radius: 12px; }
	    .profile-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
	    .profile-controls { display: inline-flex; gap: 6px; padding: 4px; border-radius: 999px; background: rgba(219,234,254,.72); border: 1px solid rgba(37,99,235,.10); }
	    .profile-controls button { border: 0; background: transparent; color: var(--muted); padding: 6px 9px; border-radius: 999px; font-size: 12px; font-weight: 900; cursor: pointer; }
	    .profile-controls button.active { background: #2563eb; color: #fff; box-shadow: 0 8px 18px rgba(37,99,235,.20); }
	    .profile-history { margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(37,99,235,.12); }
	    .profile-history-summary { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; color: var(--muted); font-size: 13px; font-weight: 850; }
	    .mini-table { width: 100%; border-collapse: collapse; font-size: 12px; }
	    .mini-table th, .mini-table td { padding: 7px 6px; border-bottom: 1px solid rgba(37,99,235,.08); white-space: nowrap; }
	    .mini-table th { color: var(--muted); text-transform: none; letter-spacing: 0; }
	    .mini-table td { color: var(--text); }
	    .mini-table .positive { color: var(--green); font-weight: 900; }
	    .mini-table .negative { color: var(--red); font-weight: 900; }
	    .mini-table .neutral { color: var(--text); font-weight: 850; }
	    .empty { color: var(--muted); padding: 28px; border: 1px dashed var(--line); border-radius: 20px; }
	    @media (max-width: 1120px) { .app-shell { grid-template-columns: 1fr; } .sidebar { position: static; height: auto; } .nav-list { grid-template-columns: repeat(3, 1fr); } .profile-grid { grid-template-columns: repeat(2, 1fr); } }
	    @media (max-width: 980px) { .hero { display: block; } .hero a { display: inline-flex; margin-top: 16px; } .grid { grid-template-columns: repeat(2, 1fr); } .split { grid-template-columns: 1fr; } .chart-zoom { grid-template-columns: 1fr; } #chartWindowLabel { text-align: left; } }
	    @media (max-width: 620px) { .app-shell { width: min(100% - 22px, 1480px); } .grid, .nav-list, .profile-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">AI</span>
        <div>股智雷達 <small>STOCK LAB</small></div>
      </div>
      <nav class="nav-list" aria-label="個股分析功能">
        <a class="nav-item" href="/"><span class="nav-icon">⌂</span>看盤首頁</a>
	        <a class="nav-item active" href="#chart"><span class="nav-icon">▥</span>K 線均線</a>
	        <a class="nav-item" href="#riskPanel"><span class="nav-icon">◎</span>入場風控</a>
	        <a class="nav-item" href="#profilePanel"><span class="nav-icon">◌</span>延伸分析</a>
	        <a class="nav-item" href="/backtest"><span class="nav-icon">↗</span>回測中心</a>
	        <a class="nav-item" href="/notify"><span class="nav-icon">✉</span>推播中心</a>
        <a class="nav-item" href="#indicatorsPanel"><span class="nav-icon">▣</span>技術指標</a>
        <a class="nav-item" href="/#signalsPanel"><span class="nav-icon">✦</span>今日訊號</a>
        <a class="nav-item" href="/#modelsPanel"><span class="nav-icon">◇</span>AI 模型</a>
      </nav>
      <div class="sidebar-note">個股頁保留看盤重點：K 線、量能、MACD、入場價、停損與停利。</div>
    </aside>

  <main class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Technical Detail</p>
	        <h1 id="title">個股技術指標分析</h1>
	        <p id="subtitle">載入中...</p>
	      </div>
	      <div class="actions">
	        <button id="autoRefreshButton" class="auto-refresh-button active" onclick="toggleAutoRefresh()">⏱ 30秒自動刷新：開</button>
	        <a class="button" href="/">← 回到總覽</a>
	      </div>
	    </section>

    <section class="grid" id="summary"></section>

    <section class="split">
      <article class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Price / MA</p>
            <h2>紅綠 K 線與技術圖</h2>
          </div>
          <div class="chart-controls">
            <button data-interval="1d" class="active">日線</button>
            <button data-interval="1m">1分</button>
            <button data-interval="5m">5分</button>
            <button data-interval="30m">30分</button>
            <button data-interval="60m">60分</button>
            <select id="period"></select>
          </div>
        </div>
        <div class="status" id="chartStatus">載入 K 線...</div>
        <div class="chart-inspector" id="chartInspector">移動滑鼠查看開高低收、量能與 MACD。</div>
        <div class="chart-zoom">
          <div class="zoom-buttons">
            <button id="zoomIn" type="button">＋ 放大</button>
            <button id="zoomOut" type="button">－ 縮小</button>
            <button id="latestView" type="button">跳到最新</button>
          </div>
          <label class="range-label">左右移動 <input id="chartRange" type="range" min="0" max="0" value="0"></label>
          <span id="chartWindowLabel">—</span>
        </div>
        <div id="chart"></div>
      </article>

      <article class="panel" id="riskPanel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Risk Plan</p>
            <h2>入場與風控</h2>
          </div>
        </div>
        <div class="risk-grid" id="risk"></div>
      </article>
	    </section>

	    <section class="panel" id="profilePanel">
	      <div class="panel-head">
	        <div>
	          <p class="eyebrow">Extended Analysis</p>
	          <h2>技術、AI、基本面與籌碼框架</h2>
	        </div>
	        <div class="status">可算先顯示，未接資料清楚標示</div>
	      </div>
	      <div class="profile-grid" id="marketProfile"></div>
	    </section>

	    <section class="panel" id="indicatorsPanel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">Indicators</p>
          <h2>完整技術指標</h2>
        </div>
        <div class="status">第 2 階段計算結果</div>
      </div>
      <div class="table-wrap"><table id="indicators"></table></div>
    </section>
  </main>
  </div>

  <script>
    const ticker = __TICKER__;
    const pageMode = __MODE__;
    const periodOptions = {
      "1d": [["3mo", "3月"], ["6mo", "6月"], ["1y", "1年"], ["2y", "2年"], ["5y", "5年"], ["all", "全部"]],
      "1m": [["1d", "1日"], ["5d", "5日"], ["7d", "7日"]],
      "5m": [["1d", "1日"], ["5d", "5日"], ["1mo", "1月"], ["60d", "60日"]],
      "30m": [["5d", "5日"], ["1mo", "1月"], ["60d", "60日"]],
      "60m": [["1mo", "1月"], ["3mo", "3月"], ["6mo", "6月"], ["1y", "1年"], ["2y", "2年"]],
    };
    const defaultPeriods = {"1d": "6mo", "1m": "1d", "5m": "5d", "30m": "1mo", "60m": "3mo"};
    let currentInterval = "1d";
    let currentPeriod = "6mo";
	    let chartPayload = null;
	    let chartWindow = 120;
	    let chartStart = 0;
	    let chartControlsReady = false;
	    const refreshIntervalMs = 30000;
	    let autoRefreshEnabled = true;
	    let autoRefreshTimer = null;
	    let isRefreshing = false;
	    let profileCards = [];
	    let profilePeriods = {};

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    function fmt(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }

    function pct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
      return `${(Number(value) * 100).toFixed(1)}%`;
    }

	    async function loadStock() {
	      initChartControls();
	      await refreshStock(false);
	    }

		    async function refreshStock(preserveChartView = true) {
		      if (isRefreshing) return;
		      isRefreshing = true;
		      try {
		        const response = await fetch(`/api/technical/${encodeURIComponent(ticker)}?mode=${pageMode}`);
		        if (!response.ok) throw new Error(await response.text());
	        const data = await response.json();
	        document.title = `${data.stock_name} 技術指標分析`;
	        document.getElementById("title").textContent = `${data.stock_name} 技術指標分析`;
	        document.getElementById("subtitle").textContent = `資料日期：${data.latest_date}｜自動刷新 ${autoRefreshEnabled ? "開啟" : "關閉"}`;
		        renderSummary(data.summary_cards);
		        renderRisk(data.risk);
		        renderMarketProfile(data.market_profile);
		        renderIndicators(data.indicators);
	        await loadChart({preserveView: preserveChartView});
	      } finally {
	        isRefreshing = false;
	      }
	    }

    function renderSummary(cards) {
      document.getElementById("summary").innerHTML = cards.map(card => `
        <article class="card ${escapeHtml(card.tone)}">
          <span>${escapeHtml(card.title)}</span>
          <strong>${escapeHtml(card.value)}</strong>
          <small><span class="tone">${escapeHtml(card.status)}</span></small>
        </article>
      `).join("");
    }

	    function renderRisk(risk) {
      const rows = [
        ["推薦評分", risk.recommendation_score === undefined ? "—" : `${risk.recommendation_score} 分`],
        ["入場推薦價", fmt(risk.entry_price)],
        ["停損價", fmt(risk.stop_loss)],
        ["停利價", fmt(risk.take_profit)],
        ["建議股數", risk.position_size === undefined ? "—" : Number(risk.position_size).toLocaleString()],
      ];
      document.getElementById("risk").innerHTML = rows.map(([label, value]) => `
        <div class="risk-item"><span>${label}</span><strong>${value}</strong></div>
	      `).join("");
	    }

	    function renderMarketProfile(cards) {
	      const profile = document.getElementById("marketProfile");
	      if (!cards?.length) {
	        profile.innerHTML = `<div class="empty">尚無延伸分析資料。</div>`;
	        return;
	      }
	      profileCards = cards;
	      profile.innerHTML = cards.map((card, index) => `
	        <article class="profile-card">
	          <div class="profile-card-head">
	            <div>
	              <h3>${escapeHtml(card.title)}</h3>
	              <p>${escapeHtml(card.subtitle)}</p>
	            </div>
	            ${renderProfileControls(card, index)}
	          </div>
	          ${(card.rows || []).map(row => `
	            <div class="profile-row ${escapeHtml(row.tone || "neutral")}">
	              <span>${escapeHtml(row.label)}</span>
	              <div>
	                <strong>${escapeHtml(row.value)}</strong>
	                <small>${escapeHtml(row.note)}</small>
	              </div>
	            </div>
	          `).join("")}
	          ${renderProfileHistory(card)}
	        </article>
	      `).join("");
	    }

	    function renderProfileControls(card, index) {
	      const periods = card.controls?.periods || [];
	      if (!periods.length) return "";
	      const selected = profilePeriods[card.key] || card.controls?.default || periods[0];
	      return `<div class="profile-controls" aria-label="${escapeHtml(card.title)} 天數切換">
	        ${periods.map(days => `
	          <button type="button" class="${Number(selected) === Number(days) ? "active" : ""}" onclick="setProfilePeriod(${index}, ${Number(days)})">${Number(days)}日</button>
	        `).join("")}
	      </div>`;
	    }

	    function setProfilePeriod(index, days) {
	      const card = profileCards[index];
	      if (!card?.key) return;
	      profilePeriods[card.key] = Number(days);
	      renderMarketProfile(profileCards);
	    }

	    function renderProfileHistory(card) {
	      if (!card.controls?.type || !Array.isArray(card.history) || !card.history.length) return "";
	      const selected = Number(profilePeriods[card.key] || card.controls.default || 1);
	      const rows = card.history.slice(0, selected);
	      if (card.controls.type === "institutional") return renderInstitutionalHistory(rows, selected, card.history.length);
	      if (card.controls.type === "margin") return renderMarginHistory(rows, selected, card.history.length);
	      return "";
	    }

	    function renderInstitutionalHistory(rows, selected, available) {
	      const total = rows.reduce((sum, row) => sum + (Number(row.total_net) || 0), 0);
	      return `<div class="profile-history">
	        <div class="profile-history-summary">
	          <span>近 ${selected} 日明細</span>
	          <strong class="${toneClass(total)}">合計 ${formatShareLots(total)}</strong>
	        </div>
	        <div class="table-wrap"><table class="mini-table">
	          <thead><tr><th>日期</th><th>外資</th><th>投信</th><th>自營商</th><th>合計</th></tr></thead>
	          <tbody>${rows.map(row => `
	            <tr>
	              <td>${escapeHtml(row.date || "—")}</td>
	              <td class="${toneClass(row.foreign_net)}">${formatShareLots(row.foreign_net)}</td>
	              <td class="${toneClass(row.investment_trust_net)}">${formatShareLots(row.investment_trust_net)}</td>
	              <td class="${toneClass(row.dealer_net)}">${formatShareLots(row.dealer_net)}</td>
	              <td class="${toneClass(row.total_net)}">${formatShareLots(row.total_net)}</td>
	            </tr>
	          `).join("")}</tbody>
	        </table></div>
	        <small>已累積 ${available} 日；若資料源尚未更新，會保留最近成功抓取結果。</small>
	      </div>`;
	    }

	    function renderMarginHistory(rows, selected, available) {
	      const marginChange = rows.reduce((sum, row) => sum + (Number(row.margin_change) || 0), 0);
	      const shortChange = rows.reduce((sum, row) => sum + (Number(row.short_change) || 0), 0);
	      return `<div class="profile-history">
	        <div class="profile-history-summary">
	          <span>近 ${selected} 日明細</span>
	          <strong>資 ${formatLots(marginChange)}｜券 ${formatLots(shortChange)}</strong>
	        </div>
	        <div class="table-wrap"><table class="mini-table">
	          <thead><tr><th>日期</th><th>融資餘額</th><th>資增減</th><th>融券餘額</th><th>券增減</th><th>券資比</th></tr></thead>
	          <tbody>${rows.map(row => `
	            <tr>
	              <td>${escapeHtml(row.date || "—")}</td>
	              <td>${formatAbsLots(row.margin_balance)}</td>
	              <td class="${toneClass(row.margin_change)}">${formatLots(row.margin_change)}</td>
	              <td>${formatAbsLots(row.short_balance)}</td>
	              <td class="${toneClass(row.short_change, false)}">${formatLots(row.short_change)}</td>
	              <td>${Number.isFinite(Number(row.short_margin_ratio)) ? `${(Number(row.short_margin_ratio) * 100).toFixed(1)}%` : "—"}</td>
	            </tr>
	          `).join("")}</tbody>
	        </table></div>
	        <small>已累積 ${available} 日；融資融券需等交易所更新後才會增加新日資料。</small>
	      </div>`;
	    }

	    function toneClass(value, positiveIsGood = true) {
	      const number = Number(value);
	      if (!Number.isFinite(number) || number === 0) return "neutral";
	      const good = positiveIsGood ? number > 0 : number < 0;
	      return good ? "positive" : "negative";
	    }

	    function formatShareLots(value) {
	      const number = Number(value);
	      if (!Number.isFinite(number)) return "—";
	      return formatLots(number / 1000);
	    }

	    function formatLots(value) {
	      const number = Number(value);
	      if (!Number.isFinite(number)) return "—";
	      const sign = number > 0 ? "+" : number < 0 ? "-" : "";
	      return `${sign}${Math.abs(number).toLocaleString(undefined, { maximumFractionDigits: 0 })} 張`;
	    }

	    function formatAbsLots(value) {
	      const number = Number(value);
	      if (!Number.isFinite(number)) return "—";
	      return `${number.toLocaleString(undefined, { maximumFractionDigits: 0 })} 張`;
	    }

	    function renderIndicators(rows) {
      document.getElementById("indicators").innerHTML =
        `<thead><tr><th>指標</th><th>數值</th><th>欄位</th></tr></thead>` +
        `<tbody>${rows.map(row => `<tr>
          <td>${escapeHtml(row.label)}</td>
          <td>${escapeHtml(row.display)}</td>
          <td>${escapeHtml(row.field)}</td>
        </tr>`).join("")}</tbody>`;
    }

	    function initChartControls() {
	      if (chartControlsReady) return;
	      chartControlsReady = true;
	      document.querySelectorAll("[data-interval]").forEach(button => {
        button.addEventListener("click", async () => {
          currentInterval = button.dataset.interval;
          currentPeriod = defaultPeriods[currentInterval];
          document.querySelectorAll("[data-interval]").forEach(item => item.classList.toggle("active", item === button));
          updatePeriodOptions();
	          await loadChart();
	        });
	      });
      document.getElementById("period").addEventListener("change", async event => {
        currentPeriod = event.target.value;
        await loadChart();
      });
      document.getElementById("zoomIn").addEventListener("click", () => zoomChart(0.65));
      document.getElementById("zoomOut").addEventListener("click", () => zoomChart(1.45));
      document.getElementById("latestView").addEventListener("click", () => {
        const total = chartPayload?.candles?.length || 0;
        chartStart = Math.max(0, total - chartWindow);
        drawChart();
      });
      document.getElementById("chartRange").addEventListener("input", event => {
        chartStart = Number(event.target.value);
        drawChart();
      });
      updatePeriodOptions();
    }

    function updatePeriodOptions() {
      const select = document.getElementById("period");
      select.innerHTML = periodOptions[currentInterval].map(([value, label]) =>
        `<option value="${value}" ${value === currentPeriod ? "selected" : ""}>${label}</option>`
      ).join("");
    }

		    async function loadChart(options = {}) {
	      document.getElementById("chartStatus").textContent = "K 線資料載入中...";
	      const response = await fetch(`/api/chart/${encodeURIComponent(ticker)}?mode=${pageMode}&interval=${currentInterval}&period=${currentPeriod}`);
	      if (!response.ok) throw new Error(await response.text());
	      const payload = await response.json();
	      currentPeriod = payload.period;
	      updatePeriodOptions();
	      renderChart(payload, options);
	      document.getElementById("chartStatus").textContent = `${payload.interval}｜${payload.period}｜資料來源：${payload.source}｜${new Date().toLocaleTimeString("zh-TW", {hour12: false})} 更新`;
	    }

    function numeric(value) {
      if (value === null || value === undefined || value === "") return NaN;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : NaN;
    }

    function defaultWindowSize(interval, total) {
      const sizes = {"1d": 120, "1m": 90, "5m": 120, "30m": 140, "60m": 160};
      return Math.min(total, sizes[interval] || 120);
    }

	    function renderChart(payload, options = {}) {
	      const previousTotal = chartPayload?.candles?.length || 0;
	      const previousWindow = chartWindow;
	      const previousStart = chartStart;
	      const wasAtLatest = previousTotal === 0 || previousStart + previousWindow >= previousTotal - 1;
	      chartPayload = payload;
	      const total = chartPayload.candles?.length || 0;
	      if (options.preserveView && previousTotal > 0) {
	        chartWindow = Math.max(1, Math.min(total, previousWindow));
	        chartStart = wasAtLatest ? Math.max(0, total - chartWindow) : Math.max(0, Math.min(total - chartWindow, previousStart));
	      } else {
	        chartWindow = defaultWindowSize(payload.interval, total);
	        chartStart = Math.max(0, total - chartWindow);
	      }
	      drawChart();
	    }

	    function updateAutoRefreshButton() {
	      const button = document.getElementById("autoRefreshButton");
	      button.classList.toggle("active", autoRefreshEnabled);
	      button.textContent = autoRefreshEnabled ? "⏱ 30秒自動刷新：開" : "⏸ 30秒自動刷新：關";
	    }

		    function scheduleAutoRefresh() {
		      if (autoRefreshTimer) clearInterval(autoRefreshTimer);
		      if (autoRefreshEnabled) {
		        autoRefreshTimer = setInterval(() => refreshStock(true).catch(error => {
		          document.getElementById("subtitle").textContent = `自動刷新失敗：${error.message}`;
		        }), refreshIntervalMs);
		      }
		      updateAutoRefreshButton();
		    }

	    function toggleAutoRefresh() {
	      autoRefreshEnabled = !autoRefreshEnabled;
	      document.getElementById("subtitle").textContent = autoRefreshEnabled ? "30 秒自動刷新已開啟" : "30 秒自動刷新已關閉";
	      scheduleAutoRefresh();
	    }

    function zoomChart(factor) {
      if (!chartPayload) return;
      const total = chartPayload.candles.length;
      if (!total) return;
      const center = chartStart + chartWindow / 2;
      chartWindow = Math.max(24, Math.min(total, Math.round(chartWindow * factor)));
      chartStart = Math.max(0, Math.min(total - chartWindow, Math.round(center - chartWindow / 2)));
      drawChart();
    }

    function updateRangeControl(total) {
      const range = document.getElementById("chartRange");
      range.max = Math.max(0, total - chartWindow);
      range.value = chartStart;
      range.disabled = total <= chartWindow;
      document.getElementById("chartWindowLabel").textContent = `顯示 ${chartWindow}/${total} 根｜可滾輪縮放`;
    }

    function drawChart() {
      const allCandles = chartPayload?.candles || [];
      updateRangeControl(allCandles.length);
      const candles = allCandles.slice(chartStart, chartStart + chartWindow);
      if (!candles.length) {
        document.getElementById("chart").innerHTML = `<div class="empty">尚無走勢資料。</div>`;
        document.getElementById("chartInspector").textContent = "尚無走勢資料。";
        return;
      }
      const width = 1120, height = 700, left = 68, right = 70;
      const priceTop = 52, priceHeight = 360;
      const volumeTop = 452, volumeHeight = 88;
      const macdTop = 590, macdHeight = 82;
      const plotWidth = width - left - right;
      const step = plotWidth / Math.max(1, candles.length);
      const candleWidth = Math.max(3, Math.min(14, step * 0.68));
      const upColor = "#ef4444";
	      const downColor = "#16a34a";
	      const ma20Color = "#f59e0b";
	      const ma60Color = "#2563eb";
	      const ma120Color = "#9333ea";
	      const macdColor = "#0891b2";
      const signalColor = "#f59e0b";
      const gridColor = "#d7e3f4";
      const textColor = "#475569";
      const strongText = "#0f172a";
      const latestColor = "#0ea5e9";
	      const priceValues = candles.flatMap(row => [row.open, row.high, row.low, row.close, row.sma20, row.sma60, row.sma120].map(numeric).filter(Number.isFinite));
      let priceMin = Math.min(...priceValues);
      let priceMax = Math.max(...priceValues);
      if (!Number.isFinite(priceMin) || !Number.isFinite(priceMax)) {
        document.getElementById("chart").innerHTML = `<div class="empty">K 線價格資料不足。</div>`;
        return;
      }
      const pad = Math.max((priceMax - priceMin) * 0.06, priceMax * 0.002);
      priceMin -= pad;
      priceMax += pad;
      const priceRange = priceMax - priceMin || 1;
      const volumeMax = Math.max(...candles.map(row => numeric(row.volume)).filter(Number.isFinite), 1);
      const macdValues = candles.flatMap(row => [row.macd, row.macdSignal, row.macdHist].map(numeric).filter(Number.isFinite));
      const macdBase = macdValues.length ? macdValues : [0];
      let macdMin = Math.min(...macdBase, 0);
      let macdMax = Math.max(...macdBase, 0);
      const macdPad = Math.max((macdMax - macdMin) * 0.18, 0.01);
      macdMin -= macdPad;
      macdMax += macdPad;
      const macdRange = macdMax - macdMin || 1;

      const x = index => left + index * step + step / 2;
      const priceY = value => priceTop + (priceMax - value) / priceRange * priceHeight;
      const volumeY = value => volumeTop + volumeHeight - value / volumeMax * volumeHeight;
      const macdY = value => macdTop + (macdMax - value) / macdRange * macdHeight;
      const zeroY = macdY(0);
      const latest = candles[candles.length - 1];
      const latestClose = numeric(latest.close);

      function candleTone(row) {
        const open = numeric(row.open), close = numeric(row.close);
        if (![open, close].every(Number.isFinite)) return "neutral";
        return close >= open ? "up" : "down";
      }

      function inspectorHtml(row) {
        if (!row) return "移動滑鼠查看開高低收、量能與 MACD。";
        const tone = candleTone(row);
        const macdHist = numeric(row.macdHist);
        return `
          <span>${escapeHtml(row.date)}</span>
          <span class="${tone}">${tone === "up" ? "紅K" : tone === "down" ? "綠K" : "平盤"}</span>
          <span>開 ${fmt(row.open)}</span>
          <span>高 ${fmt(row.high)}</span>
          <span>低 ${fmt(row.low)}</span>
          <span>收 ${fmt(row.close)}</span>
          <span>量 ${fmt(numeric(row.volume) / 1000, 0)} 張</span>
          <span>MACD柱 ${Number.isFinite(macdHist) ? fmt(macdHist) : "—"}</span>
        `;
      }
      document.getElementById("chartInspector").innerHTML = inspectorHtml(latest);

      function segments(key, yFunc, color, widthValue) {
        const output = [];
        let points = [];
        candles.forEach((row, index) => {
          const value = numeric(row[key]);
          if (!Number.isFinite(value)) {
            if (points.length > 1) output.push(points);
            points = [];
            return;
          }
          points.push(`${x(index).toFixed(2)},${yFunc(value).toFixed(2)}`);
        });
        if (points.length > 1) output.push(points);
        return output.map(pointsGroup =>
          `<polyline points="${pointsGroup.join(" ")}" fill="none" stroke="${color}" stroke-width="${widthValue}" stroke-linecap="round" stroke-linejoin="round"/>`
        ).join("");
      }

      const priceGrid = [0, .25, .5, .75, 1].map(ratio => {
        const yValue = priceTop + priceHeight * ratio;
        const label = priceMax - priceRange * ratio;
        return `<line x1="${left}" y1="${yValue}" x2="${width - right}" y2="${yValue}" stroke="${gridColor}" stroke-width="1"/>
          <text x="${width - right + 10}" y="${yValue + 4}" fill="${textColor}" font-size="12">${fmt(label)}</text>`;
      }).join("");

      const verticalGrid = [0, .25, .5, .75, 1].map(ratio => {
        const xValue = left + plotWidth * ratio;
        return `<line x1="${xValue}" y1="${priceTop}" x2="${xValue}" y2="${macdTop + macdHeight}" stroke="${gridColor}" stroke-width="1" opacity=".55"/>`;
      }).join("");

      const highPoint = candles.reduce((best, row, index) => {
        const value = numeric(row.high);
        return Number.isFinite(value) && value > best.value ? {value, index, row} : best;
      }, {value: -Infinity, index: 0, row: null});
      const lowPoint = candles.reduce((best, row, index) => {
        const value = numeric(row.low);
        return Number.isFinite(value) && value < best.value ? {value, index, row} : best;
      }, {value: Infinity, index: 0, row: null});
      const highLowLabels = [
        {point: highPoint, label: "最高", color: upColor, yOffset: -10},
        {point: lowPoint, label: "最低", color: downColor, yOffset: 20},
      ].map(item => item.point.row ? `
        <g>
          <circle cx="${x(item.point.index)}" cy="${priceY(item.point.value)}" r="4" fill="${item.color}"/>
          <text x="${x(item.point.index)}" y="${priceY(item.point.value) + item.yOffset}" fill="${item.color}" font-size="12" font-weight="900" text-anchor="middle">${item.label} ${fmt(item.point.value)}</text>
        </g>` : "").join("");

      const latestLine = Number.isFinite(latestClose) ? `
        <line x1="${left}" y1="${priceY(latestClose)}" x2="${width - right}" y2="${priceY(latestClose)}" stroke="${latestColor}" stroke-width="1.4" stroke-dasharray="6 5" opacity=".9"/>
        <rect x="${width - right + 8}" y="${priceY(latestClose) - 13}" width="58" height="24" rx="7" fill="${latestColor}"/>
        <text x="${width - right + 37}" y="${priceY(latestClose) + 4}" fill="#fff" font-size="12" font-weight="900" text-anchor="middle">${fmt(latestClose)}</text>
      ` : "";

      const candleShapes = candles.map((row, index) => {
        const open = numeric(row.open), high = numeric(row.high), low = numeric(row.low), close = numeric(row.close);
        if (![open, high, low, close].every(Number.isFinite)) return "";
        const color = close >= open ? upColor : downColor;
        const cx = x(index);
        const yOpen = priceY(open), yClose = priceY(close);
        const bodyY = Math.min(yOpen, yClose);
        const bodyH = Math.max(2, Math.abs(yClose - yOpen));
        return `<g>
          <title>${escapeHtml(row.date)} 開:${fmt(open)} 高:${fmt(high)} 低:${fmt(low)} 收:${fmt(close)}</title>
          <line x1="${cx}" y1="${priceY(high)}" x2="${cx}" y2="${priceY(low)}" stroke="${color}" stroke-width="1.6"/>
          <rect x="${cx - candleWidth / 2}" y="${bodyY}" width="${candleWidth}" height="${bodyH}" rx="1.5" fill="${color}"/>
        </g>`;
      }).join("");

      const volumeBars = candles.map((row, index) => {
        const value = numeric(row.volume);
        const open = numeric(row.open), close = numeric(row.close);
        if (!Number.isFinite(value)) return "";
        const color = close >= open ? upColor : downColor;
        const barY = volumeY(value);
        return `<rect x="${x(index) - candleWidth / 2}" y="${barY}" width="${candleWidth}" height="${Math.max(1, volumeTop + volumeHeight - barY)}" fill="${color}" opacity=".6"/>`;
      }).join("");

      const macdBars = candles.map((row, index) => {
        const value = numeric(row.macdHist);
        if (!Number.isFinite(value)) return "";
        const color = value >= 0 ? upColor : downColor;
        const yValue = macdY(value);
        const top = Math.min(yValue, zeroY);
        const barHeight = Math.max(1, Math.abs(yValue - zeroY));
        return `<rect x="${x(index) - candleWidth / 2}" y="${top}" width="${candleWidth}" height="${barHeight}" fill="${color}" opacity=".55"/>`;
      }).join("");

      const midIndex = Math.floor((candles.length - 1) / 2);
      const legend = `
        <g transform="translate(${left}, 28)">
          <circle r="5" fill="${upColor}"></circle><text x="12" y="5" fill="${textColor}" font-size="13">紅K上漲</text>
          <circle cx="92" r="5" fill="${downColor}"></circle><text x="104" y="5" fill="${textColor}" font-size="13">綠K下跌</text>
	          <circle cx="198" r="5" fill="${ma20Color}"></circle><text x="210" y="5" fill="${textColor}" font-size="13">20均</text>
	          <circle cx="278" r="5" fill="${ma60Color}"></circle><text x="290" y="5" fill="${textColor}" font-size="13">60均</text>
	          <circle cx="358" r="5" fill="${ma120Color}"></circle><text x="370" y="5" fill="${textColor}" font-size="13">120均</text>
        </g>
      `;

      document.getElementById("chart").innerHTML = `
        <svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="紅綠K線、成交量與MACD">
          <rect x="0" y="0" width="${width}" height="${height}" rx="20" fill="#ffffff"/>
          ${legend}
          <text x="${left}" y="${priceTop - 14}" fill="${strongText}" font-size="13" font-weight="800">價格</text>
          ${verticalGrid}
          ${priceGrid}
          <line x1="${left}" y1="${priceTop}" x2="${left}" y2="${priceTop + priceHeight}" stroke="#b8c9e6"/>
          <line x1="${left}" y1="${priceTop + priceHeight}" x2="${width - right}" y2="${priceTop + priceHeight}" stroke="#b8c9e6"/>
          ${candleShapes}
	          ${segments("sma20", priceY, ma20Color, 2.6)}
	          ${segments("sma60", priceY, ma60Color, 2.6)}
	          ${segments("sma120", priceY, ma120Color, 2.4)}
          ${highLowLabels}
          ${latestLine}

          <text x="${left}" y="${volumeTop - 14}" fill="${strongText}" font-size="13" font-weight="800">成交量</text>
          <text x="${width - right}" y="${volumeTop - 14}" fill="${textColor}" font-size="12" text-anchor="end">高量 ${fmt(volumeMax / 1000, 0)} 張</text>
          <line x1="${left}" y1="${volumeTop + volumeHeight}" x2="${width - right}" y2="${volumeTop + volumeHeight}" stroke="#b8c9e6"/>
          ${volumeBars}

          <text x="${left}" y="${macdTop - 14}" fill="${strongText}" font-size="13" font-weight="800">MACD</text>
          <line x1="${left}" y1="${zeroY}" x2="${width - right}" y2="${zeroY}" stroke="#b8c9e6"/>
          ${macdBars}
          ${segments("macd", macdY, macdColor, 2.6)}
          ${segments("macdSignal", macdY, signalColor, 2.6)}

          <line id="hoverX" x1="0" y1="${priceTop}" x2="0" y2="${macdTop + macdHeight}" stroke="#0f172a" stroke-width="1" opacity=".25" visibility="hidden"/>
          <line id="hoverY" x1="${left}" y1="0" x2="${width - right}" y2="0" stroke="#0f172a" stroke-width="1" opacity=".20" visibility="hidden"/>
          <circle id="hoverDot" cx="0" cy="0" r="5" fill="#0ea5e9" stroke="#ffffff" stroke-width="2" visibility="hidden"/>

          <text x="${left}" y="${height - 14}" fill="${textColor}" font-size="12">${escapeHtml(candles[0].date)}</text>
          <text x="${x(midIndex)}" y="${height - 14}" fill="${textColor}" font-size="12" text-anchor="middle">${escapeHtml(candles[midIndex]?.date || "")}</text>
          <text x="${width - right}" y="${height - 14}" fill="${textColor}" font-size="12" text-anchor="end">${escapeHtml(candles[candles.length - 1].date)}</text>
        </svg>`;

      const svg = document.querySelector("#chart svg");
      const hoverX = svg?.querySelector("#hoverX");
      const hoverY = svg?.querySelector("#hoverY");
      const hoverDot = svg?.querySelector("#hoverDot");
      const setHoverVisibility = visible => {
        [hoverX, hoverY, hoverDot].forEach(element => element?.setAttribute("visibility", visible ? "visible" : "hidden"));
      };
      svg?.addEventListener("mousemove", event => {
        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        const cursor = point.matrixTransform(svg.getScreenCTM().inverse());
        if (cursor.x < left || cursor.x > width - right || cursor.y < priceTop || cursor.y > macdTop + macdHeight) {
          setHoverVisibility(false);
          return;
        }
        const index = Math.max(0, Math.min(candles.length - 1, Math.floor((cursor.x - left) / step)));
        const row = candles[index];
        const close = numeric(row.close);
        const cx = x(index);
        const cy = Number.isFinite(close) ? priceY(close) : cursor.y;
        hoverX?.setAttribute("x1", cx);
        hoverX?.setAttribute("x2", cx);
        hoverY?.setAttribute("y1", cy);
        hoverY?.setAttribute("y2", cy);
        hoverDot?.setAttribute("cx", cx);
        hoverDot?.setAttribute("cy", cy);
        setHoverVisibility(true);
        document.getElementById("chartInspector").innerHTML = inspectorHtml(row);
      });
      svg?.addEventListener("mouseleave", () => {
        setHoverVisibility(false);
        document.getElementById("chartInspector").innerHTML = inspectorHtml(latest);
      });
      svg?.addEventListener("wheel", event => {
        event.preventDefault();
        zoomChart(event.deltaY > 0 ? 1.2 : 0.8);
      }, {passive: false});
    }

	    loadStock().catch(error => {
	      document.getElementById("subtitle").textContent = `載入失敗：${error.message}`;
	    });
	    scheduleAutoRefresh();
  </script>
</body>
</html>"""
        .replace("__TICKER__", json.dumps(ticker))
        .replace("__MODE__", json.dumps(mode))
    )
