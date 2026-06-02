from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import backtest_signal_history
from .data import build_main_df, generate_demo_main_df
from .features import add_feature_set
from .model import latest_signals, predict_signals, save_model, train_model
from .notify import format_signal_message, maybe_send_telegram
from .risk import build_risk_plan
from .sentiment import build_sentiment_data, merge_sentiment
from .universe import add_stock_names, ticker_group_map, ticker_name_map


def run_pipeline(
    config: dict[str, Any],
    demo: bool = False,
    output_dir: str | Path = "artifacts",
    send_notification: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if demo:
        main_df = generate_demo_main_df(config["data"]["tickers"])
    else:
        main_df = build_main_df(config)

    main_df = add_stock_names(main_df, config)
    news_items, sentiment = build_sentiment_data(config, include_rss=not demo)
    main_df = merge_sentiment(main_df, sentiment)
    featured = add_feature_set(
        main_df,
        horizon_days=int(config["features"]["horizon_days"]),
        benchmark_column=config["features"]["benchmark_column"],
    )

    training_result = train_model(featured, config)
    save_model(training_result, output_path / "model.joblib")

    signal_history = predict_signals(
        training_result.model,
        featured,
        training_result.feature_columns,
        buy_threshold=float(config["model"]["buy_threshold"]),
        sell_threshold=float(config["model"]["sell_threshold"]),
    )
    latest = latest_signals(signal_history)
    risk_plan = build_risk_plan(latest, config)
    test_start = training_result.predictions["date"].min()
    backtest_history = signal_history[signal_history["date"] >= test_start]
    equity_curve, backtest_metrics = backtest_signal_history(backtest_history, config)
    message = format_signal_message(risk_plan, backtest_metrics)
    sent = maybe_send_telegram(config, message) if send_notification else False

    main_df.to_csv(output_path / "main_df.csv", index=False)
    featured.to_csv(output_path / "features.csv", index=False)
    signal_history.to_csv(output_path / "signal_history.csv", index=False)
    risk_plan.to_csv(output_path / "latest_risk_plan.csv", index=False)
    equity_curve.to_csv(output_path / "equity_curve.csv", index=False)
    _build_data_coverage(config, main_df).to_csv(output_path / "data_coverage.csv", index=False)
    news_items.to_csv(output_path / "news_items.csv", index=False)
    sentiment.to_csv(output_path / "daily_sentiment.csv", index=False)
    training_result.model_comparison.to_csv(output_path / "model_comparison.csv", index=False)
    training_result.feature_importance.to_csv(output_path / "feature_importance.csv", index=False)
    pd.DataFrame([training_result.metrics | backtest_metrics]).to_csv(output_path / "metrics.csv", index=False)

    return {
        "main_df": main_df,
        "featured": featured,
        "training_metrics": training_result.metrics,
        "backtest_metrics": backtest_metrics,
        "signals": risk_plan,
        "equity_curve": equity_curve,
        "news_items": news_items,
        "sentiment": sentiment,
        "model_comparison": training_result.model_comparison,
        "feature_importance": training_result.feature_importance,
        "message": message,
        "telegram_sent": sent,
    }


def _build_data_coverage(config: dict[str, Any], main_df: pd.DataFrame) -> pd.DataFrame:
    names = ticker_name_map(config)
    configured_tickers = pd.Series(config["data"]["tickers"], name="ticker").drop_duplicates()
    if main_df.empty:
        coverage = pd.DataFrame({"ticker": configured_tickers})
        coverage["rows"] = 0
        coverage["first_date"] = pd.NaT
        coverage["last_date"] = pd.NaT
    else:
        observed = (
            main_df.groupby("ticker", as_index=False)
            .agg(rows=("date", "size"), first_date=("date", "min"), last_date=("date", "max"))
            .sort_values("ticker")
        )
        coverage = configured_tickers.to_frame().merge(observed, on="ticker", how="left")

    coverage["stock_name"] = coverage["ticker"].map(names).fillna(coverage["ticker"])
    coverage["stock_group"] = coverage["ticker"].map(ticker_group_map(config)).fillna("未分類")
    coverage["rows"] = coverage["rows"].fillna(0).astype(int)
    coverage["loaded"] = coverage["rows"].gt(0)
    return coverage[["ticker", "stock_name", "stock_group", "loaded", "rows", "first_date", "last_date"]]
