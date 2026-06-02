from __future__ import annotations

import math

import pandas as pd

from tw_ai_quant.backtest import calculate_metrics
from tw_ai_quant.backtest import backtest_signal_history


def test_calculate_metrics_includes_first_day_return() -> None:
    returns = pd.Series([0.10])
    equity = pd.Series([1_100_000.0])

    metrics = calculate_metrics(returns, equity, annual_days=252)

    assert math.isclose(metrics["total_return"], 0.10)
    assert math.isclose(metrics["cagr"], (1.10**252) - 1)
    assert math.isnan(metrics["sharpe"])
    assert math.isclose(metrics["max_drawdown"], 0.0)


def test_backtest_signal_history_applies_stop_loss_before_take_profit() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["2330.TW"],
            "ml_signal": ["BUY"],
            "close": [100.0],
            "future_return": [0.08],
            "future_high_max": [120.0],
            "future_low_min": [85.0],
            "atr_14": [5.0],
        }
    )
    config = {
        "backtest": {"initial_capital": 1_000_000, "cost_rate": 0.0015, "annual_trading_days": 252},
        "risk": {"atr_stop_multiplier": 2.0, "take_profit_r_multiple": 2.0},
    }

    curve, metrics = backtest_signal_history(history, config)

    assert math.isclose(curve.loc[0, "strategy_return"], -0.1015)
    assert curve.loc[0, "stop_hit_count"] == 1
    assert curve.loc[0, "take_profit_hit_count"] == 1
    assert math.isclose(metrics["total_return"], -0.1015)
