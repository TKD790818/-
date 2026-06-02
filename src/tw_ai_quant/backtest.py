from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def backtest_signal_history(signal_history: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    backtest_config = config["backtest"]
    risk_config = config["risk"]
    initial_capital = float(backtest_config["initial_capital"])
    cost_rate = float(backtest_config["cost_rate"])
    annual_days = int(backtest_config["annual_trading_days"])
    atr_multiplier = float(risk_config["atr_stop_multiplier"])
    take_profit_r_multiple = float(risk_config["take_profit_r_multiple"])

    history = signal_history.dropna(subset=["future_return"]).copy()
    history["position"] = history["ml_signal"].eq("BUY").astype(float)
    history["realized_return"] = history["future_return"]

    has_risk_columns = {"future_high_max", "future_low_min", "atr_14"}.issubset(history.columns)
    if has_risk_columns:
        atr_risk = history["atr_14"] * atr_multiplier
        stop_return = -atr_risk / history["close"]
        take_profit_return = atr_risk * take_profit_r_multiple / history["close"]
        stop_hit = history["future_low_min"].le(history["close"] - atr_risk)
        take_profit_hit = history["future_high_max"].ge(history["close"] + atr_risk * take_profit_r_multiple)

        history.loc[history["position"].eq(1) & take_profit_hit, "realized_return"] = take_profit_return
        history.loc[history["position"].eq(1) & stop_hit, "realized_return"] = stop_return
        history.loc[history["position"].eq(1) & stop_hit & take_profit_hit, "realized_return"] = stop_return

    returns = history.pivot(index="date", columns="ticker", values="realized_return").fillna(0.0)
    positions = history.pivot(index="date", columns="ticker", values="position").reindex_like(returns).fillna(0.0)
    active_counts = positions.sum(axis=1).replace(0, np.nan)
    weights = positions.div(active_counts, axis=0).fillna(0.0)
    gross_return = (weights * returns).sum(axis=1)
    turnover_changes = weights.diff().abs().fillna(0.0)
    if not turnover_changes.empty:
        turnover_changes.iloc[0] = weights.iloc[0].abs()
    turnover = turnover_changes.sum(axis=1)
    strategy_return = gross_return - turnover * cost_rate
    equity = initial_capital * (1 + strategy_return).cumprod()
    stop_hits = history.pivot(index="date", columns="ticker", values="position").reindex_like(returns).fillna(0.0)
    stop_hits = stop_hits * 0
    take_profit_hits = stop_hits.copy()
    if has_risk_columns:
        active_stop_hits = (stop_hit & history["position"].eq(1)).astype(float)
        active_take_profit_hits = (take_profit_hit & history["position"].eq(1)).astype(float)
        stop_hits = history.assign(stop_hit=active_stop_hits).pivot(index="date", columns="ticker", values="stop_hit").reindex_like(returns).fillna(0.0)
        take_profit_hits = history.assign(take_profit_hit=active_take_profit_hits).pivot(index="date", columns="ticker", values="take_profit_hit").reindex_like(returns).fillna(0.0)

    curve = pd.DataFrame(
        {
            "date": strategy_return.index,
            "strategy_return": strategy_return.values,
            "turnover": turnover.values,
            "equity": equity.values,
            "stop_hit_count": stop_hits.sum(axis=1).astype(int).values,
            "take_profit_hit_count": take_profit_hits.sum(axis=1).astype(int).values,
        }
    )
    metrics = calculate_metrics(curve["strategy_return"], curve["equity"], annual_days)
    return curve, metrics


def calculate_metrics(returns: pd.Series, equity: pd.Series, annual_days: int) -> dict[str, float]:
    if returns.empty:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan"), "total_return": float("nan")}

    total_return = (1 + returns).prod() - 1
    years = max(len(returns) / annual_days, 1 / annual_days)
    cagr = (1 + total_return) ** (1 / years) - 1
    volatility = returns.std(ddof=0)
    sharpe = np.sqrt(annual_days) * returns.mean() / volatility if volatility > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(total_return),
    }
