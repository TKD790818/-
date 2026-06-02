from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_risk_plan(signals: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    risk_config = config["risk"]
    capital = float(risk_config["capital"])
    risk_per_trade = float(risk_config["risk_per_trade"])
    max_position_pct = float(risk_config["max_position_pct"])
    lot_size = int(risk_config["lot_size"])
    atr_multiplier = float(risk_config["atr_stop_multiplier"])
    take_profit_r_multiple = float(risk_config["take_profit_r_multiple"])

    plan = signals.copy()
    plan["risk_budget"] = capital * risk_per_trade
    plan["entry_price"] = plan["close"]
    long_watchlist = plan["ml_signal"].isin(["BUY", "HOLD"])
    atr_risk = (plan["atr_14"] * atr_multiplier).where(long_watchlist)
    plan["stop_loss"] = (plan["entry_price"] - atr_risk).clip(lower=0)
    plan["risk_per_share"] = (plan["entry_price"] - plan["stop_loss"]).clip(lower=0)
    plan["take_profit"] = (plan["entry_price"] + plan["risk_per_share"] * take_profit_r_multiple).where(long_watchlist)
    risk_shares = np.floor(plan["risk_budget"] / plan["risk_per_share"].replace(0, np.nan))
    capital_shares = np.floor((capital * max_position_pct) / plan["close"])
    raw_position_size = np.where(
        plan["ml_signal"].eq("BUY"),
        np.minimum(risk_shares.fillna(0), capital_shares.fillna(0)),
        0,
    ).astype(int)
    plan["position_size"] = (raw_position_size // lot_size * lot_size).astype(int)
    plan["position_value"] = plan["position_size"] * plan["close"]
    plan["portfolio_weight"] = plan["position_value"] / capital
    return plan
