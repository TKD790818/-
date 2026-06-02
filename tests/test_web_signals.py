from __future__ import annotations

from tw_ai_quant.web import _apply_intraday_signal_snapshot


def test_apply_intraday_signal_snapshot_reprices_signal() -> None:
    record = {
        "ticker": "2330.TW",
        "stock_name": "台積電",
        "close": 100.0,
        "prob_up": 0.62,
        "prob_down": 0.38,
        "ml_signal": "BUY",
        "atr_14": 5.0,
        "sma_5": 105.0,
        "sma_20": 100.0,
        "sma_60": 90.0,
        "macd": 1.2,
        "macd_signal": 0.8,
        "volume": 1_000_000,
        "volume_ma_20": 900_000,
        "volume_ratio_20": 1.0,
        "prev_high_20": 108.0,
        "prev_high_60": 115.0,
        "rsi_14": 62.0,
        "entry_price": 100.0,
        "risk_per_share": 10.0,
        "stop_loss": 90.0,
        "take_profit": 120.0,
    }
    snapshot = {
        "current_price": 110.0,
        "intraday_volume": 1_800_000,
        "volume_ratio": 1.8,
        "intraday_return": 0.03,
        "vwap": 108.0,
        "vwap_gap": 0.0185,
    }
    risk_config = {
        "capital": 1_000_000,
        "risk_per_trade": 0.01,
        "max_position_pct": 0.2,
        "lot_size": 1000,
        "atr_stop_multiplier": 2.0,
        "take_profit_r_multiple": 2.0,
    }

    updated = _apply_intraday_signal_snapshot(record, snapshot, risk_config, "Yahoo Finance 1m", "", "2026-06-02 12:00:00")

    assert updated["analysis_close"] == 100.0
    assert updated["close"] == 110.0
    assert updated["current_price"] == 110.0
    assert updated["entry_price"] == 110.0
    assert updated["stop_loss"] == 100.0
    assert updated["take_profit"] == 130.0
    assert updated["volume_ratio_20"] == 1.8
    assert updated["intraday_synced"] is True
    assert updated["position_size"] == 1000
    assert updated["recommendation_score_detail"]["conditions"]["trend_up"] is True
