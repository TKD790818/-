from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
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
    "beta_60",
    "volume_ratio_20",
    "daily_return",
    "benchmark_return",
    "vix_close",
    "us10y_close",
    "cpi",
    "gdp",
    "pmi",
    "sentiment_score",
]

FEATURE_LABELS = {
    "sma_5": "5日均線",
    "sma_20": "20日均線",
    "sma_60": "60日均線",
    "sma_120": "120日均線",
    "ema_12": "12日EMA",
    "ema_26": "26日EMA",
    "rsi_14": "RSI相對強弱",
    "kd_k": "KD-K值",
    "kd_d": "KD-D值",
    "macd": "MACD差離值",
    "macd_signal": "MACD訊號線",
    "macd_hist": "MACD柱狀體",
    "bb_width": "布林通道寬度",
    "bb_percent": "布林通道位置",
    "atr_14": "ATR波動度",
    "momentum_10": "10日動能",
    "obv": "OBV能量潮",
    "vwap_20": "20日VWAP",
    "beta_60": "60日Beta",
    "volume_ratio_20": "20日量能比",
    "daily_return": "個股日報酬",
    "benchmark_return": "大盤日報酬",
    "vix_close": "VIX恐慌指數",
    "us10y_close": "美國10年債殖利率",
    "cpi": "CPI消費者物價",
    "gdp": "GDP經濟成長",
    "pmi": "PMI採購經理人指數",
    "sentiment_score": "新聞情緒分數",
}


def add_feature_set(
    main_df: pd.DataFrame,
    horizon_days: int = 1,
    benchmark_column: str = "benchmark_close",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, part in main_df.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        enriched = part.copy()
        close = enriched["close"]
        high = enriched["high"]
        low = enriched["low"]
        volume = enriched["volume"]

        enriched["daily_return"] = close.pct_change()
        if benchmark_column in enriched:
            enriched["benchmark_return"] = enriched[benchmark_column].pct_change()
        else:
            enriched["benchmark_return"] = np.nan

        enriched["sma_5"] = close.rolling(5).mean()
        enriched["sma_20"] = close.rolling(20).mean()
        enriched["sma_60"] = close.rolling(60).mean()
        enriched["sma_120"] = close.rolling(120).mean()
        enriched["ema_12"] = close.ewm(span=12, adjust=False).mean()
        enriched["ema_26"] = close.ewm(span=26, adjust=False).mean()
        enriched["rsi_14"] = _rsi(close, 14)

        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        raw_k = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
        enriched["kd_k"] = raw_k.rolling(3).mean()
        enriched["kd_d"] = enriched["kd_k"].rolling(3).mean()

        enriched["macd"] = enriched["ema_12"] - enriched["ema_26"]
        enriched["macd_signal"] = enriched["macd"].ewm(span=9, adjust=False).mean()
        enriched["macd_hist"] = enriched["macd"] - enriched["macd_signal"]

        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        enriched["bb_width"] = (bb_upper - bb_lower) / bb_middle
        enriched["bb_percent"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        enriched["atr_14"] = true_range.rolling(14).mean()

        enriched["momentum_10"] = close / close.shift(10) - 1
        enriched["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        typical_price = (high + low + close) / 3
        enriched["vwap_20"] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        enriched["volume_ratio_20"] = volume / volume.rolling(20).mean()

        benchmark_variance = enriched["benchmark_return"].rolling(60).var()
        covariance = enriched["daily_return"].rolling(60).cov(enriched["benchmark_return"])
        enriched["beta_60"] = covariance / benchmark_variance.replace(0, np.nan)

        future_highs = pd.concat([high.shift(-step) for step in range(1, horizon_days + 1)], axis=1)
        future_lows = pd.concat([low.shift(-step) for step in range(1, horizon_days + 1)], axis=1)
        enriched["future_high_max"] = future_highs.max(axis=1)
        enriched["future_low_min"] = future_lows.min(axis=1)
        enriched["future_return"] = close.shift(-horizon_days) / close - 1
        enriched["target_up"] = (enriched["future_return"] > 0).astype(float)
        enriched.loc[enriched["future_return"].isna(), "target_up"] = np.nan
        frames.append(enriched)

    featured = pd.concat(frames, ignore_index=True)
    return featured.sort_values(["date", "ticker"]).reset_index(drop=True)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in BASE_FEATURE_COLUMNS
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])
    ]


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
