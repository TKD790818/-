from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]

    rename_map = {
        "datetime": "date",
        "symbol": "ticker",
        "stock_id": "ticker",
        "adj_close": "adj_close",
        "adjclose": "adj_close",
    }
    normalized = normalized.rename(columns=rename_map)

    missing = [column for column in PRICE_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Price data missing columns: {missing}")

    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
    normalized["ticker"] = normalized["ticker"].astype(str)
    numeric_columns = [column for column in normalized.columns if column not in {"date", "ticker"}]
    normalized[numeric_columns] = normalized[numeric_columns].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close", "volume"])
    normalized = normalized[normalized["close"].gt(0)]
    return normalized.sort_values(["date", "ticker"]).reset_index(drop=True)


def read_prices_csv(path: str | Path) -> pd.DataFrame:
    return normalize_price_frame(pd.read_csv(path))


def fetch_yfinance_prices(tickers: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    ticker_list = list(tickers)
    raw = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No price data returned from yfinance")

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in ticker_list:
            if ticker not in raw.columns.get_level_values(0):
                continue
            part = raw[ticker].reset_index()
            part["ticker"] = ticker
            frames.append(part)
    else:
        part = raw.reset_index()
        part["ticker"] = ticker_list[0]
        frames.append(part)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    normalized = normalize_price_frame(combined)
    if normalized.empty:
        raise RuntimeError("No valid price rows returned from yfinance")
    return normalized


def fetch_yfinance_close(symbol: str, start: str, end: str | None, column_name: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["date", column_name])

    close_series = _extract_close_series(raw)
    close = close_series.rename(column_name).reset_index().rename(columns={"Date": "date"})
    close["date"] = pd.to_datetime(close["date"]).dt.tz_localize(None)
    close[column_name] = pd.to_numeric(close[column_name], errors="coerce")
    return close.dropna(subset=[column_name])


def _extract_close_series(raw: pd.DataFrame) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("yfinance response does not include Close column")
        return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close

    close = raw["Close"]
    return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close


def read_macro_csv(path: str | Path) -> pd.DataFrame:
    macro = pd.read_csv(path)
    macro.columns = [str(column).strip().lower().replace(" ", "_") for column in macro.columns]
    if "date" not in macro.columns:
        raise ValueError("Macro CSV must include date column")
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    for column in macro.columns:
        if column != "date":
            macro[column] = pd.to_numeric(macro[column], errors="coerce")
    return macro.sort_values("date")


def build_main_df(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]

    if data_config.get("prices_csv"):
        prices = read_prices_csv(data_config["prices_csv"])
    else:
        prices = fetch_yfinance_prices(data_config["tickers"], data_config["start"], data_config.get("end"))

    start = data_config["start"]
    end = data_config.get("end")

    main_df = prices.copy()
    benchmark = data_config.get("benchmark")
    if benchmark:
        benchmark_df = fetch_yfinance_close(benchmark, start, end, "benchmark_close")
        main_df = main_df.merge(benchmark_df, on="date", how="left")

    vix = data_config.get("vix")
    if vix:
        vix_df = fetch_yfinance_close(vix, start, end, "vix_close")
        main_df = main_df.merge(vix_df, on="date", how="left")

    us10y = data_config.get("us10y")
    if us10y:
        us10y_df = fetch_yfinance_close(us10y, start, end, "us10y_close")
        main_df = main_df.merge(us10y_df, on="date", how="left")

    if data_config.get("macro_csv"):
        macro = read_macro_csv(data_config["macro_csv"])
        main_df = pd.merge_asof(
            main_df.sort_values("date"),
            macro.sort_values("date"),
            on="date",
            direction="backward",
        )

    main_df = main_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    market_columns = [column for column in ["benchmark_close", "vix_close", "us10y_close"] if column in main_df]
    if market_columns:
        main_df[market_columns] = main_df.groupby("ticker")[market_columns].ffill().bfill()

    return main_df


def generate_demo_main_df(
    tickers: Iterable[str] = ("2330.TW", "2317.TW", "2454.TW"),
    periods: int = 520,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    benchmark_returns = rng.normal(0.00035, 0.011, periods)
    benchmark_close = 15_000 * np.cumprod(1 + benchmark_returns)
    vix_close = np.clip(18 - benchmark_returns * 320 + rng.normal(0, 1.8, periods), 8, 55)
    us10y_close = np.clip(3.8 + rng.normal(0, 0.04, periods).cumsum() / 10, 0.5, 7.0)
    cpi = 2.0 + np.sin(np.arange(periods) / 60) * 0.7 + rng.normal(0, 0.05, periods)
    gdp = 2.8 + np.sin(np.arange(periods) / 90) * 1.2 + rng.normal(0, 0.08, periods)
    pmi = 50 + np.sin(np.arange(periods) / 35) * 4 + rng.normal(0, 0.8, periods)

    frames: list[pd.DataFrame] = []
    for index, ticker in enumerate(tickers):
        alpha = 0.00005 + (index % 10) * 0.00002
        beta = 0.75 + (index % 12) * 0.06
        idiosyncratic = rng.normal(0, 0.012 + (index % 8) * 0.0015, periods)
        returns = alpha + beta * benchmark_returns + idiosyncratic
        base_price = 25 + (index % 18) * 18 + rng.uniform(0, 12)
        close = base_price * np.cumprod(1 + returns)
        open_price = close * (1 + rng.normal(0, 0.004, periods))
        high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.018, periods))
        low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.018, periods))
        volume = rng.integers(8_000_000, 80_000_000, periods) * (1 + np.abs(returns) * 12)
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume.astype(int),
                "benchmark_close": benchmark_close,
                "vix_close": vix_close,
                "us10y_close": us10y_close,
                "cpi": cpi,
                "gdp": gdp,
                "pmi": pmi,
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
