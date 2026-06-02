from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


PROFILE_CACHE_SECONDS = 30 * 60
REQUEST_TIMEOUT_SECONDS = 8
TWSE_FUNDAMENTAL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date=&selectType=ALLBUT0999"
TWSE_MARGIN_URL = "https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date=&selectType=ALL"
TPEX_FUNDAMENTAL_URL = "https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php?l=zh-tw&o=json"
TPEX_INSTITUTIONAL_URL = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json"
TPEX_MARGIN_URL = "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&o=json"
TDCC_CHIP_URL = "https://openapi.tdcc.com.tw/v1/opendata/1-5"
HEADERS = {"User-Agent": "Mozilla/5.0 tw-ai-quant/0.1"}
_MEMORY_CACHE: dict[str, tuple[float, Any]] = {}


def load_market_profile(ticker: str, cache_dir: Path) -> dict[str, Any]:
    code = ticker.split(".", 1)[0].strip()
    market = "tpex" if ticker.endswith(".TWO") else "twse"
    profile = {
        "ticker": ticker,
        "code": code,
        "market": "上櫃" if market == "tpex" else "上市",
        "fundamental": {},
        "institutional": {},
        "margin": {},
        "chips": {},
        "errors": [],
    }

    try:
        profile["fundamental"] = _fundamental_profile(code, market, cache_dir)
    except Exception as exc:
        profile["errors"].append(f"fundamental:{exc}")

    try:
        profile["institutional"] = _institutional_profile(code, market, cache_dir)
    except Exception as exc:
        profile["errors"].append(f"institutional:{exc}")

    try:
        profile["margin"] = _margin_profile(code, market, cache_dir)
    except Exception as exc:
        profile["errors"].append(f"margin:{exc}")

    try:
        profile["chips"] = _chip_profile(code, cache_dir)
    except Exception as exc:
        profile["errors"].append(f"chips:{exc}")

    return profile


def _fundamental_profile(code: str, market: str, cache_dir: Path) -> dict[str, Any]:
    if market == "twse":
        rows = _fetch_json("twse_fundamental", TWSE_FUNDAMENTAL_URL, cache_dir)
        row = _find_dict_row(rows, "Code", code)
        if not row:
            return {}
        return {
            "source": "TWSE",
            "date": _source_date(row.get("Date")),
            "pe": _to_number(row.get("PEratio")),
            "pb": _to_number(row.get("PBratio")),
            "dividend_yield": _to_number(row.get("DividendYield")),
            "dividend_per_share": None,
            "dividend_year": None,
            "report_period": None,
        }

    data = _fetch_json("tpex_fundamental", TPEX_FUNDAMENTAL_URL, cache_dir, verify=False)
    table = _first_table(data)
    row = _find_table_row(table, code)
    if not row:
        return {}
    return {
        "source": "TPEx",
        "date": _source_date(data.get("date") or table.get("date")),
        "pe": _to_number(_at(row, 2)),
        "pb": _to_number(_at(row, 6)),
        "dividend_yield": _to_number(_at(row, 5)),
        "dividend_per_share": _to_number(_at(row, 3)),
        "dividend_year": _at(row, 4),
        "report_period": _at(row, 7),
    }


def _institutional_profile(code: str, market: str, cache_dir: Path) -> dict[str, Any]:
    if market == "twse":
        data = _fetch_json("twse_institutional", TWSE_INSTITUTIONAL_URL, cache_dir)
        rows = _table_records(data)
        row = _find_dict_row(rows, "證券代號", code)
        if not row:
            return {}
        return {
            "source": "TWSE",
            "date": _source_date(data.get("date")),
            "foreign_net": _to_number(row.get("外陸資買賣超股數(不含外資自營商)")),
            "investment_trust_net": _to_number(row.get("投信買賣超股數")),
            "dealer_net": _to_number(row.get("自營商買賣超股數")),
            "total_net": _to_number(row.get("三大法人買賣超股數")),
        }

    data = _fetch_json("tpex_institutional", TPEX_INSTITUTIONAL_URL, cache_dir, verify=False)
    table = _first_table(data)
    row = _find_table_row(table, code)
    if not row:
        return {}
    return {
        "source": "TPEx",
        "date": _source_date(data.get("date") or table.get("date")),
        "foreign_net": _to_number(_at(row, 4)),
        "investment_trust_net": _to_number(_at(row, 13)),
        "dealer_net": _to_number(_at(row, 22)),
        "total_net": _to_number(_at(row, 23)),
    }


def _margin_profile(code: str, market: str, cache_dir: Path) -> dict[str, Any]:
    if market == "twse":
        data = _fetch_json("twse_margin", TWSE_MARGIN_URL, cache_dir)
        tables = data.get("tables") if isinstance(data, dict) else []
        table = tables[1] if isinstance(tables, list) and len(tables) > 1 else {}
        row = _find_table_row(table, code)
        if not row:
            return {}
        margin_balance = _to_number(_at(row, 6))
        previous_margin = _to_number(_at(row, 5))
        short_balance = _to_number(_at(row, 12))
        previous_short = _to_number(_at(row, 11))
        return {
            "source": "TWSE",
            "date": _source_date(data.get("date")),
            "margin_balance": margin_balance,
            "margin_change": _difference(margin_balance, previous_margin),
            "margin_usage_rate": None,
            "short_balance": short_balance,
            "short_change": _difference(short_balance, previous_short),
            "short_usage_rate": None,
            "short_margin_ratio": _ratio(short_balance, margin_balance),
        }

    data = _fetch_json("tpex_margin", TPEX_MARGIN_URL, cache_dir, verify=False)
    table = _first_table(data)
    row = _find_table_row(table, code)
    if not row:
        return {}
    margin_balance = _to_number(_at(row, 6))
    previous_margin = _to_number(_at(row, 2))
    short_balance = _to_number(_at(row, 14))
    previous_short = _to_number(_at(row, 10))
    return {
        "source": "TPEx",
        "date": _source_date(data.get("date") or table.get("date")),
        "margin_balance": margin_balance,
        "margin_change": _difference(margin_balance, previous_margin),
        "margin_usage_rate": _to_number(_at(row, 8)),
        "short_balance": short_balance,
        "short_change": _difference(short_balance, previous_short),
        "short_usage_rate": _to_number(_at(row, 16)),
        "short_margin_ratio": _ratio(short_balance, margin_balance),
    }


def _chip_profile(code: str, cache_dir: Path) -> dict[str, Any]:
    rows = _fetch_json("tdcc_chips", TDCC_CHIP_URL, cache_dir)
    stock_rows = []
    for row in rows if isinstance(rows, list) else []:
        normalized = {str(key).lstrip("\ufeff"): value for key, value in row.items()}
        if str(normalized.get("證券代號", "")).strip() == code:
            stock_rows.append(normalized)
    if not stock_rows:
        return {}

    by_level = {
        int(level): row
        for row in stock_rows
        if (level := str(row.get("持股分級", "")).strip()).isdigit()
    }
    total = by_level.get(17, {})
    odd_lot = by_level.get(1, {})
    small_levels = [by_level[level] for level in range(1, 6) if level in by_level]
    large_levels = [by_level[level] for level in range(12, 17) if level in by_level]
    thousand_plus = by_level.get(15, {})
    return {
        "source": "TDCC",
        "date": _source_date((stock_rows[0] or {}).get("資料日期")),
        "total_holders": _to_number(total.get("人數")),
        "total_shares": _to_number(total.get("股數")),
        "odd_lot_holders": _to_number(odd_lot.get("人數")),
        "odd_lot_ratio": _to_number(odd_lot.get("占集保庫存數比例%")),
        "small_holder_ratio": _sum_numbers(row.get("占集保庫存數比例%") for row in small_levels),
        "large_holder_count": _sum_numbers(row.get("人數") for row in large_levels),
        "large_holder_ratio": _sum_numbers(row.get("占集保庫存數比例%") for row in large_levels),
        "thousand_lot_holders": _to_number(thousand_plus.get("人數")),
        "thousand_lot_ratio": _to_number(thousand_plus.get("占集保庫存數比例%")),
    }


def _fetch_json(key: str, url: str, cache_dir: Path, verify: bool = True) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    memory_key = str(cache_path)
    now = time.time()
    cached = _MEMORY_CACHE.get(memory_key)
    if cached and now - cached[0] <= PROFILE_CACHE_SECONDS:
        return cached[1]

    if cache_path.exists() and now - cache_path.stat().st_mtime <= PROFILE_CACHE_SECONDS:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        _MEMORY_CACHE[memory_key] = (now, data)
        return data

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS, verify=verify)
        response.raise_for_status()
        data = response.json()
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        _MEMORY_CACHE[memory_key] = (now, data)
        return data
    except Exception:
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            _MEMORY_CACHE[memory_key] = (now, data)
            return data
        raise


def _table_records(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    fields = data.get("fields") or []
    rows = data.get("data") or []
    return [dict(zip(fields, row, strict=False)) for row in rows]


def _first_table(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    tables = data.get("tables")
    if not isinstance(tables, list) or not tables:
        return {}
    return tables[0] if isinstance(tables[0], dict) else {}


def _find_dict_row(rows: Any, key: str, code: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get(key, "")).strip() == code:
            return row
    return None


def _find_table_row(table: dict[str, Any], code: str) -> list[Any] | None:
    rows = table.get("data") if isinstance(table, dict) else []
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, list) and str(_at(row, 0)).strip() == code:
            return row
    return None


def _at(row: list[Any], index: int) -> Any:
    return row[index] if index < len(row) else None


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    if not text or text.upper() in {"N/A", "NA", "NULL", "--", "－"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sum_numbers(values: Any) -> float | None:
    total = 0.0
    has_value = False
    for value in values:
        number = _to_number(value)
        if number is not None:
            total += number
            has_value = True
    return total if has_value else None


def _difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _source_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 3 and parts[0].isdigit():
            year = int(parts[0])
            if year < 1911:
                year += 1911
            return f"{year:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) == 7:
        year = int(digits[:3]) + 1911
        return f"{year:04d}-{int(digits[3:5]):02d}-{int(digits[5:7]):02d}"
    if len(digits) == 8:
        return f"{int(digits[:4]):04d}-{int(digits[4:6]):02d}-{int(digits[6:8]):02d}"
    return text
