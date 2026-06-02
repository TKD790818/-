from __future__ import annotations

import re
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd
import certifi

from .universe import ticker_name_map


POSITIVE_TERMS = {
    "上修",
    "利多",
    "成長",
    "突破",
    "創高",
    "優於預期",
    "擴產",
    "接單",
    "營收增加",
    "獲利",
    "看好",
    "強勁",
    "旺季",
    "優於",
    "升級",
    "買超",
    "創新高",
    "漲停",
    "轉盈",
    "復甦",
    "動能",
}
NEGATIVE_TERMS = {
    "下修",
    "利空",
    "衰退",
    "跌破",
    "虧損",
    "低於預期",
    "裁員",
    "減產",
    "營收減少",
    "訴訟",
    "看淡",
    "疲弱",
    "淡季",
    "低於",
    "降評",
    "賣超",
    "跌停",
    "轉虧",
    "下滑",
    "放緩",
    "裁罰",
}
NEWS_COLUMNS = ["date", "ticker", "stock_name", "source", "title", "summary", "content", "url"]


class LexiconSentimentAnalyzer:
    def score_text(self, text: str) -> float:
        content = text or ""
        positive = sum(term in content for term in POSITIVE_TERMS)
        negative = sum(term in content for term in NEGATIVE_TERMS)
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total


def read_news_csv(path: str | Path) -> pd.DataFrame:
    news = pd.read_csv(path)
    news.columns = [str(column).strip().lower().replace(" ", "_") for column in news.columns]
    if "date" not in news.columns:
        raise ValueError("News CSV must include date column")
    if "ticker" not in news.columns:
        news["ticker"] = "MARKET"
    if "source" not in news.columns:
        news["source"] = "CSV"
    if "url" not in news.columns:
        news["url"] = ""
    for column in ["title", "content", "summary"]:
        if column not in news.columns:
            news[column] = ""
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    news["ticker"] = news["ticker"].astype(str)
    return news


def fetch_rss_news(config: dict[str, Any]) -> pd.DataFrame:
    sources = config["data"].get("news_rss_sources") or []
    if not sources:
        return _empty_news_frame()

    max_items = int(config.get("sentiment", {}).get("max_items_per_source", 50))
    timeout = int(config.get("sentiment", {}).get("request_timeout", 15))
    frames: list[pd.DataFrame] = []
    for source in sources:
        name = str(source.get("name") or source.get("url") or "RSS")
        url = str(source.get("url") or "")
        if not url:
            continue
        try:
            frames.append(_fetch_one_rss(name, url, max_items, timeout))
        except Exception:
            continue

    if not frames:
        return _empty_news_frame()
    return pd.concat(frames, ignore_index=True)


def build_news_frame(config: dict[str, Any], include_rss: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    news_csv = config["data"].get("news_csv")
    if news_csv:
        frames.append(read_news_csv(news_csv))

    if include_rss:
        rss_news = fetch_rss_news(config)
        if not rss_news.empty:
            frames.append(rss_news)

    if not frames:
        return _empty_news_frame()

    news = pd.concat(frames, ignore_index=True)
    news = _normalize_news_frame(news, config)
    analyzer = LexiconSentimentAnalyzer()
    text = news[["title", "summary", "content"]].fillna("").agg(" ".join, axis=1)
    news["sentiment_score"] = text.map(analyzer.score_text)
    news["sentiment_label"] = news["sentiment_score"].map(_sentiment_label)
    return news.sort_values(["date", "ticker", "source"], ascending=[False, True, True]).reset_index(drop=True)


def build_sentiment_data(config: dict[str, Any], include_rss: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    news = build_news_frame(config, include_rss=include_rss)
    daily = aggregate_daily_sentiment(news)
    return news, daily


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["date", "ticker", "sentiment_score", "news_count"])

    daily = (
        news.groupby(["date", "ticker"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), news_count=("sentiment_score", "size"))
        .sort_values(["date", "ticker"])
    )
    return daily


def build_daily_sentiment(config: dict[str, Any]) -> pd.DataFrame:
    _, daily = build_sentiment_data(config)
    return daily


def merge_sentiment(main_df: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    if sentiment.empty:
        enriched = main_df.copy()
        enriched["sentiment_score"] = 0.0
        enriched["news_count"] = 0
        return enriched

    stock_sentiment = sentiment[sentiment["ticker"] != "MARKET"]
    market_sentiment = sentiment[sentiment["ticker"] == "MARKET"][["date", "sentiment_score"]].rename(
        columns={"sentiment_score": "market_sentiment_score"}
    )

    enriched = main_df.merge(stock_sentiment, on=["date", "ticker"], how="left")
    enriched = enriched.merge(market_sentiment, on="date", how="left")
    enriched["sentiment_score"] = enriched["sentiment_score"].fillna(enriched["market_sentiment_score"]).fillna(0.0)
    enriched["news_count"] = enriched["news_count"].fillna(0).astype(int)
    return enriched.drop(columns=["market_sentiment_score"], errors="ignore")


def _fetch_one_rss(name: str, url: str, max_items: int, timeout: int) -> pd.DataFrame:
    request = urllib.request.Request(url, headers={"User-Agent": "tw-ai-quant/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        payload = response.read()

    root = ET.fromstring(payload)
    rows = [_rss_item_to_row(name, item) for item in root.findall(".//item")[:max_items]]
    if not rows:
        rows = [_atom_entry_to_row(name, entry) for entry in root.findall("{http://www.w3.org/2005/Atom}entry")[:max_items]]
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def _rss_item_to_row(name: str, item: ET.Element) -> dict[str, Any]:
    return {
        "date": _parse_news_date(_text(item, "pubDate")),
        "ticker": "MARKET",
        "stock_name": "市場",
        "source": name,
        "title": _clean_html(_text(item, "title")),
        "summary": _clean_html(_text(item, "description")),
        "content": _clean_html(_text(item, "content:encoded")),
        "url": _text(item, "link"),
    }


def _atom_entry_to_row(name: str, entry: ET.Element) -> dict[str, Any]:
    namespace = "{http://www.w3.org/2005/Atom}"
    link = entry.find(f"{namespace}link")
    return {
        "date": _parse_news_date(_text(entry, f"{namespace}updated") or _text(entry, f"{namespace}published")),
        "ticker": "MARKET",
        "stock_name": "市場",
        "source": name,
        "title": _clean_html(_text(entry, f"{namespace}title")),
        "summary": _clean_html(_text(entry, f"{namespace}summary")),
        "content": _clean_html(_text(entry, f"{namespace}content")),
        "url": link.attrib.get("href", "") if link is not None else "",
    }


def _normalize_news_frame(news: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    normalized = news.copy()
    for column in NEWS_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "" if column not in {"date", "ticker"} else None
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    normalized["date"] = normalized["date"].fillna(pd.Timestamp.now().normalize())
    for column in ["title", "summary", "content", "source", "url"]:
        normalized[column] = normalized[column].fillna("").astype(str)
    tagged = _tag_news_tickers(normalized, config)
    return tagged.drop_duplicates(subset=["date", "ticker", "title", "url"]).reset_index(drop=True)


def _tag_news_tickers(news: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    names = ticker_name_map(config)
    rows: list[dict[str, Any]] = []
    for _, row in news.iterrows():
        row_dict = row.to_dict()
        explicit_ticker = str(row_dict.get("ticker") or "")
        if explicit_ticker and explicit_ticker != "MARKET":
            row_dict["stock_name"] = names.get(explicit_ticker, row_dict.get("stock_name") or explicit_ticker)
            rows.append(row_dict)
            continue

        content = " ".join(str(row_dict.get(column, "")) for column in ["title", "summary", "content"])
        matched = _matched_tickers(content, names)
        if not matched:
            row_dict["ticker"] = "MARKET"
            row_dict["stock_name"] = "市場"
            rows.append(row_dict)
            continue

        for ticker in matched:
            tagged = row_dict.copy()
            tagged["ticker"] = ticker
            tagged["stock_name"] = names.get(ticker, ticker)
            rows.append(tagged)

    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def _matched_tickers(content: str, names: dict[str, str]) -> list[str]:
    matched: list[str] = []
    for ticker, stock_name in names.items():
        code = ticker.split(".")[0]
        if code in content or stock_name in content:
            matched.append(ticker)
    return matched


def _parse_news_date(value: str) -> pd.Timestamp:
    if not value:
        return pd.Timestamp.now().normalize()
    try:
        parsed = parsedate_to_datetime(value)
        return pd.Timestamp(parsed).tz_localize(None).normalize()
    except Exception:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return pd.Timestamp.now().normalize()
        if getattr(parsed, "tzinfo", None) is not None:
            parsed = parsed.tz_localize(None)
        return pd.Timestamp(parsed).normalize()


def _clean_html(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", value or "")
    cleaned = unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _text(element: ET.Element, tag: str) -> str:
    found = element.find(tag)
    if found is None or found.text is None:
        return ""
    return found.text.strip()


def _sentiment_label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS + ["sentiment_score", "sentiment_label"])
