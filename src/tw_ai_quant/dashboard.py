from __future__ import annotations

from datetime import datetime
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from .features import feature_label


METRIC_LABELS = {
    "cagr": "年化報酬 CAGR",
    "sharpe": "夏普比率",
    "max_drawdown": "最大回撤",
    "total_return": "總報酬",
    "accuracy": "模型準確率",
    "roc_auc": "ROC AUC",
}
PERCENT_METRICS = {"cagr", "max_drawdown", "total_return", "accuracy", "roc_auc"}
SIGNAL_ORDER = ["BUY", "HOLD", "SELL"]
SIGNAL_LABELS = {"BUY": "🟢 買進", "HOLD": "🟡 觀望", "SELL": "🔴 賣出"}
MODEL_LABELS = {
    "random_forest": "隨機森林",
    "extra_trees": "極端隨機森林",
    "gradient_boosting": "梯度提升樹",
    "logistic_regression": "邏輯回歸",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


def generate_dashboard(artifacts_dir: str | Path = "artifacts", report_dir: str | Path = "reports") -> Path:
    artifacts_path = Path(artifacts_dir)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    metrics = _read_csv(artifacts_path / "metrics.csv")
    latest_signals = _read_csv(artifacts_path / "latest_risk_plan.csv")
    equity_curve = _read_csv(artifacts_path / "equity_curve.csv")
    signal_history = _read_csv(artifacts_path / "signal_history.csv")
    data_coverage = _read_csv(artifacts_path / "data_coverage.csv")
    news_items = _read_csv(artifacts_path / "news_items.csv")
    daily_sentiment = _read_csv(artifacts_path / "daily_sentiment.csv")
    model_comparison = _read_csv(artifacts_path / "model_comparison.csv")
    feature_importance = _read_csv(artifacts_path / "feature_importance.csv")

    dashboard_path = report_path / "dashboard.html"
    dashboard_path.write_text(
        _render_dashboard(
            metrics,
            latest_signals,
            equity_curve,
            signal_history,
            data_coverage,
            news_items,
            daily_sentiment,
            model_comparison,
            feature_importance,
            artifacts_path,
        ),
        encoding="utf-8",
    )
    return dashboard_path


def serve_dashboard(
    artifacts_dir: str | Path = "artifacts",
    report_dir: str | Path = "reports",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    dashboard_path = generate_dashboard(artifacts_dir, report_dir)
    handler = partial(SimpleHTTPRequestHandler, directory=str(dashboard_path.parent))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard: http://{host}:{port}/{dashboard_path.name}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped dashboard server.")
    finally:
        server.server_close()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _render_dashboard(
    metrics: pd.DataFrame,
    latest_signals: pd.DataFrame,
    equity_curve: pd.DataFrame,
    signal_history: pd.DataFrame,
    data_coverage: pd.DataFrame,
    news_items: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    artifacts_path: Path,
) -> str:
    metrics_row = metrics.iloc[0].to_dict() if not metrics.empty else {}
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal_counts = _signal_counts(latest_signals)
    loaded_count, total_count = _coverage_counts(data_coverage, latest_signals)
    sentiment_summary = _sentiment_summary(news_items, daily_sentiment)

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台股 AI 量化儀表板</title>
  <style>{_style()}</style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">TW AI Quant System</p>
        <h1>台股 AI 量化分析儀表板</h1>
        <p class="subtitle">整合 main_df、技術指標、情緒因子、AI 訊號、ATR 風控與回測績效。</p>
      </div>
      <div class="hero-card">
        <span>最後更新</span>
        <strong>{escape(generated_at)}</strong>
        <small>資料來源：{escape(str(artifacts_path))}</small>
        <small>股票資料：{loaded_count}/{total_count} 檔</small>
      </div>
    </section>

    <section class="metric-grid">
      {_metric_cards(metrics_row)}
    </section>

    <section class="content-grid">
      <article class="panel chart-panel">
        <div class="panel-header">
          <div>
            <p class="eyebrow">Backtest</p>
            <h2>資產曲線</h2>
          </div>
          <span class="pill">Equity Curve</span>
        </div>
        {_equity_svg(equity_curve)}
      </article>

      <article class="panel">
        <div class="panel-header">
          <div>
            <p class="eyebrow">Signal Mix</p>
            <h2>最新訊號分布</h2>
          </div>
        </div>
        {_signal_count_cards(signal_counts)}
      </article>
    </section>

    <section class="content-grid">
      <article class="panel">
        <div class="panel-header">
          <div>
            <p class="eyebrow">Model Arena</p>
            <h2>AI 模型比較</h2>
          </div>
        </div>
        {_model_comparison_table(model_comparison)}
      </article>

      <article class="panel">
        <div class="panel-header">
          <div>
            <p class="eyebrow">Feature Importance</p>
            <h2>特徵重要度</h2>
          </div>
        </div>
        {_feature_importance_list(feature_importance)}
      </article>
    </section>

    <section class="panel">
      <div class="panel-header">
        <div>
          <p class="eyebrow">News Sentiment</p>
          <h2>新聞情緒因子</h2>
        </div>
        <span class="pill">{sentiment_summary["news_count"]} 則新聞</span>
      </div>
      {_sentiment_cards(sentiment_summary)}
      {_news_table(news_items)}
    </section>

    <section class="panel">
      <div class="panel-header">
        <div>
          <p class="eyebrow">Risk Plan</p>
          <h2>最新 AI 訊號與風控</h2>
        </div>
        <span class="pill">{len(latest_signals)} 檔股票</span>
      </div>
      {_signals_table(latest_signals)}
    </section>

    <section class="panel">
      <div class="panel-header">
        <div>
          <p class="eyebrow">Data Coverage</p>
          <h2>資料抓取狀態</h2>
        </div>
        <span class="pill">{loaded_count}/{total_count} 檔成功</span>
      </div>
      {_coverage_table(data_coverage)}
    </section>

    <section class="panel">
      <div class="panel-header">
        <div>
          <p class="eyebrow">History</p>
          <h2>近期訊號摘要</h2>
        </div>
      </div>
      {_history_summary(signal_history)}
    </section>
  </main>
</body>
</html>
"""


def _style() -> str:
    return """
:root {
  color-scheme: dark;
  --bg: #08111f;
  --panel: rgba(13, 24, 43, 0.86);
  --panel-strong: rgba(20, 36, 61, 0.92);
  --line: rgba(148, 163, 184, 0.18);
  --text: #e5eefc;
  --muted: #8ea3bf;
  --cyan: #22d3ee;
  --green: #35d07f;
  --yellow: #f8c555;
  --red: #ff6b7a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--text);
  background:
    radial-gradient(circle at 15% 10%, rgba(34, 211, 238, 0.22), transparent 35%),
    radial-gradient(circle at 85% 0%, rgba(53, 208, 127, 0.16), transparent 32%),
    linear-gradient(135deg, #07101d 0%, #0b1426 52%, #07111e 100%);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
}
.shell { width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 40px 0; }
.hero { display: flex; align-items: stretch; justify-content: space-between; gap: 24px; margin-bottom: 24px; }
.eyebrow { margin: 0 0 8px; color: var(--cyan); font-size: 12px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
h1, h2, p { margin-top: 0; }
h1 { margin-bottom: 12px; font-size: clamp(34px, 5vw, 62px); line-height: 1.02; letter-spacing: -0.04em; }
h2 { margin-bottom: 0; font-size: 22px; letter-spacing: -0.02em; }
.subtitle { max-width: 760px; margin-bottom: 0; color: var(--muted); font-size: 18px; line-height: 1.7; }
.hero-card, .panel, .metric-card {
  border: 1px solid var(--line);
  background: var(--panel);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(20px);
}
.hero-card { min-width: 260px; padding: 22px; border-radius: 26px; display: flex; flex-direction: column; justify-content: center; }
.hero-card span, .hero-card small { color: var(--muted); }
.hero-card strong { margin: 8px 0; font-size: 24px; }
.metric-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 16px; }
.metric-card { padding: 18px; border-radius: 22px; }
.metric-card span { color: var(--muted); font-size: 13px; }
.metric-card strong { display: block; margin-top: 10px; font-size: 26px; letter-spacing: -0.03em; }
.content-grid { display: grid; grid-template-columns: 1.8fr 1fr; gap: 16px; margin-bottom: 16px; }
.panel { padding: 22px; border-radius: 28px; overflow: hidden; }
.panel-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
.pill, .badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 800; }
.pill { color: #a7f3ff; background: rgba(34, 211, 238, 0.12); border: 1px solid rgba(34, 211, 238, 0.25); }
.badge.buy { color: #bbf7d0; background: rgba(53, 208, 127, 0.14); }
.badge.hold { color: #fde68a; background: rgba(248, 197, 85, 0.14); }
.badge.sell { color: #fecdd3; background: rgba(255, 107, 122, 0.14); }
.badge.positive { color: #bbf7d0; background: rgba(53, 208, 127, 0.14); }
.badge.neutral { color: #fde68a; background: rgba(248, 197, 85, 0.14); }
.badge.negative { color: #fecdd3; background: rgba(255, 107, 122, 0.14); }
.signal-cards { display: grid; gap: 12px; }
.signal-card { padding: 18px; border-radius: 20px; background: var(--panel-strong); border: 1px solid var(--line); }
.signal-card span { color: var(--muted); font-size: 13px; }
.signal-card strong { display: block; margin-top: 6px; font-size: 34px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 14px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { color: var(--muted); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }
td { color: #dce7f7; }
.chart { width: 100%; min-height: 280px; }
.empty { color: var(--muted); padding: 28px; border: 1px dashed var(--line); border-radius: 20px; }
.history-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.history-item { padding: 16px; border-radius: 18px; border: 1px solid var(--line); background: rgba(15, 23, 42, 0.55); }
.history-item span { color: var(--muted); font-size: 13px; }
.history-item strong { display: block; margin-top: 8px; font-size: 24px; }
.sentiment-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.sentiment-card { padding: 16px; border-radius: 18px; border: 1px solid var(--line); background: rgba(15, 23, 42, 0.55); }
.sentiment-card span { color: var(--muted); font-size: 13px; }
.sentiment-card strong { display: block; margin-top: 8px; font-size: 24px; }
.importance-list { display: grid; gap: 12px; }
.importance-row { display: grid; gap: 8px; }
.importance-meta { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: #dce7f7; font-size: 14px; }
.importance-bar { height: 9px; overflow: hidden; border-radius: 999px; background: rgba(148, 163, 184, 0.18); }
.importance-bar span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--cyan), var(--green)); }
.model-selected { color: #bbf7d0; font-weight: 900; }
@media (max-width: 980px) {
  .hero, .content-grid { grid-template-columns: 1fr; display: grid; }
  .metric-grid, .sentiment-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 620px) {
  .shell { width: min(100% - 24px, 1180px); padding: 24px 0; }
  .metric-grid, .history-grid, .sentiment-grid { grid-template-columns: 1fr; }
}
"""


def _metric_cards(metrics_row: dict[str, Any]) -> str:
    cards = []
    for key, label in METRIC_LABELS.items():
        cards.append(
            f"""<article class="metric-card">
  <span>{escape(label)}</span>
  <strong>{escape(_format_metric(key, metrics_row.get(key)))}</strong>
</article>"""
        )
    return "\n".join(cards)


def _signal_counts(latest_signals: pd.DataFrame) -> dict[str, int]:
    if latest_signals.empty or "ml_signal" not in latest_signals:
        return {signal: 0 for signal in SIGNAL_ORDER}
    counts = latest_signals["ml_signal"].value_counts().to_dict()
    return {signal: int(counts.get(signal, 0)) for signal in SIGNAL_ORDER}


def _coverage_counts(data_coverage: pd.DataFrame, latest_signals: pd.DataFrame) -> tuple[int, int]:
    if not data_coverage.empty and "loaded" in data_coverage:
        loaded = int(data_coverage["loaded"].astype(bool).sum())
        return loaded, len(data_coverage)
    return len(latest_signals), len(latest_signals)


def _sentiment_summary(news_items: pd.DataFrame, daily_sentiment: pd.DataFrame) -> dict[str, str]:
    if news_items.empty:
        return {"news_count": "0", "average": "—", "positive": "0", "negative": "0"}

    scores = pd.to_numeric(news_items.get("sentiment_score"), errors="coerce").dropna()
    average = scores.mean() if not scores.empty else float("nan")
    labels = news_items.get("sentiment_label", pd.Series(dtype=str)).value_counts().to_dict()
    stock_count = 0 if daily_sentiment.empty else int(daily_sentiment["ticker"].nunique())
    return {
        "news_count": f"{len(news_items):,}",
        "average": "—" if pd.isna(average) else f"{average:+.2f}",
        "positive": f"{int(labels.get('positive', 0)):,}",
        "negative": f"{int(labels.get('negative', 0)):,}",
        "stock_count": f"{stock_count:,}",
    }


def _sentiment_cards(summary: dict[str, str]) -> str:
    items = [
        ("新聞筆數", summary["news_count"]),
        ("平均情緒", summary["average"]),
        ("利多新聞", summary["positive"]),
        ("利空新聞", summary["negative"]),
    ]
    rendered = "".join(
        f"""<div class="sentiment-card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"""
        for label, value in items
    )
    return f"""<div class="sentiment-grid">{rendered}</div>"""


def _news_table(news_items: pd.DataFrame) -> str:
    if news_items.empty:
        return """<div class="empty">尚未抓到新聞資料。</div>"""

    latest = news_items.sort_values("date", ascending=False).head(12)
    columns = [
        ("date", "日期"),
        ("source", "來源"),
        ("stock_name", "股名"),
        ("sentiment_label", "情緒"),
        ("sentiment_score", "分數"),
        ("title", "標題"),
    ]
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    rows = []
    for _, row in latest.iterrows():
        cells = []
        for key, _ in columns:
            if key == "sentiment_label":
                label = str(row.get(key, "neutral"))
                cells.append(f'<td><span class="badge {escape(label)}">{escape(_sentiment_text(label))}</span></td>')
            else:
                cells.append(f"<td>{escape(_format_cell(key, row.get(key)))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def _model_comparison_table(model_comparison: pd.DataFrame) -> str:
    if model_comparison.empty:
        return """<div class="empty">尚未產生模型比較資料。</div>"""

    columns = [
        ("selected", "最佳"),
        ("model", "模型"),
        ("status", "狀態"),
        ("accuracy", "準確率"),
        ("roc_auc", "ROC AUC"),
        ("selection_score", "選模分數"),
        ("error", "錯誤"),
    ]
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    rows = []
    for _, row in model_comparison.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if key == "selected":
                cells.append(f"""<td class="model-selected">{'最佳' if _is_truthy(value) else ''}</td>""")
            elif key == "model":
                cells.append(f"<td>{escape(_model_label(str(value)))}</td>")
            elif key == "status":
                cells.append(f"<td>{escape(_model_status_text(str(value)))}</td>")
            else:
                cells.append(f"<td>{escape(_format_cell(key, value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def _feature_importance_list(feature_importance: pd.DataFrame) -> str:
    if feature_importance.empty:
        return """<div class="empty">目前選出的模型沒有可直接讀取的特徵重要度。</div>"""

    rows = []
    for _, row in feature_importance.head(10).iterrows():
        pct = pd.to_numeric(row.get("importance_pct"), errors="coerce")
        pct = 0.0 if pd.isna(pct) else float(pct)
        width = max(2.0, min(100.0, pct * 100))
        label = str(row.get("feature_label") or feature_label(str(row.get("feature", ""))))
        rows.append(
            f"""<div class="importance-row">
  <div class="importance-meta"><span>{escape(label)}</span><strong>{pct:.1%}</strong></div>
  <div class="importance-bar"><span style="width: {width:.1f}%"></span></div>
</div>"""
        )
    return f"""<div class="importance-list">{''.join(rows)}</div>"""


def _signal_count_cards(signal_counts: dict[str, int]) -> str:
    cards = []
    for signal in SIGNAL_ORDER:
        cards.append(
            f"""<div class="signal-card">
  <span>{escape(SIGNAL_LABELS.get(signal, signal))}</span>
  <strong>{signal_counts[signal]}</strong>
</div>"""
        )
    return f"""<div class="signal-cards">{''.join(cards)}</div>"""


def _signals_table(latest_signals: pd.DataFrame) -> str:
    if latest_signals.empty:
        return """<div class="empty">還沒有訊號資料。請先執行 demo 或 run。</div>"""

    columns = [
        ("date", "日期"),
        ("stock_name", "股名"),
        ("close", "目前價"),
        ("prob_up", "上漲機率"),
        ("prob_down", "下跌機率"),
        ("ml_signal", "燈號"),
        ("entry_price", "入場推薦價"),
        ("stop_loss", "停損價"),
        ("take_profit", "停利價"),
        ("position_size", "股數"),
    ]
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    rows = []
    for _, row in latest_signals.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if key == "ml_signal":
                signal = str(value)
                cells.append(
                    f'<td><span class="badge {escape(signal.lower())}">{escape(SIGNAL_LABELS.get(signal, signal))}</span></td>'
                )
            else:
                cells.append(f"<td>{escape(_format_cell(key, value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def _coverage_table(data_coverage: pd.DataFrame) -> str:
    if data_coverage.empty:
        return """<div class="empty">尚未產生資料覆蓋率檔案。</div>"""

    failed = data_coverage[~data_coverage["loaded"].astype(bool)].copy()
    if failed.empty:
        return """<div class="empty">所有股票資料都已成功載入。</div>"""

    columns = [("ticker", "代號"), ("stock_name", "股名"), ("rows", "筆數")]
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    rows = []
    for _, row in failed.iterrows():
        cells = "".join(f"<td>{escape(_format_cell(key, row.get(key)))}</td>" for key, _ in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"""<p class="subtitle small">以下股票目前沒有成功抓到 Yahoo Finance 資料，可檢查是否改代號、下市櫃、或需改 `.TW/.TWO`。</p><div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def _equity_svg(equity_curve: pd.DataFrame) -> str:
    if equity_curve.empty or "equity" not in equity_curve:
        return """<div class="empty">還沒有資產曲線資料。</div>"""

    curve = equity_curve.dropna(subset=["equity"]).reset_index(drop=True)
    if len(curve) < 2:
        return """<div class="empty">資產曲線資料不足。</div>"""

    width = 900
    height = 280
    padding_x = 48
    padding_y = 28
    values = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        min_value *= 0.98
        max_value *= 1.02

    points = []
    for index, value in values.reset_index(drop=True).items():
        x = padding_x + index / (len(values) - 1) * (width - padding_x * 2)
        y = height - padding_y - (float(value) - min_value) / (max_value - min_value) * (height - padding_y * 2)
        points.append(f"{x:.2f},{y:.2f}")

    start_date = _format_cell("date", curve.iloc[0].get("date", ""))
    end_date = _format_cell("date", curve.iloc[-1].get("date", ""))
    start_value = _format_number(values.iloc[0])
    end_value = _format_number(values.iloc[-1])
    min_label = _format_number(min_value)
    max_label = _format_number(max_value)

    return f"""<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="資產曲線">
  <defs>
    <linearGradient id="equityLine" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#35d07f"/>
    </linearGradient>
  </defs>
  <line x1="{padding_x}" y1="{padding_y}" x2="{padding_x}" y2="{height - padding_y}" stroke="rgba(148,163,184,.25)"/>
  <line x1="{padding_x}" y1="{height - padding_y}" x2="{width - padding_x}" y2="{height - padding_y}" stroke="rgba(148,163,184,.25)"/>
  <polyline points="{' '.join(points)}" fill="none" stroke="url(#equityLine)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="{padding_x}" y="18" fill="#8ea3bf" font-size="12">{escape(max_label)}</text>
  <text x="{padding_x}" y="{height - 6}" fill="#8ea3bf" font-size="12">{escape(min_label)}</text>
  <text x="{padding_x}" y="{height - padding_y + 22}" fill="#8ea3bf" font-size="12">{escape(start_date)}</text>
  <text x="{width - padding_x}" y="{height - padding_y + 22}" fill="#8ea3bf" font-size="12" text-anchor="end">{escape(end_date)}</text>
  <text x="{padding_x}" y="{padding_y + 22}" fill="#e5eefc" font-size="13">起始 {escape(start_value)}</text>
  <text x="{width - padding_x}" y="{padding_y + 22}" fill="#e5eefc" font-size="13" text-anchor="end">最新 {escape(end_value)}</text>
</svg>"""


def _history_summary(signal_history: pd.DataFrame) -> str:
    if signal_history.empty:
        return """<div class="empty">還沒有歷史訊號資料。</div>"""

    latest_date = _format_cell("date", signal_history["date"].max()) if "date" in signal_history else "—"
    total_rows = len(signal_history)
    buy_ratio = 0.0
    if "ml_signal" in signal_history and total_rows:
        buy_ratio = float(signal_history["ml_signal"].eq("BUY").mean())

    items = [
        ("最新交易日", latest_date),
        ("歷史訊號筆數", f"{total_rows:,}"),
        ("BUY 訊號比例", f"{buy_ratio:.1%}"),
    ]
    rendered = "".join(
        f"""<div class="history-item"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"""
        for label, value in items
    )
    return f"""<div class="history-grid">{rendered}</div>"""


def _format_metric(key: str, value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "—"
    if key in PERCENT_METRICS:
        return f"{float(numeric):.2%}"
    return f"{float(numeric):.2f}"


def _format_cell(key: str, value: Any) -> str:
    if pd.isna(value):
        return "—"
    if key == "date":
        return str(value)[:10]
    if key in {"prob_up", "prob_down", "accuracy", "roc_auc", "selection_score", "importance_pct"}:
        numeric = pd.to_numeric(value, errors="coerce")
        return "—" if pd.isna(numeric) else f"{float(numeric):.1%}"
    if key in {"close", "entry_price", "stop_loss", "take_profit"}:
        numeric = pd.to_numeric(value, errors="coerce")
        return "—" if pd.isna(numeric) else f"{float(numeric):,.2f}"
    if key == "position_size":
        numeric = pd.to_numeric(value, errors="coerce")
        return "—" if pd.isna(numeric) else f"{int(numeric):,}"
    if key == "sentiment_score":
        numeric = pd.to_numeric(value, errors="coerce")
        return "—" if pd.isna(numeric) else f"{float(numeric):+.2f}"
    return str(value)


def _format_number(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "—"
    return f"{float(numeric):,.0f}"


def _sentiment_text(label: str) -> str:
    return {"positive": "利多", "negative": "利空", "neutral": "中性"}.get(label, label)


def _model_status_text(status: str) -> str:
    return {"success": "成功", "failed": "失敗"}.get(status, status)


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
