from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


def format_signal_message(signals: pd.DataFrame, metrics: dict[str, float] | None = None) -> str:
    lines = ["📈 台股 AI 量化訊號"]
    if metrics:
        lines.append(
            "回測："
            f"CAGR {metrics.get('cagr', float('nan')):.2%}｜"
            f"Sharpe {metrics.get('sharpe', float('nan')):.2f}｜"
            f"MDD {metrics.get('max_drawdown', float('nan')):.2%}"
        )

    for _, row in signals.sort_values("prob_up", ascending=False).iterrows():
        position_size = int(row.get("position_size", 0))
        stop_loss = row.get("stop_loss")
        take_profit = row.get("take_profit")
        stock_label = f"{row['ticker']} {row.get('stock_name', '')}".strip()
        risk_text = ""
        if row["ml_signal"] == "BUY" and position_size > 0:
            risk_text = f"｜張數 {position_size // 1000}｜停損 {stop_loss:.2f}｜停利 {take_profit:.2f}"
        lines.append(f"{stock_label}：{row['ml_signal']}｜上漲機率 {row['prob_up']:.1%}{risk_text}")

    return "\n".join(lines)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else "連線失敗"
        raise RuntimeError(f"Telegram 推播失敗：{status}，請檢查 Token、Chat ID 或網路連線。") from None


def maybe_send_telegram(config: dict[str, Any], text: str) -> bool:
    telegram_config = config["telegram"]
    if not telegram_config.get("enabled"):
        return False

    bot_token = os.getenv(telegram_config["bot_token_env"], "")
    chat_id = os.getenv(telegram_config["chat_id_env"], "")
    if not bot_token or not chat_id:
        return False

    send_telegram_message(bot_token, chat_id, text)
    return True
