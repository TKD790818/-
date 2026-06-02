from __future__ import annotations

from pathlib import Path

import pandas as pd

from tw_ai_quant.config import load_config
from tw_ai_quant.pipeline import run_pipeline


def test_demo_pipeline_skips_rss_fetch(monkeypatch, tmp_path: Path) -> None:
    config = load_config("configs/example.yml")

    def fail_fetch(_: dict[str, object]) -> pd.DataFrame:
        raise AssertionError("demo mode should not fetch RSS news")

    monkeypatch.setattr("tw_ai_quant.sentiment.fetch_rss_news", fail_fetch)

    result = run_pipeline(config, demo=True, output_dir=tmp_path, send_notification=False)

    assert result["news_items"].empty
    assert result["sentiment"].empty
    assert not result["signals"].empty
    assert (tmp_path / "metrics.csv").exists()
