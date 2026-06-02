from __future__ import annotations

import argparse
from pathlib import Path

from .automation import run_automation_job, run_scheduler
from .config import load_config
from .dashboard import generate_dashboard, serve_dashboard
from .pipeline import run_pipeline
from .web import run_web_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taiwan AI quant analysis pipeline")
    parser.add_argument("--config", default="configs/example.yml", help="Path to YAML config")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    parser.add_argument("--report-dir", default="reports", help="Directory for dashboard HTML")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run a full offline demo with synthetic market data")
    subparsers.add_parser("run", help="Run with configured real data sources")
    subparsers.add_parser("report", help="Generate dashboard HTML from existing artifacts")
    serve = subparsers.add_parser("serve", help="Generate and serve dashboard locally")
    serve.add_argument("--host", default="127.0.0.1", help="Dashboard server host")
    serve.add_argument("--port", type=int, default=8000, help="Dashboard server port")
    web = subparsers.add_parser("web", help="Start FastAPI web app")
    web.add_argument("--host", default="127.0.0.1", help="Web app host")
    web.add_argument("--port", type=int, default=8000, help="Web app port")
    job = subparsers.add_parser("job", help="Run one automation job")
    job.add_argument("name", choices=["intraday", "after_close", "evening_notify"], help="Job to run")
    job.add_argument("--mode", choices=["demo", "real"], default="real", help="Data mode")
    job.add_argument("--daytrade-limit", type=int, default=None, help="Intraday candidate limit")
    schedule = subparsers.add_parser("schedule", help="Run the local automation scheduler loop")
    schedule.add_argument("--mode", choices=["demo", "real"], default="real", help="Data mode")
    schedule.add_argument("--poll-seconds", type=int, default=60, help="Scheduler polling interval")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "report":
        dashboard_path = generate_dashboard(args.output_dir, args.report_dir)
        print(f"Dashboard generated: {dashboard_path}")
        return
    if args.command == "serve":
        serve_dashboard(args.output_dir, args.report_dir, args.host, args.port)
        return
    if args.command == "web":
        run_web_app(args.config, args.host, args.port)
        return
    if args.command == "job":
        result = run_automation_job(args.name, args.config, args.mode, args.daytrade_limit)
        print(f"{result.job}: {result.status} - {result.message}")
        for key, value in result.output.items():
            print(f"- {key}: {value}")
        return
    if args.command == "schedule":
        run_scheduler(args.config, args.mode, args.poll_seconds)
        return

    config = load_config(args.config)
    result = run_pipeline(config, demo=args.command == "demo", output_dir=Path(args.output_dir))
    dashboard_path = generate_dashboard(args.output_dir, args.report_dir)

    print("Training metrics")
    for key, value in result["training_metrics"].items():
        if isinstance(value, (int, float)):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")

    print("\nBacktest metrics")
    for key, value in result["backtest_metrics"].items():
        print(f"- {key}: {value:.4f}")

    print("\nLatest signals")
    printable = result["signals"][
        [
            "date",
            "stock_name",
            "close",
            "prob_up",
            "prob_down",
            "ml_signal",
            "entry_price",
            "stop_loss",
            "take_profit",
            "position_size",
        ]
    ]
    print(printable.to_string(index=False))

    print("\nTelegram message")
    print(result["message"])
    print(f"\nDashboard generated: {dashboard_path}")


if __name__ == "__main__":
    main()
