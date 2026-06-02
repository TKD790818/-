from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from .config import load_config
from .dashboard import generate_dashboard
from .notify import send_telegram_message
from .pipeline import run_pipeline


JobName = Literal["intraday", "after_close", "evening_notify"]

DEFAULT_SCHEDULE = {
    "intraday": {
        "enabled": True,
        "start": "09:05",
        "end": "13:25",
        "interval_minutes": 30,
        "daytrade_limit": 50,
    },
    "after_close": {
        "enabled": True,
        "time": "14:30",
    },
    "evening_notify": {
        "enabled": True,
        "time": "20:30",
    },
}


@dataclass(frozen=True)
class JobResult:
    job: JobName
    status: str
    message: str
    started_at: str
    finished_at: str
    output: dict[str, object]


def schedule_settings(config: dict[str, Any]) -> dict[str, Any]:
    schedule = json.loads(json.dumps(DEFAULT_SCHEDULE))
    override = config.get("schedule", {})
    if isinstance(override, dict):
        for name, settings in override.items():
            if name in schedule and isinstance(settings, dict):
                schedule[name].update(settings)
    return schedule


def automation_status(config_path: str | Path, mode: str = "real") -> dict[str, object]:
    config = load_config(config_path)
    output_dir, report_dir = _paths(mode)
    state = _read_state(output_dir)
    heartbeat = _read_scheduler_heartbeat(output_dir)
    heartbeat_age = _scheduler_heartbeat_age_seconds(config, heartbeat)
    scheduler_process_active = _scheduler_process_is_running(heartbeat)
    return {
        "mode": mode,
        "timezone": str(config.get("project", {}).get("timezone", "Asia/Taipei")),
        "now": _now(config).strftime("%Y-%m-%d %H:%M:%S"),
        "schedule": schedule_settings(config),
        "state": state,
        "artifacts": {
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "signals_exists": (output_dir / "latest_risk_plan.csv").exists(),
            "daytrade_exists": (output_dir / "daytrade_latest.csv").exists(),
            "dashboard_exists": (report_dir / "dashboard.html").exists(),
            "scheduler_log_exists": (output_dir / "scheduler.log").exists(),
            "scheduler_heartbeat_exists": bool(heartbeat),
            "scheduler_active": scheduler_process_active and heartbeat_age is not None and heartbeat_age <= 180,
            "scheduler_process_active": scheduler_process_active,
            "scheduler_heartbeat_age_seconds": heartbeat_age,
            "scheduler_heartbeat": heartbeat,
        },
    }


def run_automation_job(
    job: JobName,
    config_path: str | Path,
    mode: str = "real",
    daytrade_limit: int | None = None,
) -> JobResult:
    started = _now(load_config(config_path))
    try:
        if job == "intraday":
            output = _run_intraday(config_path, mode, daytrade_limit)
            message = "盤中行情與當沖推薦已更新。"
        elif job == "after_close":
            output = _run_after_close(config_path, mode)
            message = "收盤後完整 AI 分析、回測與報告已完成。"
        elif job == "evening_notify":
            output = _run_evening_notify(config_path, mode)
            message = str(output.get("message", "晚間 Telegram 推播任務已完成。"))
        else:
            raise ValueError(f"Unsupported job: {job}")
        result = JobResult(job, "success", message, _format_dt(started), _format_dt(_now(load_config(config_path))), output)
    except Exception as exc:
        result = JobResult(job, "failed", str(exc), _format_dt(started), _format_dt(_now(load_config(config_path))), {})

    output_dir, _ = _paths(mode)
    _write_state(output_dir, result)
    return result


def run_scheduler(
    config_path: str | Path,
    mode: str = "real",
    poll_seconds: int = 60,
) -> None:
    config = load_config(config_path)
    settings = schedule_settings(config)
    last_intraday_at: datetime | None = None
    last_after_close_day: date | None = None
    last_evening_day: date | None = None

    while True:
        config = load_config(config_path)
        settings = schedule_settings(config)
        now = _now(config)
        output_dir, _ = _paths(mode)
        _write_scheduler_heartbeat(output_dir, config, settings, poll_seconds)
        if _is_weekday(now):
            intraday = settings["intraday"]
            if intraday.get("enabled") and _within_time_window(now, str(intraday["start"]), str(intraday["end"])):
                interval = int(intraday.get("interval_minutes", 30))
                if last_intraday_at is None or (now - last_intraday_at).total_seconds() >= interval * 60:
                    result = run_automation_job("intraday", config_path, mode, int(intraday.get("daytrade_limit", 50)))
                    print(_result_line(result), flush=True)
                    last_intraday_at = now

            after_close = settings["after_close"]
            if after_close.get("enabled") and _is_time_due(now, str(after_close["time"])) and last_after_close_day != now.date():
                result = run_automation_job("after_close", config_path, mode)
                print(_result_line(result), flush=True)
                last_after_close_day = now.date()

            evening = settings["evening_notify"]
            if evening.get("enabled") and _is_time_due(now, str(evening["time"])) and last_evening_day != now.date():
                result = run_automation_job("evening_notify", config_path, mode)
                print(_result_line(result), flush=True)
                last_evening_day = now.date()

        time.sleep(max(10, poll_seconds))


def _run_intraday(config_path: str | Path, mode: str, daytrade_limit: int | None) -> dict[str, object]:
    from .web import _daytrade_payload

    config = load_config(config_path)
    output_dir, _ = _paths(mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = daytrade_limit or int(schedule_settings(config)["intraday"].get("daytrade_limit", 50))
    payload = _daytrade_payload(mode, str(config_path), limit=max(1, min(limit, 50)))
    (output_dir / "daytrade_latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(payload.get("items", [])).to_csv(output_dir / "daytrade_latest.csv", index=False)
    return {
        "source": payload.get("source"),
        "universe_count": payload.get("universe_count", 0),
        "scored_count": payload.get("scored_count", 0),
        "item_count": len(payload.get("items", [])),
        "warning": payload.get("warning", ""),
    }


def _run_after_close(config_path: str | Path, mode: str) -> dict[str, object]:
    config = load_config(config_path)
    output_dir, report_dir = _paths(mode)
    result = run_pipeline(config, demo=mode == "demo", output_dir=output_dir, send_notification=False)
    dashboard = generate_dashboard(output_dir, report_dir)
    return {
        "signals": int(len(result["signals"])),
        "dashboard": str(dashboard),
        "selected_model": result["training_metrics"].get("selected_model"),
        "cagr": result["backtest_metrics"].get("cagr"),
        "sharpe": result["backtest_metrics"].get("sharpe"),
        "max_drawdown": result["backtest_metrics"].get("max_drawdown"),
    }


def _run_evening_notify(config_path: str | Path, mode: str) -> dict[str, object]:
    from .web import _notification_payload, _telegram_status

    config = load_config(config_path)
    status = _telegram_status(config, include_secret=True)
    payload = _notification_payload(mode, str(config_path))
    if not status.get("ready"):
        return {
            "sent": False,
            "message": f"Telegram 尚未送出：{status.get('reason')}",
            "message_length": len(str(payload.get("message", ""))),
        }
    send_telegram_message(str(status["bot_token"]), str(status["chat_id"]), str(payload["message"]))
    return {
        "sent": True,
        "message": "晚間 Telegram 推播已送出。",
        "message_length": len(str(payload.get("message", ""))),
    }


def _paths(mode: str) -> tuple[Path, Path]:
    if mode == "demo":
        return Path("artifacts"), Path("reports")
    return Path("artifacts_real"), Path("reports_real")


def _read_state(output_dir: Path) -> dict[str, object]:
    path = output_dir / "automation_state.json"
    if not path.exists():
        return {"jobs": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"jobs": {}, "warning": "automation_state.json 無法解析，已忽略。"}


def _write_state(output_dir: Path, result: JobResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _read_state(output_dir)
    jobs = state.setdefault("jobs", {})
    if isinstance(jobs, dict):
        jobs[result.job] = {
            "status": result.status,
            "message": result.message,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "output": result.output,
        }
    state["updated_at"] = result.finished_at
    (output_dir / "automation_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_scheduler_heartbeat(output_dir: Path) -> dict[str, object]:
    path = output_dir / "scheduler_heartbeat.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {"warning": "scheduler_heartbeat.json 無法解析，已忽略。"}


def _write_scheduler_heartbeat(output_dir: Path, config: dict[str, Any], settings: dict[str, Any], poll_seconds: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _format_dt(_now(config)),
        "pid": os.getpid(),
        "poll_seconds": poll_seconds,
        "timezone": str(config.get("project", {}).get("timezone", "Asia/Taipei")),
        "schedule": settings,
    }
    (output_dir / "scheduler_heartbeat.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _scheduler_heartbeat_age_seconds(config: dict[str, Any], heartbeat: dict[str, object]) -> int | None:
    updated_at = heartbeat.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        return None
    try:
        updated = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    timezone = ZoneInfo(str(config.get("project", {}).get("timezone", "Asia/Taipei")))
    updated = updated.replace(tzinfo=timezone)
    return max(0, int((_now(config) - updated).total_seconds()))


def _scheduler_process_is_running(heartbeat: dict[str, object]) -> bool:
    pid = heartbeat.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _now(config: dict[str, Any]) -> datetime:
    timezone = ZoneInfo(str(config.get("project", {}).get("timezone", "Asia/Taipei")))
    return datetime.now(timezone)


def _format_dt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _parse_time(value: str) -> datetime_time:
    hour, minute = value.split(":", 1)
    return datetime_time(hour=int(hour), minute=int(minute))


def _within_time_window(now: datetime, start: str, end: str) -> bool:
    current = now.time()
    return _parse_time(start) <= current <= _parse_time(end)


def _is_time_due(now: datetime, scheduled: str) -> bool:
    return now.time() >= _parse_time(scheduled)


def _is_weekday(now: datetime) -> bool:
    return now.weekday() < 5


def _result_line(result: JobResult) -> str:
    return f"[{result.finished_at}] {result.job}: {result.status} - {result.message}"
