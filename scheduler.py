"""
Bo lap lich chay nen trong app Streamlit.

Co che:
    - Dinh ky (mac dinh 10 phut) quet thu muc data/incoming/.
    - Chi tinh cac CSV co mtime > thoi diem retrain gan nhat -> dem "so dong moi".
    - Neu so dong moi >= MIN_NEW_ROWS (mac dinh 100) -> goi retrain.run_retrain().
    - Trang thai (lan check cuoi, lan retrain cuoi, ly do trigger) luu o
      data/scheduler_state.json.

Singleton scope la module-level (mot instance moi process). Streamlit rerun
script lien tuc nen mo-dun nay dung cờ noi tai de tranh tao trung scheduler.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

import retrain as rt


BASE_DIR = Path(__file__).resolve().parent
INCOMING_DIR = BASE_DIR / "data" / "incoming"
STATE_FILE = BASE_DIR / "data" / "scheduler_state.json"

MIN_NEW_ROWS = 100
DEFAULT_INTERVAL_MINUTES = 10
JOB_ID = "retrain_check"

_scheduler: Optional[BackgroundScheduler] = None
_init_lock = threading.Lock()
_running_lock = threading.Lock()


def _default_state() -> dict:
    return {
        "last_check_at": None,
        "last_retrain_at": None,
        "last_trigger_reason": None,
        "interval_minutes": DEFAULT_INTERVAL_MINUTES,
        "min_new_rows": MIN_NEW_ROWS,
    }


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return {**_default_state(), **json.loads(STATE_FILE.read_text(encoding="utf-8"))}
        except json.JSONDecodeError:
            pass
    return _default_state()


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _last_retrain_dt(state: dict) -> Optional[datetime]:
    s = state.get("last_retrain_at")
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def count_new_rows() -> tuple[int, list[str]]:
    """Dem so dong va liet ke file CSV moi them vao incoming/ ke tu lan retrain cuoi."""
    state = _load_state()
    cutoff = _last_retrain_dt(state)
    if not INCOMING_DIR.exists():
        return 0, []

    new_rows = 0
    new_files: list[str] = []
    for p in sorted(INCOMING_DIR.glob("*.csv")):
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if cutoff is not None and mtime <= cutoff:
            continue
        try:
            n = len(pd.read_csv(p))
        except Exception:
            continue
        new_rows += n
        new_files.append(p.name)
    return new_rows, new_files


def tick() -> None:
    """Mot lan kiem tra: dem du lieu moi va trigger retrain neu vuot nguong."""
    if not _running_lock.acquire(blocking=False):
        return

    try:
        state = _load_state()
        state["last_check_at"] = datetime.now().isoformat(timespec="seconds")

        threshold = int(state.get("min_new_rows", MIN_NEW_ROWS))
        new_rows, new_files = count_new_rows()

        if new_rows >= threshold:
            try:
                rt.run_retrain()
                state["last_retrain_at"] = datetime.now().isoformat(timespec="seconds")
                state["last_trigger_reason"] = (
                    f"Trigger: {new_rows} dong moi tu {len(new_files)} file "
                    f">= nguong {threshold}"
                )
            except Exception as e:
                state["last_trigger_reason"] = f"Retrain that bai: {e}"
        else:
            state["last_trigger_reason"] = (
                f"Bo qua: {new_rows} dong moi < nguong {threshold}"
            )

        _save_state(state)
    finally:
        _running_lock.release()


def start(
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    min_new_rows: int = MIN_NEW_ROWS,
) -> BackgroundScheduler:
    """Khoi tao + chay scheduler. An toan khi goi nhieu lan (idempotent)."""
    global _scheduler
    with _init_lock:
        state = _load_state()
        state["interval_minutes"] = int(interval_minutes)
        state["min_new_rows"] = int(min_new_rows)
        _save_state(state)

        if _scheduler is not None and _scheduler.running:
            job = _scheduler.get_job(JOB_ID)
            if job is not None:
                _scheduler.reschedule_job(JOB_ID, trigger="interval", minutes=interval_minutes)
            return _scheduler

        sch = BackgroundScheduler(daemon=True)
        sch.add_job(
            tick,
            trigger="interval",
            minutes=interval_minutes,
            id=JOB_ID,
            replace_existing=True,
            next_run_time=datetime.now(),
        )
        sch.start()
        _scheduler = sch
        return sch


def stop() -> None:
    global _scheduler
    with _init_lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=False)
        _scheduler = None


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


def run_now() -> None:
    """Buoc scheduler chay tick ngay lap tuc (khong cho toi chu ky tiep theo)."""
    if _scheduler is None or not _scheduler.running:
        tick()
        return
    _scheduler.modify_job(JOB_ID, next_run_time=datetime.now())


def get_status() -> dict:
    state = _load_state()
    new_rows, new_files = count_new_rows()
    next_run = None
    if _scheduler is not None and _scheduler.running:
        job = _scheduler.get_job(JOB_ID)
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat(timespec="seconds")
    return {
        "running": is_running(),
        "interval_minutes": state.get("interval_minutes", DEFAULT_INTERVAL_MINUTES),
        "min_new_rows": state.get("min_new_rows", MIN_NEW_ROWS),
        "last_check_at": state.get("last_check_at"),
        "last_retrain_at": state.get("last_retrain_at"),
        "last_trigger_reason": state.get("last_trigger_reason"),
        "pending_new_rows": new_rows,
        "pending_new_files": new_files,
        "next_run_at": next_run,
    }
