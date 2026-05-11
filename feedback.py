"""
Luu phan hoi (dung/sai) cua nguoi dung ve ket qua du doan.

Moi phan hoi append vao hai file:
    1. data/incoming/feedback.csv - file CHUNG, moi dong la 1 ban ghi
       gom 21 dac trung + Diabetes_binary (theo nhan thuc te do nguoi
       dung xac nhan). Retrain va scheduler doc file nay nhu nguon du
       lieu huan luyen bo sung.
    2. data/feedback/feedback_log.csv - append mot dong day du metadata
       (nhan du doan, xac suat, nhan thuc te, is_correct, ghi chu, dac trung)
       phuc vu hien thi tren UI.

Scheduler theo doi rieng so dong "feedback.csv" qua co che row-count
snapshot (xem scheduler.py) thay vi mtime, vi append lien tuc se lam
mtime cap nhat moi lan ghi.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INCOMING_DIR = BASE_DIR / "data" / "incoming"
FEEDBACK_DIR = BASE_DIR / "data" / "feedback"
TRAINING_PATH = INCOMING_DIR / "feedback.csv"
LOG_PATH = FEEDBACK_DIR / "feedback_log.csv"
TARGET_COL = "Diabetes_binary"


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def save(
    values: dict,
    predicted_label: int,
    predicted_proba: float,
    actual_label: int,
    note: str = "",
) -> Path:
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    training_row = {**values, TARGET_COL: int(actual_label)}
    _append_csv(TRAINING_PATH, pd.DataFrame([training_row]))

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predicted_label": int(predicted_label),
        "predicted_proba": float(predicted_proba),
        "actual_label": int(actual_label),
        "is_correct": int(predicted_label == actual_label),
        "note": (note or "").strip(),
        **values,
    }
    _append_csv(LOG_PATH, pd.DataFrame([log_row]))

    return TRAINING_PATH


def load_log(limit: Optional[int] = None) -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH)
    df = df.iloc[::-1].reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df


def stats() -> dict:
    if not LOG_PATH.exists():
        return {"total": 0, "correct": 0, "wrong": 0, "accuracy": None}
    df = pd.read_csv(LOG_PATH)
    n = len(df)
    correct = int(df["is_correct"].sum()) if n else 0
    return {
        "total": n,
        "correct": correct,
        "wrong": n - correct,
        "accuracy": (correct / n) if n else None,
    }
