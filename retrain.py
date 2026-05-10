"""
Pipeline tai huan luyen lien tuc theo co che Champion - Challenger.

Quy trinh:
    1. Doc du lieu nen + bo sung du lieu moi tu data/incoming/ (neu co).
    2. Doi chieu schema & phat hien drift co ban so voi data_profile.csv goc.
    3. Huan luyen 5 mo hinh (Logistic, DT, RF, GB, KNN) bang GridSearchCV.
    4. So sanh recall cua mo hinh moi (challenger) voi mo hinh hien tai (champion).
    5. Promote neu thang, snapshot tat ca artefact vao outputs/runs/run_<timestamp>/.

Co the chay tu CLI:
    python retrain.py
hoac goi tu Streamlit:
    from retrain import run_retrain
    info = run_retrain()
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
INCOMING_DIR = BASE_DIR / "data" / "incoming"
CHAMPION_DIR = BASE_DIR / "outputs" / "diabetes_brfss_v1"
RUNS_DIR = BASE_DIR / "outputs" / "runs"
TARGET_COL = "Diabetes_binary"
RANDOM_STATE = 42
SELECTION_METRIC = "recall"
DRIFT_THRESHOLD = 0.15  # |delta_mean / mean_baseline| > 15% se canh bao


@dataclass
class RetrainSummary:
    timestamp: str
    n_rows: int
    new_rows_added: int
    schema_ok: bool
    schema_issues: list
    drift_columns: list
    challenger_model: str
    challenger_recall: float
    champion_model: Optional[str]
    champion_recall: Optional[float]
    promoted: bool
    decision_reason: str
    run_dir: str


def _ensure_dirs() -> None:
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_combined_dataset() -> tuple[pd.DataFrame, int]:
    """Doc CSV goc, sau do append cac file moi trong data/incoming/."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay du lieu goc: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    new_rows = 0
    if INCOMING_DIR.exists():
        for csv_path in sorted(INCOMING_DIR.glob("*.csv")):
            extra = pd.read_csv(csv_path)
            extra = extra[df.columns.intersection(extra.columns)]
            df = pd.concat([df, extra], ignore_index=True)
            new_rows += len(extra)
            print(f"Da nap them {len(extra)} dong tu {csv_path.name}")

    df = df.drop_duplicates().reset_index(drop=True)
    return df, new_rows


def validate_schema(df: pd.DataFrame, profile_path: Path) -> tuple[bool, list[str]]:
    """Doi chieu cot, kieu du lieu, mien gia tri voi data_profile.csv goc."""
    issues: list[str] = []
    if not profile_path.exists():
        return True, ["Chua co data_profile.csv goc - bo qua kiem tra schema"]

    profile = pd.read_csv(profile_path).set_index("column")
    expected_cols = set(profile.index)
    actual_cols = set(df.columns)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    if missing:
        issues.append(f"Thieu cot: {sorted(missing)}")
    if extra:
        issues.append(f"Cot phat sinh ngoai schema: {sorted(extra)}")

    if TARGET_COL in df.columns:
        target_vals = set(df[TARGET_COL].dropna().unique().astype(float))
        if not target_vals.issubset({0.0, 1.0}):
            issues.append(f"Target ngoai mien {{0,1}}: {target_vals}")

    return len(issues) == 0, issues


def detect_drift(df_new: pd.DataFrame, profile_path: Path) -> list[dict]:
    """So sanh trung binh tung cot so voi baseline. Tra ve danh sach cot drift."""
    if not profile_path.exists():
        return []
    baseline = pd.read_csv(profile_path).set_index("column")
    drifted: list[dict] = []
    for col in df_new.columns:
        if col not in baseline.index:
            continue
        if not bool(baseline.loc[col, "is_numeric"]):
            continue
        old_mean = baseline.loc[col, "mean"]
        if pd.isna(old_mean) or old_mean == 0:
            continue
        new_mean = float(df_new[col].mean())
        ratio = abs(new_mean - old_mean) / abs(old_mean)
        if ratio > DRIFT_THRESHOLD:
            drifted.append(
                {
                    "column": col,
                    "baseline_mean": float(old_mean),
                    "new_mean": new_mean,
                    "delta_ratio": ratio,
                }
            )
    return drifted


def build_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "is_numeric": bool(is_num),
                "missing_count": int(s.isna().sum()),
                "missing_ratio": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
                "min": float(non_null.min()) if is_num and not non_null.empty else None,
                "max": float(non_null.max()) if is_num and not non_null.empty else None,
                "mean": float(non_null.mean()) if is_num and not non_null.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _build_grids() -> dict:
    return {
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                ]
            ),
            "params": {"model__C": [0.1, 1, 10]},
        },
        "decision_tree": {
            "pipeline": Pipeline([("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            "params": {
                "model__max_depth": [5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
        },
        "random_forest": {
            "pipeline": Pipeline([("model", RandomForestClassifier(random_state=RANDOM_STATE))]),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
            },
        },
        "gradient_boosting": {
            "pipeline": Pipeline(
                [("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
        },
        "knn": {
            "pipeline": Pipeline(
                [("scaler", StandardScaler()), ("model", KNeighborsClassifier())]
            ),
            "params": {
                "model__n_neighbors": [5, 7, 11],
                "model__weights": ["uniform", "distance"],
            },
        },
    }


def _train_all(X_train, y_train, X_test, y_test):
    results = []
    all_models = {}
    best = {"name": "", "model": None, "recall": -1.0, "precision": -1.0}

    for name, cfg in _build_grids().items():
        print(f"\n[Challenger] Dang train: {name}")
        grid = GridSearchCV(
            estimator=cfg["pipeline"],
            param_grid=cfg["params"],
            cv=3,
            scoring="recall",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        all_models[name] = model
        results.append(
            {
                "model": name,
                "best_params": grid.best_params_,
                "cv_recall": float(grid.best_score_),
                "test_recall": rec,
                "test_precision": prec,
                "accuracy": acc,
            }
        )
        if (rec > best["recall"]) or (
            np.isclose(rec, best["recall"]) and prec > best["precision"]
        ):
            best.update({"name": name, "model": model, "recall": rec, "precision": prec})

    return all_models, pd.DataFrame(results), best


def _read_champion_recall() -> tuple[Optional[str], Optional[float]]:
    """Tra ve (model_name, recall) cua champion hien tai, neu co."""
    metadata_path = CHAMPION_DIR / "run_metadata.json"
    if not metadata_path.exists():
        return None, None
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("best_model"), meta.get("best_recall")


def _write_artifacts(
    output_dir: Path,
    all_models: dict,
    best: dict,
    results_df: pd.DataFrame,
    metadata: dict,
    y_test,
    profile_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for name, model in all_models.items():
        joblib.dump(model, models_dir / f"{name}.pkl")
    joblib.dump(best["model"], output_dir / "best_model.pkl")
    joblib.dump(all_models, output_dir / "all_models.pkl")

    results_df.sort_values(by="test_recall", ascending=False).to_csv(
        output_dir / "model_results.csv", index=False
    )
    profile_df.to_csv(output_dir / "data_profile.csv", index=False)

    y_pred = best["model"].predict(metadata["_X_test"])
    report_text = classification_report(y_test, y_pred)
    matrix_text = str(confusion_matrix(y_test, y_pred))
    with (output_dir / "best_model_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Best model: {best['name']}\n")
        f.write(f"Selection metric: {SELECTION_METRIC}\n")
        f.write(f"Best recall (class 1): {best['recall']:.6f}\n")
        f.write(f"Best precision (class 1): {best['precision']:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report_text)
        f.write("\nConfusion matrix:\n")
        f.write(matrix_text)
        f.write("\n")

    clean_meta = {k: v for k, v in metadata.items() if not k.startswith("_")}
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(clean_meta, f, ensure_ascii=False, indent=2)


def run_retrain(force_promote: bool = False) -> RetrainSummary:
    """Chay mot chu ky retrain. Tra ve summary co the dem hien thi tren UI."""
    _ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{timestamp}"

    df, new_rows = load_combined_dataset()
    profile_path_existing = CHAMPION_DIR / "data_profile.csv"

    schema_ok, schema_issues = validate_schema(df, profile_path_existing)
    drifted = detect_drift(df, profile_path_existing)
    profile_df = build_data_profile(df)

    X = df.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Du lieu chua sach: con NaN sau khi ep kieu so.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    all_models, results_df, best = _train_all(X_train, y_train, X_test, y_test)

    champion_name, champion_recall = _read_champion_recall()
    promoted = False
    if force_promote or champion_recall is None:
        promoted = True
        reason = "Khoi tao champion lan dau" if champion_recall is None else "Force promote tu user"
    elif best["recall"] >= champion_recall:
        promoted = True
        reason = (
            f"Challenger {best['name']} (recall={best['recall']:.4f}) "
            f">= champion {champion_name} (recall={champion_recall:.4f})"
        )
    else:
        reason = (
            f"Challenger {best['name']} (recall={best['recall']:.4f}) "
            f"< champion {champion_name} (recall={champion_recall:.4f}) -> giu nguyen"
        )

    metadata = {
        "timestamp": timestamp,
        "dataset_path": str(DATA_PATH),
        "target_column": TARGET_COL,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "new_rows_added": new_rows,
        "schema_ok": schema_ok,
        "schema_issues": schema_issues,
        "drift_columns": drifted,
        "feature_columns": list(X.columns),
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "best_model": best["name"],
        "best_recall": float(best["recall"]),
        "best_precision": float(best["precision"]),
        "selection_metric": SELECTION_METRIC,
        "promoted": promoted,
        "decision_reason": reason,
        "previous_champion": champion_name,
        "previous_champion_recall": champion_recall,
        "_X_test": X_test,
    }

    _write_artifacts(run_dir, all_models, best, results_df, metadata, y_test, profile_df)
    print(f"\n[Run snapshot] Da luu vao {run_dir}")

    if promoted:
        print(f"[Promote] {reason}")
        for item in CHAMPION_DIR.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        for item in run_dir.iterdir():
            target = CHAMPION_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    else:
        print(f"[Hold] {reason}")

    summary = RetrainSummary(
        timestamp=timestamp,
        n_rows=int(df.shape[0]),
        new_rows_added=new_rows,
        schema_ok=schema_ok,
        schema_issues=schema_issues,
        drift_columns=drifted,
        challenger_model=best["name"],
        challenger_recall=float(best["recall"]),
        champion_model=champion_name,
        champion_recall=champion_recall,
        promoted=promoted,
        decision_reason=reason,
        run_dir=str(run_dir),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def list_runs() -> list[dict]:
    """Liet ke cac run da chay (de hien thi trong trang Quan tri)."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.glob("run_*"), reverse=True):
        s = d / "summary.json"
        if s.exists():
            runs.append(json.loads(s.read_text(encoding="utf-8")))
    return runs


if __name__ == "__main__":
    res = run_retrain()
    print("\n=== KET QUA RETRAIN ===")
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
