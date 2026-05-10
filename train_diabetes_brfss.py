import json
from pathlib import Path

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


DATA_PATH = Path("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
OUTPUT_DIR = Path("outputs/diabetes_brfss_v1")
TARGET_COL = "Diabetes_binary"
RANDOM_STATE = 42
SELECTION_METRIC = "recall"


def build_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile_rows = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        profile_rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "is_numeric": bool(is_numeric),
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
                "min": float(non_null.min()) if is_numeric and not non_null.empty else None,
                "max": float(non_null.max()) if is_numeric and not non_null.empty else None,
                "mean": float(non_null.mean()) if is_numeric and not non_null.empty else None,
            }
        )
    return pd.DataFrame(profile_rows)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for old_item in models_dir.iterdir():
        if old_item.is_file() or old_item.is_symlink():
            old_item.unlink()

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Khong tim thay cot target '{TARGET_COL}' trong du lieu.")

    n_rows_before_dedup = int(df.shape[0])
    df = df.drop_duplicates().reset_index(drop=True)
    n_rows_after_dedup = int(df.shape[0])
    dropped_duplicates = n_rows_before_dedup - n_rows_after_dedup
    print(
        f"Da drop duplicate: {dropped_duplicates} dong "
        f"(tu {n_rows_before_dedup} con {n_rows_after_dedup})"
    )

    # Tao profile de xac nhan schema cua data moi khac data cu
    data_profile = build_data_profile(df)
    data_profile_path = OUTPUT_DIR / "data_profile.csv"
    data_profile.to_csv(data_profile_path, index=False)

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = pd.to_numeric(y, errors="coerce")

    if X.isna().any().any() or y.isna().any():
        missing_feature_cells = int(X.isna().sum().sum())
        missing_target_cells = int(y.isna().sum())
        raise ValueError(
            "Du lieu co gia tri khong hop le sau khi ep kieu so. "
            f"Missing feature cells: {missing_feature_cells}, "
            f"missing target cells: {missing_target_cells}"
        )

    unique_target = sorted(np.unique(y))
    if not set(unique_target).issubset({0.0, 1.0}):
        raise ValueError(
            f"Target '{TARGET_COL}' khong phai binary 0/1. Gia tri tim thay: {unique_target}"
        )
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
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
            "pipeline": Pipeline(
                [("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]
            ),
            "params": {
                "model__max_depth": [5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                [("model", RandomForestClassifier(random_state=RANDOM_STATE))]
            ),
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
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "params": {
                "model__n_neighbors": [5, 7, 11],
                "model__weights": ["uniform", "distance"],
            },
        },
    }

    results = []
    all_models = {}
    best_model = None
    best_model_name = ""
    best_recall = 0.0
    best_precision = 0.0

    for model_name, config in models.items():
        print(f"\nDang train: {model_name}")
        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            cv=3,
            scoring="recall",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Luu model: {model_path}")
        print(f"Best params: {grid.best_params_}")
        print(f"Recall (class 1): {rec:.6f}")
        print(f"Precision (class 1): {prec:.6f}")
        print(f"Accuracy: {acc:.6f}")

        results.append(
            {
                "model": model_name,
                "best_params": grid.best_params_,
                "cv_recall": float(grid.best_score_),
                "test_recall": rec,
                "test_precision": prec,
                "accuracy": acc,
            }
        )
        all_models[model_name] = model

        # Uu tien recall cho bai toan sang loc tieu duong.
        # Neu recall bang nhau thi uu tien precision cao hon.
        if (rec > best_recall) or (
            np.isclose(rec, best_recall) and prec > best_precision
        ):
            best_recall = rec
            best_precision = prec
            best_model = model
            best_model_name = model_name

    results_df = pd.DataFrame(results).sort_values(by="test_recall", ascending=False)
    results_path = OUTPUT_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False)

    if best_model is None:
        raise RuntimeError("Khong tim duoc model tot nhat.")

    best_model_path = OUTPUT_DIR / "best_model.pkl"
    all_models_path = OUTPUT_DIR / "all_models.pkl"
    report_path = OUTPUT_DIR / "best_model_report.txt"
    metadata_path = OUTPUT_DIR / "run_metadata.json"

    y_pred_best = best_model.predict(X_test)
    report_text = classification_report(y_test, y_pred_best)
    matrix_text = str(confusion_matrix(y_test, y_pred_best))

    joblib.dump(best_model, best_model_path)
    joblib.dump(all_models, all_models_path)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"Selection metric: {SELECTION_METRIC}\n")
        f.write(f"Best recall (class 1): {best_recall:.6f}\n")
        f.write(f"Best precision (class 1): {best_precision:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report_text)
        f.write("\nConfusion matrix:\n")
        f.write(matrix_text)
        f.write("\n")

    metadata = {
        "dataset_path": str(DATA_PATH),
        "target_column": TARGET_COL,
        "n_rows_before_dedup": n_rows_before_dedup,
        "n_rows_after_dedup": n_rows_after_dedup,
        "dropped_duplicates": dropped_duplicates,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "feature_columns": list(X.columns),
        "data_profile_path": str(data_profile_path),
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "best_model": best_model_name,
        "best_recall": float(best_recall),
        "best_precision": float(best_precision),
        "selection_metric": SELECTION_METRIC,
        "output_dir": str(OUTPUT_DIR),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nHoan tat train!")
    print("\n=== MODEL TOT NHAT CHO PHAN LOAI TIEU DUONG ===")
    print(f"Tieu chi chon: {SELECTION_METRIC}")
    print(f"Ten model: {best_model_name}")
    print(f"Recall (class 1): {best_recall:.6f}")
    print(f"Precision (class 1): {best_precision:.6f}")
    print(f"Data profile: {data_profile_path}")
    print(f"Ket qua tong hop: {results_path}")
    print(f"Bao cao model tot nhat: {report_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
