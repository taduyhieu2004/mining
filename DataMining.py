# 1. IMPORT THU VIEN
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. LOAD DATA
df = pd.read_csv("heart_disease_risk_dataset_earlymed.csv")

X = df.drop("Heart_Risk", axis=1)
y = df["Heart_Risk"]

# 3. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. KHAI BAO MODEL + PARAM GRID

models = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1, 10]
        }
    },

    "Decision Tree": {
        "pipeline": Pipeline([
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "params": {
            "model__max_depth": [5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("model", RandomForestClassifier(random_state=42))
        ]),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_split": [2, 5]
        }
    },

    "Gradient Boosting": {
        "pipeline": Pipeline([
            ("model", GradientBoostingClassifier(random_state=42))
        ]),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5]
        }
    },

    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "params": {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"]
        }
    },

    "SVM": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"]
        }
    }
}

# 5. TRAIN + GRID SEARCH
results = []
all_models = {}

best_model = None
best_score = 0

for name, config in models.items():
    print(f"\n Training {name}...")

    grid = GridSearchCV(
        config["pipeline"],
        config["params"],
        cv=3,
        scoring="accuracy",
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    # Best model
    model = grid.best_estimator_

    filename = name.replace(" ", "_").lower() + ".pkl"
    joblib.dump(model, filename)
    print(f"Saved: {filename}")

    all_models[name] = model

    # Predict
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": acc
    })

    # Luu model tot nhat
    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

# 6. KET QUA SO SANH
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
results_df.to_csv("model_results.csv", index=False)

print("\n MODEL COMPARISON:")
print(results_df)

# 7. MODEL TOT NHAT
print("\n BEST MODEL:", best_model_name)
print("Best Accuracy:", best_score)

# 8. CLASSIFICATION REPORT
y_pred_best = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# 9. LUU MODEL TOT NHAT
import joblib
joblib.dump(best_model, "best_model.pkl")
joblib.dump(all_models, "all_models.pkl")

print("\n Done!")
