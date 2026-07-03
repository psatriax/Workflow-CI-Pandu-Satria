import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "stroke_preprocessing_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "best_scaler.pkl")

# =====================================================
# MLFLOW
# =====================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Stroke Prediction")

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(DATASET)

X = df.drop("stroke", axis=1)
y = df["stroke"]

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()

numeric_cols = [
    "age",
    "avg_glucose_level",
    "bmi",
]

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# =====================================================
# GRID SEARCH
# =====================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

# =====================================================
# TRAIN
# =====================================================

with mlflow.start_run(run_name="RandomForest_Tuning"):

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print("Best Parameter")
    print(grid.best_params_)

    print()
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # =============================
    # Manual Logging
    # =============================

    mlflow.log_params(grid.best_params_)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nTraining tuning selesai.")