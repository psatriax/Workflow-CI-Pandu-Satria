import warnings
warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlflow.models.signature import infer_signature


# ==========================================================
# DATASET
# ==========================================================

DATASET = "membangun_model/stroke_preprocessing_clean.csv"

df = pd.read_csv(DATASET)

X = df.drop("stroke", axis=1)
y = df["stroke"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ==========================================================
# SCALER
# ==========================================================

num_cols = [
    "age",
    "avg_glucose_level",
    "bmi",
]

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(
    X_train[num_cols]
)

X_test[num_cols] = scaler.transform(
    X_test[num_cols]
)


# ==========================================================
# MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


print(classification_report(y_test, y_pred))


# ==========================================================
# SAVE NATIVE MODEL
# ==========================================================

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")


# ==========================================================
# MLFLOW
# ==========================================================

mlflow.set_experiment("Stroke Prediction")

signature = infer_signature(
    X_train,
    model.predict(X_train)
)

with mlflow.start_run(run_name="RandomForest"):

    # -----------------------------
    # Parameter
    # -----------------------------

    mlflow.log_param("algorithm", "RandomForest")

    mlflow.log_param("n_estimators", 200)

    mlflow.log_param("random_state", 42)

    mlflow.log_param("class_weight", "balanced")


    # -----------------------------
    # Metrics
    # -----------------------------

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall", recall)

    mlflow.log_metric("f1_score", f1)


    # -----------------------------
    # Native Artifact
    # -----------------------------

    mlflow.log_artifact("model.pkl")

    mlflow.log_artifact("scaler.pkl")


    # -----------------------------
    # MLflow Model
    # -----------------------------

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )


print("="*60)
print("Training selesai")
print("="*60)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("="*60)