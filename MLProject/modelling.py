import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# ===========================
# MLflow
# ===========================
mlflow.set_experiment("Stroke Prediction")
mlflow.sklearn.autolog()

# ===========================
# Path
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "stroke_preprocessing_clean.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv(DATASET)

# ===========================
# Split Feature & Label
# ===========================
X = df.drop("stroke", axis=1)
y = df["stroke"]

# ===========================
# Train Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ===========================
# Scaling
# ===========================
scaler = StandardScaler()

numeric_cols = [
    "age",
    "avg_glucose_level",
    "bmi",
]

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ===========================
# Model
# ===========================
model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# ===========================
# Save Model
# ===========================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("=" * 50)
print("Training selesai")
print("Model berhasil disimpan")
print("=" * 50)