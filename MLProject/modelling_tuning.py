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
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler

from mlflow.models.signature import infer_signature


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


param_grid = {

    "n_estimators":[100,200],

    "max_depth":[5,10,None],

    "min_samples_split":[2,5],

    "class_weight":["balanced"]

}


rf = RandomForestClassifier(
    random_state=42
)


grid = GridSearchCV(

    rf,

    param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1

)

grid.fit(X_train,y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print(grid.best_params_)

print(classification_report(y_test,y_pred))


joblib.dump(best_model,"best_model.pkl")

joblib.dump(scaler,"best_scaler.pkl")



mlflow.set_experiment("Stroke Prediction")


signature = infer_signature(
    X_train,
    best_model.predict(X_train)
)

with mlflow.start_run(run_name="RandomForest_GridSearch"):

    mlflow.log_params(grid.best_params_)

    mlflow.log_metric("accuracy",accuracy)

    mlflow.log_metric("precision",precision)

    mlflow.log_metric("recall",recall)

    mlflow.log_metric("f1_score",f1)

    mlflow.log_metric(
        "best_cv_score",
        grid.best_score_
    )

    mlflow.log_artifact("best_model.pkl")

    mlflow.log_artifact("best_scaler.pkl")

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

print("="*60)
print("Training tuning selesai")
print("="*60)