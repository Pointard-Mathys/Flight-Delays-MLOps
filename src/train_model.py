from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)




TRAIN_DATA_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("artifacts/model.joblib")
TARGET_COLUMN = "ARRIVAL_DELAYED"

CATEGORICAL_FEATURES = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
NUMERIC_FEATURES = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "DISTANCE",
]



def train_model(max_iter=1000, C=1.0):

    train_df = pd.read_csv(TRAIN_DATA_PATH)

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    test_df = pd.read_csv("data/processed/test.csv")

    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    with mlflow.start_run():

        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_FEATURES,
                ),
                (
                    "numeric",
                    StandardScaler(),
                    NUMERIC_FEATURES,
                ),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=max_iter,
                        C=C,
                        class_weight="balanced"
                    ),
                ),
            ]
        )

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact("artifacts/metrics.json")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    train_model()
