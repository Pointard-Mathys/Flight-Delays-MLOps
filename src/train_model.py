import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/model.joblib")
METRICS_JSON_PATH = Path("artifacts/metrics.json")
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


def normalize_class_weight(class_weight):
    """Normalise la valeur de class_weight venant d'Airflow/CLI.

    Entrees: class_weight (str ou None, ex: "balanced", "none", "None", "").
    Sorties: "balanced", None, ou la valeur d'origine si deja valide.
    Dependances: aucune (fonction pure).
    """
    if class_weight in {"none", "None", ""}:
        return None
    return class_weight


def build_pipeline(max_iter=1000, C=1.0, class_weight="balanced") -> Pipeline:
    """Construit le pipeline sklearn (pretraitement + regression logistique).

    Entrees: max_iter, C, class_weight (hyperparametres de LogisticRegression).
    Sorties: un sklearn Pipeline non entraine.
    Dependances: aucune (fonction pure, pas d'IO, pas de MLflow).
    """
    class_weight = normalize_class_weight(class_weight)

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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    C=C,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Calcule les metriques de classification standard.

    Entrees: y_true (labels reels), y_pred (labels predits).
    Sorties: dict avec accuracy, precision, recall, f1.
    Dependances: aucune (fonction pure).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_model(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    run_name=None,
    train_data_path: Path = TRAIN_DATA_PATH,
    test_data_path: Path = TEST_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_json_path: Path = METRICS_JSON_PATH,
) -> dict:
    """Entraine le modele, logge dans MLflow et sauvegarde les artefacts.

    Entrees: max_iter, C, class_weight (hyperparametres), run_name (nom du run
        MLflow), train_data_path, test_data_path, model_path, metrics_json_path.
    Sorties: dict des metriques calculees (accuracy, precision, recall, f1).
    Dependances: lit train_data_path/test_data_path, ecrit model_path et
        metrics_json_path, logge params/metrics/artefacts dans MLflow.
    """
    train_df = pd.read_csv(train_data_path)
    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    test_df = pd.read_csv(test_data_path)
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    class_weight = normalize_class_weight(class_weight)

    mlflow.set_experiment("flight-delay-pipeline")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("class_weight", class_weight)

        model = build_pipeline(max_iter=max_iter, C=C, class_weight=class_weight)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        metrics = compute_classification_metrics(y_test, predictions)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(str(metrics_json_path))
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    return metrics


if __name__ == "__main__":
    train_model()
