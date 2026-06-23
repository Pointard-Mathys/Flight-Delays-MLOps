from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.txt")
TARGET_COLUMN = "ARRIVAL_DELAYED"


def compute_metrics(y_true, y_pred) -> dict:
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


def evaluate_model(
    test_data_path: Path = TEST_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    """Evalue le modele entraine sur le jeu de test et ecrit un rapport texte.

    Entrees: test_data_path, model_path, metrics_path.
    Sorties: dict des metriques (accuracy, precision, recall, f1).
    Dependances: lit test_data_path et model_path, ecrit metrics_path.
    """
    test_df = pd.read_csv(test_data_path)
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(model_path)
    predictions = model.predict(x_test)

    metrics = compute_metrics(y_test, predictions)

    report = classification_report(y_test, predictions, zero_division=0)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "\n".join(f"{name}: {value:.4f}" for name, value in metrics.items())
        + "\n\n"
        + report,
        encoding="utf-8",
    )

    print(f"Metrics saved to {metrics_path}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
