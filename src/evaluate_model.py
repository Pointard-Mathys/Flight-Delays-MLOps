from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.txt")
TARGET_COLUMN = "ARRIVAL_DELAYED"


def evaluate_model() -> None:
    test_df = pd.read_csv(TEST_DATA_PATH)
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    report = classification_report(y_test, predictions, zero_division=0)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        "\n".join(f"{name}: {value:.4f}" for name, value in metrics.items())
        + "\n\n"
        + report,
        encoding="utf-8",
    )

    print(f"Metrics saved to {METRICS_PATH}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    evaluate_model()
