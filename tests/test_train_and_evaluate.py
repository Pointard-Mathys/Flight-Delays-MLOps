"""Tests unitaires pour src/train_model.py et src/evaluate_model.py."""
from pathlib import Path

import joblib
import pandas as pd

from src.evaluate_model import compute_metrics, evaluate_model
from src.train_model import (
    build_pipeline,
    compute_classification_metrics,
    normalize_class_weight,
    train_model,
)


def _make_processed_csv(path: Path, n: int = 60) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "MONTH": (i % 12) + 1,
                "DAY": (i % 28) + 1,
                "DAY_OF_WEEK": (i % 7) + 1,
                "AIRLINE": ["AA", "DL", "UA"][i % 3],
                "ORIGIN_AIRPORT": ["JFK", "LAX", "ORD"][i % 3],
                "DESTINATION_AIRPORT": ["ATL", "DFW", "SFO"][i % 3],
                "SCHEDULED_DEPARTURE": 800 + i,
                "SCHEDULED_ARRIVAL": 1100 + i,
                "DISTANCE": 500 + (i * 10),
                "ARRIVAL_DELAYED": i % 2,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_normalize_class_weight_handles_none_aliases():
    """Les chaines 'none'/'None'/'' doivent etre converties en None Python."""
    assert normalize_class_weight("none") is None
    assert normalize_class_weight("None") is None
    assert normalize_class_weight("") is None
    assert normalize_class_weight("balanced") == "balanced"


def test_build_pipeline_returns_fitted_estimator(raw_flights_df):
    """build_pipeline doit produire un Pipeline sklearn entrainable et predictif."""
    from src.prepare_data import build_dataset

    dataset = build_dataset(raw_flights_df)
    x = dataset.drop(columns=["ARRIVAL_DELAYED"])
    y = dataset["ARRIVAL_DELAYED"]

    pipeline = build_pipeline(max_iter=200, C=1.0, class_weight="balanced")
    pipeline.fit(x, y)
    predictions = pipeline.predict(x)

    assert len(predictions) == len(x)
    assert set(predictions).issubset({0, 1})


def test_compute_classification_metrics_perfect_predictions():
    """Avec des predictions parfaites, toutes les metriques doivent valoir 1.0."""
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_compute_metrics_matches_train_model_metrics():
    """evaluate_model.compute_metrics doit etre coherent avec train_model.compute_classification_metrics
    (non-regression : les deux modules ne doivent jamais diverger sur la definition des metriques)."""
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 0, 1, 1]
    assert compute_metrics(y_true, y_pred) == compute_classification_metrics(y_true, y_pred)


def test_train_model_writes_model_and_metrics(tmp_path: Path, monkeypatch):
    """train_model doit produire un modele serialisable et un fichier de metriques JSON."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    metrics_json_path = tmp_path / "metrics.json"

    _make_processed_csv(train_path, n=60)
    _make_processed_csv(test_path, n=20)

    # MLflow ecrit dans un dossier local mlruns/ : on l'isole dans tmp_path
    # pour ne pas polluer le tracking store du projet pendant les tests.
    monkeypatch.chdir(tmp_path)
    train_path = Path(train_path.name)
    test_path = Path(test_path.name)

    metrics = train_model(
        max_iter=200,
        C=1.0,
        class_weight="balanced",
        run_name="pytest-run",
        train_data_path=train_path,
        test_data_path=test_path,
        model_path=model_path.relative_to(tmp_path) if model_path.is_relative_to(tmp_path) else model_path,
        metrics_json_path=metrics_json_path.relative_to(tmp_path)
        if metrics_json_path.is_relative_to(tmp_path)
        else metrics_json_path,
    )

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert (tmp_path / model_path.name).exists()
    loaded_model = joblib.load(tmp_path / model_path.name)
    assert hasattr(loaded_model, "predict")


def test_evaluate_model_end_to_end(tmp_path: Path, monkeypatch):
    """evaluate_model doit lire un modele entraine et ecrire un rapport de metriques exploitable."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    metrics_json_path = tmp_path / "metrics.json"
    metrics_txt_path = tmp_path / "metrics.txt"

    _make_processed_csv(train_path, n=60)
    _make_processed_csv(test_path, n=20)

    monkeypatch.chdir(tmp_path)
    train_model(
        max_iter=200,
        train_data_path=Path(train_path.name),
        test_data_path=Path(test_path.name),
        model_path=Path(model_path.name),
        metrics_json_path=Path(metrics_json_path.name),
    )

    metrics = evaluate_model(
        test_data_path=Path(test_path.name),
        model_path=Path(model_path.name),
        metrics_path=Path(metrics_txt_path.name),
    )

    assert metrics_txt_path.exists()
    content = metrics_txt_path.read_text(encoding="utf-8")
    assert "accuracy" in content
    assert 0.0 <= metrics["accuracy"] <= 1.0
