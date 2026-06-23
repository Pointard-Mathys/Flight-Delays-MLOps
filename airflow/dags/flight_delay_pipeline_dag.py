from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from airflow.sdk import DAG
except ImportError:
    from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator


PROJECT_DIR = Path("/opt/airflow/project")
RAW_DATA_PATH = PROJECT_DIR / "data/raw/flights.csv"
TRAIN_DATA_PATH = PROJECT_DIR / "data/processed/train.csv"
TEST_DATA_PATH = PROJECT_DIR / "data/processed/test.csv"
MODEL_PATH = PROJECT_DIR / "artifacts/model.joblib"
FINAL_MODEL_PATH = PROJECT_DIR / "artifacts/flight_delay_model.joblib"
METRICS_PATH = PROJECT_DIR / "artifacts/metrics.txt"
METRICS_JSON_PATH = PROJECT_DIR / "artifacts/metrics.json"

RAW_REQUIRED_COLUMNS = {
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "DISTANCE",
    "ARRIVAL_DELAY",
    "CANCELLED",
    "DIVERTED",
}

PROCESSED_REQUIRED_COLUMNS = {
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "DISTANCE",
    "ARRIVAL_DELAYED",
}


def _use_project_dir() -> None:
    os.chdir(PROJECT_DIR)
    project_dir = str(PROJECT_DIR)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def clean_previous_outputs() -> None:
    for path in [
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        MODEL_PATH,
        FINAL_MODEL_PATH,
        METRICS_PATH,
        METRICS_JSON_PATH,
        PROJECT_DIR / "artifacts/model_card.txt",
    ]:
        if path.exists():
            path.unlink()


def check_raw_data_exists() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset missing: {RAW_DATA_PATH}")
    if RAW_DATA_PATH.stat().st_size == 0:
        raise ValueError(f"Raw dataset is empty: {RAW_DATA_PATH}")


def force_failure_for_demo(**context) -> None:
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    if conf.get("force_failure") is True:
        raise RuntimeError("Forced failure requested by DAG run config.")


def check_raw_data_format() -> None:
    sample = pd.read_csv(RAW_DATA_PATH, nrows=100)
    missing_columns = RAW_REQUIRED_COLUMNS.difference(sample.columns)
    if missing_columns:
        raise ValueError(f"Raw dataset missing columns: {sorted(missing_columns)}")


def run_prepare_data() -> None:
    _use_project_dir()
    from src.prepare_data import prepare_data

    prepare_data()


def check_processed_data_format() -> None:
    for path in [TRAIN_DATA_PATH, TEST_DATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset missing: {path}")

        sample = pd.read_csv(path, nrows=100)
        missing_columns = PROCESSED_REQUIRED_COLUMNS.difference(sample.columns)
        if missing_columns:
            raise ValueError(f"{path.name} missing columns: {sorted(missing_columns)}")

        if sample.empty:
            raise ValueError(f"Processed dataset is empty: {path}")


def run_train_model() -> None:
    _use_project_dir()
    from src.train_model import train_model

    train_model()


def check_model_artifact() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact missing: {MODEL_PATH}")
    if MODEL_PATH.stat().st_size == 0:
        raise ValueError(f"Model artifact is empty: {MODEL_PATH}")


def run_evaluate_model() -> None:
    _use_project_dir()
    from src.evaluate_model import evaluate_model

    evaluate_model()


def check_metrics_artifact() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics artifact missing: {METRICS_PATH}")
    if METRICS_PATH.stat().st_size == 0:
        raise ValueError(f"Metrics artifact is empty: {METRICS_PATH}")


def run_save_model() -> None:
    _use_project_dir()
    from src.save_model import save_model

    save_model()


with DAG(
    dag_id="flight_delay_ml_pipeline",
    description="Prepare data, train, evaluate, and publish the flight delay model.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "mlops",
        "retries": 0,
    },
    tags=["mlops", "flight-delays"],
) as dag:
    clean_outputs = PythonOperator(
        task_id="clean_previous_outputs",
        python_callable=clean_previous_outputs,
    )

    validate_raw_exists = PythonOperator(
        task_id="check_raw_data_exists",
        python_callable=check_raw_data_exists,
    )

    failure_demo_guard = PythonOperator(
        task_id="force_failure_for_demo",
        python_callable=force_failure_for_demo,
    )

    validate_raw_format = PythonOperator(
        task_id="check_raw_data_format",
        python_callable=check_raw_data_format,
    )

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=run_prepare_data,
    )

    validate_processed_format = PythonOperator(
        task_id="check_processed_data_format",
        python_callable=check_processed_data_format,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=run_train_model,
    )

    validate_model = PythonOperator(
        task_id="check_model_artifact",
        python_callable=check_model_artifact,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluate_model,
    )

    validate_metrics = PythonOperator(
        task_id="check_metrics_artifact",
        python_callable=check_metrics_artifact,
    )

    save = PythonOperator(
        task_id="save_model",
        python_callable=run_save_model,
    )

    (
        clean_outputs
        >> validate_raw_exists
        >> failure_demo_guard
        >> validate_raw_format
        >> prepare
        >> validate_processed_format
        >> train
        >> validate_model
        >> evaluate
        >> validate_metrics
        >> save
    )
