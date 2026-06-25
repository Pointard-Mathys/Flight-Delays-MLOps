from calendar import monthrange
from collections import deque
from functools import lru_cache
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException


MODEL_PATHS = [
    Path("artifacts/flight_delay_model.joblib"),
    Path("artifacts/model.joblib"),
]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

REFERENCE_DISTANCE_MEAN = 800.0
DRIFT_DISTANCE_THRESHOLD = 0.35
DRIFT_WINDOW_SIZE = 20
ANOMALY_DISTANCE_THRESHOLD = 5000
ANOMALY_DEPARTURE_EARLY = 300
ANOMALY_DEPARTURE_LATE = 2300
distance_window: deque[int] = deque(maxlen=DRIFT_WINDOW_SIZE)
metrics_state: dict[str, int | float | bool | None] = {
    "requests_total": 0,
    "predictions_total": 0,
    "validation_errors_total": 0,
    "internal_errors_total": 0,
    "anomalies_total": 0,
    "drift_alerts_total": 0,
    "last_prediction_probability": None,
    "last_latency_ms": None,
    "distance_window_mean": None,
    "distance_drift_score": None,
    "drift_detected": False,
}

FEATURE_COLUMNS = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "DISTANCE",
]

app = FastAPI(
    title="Flight Delays API",
    description="API d'inference pour predire si un vol aura plus de 15 minutes de retard a l'arrivee.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


class RootResponse(BaseModel):
    status: str
    service: str
    docs: str


class HealthResponse(BaseModel):
    status: str
    model_available: bool


class MetricsResponse(BaseModel):
    requests_total: int
    predictions_total: int
    validation_errors_total: int
    internal_errors_total: int
    anomalies_total: int
    drift_alerts_total: int
    last_prediction_probability: float | None
    last_latency_ms: float | None
    distance_window_size: int
    distance_window_mean: float | None
    distance_drift_score: float | None
    drift_detected: bool


class FlightPredictionRequest(BaseModel):
    month: int = Field(..., ge=1, le=12, examples=[6])
    day: int = Field(..., ge=1, le=31, examples=[18])
    day_of_week: int = Field(..., ge=1, le=7, examples=[4])
    airline: str = Field(..., min_length=2, max_length=5, examples=["AA"])
    origin_airport: str = Field(..., min_length=3, max_length=5, examples=["JFK"])
    destination_airport: str = Field(..., min_length=3, max_length=5, examples=["LAX"])
    scheduled_departure: int = Field(..., ge=0, le=2359, examples=[930])
    scheduled_arrival: int = Field(..., ge=0, le=2359, examples=[1230])
    distance: int = Field(..., gt=0, le=10000, examples=[2475])

    @field_validator("airline", "origin_airport", "destination_airport")
    @classmethod
    def normalize_codes(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError("Les codes compagnie/aeroport doivent etre alphanumeriques.")
        return cleaned

    @field_validator("scheduled_departure", "scheduled_arrival")
    @classmethod
    def validate_hhmm_time(cls, value: int) -> int:
        minutes = value % 100
        if minutes >= 60:
            raise ValueError("L'heure doit etre au format HHMM avec des minutes entre 00 et 59.")
        return value

    @model_validator(mode="after")
    def validate_business_rules(self) -> "FlightPredictionRequest":
        if self.origin_airport == self.destination_airport:
            raise ValueError("L'aeroport d'origine doit etre different de l'aeroport de destination.")

        last_day = monthrange(2024, self.month)[1]
        if self.day > last_day:
            raise ValueError(f"Le mois {self.month} ne contient pas {self.day} jours.")
        return self

    def to_model_row(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "MONTH": self.month,
                    "DAY": self.day,
                    "DAY_OF_WEEK": self.day_of_week,
                    "AIRLINE": self.airline,
                    "ORIGIN_AIRPORT": self.origin_airport,
                    "DESTINATION_AIRPORT": self.destination_airport,
                    "SCHEDULED_DEPARTURE": self.scheduled_departure,
                    "SCHEDULED_ARRIVAL": self.scheduled_arrival,
                    "DISTANCE": self.distance,
                }
            ],
            columns=FEATURE_COLUMNS,
        )


class PredictionResponse(BaseModel):
    arrival_delayed: bool
    delay_risk: int = Field(..., ge=0, le=1)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    model_path: str
    anomaly_detected: bool
    anomaly_reasons: list[str]
    drift_detected: bool
    distance_drift_score: float | None


def log_event(level: int, event: str, **fields: Any) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, ensure_ascii=False))


def detect_point_anomaly(payload: FlightPredictionRequest) -> list[str]:
    reasons = []
    if payload.distance > ANOMALY_DISTANCE_THRESHOLD:
        reasons.append(f"distance>{ANOMALY_DISTANCE_THRESHOLD}")
    if payload.scheduled_departure < ANOMALY_DEPARTURE_EARLY:
        reasons.append(f"scheduled_departure<{ANOMALY_DEPARTURE_EARLY}")
    if payload.scheduled_departure > ANOMALY_DEPARTURE_LATE:
        reasons.append(f"scheduled_departure>{ANOMALY_DEPARTURE_LATE}")
    return reasons


def update_drift_window(distance: int) -> tuple[bool, float | None, float | None]:
    distance_window.append(distance)
    if len(distance_window) < DRIFT_WINDOW_SIZE:
        return False, None, None

    window_mean = sum(distance_window) / len(distance_window)
    drift_score = abs(window_mean - REFERENCE_DISTANCE_MEAN) / REFERENCE_DISTANCE_MEAN
    drift_detected = drift_score >= DRIFT_DISTANCE_THRESHOLD
    return drift_detected, window_mean, drift_score


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Path]:
    for model_path in MODEL_PATHS:
        if model_path.exists():
            return joblib.load(model_path), model_path

    candidates = ", ".join(str(path) for path in MODEL_PATHS)
    raise FileNotFoundError(f"Aucun modele trouve. Chemins verifies: {candidates}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    metrics_state["validation_errors_total"] += 1
    log_event(logging.WARNING, "validation_error", path=str(request.url.path), errors=exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "La requete ne respecte pas le schema attendu.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code >= 500:
        metrics_state["internal_errors_total"] += 1
        log_event(logging.ERROR, "http_error", path=str(request.url.path), status_code=exc.status_code)
    else:
        log_event(logging.WARNING, "http_error", path=str(request.url.path), status_code=exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    metrics_state["internal_errors_total"] += 1
    log_event(logging.ERROR, "internal_server_error", path=str(request.url.path), error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Erreur interne pendant l'inference.",
        },
    )


@app.middleware("http")
async def count_requests(request: Request, call_next):
    metrics_state["requests_total"] += 1
    response = await call_next(request)
    return response


@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(
        status="ok",
        service="flight-delays-api",
        docs="/docs",
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_available=any(model_path.exists() for model_path in MODEL_PATHS),
    )


@app.get("/metrics", response_model=MetricsResponse)
def read_metrics() -> MetricsResponse:
    return MetricsResponse(
        requests_total=int(metrics_state["requests_total"]),
        predictions_total=int(metrics_state["predictions_total"]),
        validation_errors_total=int(metrics_state["validation_errors_total"]),
        internal_errors_total=int(metrics_state["internal_errors_total"]),
        anomalies_total=int(metrics_state["anomalies_total"]),
        drift_alerts_total=int(metrics_state["drift_alerts_total"]),
        last_prediction_probability=metrics_state["last_prediction_probability"],
        last_latency_ms=metrics_state["last_latency_ms"],
        distance_window_size=len(distance_window),
        distance_window_mean=metrics_state["distance_window_mean"],
        distance_drift_score=metrics_state["distance_drift_score"],
        drift_detected=bool(metrics_state["drift_detected"]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FlightPredictionRequest) -> PredictionResponse:
    start_time = perf_counter()
    try:
        anomaly_reasons = detect_point_anomaly(payload)
        anomaly_detected = bool(anomaly_reasons)
        if anomaly_detected:
            metrics_state["anomalies_total"] += 1
            log_event(logging.WARNING, "point_anomaly_detected", reasons=anomaly_reasons)

        drift_detected, distance_mean, drift_score = update_drift_window(payload.distance)
        metrics_state["distance_window_mean"] = distance_mean
        metrics_state["distance_drift_score"] = drift_score
        metrics_state["drift_detected"] = drift_detected
        if drift_detected:
            metrics_state["drift_alerts_total"] += 1
            log_event(
                logging.WARNING,
                "drift_detected",
                feature="distance",
                window_mean=distance_mean,
                reference_mean=REFERENCE_DISTANCE_MEAN,
                drift_score=drift_score,
            )

        model, model_path = load_model()
        features = payload.to_model_row()
        prediction = int(model.predict(features)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])

        latency_ms = round((perf_counter() - start_time) * 1000, 2)
        metrics_state["predictions_total"] += 1
        metrics_state["last_prediction_probability"] = probability
        metrics_state["last_latency_ms"] = latency_ms
        log_event(
            logging.INFO,
            "prediction_success",
            prediction=prediction,
            probability=probability,
            latency_ms=latency_ms,
            anomaly_detected=anomaly_detected,
            drift_detected=drift_detected,
        )

        return PredictionResponse(
            arrival_delayed=bool(prediction),
            delay_risk=prediction,
            probability=probability,
            model_path=str(model_path),
            anomaly_detected=anomaly_detected,
            anomaly_reasons=anomaly_reasons,
            drift_detected=drift_detected,
            distance_drift_score=drift_score,
        )
    except FileNotFoundError as exc:
        logger.exception("Model artifact is missing.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        metrics_state["internal_errors_total"] += 1
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Impossible de generer une prediction.",
        ) from exc
