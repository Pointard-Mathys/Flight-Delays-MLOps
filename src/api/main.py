from functools import lru_cache
from calendar import monthrange
import logging
from pathlib import Path
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
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Erreur interne pendant l'inference.",
        },
    )


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


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FlightPredictionRequest) -> PredictionResponse:
    try:
        model, model_path = load_model()
        features = payload.to_model_row()
        prediction = int(model.predict(features)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])

        return PredictionResponse(
            arrival_delayed=bool(prediction),
            delay_risk=prediction,
            probability=probability,
            model_path=str(model_path),
        )
    except FileNotFoundError as exc:
        logger.exception("Model artifact is missing.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Impossible de generer une prediction.",
        ) from exc
