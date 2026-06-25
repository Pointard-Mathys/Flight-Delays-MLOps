"""Tests de l'API FastAPI : routes, validation Pydantic, gestion d'erreurs."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "month": 6,
    "day": 18,
    "day_of_week": 4,
    "airline": "AA",
    "origin_airport": "JFK",
    "destination_airport": "LAX",
    "scheduled_departure": 930,
    "scheduled_arrival": 1230,
    "distance": 2475,
}


def test_root_returns_ok_status():
    """La route / doit repondre 200 avec un statut 'ok'."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_route_reports_model_availability():
    """La route /health doit toujours repondre 200 et exposer model_available."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_available" in response.json()


def test_metrics_route_exposes_required_indicators():
    """La route /metrics doit exposer au moins 4 indicateurs de supervision."""
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    required_fields = {
        "requests_total",
        "predictions_total",
        "validation_errors_total",
        "drift_detected",
    }
    assert required_fields.issubset(body.keys())


def test_predict_rejects_same_origin_and_destination():
    """Le validateur metier doit rejeter une requete ou origine == destination (422)."""
    payload = {**VALID_PAYLOAD, "origin_airport": "JFK", "destination_airport": "JFK"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_time_format():
    """Le validateur d'heure HHMM doit rejeter des minutes >= 60 (422)."""
    payload = {**VALID_PAYLOAD, "scheduled_departure": 999}  # minutes=99 invalide
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_month():
    """Le typage Pydantic (ge/le) doit rejeter un mois hors de [1, 12] (422)."""
    payload = {**VALID_PAYLOAD, "month": 13}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_returns_valid_prediction_when_model_available():
    """Si un modele est disponible, /predict doit renvoyer une prediction bien formee."""
    health = client.get("/health").json()
    if not health["model_available"]:
        pytest.skip("Aucun modele entraine disponible dans cet environnement de test.")

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["delay_risk"] in (0, 1)
    assert isinstance(body["anomaly_detected"], bool)


def test_predict_normalizes_airline_code_case():
    """Le validateur normalize_codes doit accepter un code compagnie en minuscule."""
    payload = {**VALID_PAYLOAD, "airline": "aa"}
    response = client.post("/predict", json=payload)
    # Ne doit pas echouer a cause de la casse (422 uniquement pour d'autres raisons).
    assert response.status_code != 422 or "airline" not in str(response.json())
