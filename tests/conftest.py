"""Fixtures partagees pour les tests du pipeline et de l'API."""
import pandas as pd
import pytest


@pytest.fixture
def raw_flights_df() -> pd.DataFrame:
    """Petit DataFrame de vols bruts couvrant les cas limites utiles aux tests:
    un vol annule, un vol derive, un vol avec valeur manquante, et des vols
    valides en retard / a l'heure."""
    return pd.DataFrame(
        {
            "MONTH": [1, 1, 1, 1, 1, 1],
            "DAY": [1, 2, 3, 4, 5, 6],
            "DAY_OF_WEEK": [4, 5, 6, 7, 1, 2],
            "AIRLINE": ["AA", "DL", "UA", "WN", "AA", "DL"],
            "ORIGIN_AIRPORT": ["JFK", "LAX", "ORD", "ATL", "JFK", "LAX"],
            "DESTINATION_AIRPORT": ["LAX", "JFK", "ATL", "ORD", "ORD", "JFK"],
            "SCHEDULED_DEPARTURE": [800, 1200, 1500, 900, 1000, 1100],
            "SCHEDULED_ARRIVAL": [1100, 1500, 1800, 1200, 1300, 1400],
            "DISTANCE": [2475, 2475, 700, 700, 800, 2475],
            "ARRIVAL_DELAY": [5, 30, None, 10, -5, 45],
            "CANCELLED": [0, 0, 0, 1, 0, 0],
            "DIVERTED": [0, 0, 0, 0, 1, 0],
        }
    )
