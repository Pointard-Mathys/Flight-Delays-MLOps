"""Tests unitaires pour src/prepare_data.py."""
from pathlib import Path

import pandas as pd

from src.prepare_data import TARGET_COLUMN, build_dataset, prepare_data


def test_build_dataset_filters_cancelled_and_diverted(raw_flights_df):
    """Les vols annules ou derives ne doivent jamais apparaitre dans le dataset final."""
    dataset = build_dataset(raw_flights_df)
    # 6 vols - 1 annule - 1 derive - 1 avec ARRIVAL_DELAY manquant = 3
    assert len(dataset) == 3


def test_build_dataset_drops_missing_values(raw_flights_df):
    """Une ligne avec ARRIVAL_DELAY manquant doit etre exclue du dataset."""
    dataset = build_dataset(raw_flights_df)
    # Le vol ORD->ATL (3eme ligne) a un ARRIVAL_DELAY manquant et n'est pas
    # annule/derive : il doit disparaitre du resultat malgre tout.
    assert not (
        (dataset["ORIGIN_AIRPORT"] == "ORD") & (dataset["DESTINATION_AIRPORT"] == "ATL")
    ).any()


def test_build_dataset_target_is_binary_threshold(raw_flights_df):
    """La cible ARRIVAL_DELAYED doit valoir 1 si le retard depasse 15 minutes, sinon 0."""
    dataset = build_dataset(raw_flights_df)
    # Vol JFK->LAX, delay=5 -> non retarde
    row_short_delay = dataset[
        (dataset["ORIGIN_AIRPORT"] == "JFK") & (dataset["DESTINATION_AIRPORT"] == "LAX")
    ].iloc[0]
    assert row_short_delay[TARGET_COLUMN] == 0

    # Vol LAX->JFK, delay=30 -> retarde
    row_long_delay = dataset[
        (dataset["ORIGIN_AIRPORT"] == "LAX") & (dataset["DESTINATION_AIRPORT"] == "JFK")
    ].iloc[0]
    assert row_long_delay[TARGET_COLUMN] == 1


def test_prepare_data_writes_train_and_test_csv(tmp_path: Path):
    """prepare_data doit ecrire deux fichiers CSV non vides aux chemins fournis."""
    raw_path = tmp_path / "flights.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    rows = []
    for i in range(40):
        rows.append(
            {
                "MONTH": 1,
                "DAY": (i % 28) + 1,
                "DAY_OF_WEEK": (i % 7) + 1,
                "AIRLINE": "AA" if i % 2 == 0 else "DL",
                "ORIGIN_AIRPORT": "JFK",
                "DESTINATION_AIRPORT": "LAX",
                "SCHEDULED_DEPARTURE": 800,
                "SCHEDULED_ARRIVAL": 1100,
                "DISTANCE": 2475,
                "ARRIVAL_DELAY": 20 if i % 3 == 0 else 5,
                "CANCELLED": 0,
                "DIVERTED": 0,
            }
        )
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    prepare_data(
        sample_size=None,
        raw_data_path=raw_path,
        train_data_path=train_path,
        test_data_path=test_path,
    )

    assert train_path.exists() and train_path.stat().st_size > 0
    assert test_path.exists() and test_path.stat().st_size > 0

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns
    # 80/20 split attendu sur 40 lignes
    assert len(train_df) == 32
    assert len(test_df) == 8
