from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/flights.csv")
PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

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
TARGET_COLUMN = "ARRIVAL_DELAYED"


def build_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les vols bruts en dataset pret pour l'entrainement.

    Entrees: raw_df (DataFrame avec FEATURE_COLUMNS, ARRIVAL_DELAY, CANCELLED, DIVERTED).
    Sorties: DataFrame filtre (vols non annules/non derives, sans NA) avec la
        colonne cible binaire ARRIVAL_DELAYED.
    Dependances: aucune (fonction pure, pas d'IO).
    """
    flights = raw_df[(raw_df["CANCELLED"] == 0) & (raw_df["DIVERTED"] == 0)]
    flights = flights.dropna(subset=FEATURE_COLUMNS + ["ARRIVAL_DELAY"])
    flights = flights.copy()
    flights[TARGET_COLUMN] = (flights["ARRIVAL_DELAY"] > 15).astype(int)
    return flights[FEATURE_COLUMNS + [TARGET_COLUMN]]


def prepare_data(
    sample_size: int | None = 200_000,
    raw_data_path: Path = RAW_DATA_PATH,
    train_data_path: Path = TRAIN_DATA_PATH,
    test_data_path: Path = TEST_DATA_PATH,
) -> None:
    """Prepare les jeux d'entrainement et de test a partir des vols bruts.

    Entrees: sample_size (nb max de lignes a lire), raw_data_path, train_data_path,
        test_data_path.
    Sorties: aucune (effet de bord : ecrit les CSV train/test sur disque).
    Dependances: lit raw_data_path, ecrit train_data_path et test_data_path.
    """
    columns = FEATURE_COLUMNS + ["ARRIVAL_DELAY", "CANCELLED", "DIVERTED"]
    raw_df = pd.read_csv(raw_data_path, usecols=columns, nrows=sample_size)

    dataset = build_dataset(raw_df)
    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset[TARGET_COLUMN],
    )

    train_data_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)

    print(f"Train data saved to {train_data_path}")
    print(f"Test data saved to {test_data_path}")


if __name__ == "__main__":
    prepare_data()
