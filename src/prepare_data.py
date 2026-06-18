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


def prepare_data(sample_size: int | None = 200_000) -> None:
    columns = FEATURE_COLUMNS + ["ARRIVAL_DELAY", "CANCELLED", "DIVERTED"]
    flights = pd.read_csv(RAW_DATA_PATH, usecols=columns, nrows=sample_size)

    flights = flights[(flights["CANCELLED"] == 0) & (flights["DIVERTED"] == 0)]
    flights = flights.dropna(subset=FEATURE_COLUMNS + ["ARRIVAL_DELAY"])
    flights[TARGET_COLUMN] = (flights["ARRIVAL_DELAY"] > 15).astype(int)

    dataset = flights[FEATURE_COLUMNS + [TARGET_COLUMN]]
    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset[TARGET_COLUMN],
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"Train data saved to {TRAIN_DATA_PATH}")
    print(f"Test data saved to {TEST_DATA_PATH}")


if __name__ == "__main__":
    prepare_data()
