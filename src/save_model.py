from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

SOURCE_MODEL_PATH = Path("artifacts/model.joblib")
FINAL_MODEL_PATH = Path("artifacts/flight_delay_model.joblib")
MODEL_CARD_PATH = Path("artifacts/model_card.txt")


def save_model() -> None:
    if not SOURCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {SOURCE_MODEL_PATH}. Run `python -m src.train_model` first."
        )

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    copy2(SOURCE_MODEL_PATH, FINAL_MODEL_PATH)

    MODEL_CARD_PATH.write_text(
        "\n".join(
            [
                "Flight Delay Model",
                f"Saved at: {datetime.now(timezone.utc).isoformat()}",
                "Target: ARRIVAL_DELAYED, where ARRIVAL_DELAY > 15 minutes",
                "Dataset: Kaggle USDOT Flight Delays",
                f"Model file: {FINAL_MODEL_PATH}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Final model saved to {FINAL_MODEL_PATH}")
    print(f"Model card saved to {MODEL_CARD_PATH}")


if __name__ == "__main__":
    save_model()
