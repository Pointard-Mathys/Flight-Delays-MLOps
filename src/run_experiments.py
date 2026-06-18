from src.train_model import train_model


EXPERIMENTS = [
    {"C": 0.1, "max_iter": 500, "class_weight": "balanced", "run_name": "logreg-c-0.1-balanced"},
    {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "run_name": "logreg-c-1.0-balanced"},
    {"C": 10.0, "max_iter": 1000, "class_weight": "none", "run_name": "logreg-c-10-none"},
]


def run_experiments() -> None:
    for experiment in EXPERIMENTS:
        print(f"Running MLflow experiment: {experiment['run_name']}")
        train_model(**experiment)


if __name__ == "__main__":
    run_experiments()
