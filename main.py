from src.prepare_data import prepare_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model


def main():
    print("=== Préparation des données ===")
    prepare_data()

    print("\n=== Entraînement du modèle ===")
    train_model()

    print("\n=== Évaluation du modèle ===")
    evaluate_model()

    print("\nProjet exécuté avec succès !")


if __name__ == "__main__":
    main()