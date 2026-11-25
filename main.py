"""Exécution du pipeline via arguments CLI."""

import argparse

import pandas as pd

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def decision_finale(model, x_test):
    """Prédit pour un client exemple et affiche la décision finale."""
    client = pd.DataFrame(
        [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]], columns=x_test.columns
    )
    prediction = model.predict(client)[0]
    proba = model.predict_proba(client)[0][1]
    print("Client testé :")
    print(client.to_string(index=False))
    print(f"Probabilité de départ : {proba * 100:.2f} %")
    if prediction == 1:
        print("DÉCISION FINALE : le client va QUITTER la banque (Exited = 1)")
    else:
        print("DÉCISION FINALE : le client va RESTER dans la banque (Exited = 0)")


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Customer Churn")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "load", "all"],
        default="all",
        help="étape à exécuter (par défaut : all)",
    )
    parser.add_argument("--data", default="Churn_Modelling.csv", help="fichier CSV")
    parser.add_argument(
        "--model", default="classifier.joblib", help="fichier du modèle"
    )
    args = parser.parse_args()

    if args.step == "load":
        model = load_model(args.model)
        print(model)
        return

    x_train, x_test, y_train, y_test = prepare_data(args.data)

    if args.step == "prepare":
        print("Données préparées :", x_train.shape, x_test.shape)

    elif args.step == "train":
        model = train_model(x_train, y_train)
        save_model(model, args.model)

    elif args.step == "evaluate":
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)

    else:
        print("=== 1. PREPARE ===")
        print("Données préparées :", x_train.shape, x_test.shape)

        print("\n=== 2. TRAIN ===")
        model = train_model(x_train, y_train)
        print("Modèle entraîné :", model)

        print("\n=== 3. SAVE ===")
        save_model(model, args.model)

        print("\n=== 4. LOAD ===")
        model = load_model(args.model)

        print("\n=== 5. EVALUATE ===")
        evaluate_model(model, x_test, y_test)

        print("\n=== 6. DÉCISION FINALE ===")
        decision_finale(model, x_test)


if __name__ == "__main__":
    main()
