"""Pipeline ML : prédiction du churn client (Churn_Modelling.csv)."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(data_filename="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge et prétraite les données."""
    df = pd.read_csv(data_filename)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    df = df.drop(columns=["Surname", "Geography"])

    x = df.drop(columns=["Exited"])
    y = df["Exited"]
    x = x.drop(columns=["RowNumber", "CustomerId"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et le retourne."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle : accuracy, matrice de confusion, rapport."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("Matrice de confusion :")
    print(matrix)
    print(classification_report(y_test, y_pred))
    return accuracy * 100, matrix


def save_model(model, filename="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé dans {filename}")


def load_model(filename="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(filename)
    print(f"Modèle chargé depuis {filename}")
    return model
