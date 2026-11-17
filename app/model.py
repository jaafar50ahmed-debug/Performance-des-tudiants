from pathlib import Path
import joblib
import pandas as pd

# Même chemin que dans training/train_model.py
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "students_model.joblib"
)


def load_model() -> dict:
    """
    Charge le fichier .joblib :
    un dictionnaire {"model": pipeline, "label_encoder": encodeur}.
    """
    return joblib.load(MODEL_PATH)


def predict(bundle: dict, features: dict) -> dict:
    """
    Renvoie la classe prédite et les probabilités pour un étudiant.

    `features` est un dictionnaire dont les clés sont les noms
    de colonnes de l'entraînement.
    """
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    # Le pipeline attend un tableau avec les mêmes noms
    # de colonnes qu'à l'entraînement.
    df = pd.DataFrame([features])

    probabilities = model.predict_proba(df)[0]
    class_id = int(probabilities.argmax())

    return {
        "class_id": class_id,  # 0 / 1
        "class_name": str(
            label_encoder.inverse_transform([class_id])[0]
        ),
        "probabilities": {
            str(name): float(p)
            for name, p in zip(
                label_encoder.classes_,
                probabilities
            )
        },
    }