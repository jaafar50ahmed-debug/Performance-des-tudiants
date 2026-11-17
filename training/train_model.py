from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "StudentsPerformance.csv"
MODEL_PATH = ROOT / "models" / "students_model.joblib"

# On prédit « reussite » si la note de maths est supérieure ou égale à ce seuil
SEUIL_REUSSITE = 60

# 1. Variables numériques
NUM_COLS = [
    "reading_score",
    "writing_score",
]

# 2. Variables ordinales
ORDINAL = {
    "parental_level_of_education": [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
    "test_preparation_course": [
        "none",
        "completed",
    ],
}

ORD_COLS = list(ORDINAL)

# 3. Variables nominales
NOM_COLS = [
    "gender",
    "race_ethnicity",
    "lunch",
]


def load_data():
    """
    Lit le fichier CSV et renvoie X et y.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}\n"
            "Placez StudentsPerformance.csv dans le dossier data/ du projet."
        )

    df = pd.read_csv(DATA_PATH)

    df.columns = [
        c.strip().lower().replace("/", "_").replace(" ", "_")
        for c in df.columns
    ]

    X = df[NUM_COLS + ORD_COLS + NOM_COLS].copy()

    for col, order in ORDINAL.items():
        inconnues = set(X[col].unique()) - set(order)
        assert not inconnues, (
            f"Modalités inattendues dans {col} : {inconnues}"
        )

    y_text = (df["math_score"] >= SEUIL_REUSSITE).map(
        {
            True: "reussite",
            False: "echec",
        }
    )

    return X, y_text


def main():
    # Chargement des données
    X, y_text = load_data()

    # Encodage de la cible
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    codes = {
        c: int(i)
        for c, i in zip(
            label_encoder.classes_,
            label_encoder.transform(label_encoder.classes_),
        )
    }

    print(f"{len(X)} étudiants. Cible encodée : {codes}")

    # Découpage train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocess = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                NUM_COLS,
            ),
            (
                "ord",
                Pipeline(
                    [
                        (
                            "encode",
                            OrdinalEncoder(
                                categories=[
                                    ORDINAL[c]
                                    for c in ORD_COLS
                                ],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
                ORD_COLS,
            ),
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore"),
                NOM_COLS,
            ),
        ]
    )

    model = Pipeline(
        [
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test),
    )

    print(f"Précision sur le jeu de test : {accuracy:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
        },
        MODEL_PATH,
    )

    print(f"Modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()