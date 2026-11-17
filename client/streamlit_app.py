import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

NIVEAUX = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]

st.set_page_config(
    page_title="Réussite en maths",
    layout="wide"
)

st.title("Réussite en maths")

st.write(
    "Saisissez le profil d'un étudiant : l'API prédit s'il réussira l'examen de maths."
)

col1, col2 = st.columns(2)

with col1:
    reading_score = st.number_input(
        "Note de lecture",
        min_value=0,
        max_value=100,
        value=85,
        step=1
    )

    writing_score = st.number_input(
        "Note d'écriture",
        min_value=0,
        max_value=100,
        value=82,
        step=1
    )

    gender = st.selectbox(
        "Genre",
        ["female", "male"]
    )

    race_ethnicity = st.selectbox(
        "Groupe",
        ["group A", "group B", "group C", "group D", "group E"],
        index=3
    )

with col2:
    parental_level_of_education = st.selectbox(
        "Niveau d'études des parents",
        NIVEAUX,
        index=4
    )

    test_preparation_course = st.selectbox(
        "Cours de préparation",
        ["none", "completed"],
        index=1
    )

    lunch = st.selectbox(
        "Repas",
        ["standard", "free/reduced"]
    )

if st.button("Prédire"):

    # Validation côté client
    if reading_score < 0 or reading_score > 100:
        st.error("La note de lecture doit être comprise entre 0 et 100.")
        st.stop()

    if writing_score < 0 or writing_score > 100:
        st.error("La note d'écriture doit être comprise entre 0 et 100.")
        st.stop()

    payload = {
        "reading_score": int(reading_score),
        "writing_score": int(writing_score),
        "parental_level_of_education": parental_level_of_education,
        "test_preparation_course": test_preparation_course,
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "lunch": lunch,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        st.success(
            f"Prédiction : {result['class_name']}"
        )

        st.subheader("Probabilités")

        st.bar_chart(
            pd.Series(result["probabilities"])
        )

        st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Impossible de joindre l'API FastAPI. Vérifiez que Uvicorn est lancé."
        )

    except requests.exceptions.Timeout:
        st.error(
            "Le serveur a mis trop de temps à répondre."
        )

    except requests.exceptions.HTTPError:
        try:
            st.error(
                f"Erreur API : {response.json()['detail']}"
            )
        except Exception:
            st.error(
                f"Erreur HTTP {response.status_code}"
            )

    except Exception as e:
        st.error(f"Erreur inattendue : {e}")