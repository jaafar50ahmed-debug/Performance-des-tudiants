import pytest
from fastapi.testclient import TestClient

from app.main import app

BON_PROFIL = {
    "reading_score": 90,
    "writing_score": 88,
    "parental_level_of_education": "bachelor's degree",
    "test_preparation_course": "completed",
    "gender": "female",
    "race_ethnicity": "group D",
    "lunch": "standard",
}

PROFIL_FAIBLE = {
    "reading_score": 40,
    "writing_score": 35,
    "parental_level_of_education": "some high school",
    "test_preparation_course": "none",
    "gender": "male",
    "race_ethnicity": "group A",
    "lunch": "free/reduced",
}


@pytest.fixture(scope="module")
def client():
    # Déclenche le lifespan de FastAPI et le chargement du modèle
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_bon_profil(client):
    response = client.post("/predict", json=BON_PROFIL)
    assert response.status_code == 200
    assert response.json()["class_name"] == "reussite"


def test_predict_profil_faible(client):
    response = client.post("/predict", json=PROFIL_FAIBLE)
    assert response.status_code == 200
    assert response.json()["class_name"] == "echec"


def test_predict_missing_field(client):
    data = {k: v for k, v in BON_PROFIL.items() if k != "lunch"}

    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_predict_out_of_range(client):
    response = client.post(
        "/predict",
        json={
            **BON_PROFIL,
            "reading_score": 150,
        },
    )

    assert response.status_code == 422


def test_predict_unknown_category(client):
    response = client.post(
        "/predict",
        json={
            **BON_PROFIL,
            "parental_level_of_education": "doctorat",
        },
    )

    assert response.status_code == 422


def test_probabilities_sum_to_one(client):
    response = client.post("/predict", json=BON_PROFIL)

    probabilities = response.json()["probabilities"]

    assert abs(sum(probabilities.values()) - 1) < 1e-6