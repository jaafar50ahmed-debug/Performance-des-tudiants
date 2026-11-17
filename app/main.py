from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import load_model, predict
from app.schemas import PredictionResponse, StudentFeatures

# Dictionnaire partagé qui contient le modèle chargé en mémoire
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ce code s'exécute UNE fois au démarrage du serveur
    state["model"] = load_model()

    yield

    # Ce code s'exécute à l'arrêt du serveur
    state.clear()


app = FastAPI(
    title="API Réussite en maths",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "API Réussite en maths opérationnelle"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: StudentFeatures):
    bundle = state.get("model")

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé",
        )

    try:
        # model_dump() transforme l'objet Pydantic
        # en dictionnaire {nom_colonne: valeur}
        return predict(
            bundle,
            features.model_dump(),
        )

    except Exception:
        # On ne renvoie jamais la trace Python au client
        raise HTTPException(
            status_code=500,
            detail="Erreur interne de prédiction",
        )