from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
app = FastAPI(title="Model Release Observatory API")
# Hardcoded Run ID for candidate_v2 (We will fetch the latest APPROVED model in a real scenario)
# For MVP, we use the MLflow search API to get the best model dynamically on startup
def load_best_model():
    experiment = mlflow.get_experiment_by_name("Breast_Cancer_Classification")
    if not experiment:
        raise RuntimeError("Experiment not found. Run training first.")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"]
    )
    if len(runs) == 0:
        raise RuntimeError("No runs found.")
    best_run = runs.iloc[0]
    model_uri = f"runs:/{best_run['run_id']}/model_artifact"
    print(f"Loading model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)
try:
    model = load_best_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
class InferenceRequest(BaseModel):
    features: list[float]
@app.post("/predict")
def predict(request: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Expected 30 features for Breast Cancer Dataset
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")
    # scikit-learn expects 2D array
    X_input = pd.DataFrame([request.features])
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
