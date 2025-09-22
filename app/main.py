from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import joblib, shap
import numpy as np

# ---- Config ----
MODEL_PATH = "models/xgb_model.joblib"
FEATURES = ["events","logins","fails","successes","offhour_events","uniq_ips"]

# ---- Load model & SHAP explainer ----
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

# ---- Schemas ----
class FeatureVector(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    label: int
    probability: float

class ExplanationItem(BaseModel):
    feature: str
    contribution: float
    value: float

class ExplanationResponse(BaseModel):
    probability: float
    base_value: float
    top_reasons: List[ExplanationItem]
    all_contributions: List[ExplanationItem]

# ---- App ----
app = FastAPI(title="XAI Forensics API", version="1.0")

def vector_from_payload(d: Dict[str, float]) -> np.ndarray:
    try:
        return np.array([[float(d[f]) for f in FEATURES]])
    except KeyError as missing:
        raise HTTPException(status_code=400, detail=f"Missing feature: {missing}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "features": FEATURES}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FeatureVector):
    X = vector_from_payload(payload.features)
    proba = float(model.predict_proba(X)[0,1])
    label = int(proba >= 0.5)
    return {"label": label, "probability": round(proba, 4)}

@app.post("/explain", response_model=ExplanationResponse)
def explain(payload: FeatureVector):
    X = vector_from_payload(payload.features)
    shap_vals = explainer.shap_values(X)  # shape: (1, n_features)
    base_value = float(explainer.expected_value)
    proba = float(model.predict_proba(X)[0,1])

    all_items = [
        ExplanationItem(feature=f, contribution=float(shap_vals[0, i]), value=float(X[0, i]))
        for i, f in enumerate(FEATURES)
    ]
    top = sorted(all_items, key=lambda it: abs(it.contribution), reverse=True)[:3]

    return ExplanationResponse(
        probability=round(proba, 4),
        base_value=round(base_value, 6),
        top_reasons=top,
        all_contributions=all_items
    )
