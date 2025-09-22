# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
import shap

# ---- Config ----
MODEL_PATH = "models/xgb_model.joblib"
FEATURES = ["events", "logins", "fails", "successes", "offhour_events", "uniq_ips"]

# ---- Load model & SHAP explainer ----
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

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

# Raw log schema for /featurize
class RawLog(BaseModel):
    timestamp: str                # e.g. "2025-09-22T13:45:00"
    user: str                     # e.g. "alice"
    src_ip: str                   # e.g. "10.0.0.12"
    event: str                    # e.g. "login", "file", "vpn", ...
    success: Optional[int] = None # 0/1 (only meaningful for login)

# ---- App ----
app = FastAPI(title="XAI Forensics API", version="1.0")

def vector_from_payload(d: Dict[str, float]) -> np.ndarray:
    """Validate order and types; return 2D numpy array with one row."""
    try:
        return np.array([[float(d[f]) for f in FEATURES]], dtype=float)
    except KeyError as missing:
        raise HTTPException(status_code=400, detail=f"Missing feature: {missing}")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="All features must be numeric.")

def _safe_expected_value(exp_val) -> float:
    """Handle SHAP expected_value being scalar or array(list) for binary classifiers."""
    if isinstance(exp_val, (list, tuple, np.ndarray)):
        # If binary classifier, index 1 is the positive class; else take first
        return float(exp_val[1] if len(exp_val) > 1 else exp_val[0])
    return float(exp_val)

def _safe_shap_values(shap_vals, class_index: int = 1):
    """Handle SHAP returning list per class or a single 2D array."""
    if isinstance(shap_vals, list):
        # pick the positive class if available
        if len(shap_vals) > class_index:
            return shap_vals[class_index]
        return shap_vals[0]
    return shap_vals

def _off_hours_from_ts(ts_ser: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts_ser, errors="coerce")
    h = t.dt.hour
    return ((h < 6) | (h >= 22)).astype(int)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "features": FEATURES}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FeatureVector):
    X = vector_from_payload(payload.features)
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    return {"label": label, "probability": round(proba, 4)}

@app.post("/explain", response_model=ExplanationResponse)
def explain(payload: FeatureVector):
    X = vector_from_payload(payload.features)
    # SHAP values (handle both legacy list-of-classes and array forms)
    shap_vals_raw = explainer.shap_values(X)
    shap_vals = _safe_shap_values(shap_vals_raw, class_index=1)
    base_value = _safe_expected_value(explainer.expected_value)
    proba = float(model.predict_proba(X)[0, 1])

    # Build contributions
    all_items = [
        ExplanationItem(
            feature=f,
            contribution=float(shap_vals[0, i]),
            value=float(X[0, i]),
        )
        for i, f in enumerate(FEATURES)
    ]
    top = sorted(all_items, key=lambda it: abs(it.contribution), reverse=True)[:3]

    return ExplanationResponse(
        probability=round(proba, 4),
        base_value=round(base_value, 6),
        top_reasons=top,
        all_contributions=all_items,
    )

@app.post("/featurize")
def featurize(logs: List[RawLog]):
    """
    Aggregate a list of raw log rows (one 'case' = e.g., a user-day slice)
    into the 6 model features your API expects.
    """
    if not logs:
        raise HTTPException(status_code=400, detail="Empty payload: provide at least one log row.")

    df = pd.DataFrame([l.model_dump() for l in logs])

    # Defensive defaults
    if "success" not in df.columns:
        df["success"] = 0

    # Derive helper flags
    df["off_hours"] = _off_hours_from_ts(df["timestamp"])
    df["failed_login"] = ((df["event"] == "login") & (df["success"] == 0)).astype(int)
    df["success_login"] = ((df["event"] == "login") & (df["success"] == 1)).astype(int)

    # Build the single feature vector
    feats = {
        "events": int(len(df)),
        "logins": int((df["event"] == "login").sum()),
        "fails": int(df["failed_login"].sum()),
        "successes": int(df["success_login"].sum()),
        "offhour_events": int(df["off_hours"].sum()),
        "uniq_ips": int(df["src_ip"].nunique()),
    }

    # Keep the same order your model expects
    ordered = {k: float(feats[k]) for k in FEATURES}
    return {"features": ordered}
