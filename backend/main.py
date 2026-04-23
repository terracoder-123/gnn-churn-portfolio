from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import pandas as pd
import pickle, json
from pathlib import Path
import onnxruntime as ort

app = FastAPI(title="Churn Prediction API (ONNX)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
scaler = None
feature_cols = None
graph_stats = None
predictions_df = None
session = None

# ── Load everything safely at startup ─────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global scaler, feature_cols, graph_stats, predictions_df, session

    BASE = Path(__file__).parent

    try:
        print("Loading scaler...")
        scaler = pickle.load(open(BASE / "scaler.pkl", "rb"))

        print("Loading feature columns...")
        feature_cols = json.load(open(BASE / "feature_cols.json"))

        print("Loading graph stats...")
        graph_stats = json.load(open(BASE.parent / "data" / "graph_stats.json"))

        print("Loading predictions...")
        predictions_df = pd.read_csv(BASE.parent / "data" / "predictions.csv")

        print("Loading ONNX model...")
        session = ort.InferenceSession(str(BASE / "model.onnx"))

        print("✅ API ready.")

    except Exception as e:
        print("❌ Startup failed:", e)
        raise e


# ── Input schema ─────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int = 0
    Partner: int = 0
    Dependents: int = 0
    PhoneService: int = 1
    PaperlessBilling: int = 0
    gender: int = 0


# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ONNX Churn API running"}


@app.get("/graph-stats")
def get_graph_stats():
    return graph_stats


@app.get("/top-at-risk")
def get_top_at_risk(n: int = 20):
    top = (
        predictions_df
        .sort_values("churn_prob", ascending=False)
        .head(n)
    )
    return top.to_dict(orient="records")


@app.get("/customer/{customer_id}")
def get_customer(customer_id: int):
    if customer_id < 0 or customer_id >= len(predictions_df):
        raise HTTPException(status_code=404, detail="Customer not found")

    row = predictions_df.iloc[customer_id]

    return {
        "customer_id": customer_id,
        "churn_prob": float(row["churn_prob"]),
        "actual_churn": int(row["actual_churn"]),
        "tenure": float(row["tenure"]),
        "monthly_charges": float(row["monthly_charges"]),
    }


@app.post("/predict")
def predict_single(features: CustomerFeatures):
    try:
        # Build feature vector
        feature_dict = {col: 0.0 for col in feature_cols}
        for key, val in features.dict().items():
            if key in feature_dict:
                feature_dict[key] = float(val)

        x = np.array([feature_dict[c] for c in feature_cols]).reshape(1, -1)
        x_scaled = scaler.transform(x).astype(np.float32)

        # Run ONNX model
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: x_scaled})

        prob = float(outputs[0][0][1])  # assuming binary output

        return {
            "churn_probability": round(prob, 4),
            "risk_level": "high" if prob > 0.6 else "medium" if prob > 0.35 else "low"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
