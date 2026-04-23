"""
GNN Churn Prediction API
Serves predictions from the trained GraphSAGE model.
Deploy to Render (free tier): https://render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import pickle, json
from pathlib import Path
import os

app = FastAPI(title="GNN Churn Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your GitHub Pages URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model definition (must match notebook) ──────────────────────
class ChurnGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def embed(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return F.relu(self.conv2(x, edge_index))


# ── Load artifacts at startup ────────────────────────────────────
BASE = Path(__file__).parent

with open(BASE / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(BASE / "feature_cols.json") as f:
    feature_cols = json.load(f)

with open(BASE.parent / "data" / "graph_stats.json") as f:
    graph_stats = json.load(f)

edge_index = torch.load(BASE / "edge_index.pt", map_location="cpu")
predictions_df = pd.read_csv(BASE.parent / "data" / "predictions.csv")

model = ChurnGNN(in_channels=len(feature_cols), hidden_channels=64, out_channels=2)
model.load_state_dict(torch.load(BASE / "model_weights.pt", map_location="cpu"))
model.eval()

# Reconstruct full node feature matrix from saved predictions + scaler
# (In production you'd load the full processed dataset)
print("API ready.")


# ── Schemas ──────────────────────────────────────────────────────
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


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "GNN Churn Prediction API"}


@app.get("/graph-stats")
def get_graph_stats():
    """Return model performance and graph structure metrics."""
    return {
        **graph_stats,
        "model": "GraphSAGE (2 layers, hidden=64)",
        "description": "Nodes = customers, edges = inferred call relationships"
    }


@app.get("/top-at-risk")
def get_top_at_risk(n: int = 20):
    """Return top N customers most at risk of churning."""
    top = (
        predictions_df
        .sort_values("churn_prob", ascending=False)
        .head(n)
        [["customer_id", "churn_prob", "tenure", "monthly_charges", "actual_churn"]]
    )
    return top.to_dict(orient="records")


@app.get("/customer/{customer_id}")
def get_customer(customer_id: int):
    """Return prediction + graph neighbors for a single customer."""
    if customer_id < 0 or customer_id >= len(predictions_df):
        raise HTTPException(status_code=404, detail="Customer not found")

    row = predictions_df.iloc[customer_id]

    # Find neighbors from edge index
    mask = edge_index[0] == customer_id
    neighbor_ids = edge_index[1][mask].tolist()[:10]  # cap at 10

    neighbors = []
    for nid in neighbor_ids:
        nb = predictions_df.iloc[nid]
        neighbors.append({
            "id": int(nid),
            "churn_prob": float(nb["churn_prob"]),
            "actual_churn": int(nb["actual_churn"])
        })

    return {
        "customer_id": customer_id,
        "churn_prob": float(row["churn_prob"]),
        "actual_churn": int(row["actual_churn"]),
        "tenure": float(row["tenure"]),
        "monthly_charges": float(row["monthly_charges"]),
        "neighbors": neighbors,
        "avg_neighbor_churn_prob": round(
            np.mean([n["churn_prob"] for n in neighbors]) if neighbors else 0.0, 4
        )
    }


@app.get("/graph-sample")
def get_graph_sample(size: int = 100):
    """
    Return a subgraph sample for D3 visualization.
    Returns nodes with churn probabilities and edges between them.
    """
    size = min(size, 300)  # cap for performance
    sample_ids = list(range(size))

    nodes = []
    for i in sample_ids:
        row = predictions_df.iloc[i]
        nodes.append({
            "id": i,
            "churn_prob": float(row["churn_prob"]),
            "actual_churn": int(row["actual_churn"]),
            "tenure": float(row["tenure"]),
        })

    # Edges within the sample
    mask = (edge_index[0] < size) & (edge_index[1] < size)
    sub = edge_index[:, mask]
    # Deduplicate undirected
    seen = set()
    edges = []
    for src, dst in sub.T.tolist():
        key = (min(src, dst), max(src, dst))
        if key not in seen:
            seen.add(key)
            edges.append({"source": src, "target": dst})

    return {"nodes": nodes, "edges": edges}


@app.post("/predict")
def predict_single(features: CustomerFeatures):
    """
    Predict churn for a new customer (without graph context).
    Note: without neighbor data this uses tabular features only.
    For graph-aware prediction, use /customer/{id}.
    """
    # Build a minimal feature vector (fill unknowns with 0)
    feature_dict = {col: 0.0 for col in feature_cols}
    for key, val in features.dict().items():
        if key in feature_dict:
            feature_dict[key] = float(val)

    x_raw = np.array([feature_dict[c] for c in feature_cols]).reshape(1, -1)
    x_scaled = scaler.transform(x_raw)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float)

    # Single-node graph (no edges) — purely feature-based
    single_edge = torch.zeros((2, 0), dtype=torch.long)

    with torch.no_grad():
        out = model(x_tensor, single_edge)
        prob = F.softmax(out, dim=1)[0, 1].item()

    return {
        "churn_probability": round(prob, 4),
        "risk_level": "high" if prob > 0.6 else "medium" if prob > 0.35 else "low",
        "note": "Graph-context not available for new customers. Use /customer/{id} for existing customers."
    }
