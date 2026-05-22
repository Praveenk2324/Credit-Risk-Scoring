from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import torch
import traceback
import os

import __main__
from src.train import TabularNet
__main__.TabularNet = TabularNet

app = FastAPI(title="Credit Risk Scoring API", version="1.0")

print("Booting up API and fetching Production model from MLflow...")
try:
    model_uri = "production_model"
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
    model.eval()
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class BorrowerData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_risk(data: BorrowerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    if len(data.features) != 12:
        raise HTTPException(status_code=400, detail="Expected exactly 10 features.")
    
    try:
        input_tensor = torch.FloatTensor([data.features])

        with torch.no_grad():
            raw_logit = model(input_tensor)
            probability = torch.sigmoid(raw_logit).item()

        risk_tier = "High Risk (Likely Default)" if probability > 0.5 else "Low Risk (Safe)"

        return {

            "default_probability": round(probability, 4),
            "risk_tier": risk_tier
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "API is live and model is ready"}