from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import torch
import os

app = FastAPI(title="Credit Risk Scoring API", version="1.0")

print("Booting up API and fetching Production model from MLflow...")
try:
    model_uri = "models:/CreditRisk_PyTorch_Model/Production"
    model = mlflow.pytorch.load_model(model_uri)
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
    if len(data.features) != 10:
        raise HTTPException(status_code=400, detail="Expected exactly 10 features.")
    
    try:
        input_tensor = torch.FLoatTensor([data.features])

        with torch.no_grad():
            raw_logit = model(input_tensor)
            probability = torch.sigmoid(raw_logit).item()

        risk_tier = "High Risk (Likely Default)" if probability > 0.5 else "Low Risk (Safe)"

        return {

            "default_probability": round(probability, 4),
            "risk_tier": risk_tier
        }

    except Exception as e:
        raise HTTPException(status_code=500, deatil=str(e))

@app.get("/")
def health_check():
    return {"status": "API is live and model is ready"}