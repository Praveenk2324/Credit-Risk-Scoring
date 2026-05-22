from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import torch
import joblib
import numpy as np
import traceback
import os
import __main__

# 1. Alias the baseline model
from src.train import TabularNet
__main__.TabularNet = TabularNet

# 2. Alias the Optuna tuned model in case it won the registry!
try:
    from src.tune import FlexibleTabularNet
    __main__.FlexibleTabularNet = FlexibleTabularNet
except ImportError:
    # Fallback just in case you saved tune.py in the root folder instead of src/
    try:
        from tune import FlexibleTabularNet
        __main__.FlexibleTabularNet = FlexibleTabularNet
    except ImportError:
        pass

app = FastAPI(title="Credit Risk Scoring API", version="1.1")

print("Booting up API and fetching artifacts from production_model...")
try:
    model_uri = "production_model"
    
    # 1. Load the Model
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
    model.eval()
    
    # 2. Load the Scaler (THE FIX)
    scaler = joblib.load(os.path.join(model_uri, "scaler.pkl"))
    
    print("✅ Model and Scaler Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    model, scaler = None, None

class BorrowerData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_risk(data: BorrowerData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded.")
        
    if len(data.features) != 12:
        raise HTTPException(status_code=400, detail=f"Expected exactly 12 features, got {len(data.features)}")
    
    try:
        # THE FIX: Convert raw features to 2D array and scale them!
        raw_array = np.array(data.features).reshape(1, -1)
        scaled_array = scaler.transform(raw_array)
        
        # Pass the SCALED data into the PyTorch Tensor
        input_tensor = torch.FloatTensor(scaled_array)

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