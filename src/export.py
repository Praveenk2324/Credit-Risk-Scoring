# src/export.py
import mlflow

# Connect to your local MLflow Registry
mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("Extracting Production model...")
mlflow.artifacts.download_artifacts(
    artifact_uri="models:/CreditRisk_PyTorch_Model/Production",
    dst_path="production_model"
)
print("✅ Model successfully extracted to the 'production_model' folder!")