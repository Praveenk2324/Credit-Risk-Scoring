# src/export.py
import mlflow
from mlflow.tracking import MlflowClient

# Connect to your local MLflow Registry
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

model_name = "CreditRisk_PyTorch_Model"

print("1. Extracting PyTorch model from Registry...")
mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{model_name}/Production",
    dst_path="production_model"
)

print("2. Locating scaler.pkl in the database...")
# Find the exact Run ID that created this Production model
versions = client.get_latest_versions(name=model_name, stages=["Production"])
run_id = versions[0].run_id

print(f"Found Run ID: {run_id}. Extracting scaler...")
try:
    # Force MLflow to download the scaler directly from the specific run
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/scaler.pkl",
        dst_path="production_model"
    )
    print("✅ PyTorch Model and Scaler successfully extracted to 'production_model'!")
except Exception as e:
    # Fallback just in case it was logged to the artifact root instead of the 'model' folder
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/scaler.pkl",
        dst_path="production_model"
    )
    print("✅ PyTorch Model and Scaler successfully extracted to 'production_model'!")