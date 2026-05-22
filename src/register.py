import mlflow
from mlflow.tracking import MlflowClient
import os

def main():
    print("Starting Model Validation and Registration...")

    client = MlflowClient()
    experiment_name = "Credit_Risk_Scoring"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
    
    print(f"Searching for the best model in experiment ID: {experiment.experiment_id}")
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_auc DESC"],
        max_results=1
    )[0]

    best_run_id = best_run.info.run_id
    best_auc = best_run.data.metrics.get("val_auc", 0)

    print(f"Best Run ID found: {best_run_id}")
    print(f"Best Validation AUC: {best_auc:.4f}")

    AUC_THRESHOLD = 0.80

    if best_auc < AUC_THRESHOLD:
        print(f"❌ Model failed validation! AUC {best_auc:.4f} is below threshold {AUC_THRESHOLD}.")
        print("Pipeline stopped. Model will NOT be registered.")
        return
    
    print(f"✅ Model passed validation (AUC > {AUC_THRESHOLD})!")

    # 4. Register the Model in MLflow
    model_name = "CreditRisk_PyTorch_Model"
    model_uri = f"runs:/{best_run_id}/model"

    print(f"Registering model as '{model_name}'...")

    registered_model = mlflow.register_model(model_uri, model_name)

    print(f"Promoting version {registered_model.version} to Production...")    
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True
    )
    print("Pipeline Complete! Model is in Production and ready to be served.")

if __name__=='__main__':
    main()