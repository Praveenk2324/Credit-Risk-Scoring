import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import mlflow
import optuna
import os

# 1. Define a flexible version of your model
class FlexibleTabularNet(nn.Module):
    def __init__(self, input_dim, hidden_size_1, hidden_size_2, dropout_rate):
        super(FlexibleTabularNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.relu2 = nn.ReLU()
        
        self.out = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.layer1(x))))
        x = self.relu2(self.bn2(self.layer2(x)))
        return self.out(x)

def load_data():
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')
    
    X_train = torch.FloatTensor(train_df.drop('SeriousDlqin2yrs', axis=1).values)
    y_train = torch.FloatTensor(train_df['SeriousDlqin2yrs'].values).view(-1, 1)
    
    X_val = torch.FloatTensor(val_df.drop('SeriousDlqin2yrs', axis=1).values)
    y_val = torch.FloatTensor(val_df['SeriousDlqin2yrs'].values).view(-1, 1)
    
    return X_train, y_train, X_val, y_val

#Optuna Objective Function

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_val, y_val = load_data()

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hidden_1 = trial.suggest_categorical("hidden_size_1", [32, 64, 128])
    hidden_2 = trial.suggest_categorical("hidden_size_2", [16, 32, 64])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = FlexibleTabularNet(X_train.shape[1], hidden_1, hidden_2, dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        
        epochs = 5
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                
            # Validation Phase
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    probs = torch.sigmoid(model(batch_X))
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            val_auc = roc_auc_score(all_targets, all_preds)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            
            # Tell Optuna how we did this epoch so it can prune bad trials early
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_auc
def main():
    print("Starting Optuna Hyperparameter Search...")
    mlflow.set_experiment("Credit_Risk_Scoring")
    
    # Create the parent MLflow run that will hold all the Optuna trials
    with mlflow.start_run(run_name="Optuna_Search"):
        # Create a study to MAXIMIZE our validation AUC
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20) # We'll run 20 different combinations
        
        print("\n--- Tuning Complete ---")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Val AUC: {study.best_trial.value:.4f}")
        print("Best Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

if __name__ == '__main__':
    main()

# Best Parameters:
#     learning_rate: 0.0020664060646579277
#     batch_size: 64
#     dropout_rate: 0.14212497750656095
#     hidden_size_1: 32
#     hidden_size_2: 64