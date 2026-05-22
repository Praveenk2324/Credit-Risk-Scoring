# src/train.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import mlflow
import yaml
import os

# 1. Update Architecture to accept dynamic parameters
class TabularNet(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2, dropout_rate):
        super(TabularNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.relu2 = nn.ReLU()
        
        self.out = nn.Linear(hidden_2, 1)

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. READ PARAMETERS FROM YAML
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)['train']
        
    lr = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']
    dropout_rate = params['architecture']['dropout_rate']
    hidden_1 = params['architecture']['hidden_1']
    hidden_2 = params['architecture']['hidden_2']

    print("Loading processed data...")
    X_train, y_train, X_val, y_val = load_data()
    
    # Apply dynamic batch size
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    # Apply dynamic architecture
    model = TabularNet(X_train.shape[1], hidden_1, hidden_2, dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    
    # Apply dynamic learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting MLflow run...")
    mlflow.set_experiment("Credit_Risk_Scoring")
    
    with mlflow.start_run():
        # We can log the entire params dictionary directly to MLflow!
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "dropout_rate": dropout_rate,
            "hidden_1": hidden_1,
            "hidden_2": hidden_2
        })
        
        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    probs = torch.sigmoid(outputs)
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_auc = roc_auc_score(all_targets, all_preds)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            
        print("Training complete! Logging model to MLflow...")
        mlflow.pytorch.log_model(model, "model")

if __name__ == '__main__':
    main()