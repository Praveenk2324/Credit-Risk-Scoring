import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import mlflow
import os

class TabularNet(nn.Module):
    def __init__(self, input_dim):
        super(TabularNet, self).__init__()

        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.layer1(x))))
        x = self.relu2(self.bn2(self.layer2(x)))
        return self.out(x)

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    print("Loading processed data...")
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')

    def prepare_data(df):
        X = df.drop('SeriousDlqin2yrs', axis=1).values
        y = df['SeriousDlqin2yrs'].values
        return torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)

    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

    
    model = TabularNet(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting MLflow run...")
    mlflow.set_experiment("Credit_Risk_Scoring")

    with mlflow.start_run():
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 5)

        epochs = 5
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

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
            
            # Log all three metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            
        print("Training complete! Logging model to MLflow...")
        mlflow.pytorch.log_model(model, "model")
       

if __name__ == '__main__':
    main()