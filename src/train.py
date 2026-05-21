import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    df = pd.read_parquet('data/processed/train.parquet')

    X = df.drop('SeriousDlqin2yrs', axis=1).values
    y = df['SeriousDlqin2yrs'].values

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = TabularNet(input_dim=X.shape[1]).to(device)

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
            running_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
        print("Training complete! Logging model to mlflow...")
        mlflow.pytorch.log_model(model, "model")

if __name__ == '__main__':
    main()