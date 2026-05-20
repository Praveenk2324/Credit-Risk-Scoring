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
