from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from src.models.architectures.lstm_baseline import LSTMBaseline

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: list[str], target: str, seq_len: int):
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.df = df.dropna().reset_index(drop=True)

    def __len__(self):
        return max(0, len(self.df) - self.seq_len - 1)

    def __getitem__(self, idx):
        window = self.df.iloc[idx: idx + self.seq_len]
        x = torch.tensor(window[self.features].values, dtype=torch.float32)
        y_row = self.df.iloc[idx + self.seq_len]
        y = torch.tensor(y_row[self.target], dtype=torch.float32)
        return x, y


def train_lstm(df: pd.DataFrame, features: list[str], target: str, seq_len: int = 120, epochs: int = 5, batch_size: int = 32, lr: float = 5e-4, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = SeqDataset(df, features, target, seq_len)
    if len(ds) == 0:
        raise ValueError("Not enough data for sequence training")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = LSTMBaseline(input_dim=len(features)).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            count += len(xb)
        print(f"Epoch {epoch}: loss={total/count:.6f}")
    return model
