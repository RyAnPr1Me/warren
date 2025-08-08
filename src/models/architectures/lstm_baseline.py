from __future__ import annotations
import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1]
        return self.head(last)
