from __future__ import annotations
import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.init_weights()
    def init_weights(self):
        for m in [self.conv1, self.conv2, self.downsample] if self.downsample else [self.conv1, self.conv2]:
            if m is not None:
                nn.init.kaiming_normal_(m.weight)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_dim: int, channels: list[int] = [32, 32, 64], k: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        c_in = input_dim
        for i, c_out in enumerate(channels):
            layers.append(TemporalBlock(c_in, c_out, k=k, dilation=2**i, dropout=dropout))
            c_in = c_out
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(c_in, c_in),
            nn.ReLU(),
            nn.Linear(c_in, 1)
        )
    def forward(self, x):  # x: (B,T,F)
        x = x.transpose(1, 2)  # (B,F,T)
        out = self.network(x)  # (B,C,T)
        last = out[:, :, -1]
        return self.head(last)
