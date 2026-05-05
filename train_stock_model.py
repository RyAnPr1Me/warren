"""
Advanced Stock Price Prediction Model Trainer (v2)

Hybrid LSTM + Multi-Head Attention + TCN architecture with:
  • Rich terminal UI: live epoch table, progress bars, colour-coded metrics
  • Interactive configuration wizard (--interactive)
  • GPU-accelerated AMP training with temperature-scaled calibration
  • Walk-forward time-series cross-validation (--walk_forward)
  • Multi-task learning: simultaneous 1/5/10/21-day horizon predictions (--multi_task)
  • Temporal Fusion Transformer architecture variant (--model_type tft)
  • Optuna hyperparameter optimisation (--tune)
  • Gradient-saliency feature importance with terminal bar chart
  • JSON config file loading (--config)
  • OneCycleLR scheduler with warm-up; early stopping with model checkpointing

Usage examples
--------------
  python train_stock_model.py                                 # quick start defaults
  python train_stock_model.py --interactive                   # interactive wizard
  python train_stock_model.py --config training_config.json   # load JSON config
  python train_stock_model.py --tune --n_trials 50            # Optuna search
  python train_stock_model.py --walk_forward --n_splits 5     # walk-forward CV
  python train_stock_model.py --model_type tft --epochs 100   # TFT architecture
  python train_stock_model.py --use_extended_symbols --save_results --visualize
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import math
import json
import time
import copy
import logging
import argparse
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt
import seaborn as sns

# ── Rich terminal UI ──────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )
    from rich.live import Live
    from rich.text import Text
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.rule import Rule
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ── Optuna (optional) ─────────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR

try:                                    # PyTorch ≥ 2.0
    from torch.amp import autocast, GradScaler as _GradScaler
    def _make_scaler():
        return _GradScaler("cuda")
    def _autocast(device_type):
        return autocast(device_type=device_type)
except (ImportError, TypeError):        # older PyTorch
    from torch.cuda.amp import autocast as _legacy_autocast, GradScaler as _GradScaler
    def _make_scaler():
        return _GradScaler()
    def _autocast(_dt):
        return _legacy_autocast()

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ── Local ─────────────────────────────────────────────────────────────────────
from stock_data_generator import (
    get_feature_engineered_stock_data,
    normalize_features,
    split_train_test,
    walk_forward_splits,
    PREDICTION_HORIZONS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "2.0.0"

BANNER = r"""
 __        __  _    ____  ____  _____  _   _
 \ \      / / / \  |  _ \|  _ \| ____|| \ | |
  \ \ /\ / / / _ \ | |_) | |_) |  _|  |  \| |
   \ V  V / / ___ \|  _ <|  _ <| |___ | |\  |
    \_/\_/ /_/   \_\_| \_\_| \_\|_____||_| \_|
"""

SYMBOLS_EXTENDED = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","INTC","AMD","TSLA","ORCL",
    "CSCO","IBM","ADBE","CRM","NFLX","PYPL","QCOM","AVGO","TXN","MU",
    "JPM","BAC","WFC","GS","MS","C","BLK","AXP","V","MA","PNC","SCHW",
    "CME","CB","MMC","TFC","USB","ALL","AIG","BK",
    "JNJ","PFE","MRK","ABBV","LLY","ABT","UNH","TMO","DHR","BMY",
    "AMGN","MDT","ISRG","GILD","CVS","VRTX","ZTS","REGN","HUM","BIIB",
    "PG","KO","PEP","WMT","COST","MCD","SBUX","NKE","DIS","HD",
    "LOW","TGT","MDLZ","CL","EL","ROST","TJX","YUM","MAR","CMG",
    "GE","HON","MMM","CAT","DE","BA","LMT","RTX","UPS","FDX",
    "UNP","CSX","ETN","EMR","ITW","PH","GD","NSC","CARR","PCAR",
    "XOM","CVX","COP","EOG","PSX","PXD","VLO","SLB","MPC","OXY",
    "T","VZ","TMUS","CMCSA","NEE","DUK","SO","D","AEP",
    "AMT","PLD","CCI","SPG","EQIX",
]

# ══════════════════════════════════════════════════════════════════════════════
# Console + Logging
# ══════════════════════════════════════════════════════════════════════════════

console = Console() if RICH_AVAILABLE else None


def _setup_logging(log_file: str = "training.log") -> logging.Logger:
    handlers = [logging.FileHandler(log_file)]
    if not RICH_AVAILABLE:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


logger = _setup_logging()


def cprint(msg: str, style: str = "") -> None:
    """Print with rich if available, else plain print."""
    if RICH_AVAILABLE and console:
        console.print(msg, style=style)
    else:
        print(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(filled: int, total: int, width: int = 20) -> str:
    f = int(width * filled / max(total, 1))
    return "█" * f + "░" * (width - f)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class StockDataset(Dataset):
    """Sliding-window time-series dataset for single-target training."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 30):
        self.features = features.astype(np.float32)
        self.targets  = targets.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx: idx + self.seq_length])
        y = torch.tensor([self.targets[idx + self.seq_length]], dtype=torch.float32)
        return x, y


class MultiTaskStockDataset(Dataset):
    """Dataset that returns targets for all prediction horizons simultaneously."""

    def __init__(self, features: np.ndarray, targets_dict: dict, seq_length: int = 30):
        self.features    = features.astype(np.float32)
        self.targets_dict = {k: v.astype(np.float32) for k, v in targets_dict.items()}
        self.seq_length  = seq_length

    def __len__(self) -> int:
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx: idx + self.seq_length])
        ys = {
            k: torch.tensor([v[idx + self.seq_length]], dtype=torch.float32)
            for k, v in self.targets_dict.items()
        }
        return x, ys


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.query  = nn.Linear(hidden_dim, hidden_dim)
        self.key    = nn.Linear(hidden_dim, hidden_dim)
        self.value  = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy  = torch.matmul(Q, K) * self.scale
        weights = self.dropout(torch.softmax(energy, dim=-1))
        out = torch.matmul(weights, V).permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.fc_out(out)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with dilation and residual skip."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv      = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(out_ch)
        self.residual  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.dropout(self.relu(self.conv(x)))
        if out.size(-1) != res.size(-1):
            n = min(out.size(-1), res.size(-1))
            out, res = out[..., -n:], res[..., -n:]
        out = (out + res).transpose(1, 2)
        return self.norm(out).transpose(1, 2)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network — core building block of TFT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: int = 0):
        super().__init__()
        self.fc1     = nn.Linear(input_dim + context_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim)
        self.gate_fc = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip    = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        inp = torch.cat([x, context], dim=-1) if context is not None else x
        h1  = torch.elu(self.fc1(inp))
        h2  = self.dropout(self.fc2(h1))
        gate = torch.sigmoid(self.gate_fc(h1))
        out = self.layer_norm(gate * h2 + self.skip(x))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class StockPredictionModel(nn.Module):
    """Hybrid BiLSTM + Multi-Head Attention + TCN model.

    Supports single-task (default) and multi-task prediction heads for all
    forecast horizons in PREDICTION_HORIZONS.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.2, bidirectional: bool = True,
                 multi_task: bool = False):
        super().__init__()
        self.multi_task   = multi_task
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention  = SelfAttention(lstm_dim, num_heads=num_heads, dropout=dropout)
        self.tcn1       = TCNBlock(lstm_dim, lstm_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2       = TCNBlock(lstm_dim, lstm_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn3       = TCNBlock(lstm_dim, lstm_dim, kernel_size=3, dilation=4, dropout=dropout)
        self.feat_proj  = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.temperature = nn.Parameter(torch.ones(1))
        fused_dim = lstm_dim + hidden_dim
        self.layer_norm1 = nn.LayerNorm(lstm_dim)
        self.layer_norm2 = nn.LayerNorm(fused_dim)
        self.fc1     = nn.Linear(fused_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if multi_task:
            # One head per forecast horizon
            self.heads = nn.ModuleDict({
                f"h{h}": nn.Linear(hidden_dim, 1) for h in PREDICTION_HORIZONS
            })
        else:
            self.fc2 = nn.Linear(hidden_dim, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        att  = self.layer_norm1(self.attention(lstm_out))
        tcn  = att.transpose(1, 2)
        tcn  = self.tcn3(self.tcn2(self.tcn1(tcn))).transpose(1, 2)
        seq  = tcn[:, -1, :]
        stat = self.feat_proj(x[:, -1, :])
        fused = self.layer_norm2(torch.cat([seq, stat], dim=1))
        return torch.relu(self.dropout(self.fc1(fused)))

    def forward(self, x: torch.Tensor):
        h = self._encode(x)
        if self.multi_task:
            return {k: head(h) / self.temperature for k, head in self.heads.items()}
        return self.fc2(h) / self.temperature


class TemporalFusionTransformer(nn.Module):
    """Simplified Temporal Fusion Transformer (TFT).

    Implements the key components: GRN-based variable selection, LSTM processing,
    and interpretable multi-head attention with a gated skip connection.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.2, multi_task: bool = False):
        super().__init__()
        self.multi_task = multi_task
        # Variable selection: project each feature through GRN + softmax weighting
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(input_dim)
        ])
        self.var_select = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Softmax(dim=-1)
        )
        self.input_proj = nn.Linear(input_dim * hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.attention = SelfAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.ff_grn    = GatedResidualNetwork(hidden_dim, hidden_dim * 2, hidden_dim, dropout=dropout)
        self.temperature = nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(hidden_dim)
        if multi_task:
            self.heads = nn.ModuleDict({
                f"h{h}": nn.Linear(hidden_dim, 1) for h in PREDICTION_HORIZONS
            })
        else:
            self.fc_out = nn.Linear(hidden_dim, 1)

    def _select_vars(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        grn_outs = torch.stack(
            [self.var_grns[i](x[:, :, i:i+1]) for i in range(D)], dim=-2
        )  # (B, T, D, H)
        weights = self.var_select(x)          # (B, T, D)
        weighted = (grn_outs * weights.unsqueeze(-1)).sum(dim=-2)  # (B, T, H)
        return weighted

    def forward(self, x: torch.Tensor):
        selected = self._select_vars(x)
        lstm_out, _ = self.lstm(selected)
        attn_in  = self.attn_grn(lstm_out)
        attn_out = self.attention(attn_in)
        post     = self.post_attn_grn(attn_out + attn_in)
        ff_out   = self.ff_grn(post)
        out      = self.norm(ff_out + post)[:, -1, :]
        if self.multi_task:
            return {k: head(out) / self.temperature for k, head in self.heads.items()}
        return self.fc_out(out) / self.temperature


def build_model(model_type: str, input_dim: int, config: dict) -> nn.Module:
    """Instantiate the requested model architecture."""
    kwargs = dict(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        multi_task=config.get("multi_task", False),
    )
    if model_type == "tft":
        return TemporalFusionTransformer(**kwargs)
    kwargs["bidirectional"] = config.get("bidirectional", True)
    return StockPredictionModel(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion, device: torch.device, scaler, scheduler=None,
                multi_task: bool = False) -> float:
    """Run one training epoch; return average loss."""
    model.train()
    total_loss, n = 0.0, 0
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device)
        optimizer.zero_grad()
        with _autocast(device.type) if use_amp else _null_ctx():
            out = model(x)
            if multi_task:
                y_dict = {k: v.to(device) for k, v in y.items()}
                loss = sum(criterion(out[k].view(-1), y_dict[k].view(-1))
                           for k in y_dict) / len(y_dict)
            else:
                loss = criterion(out.view(-1), y.to(device).view(-1))
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


class _null_ctx:
    """Null context manager for non-AMP paths."""
    def __enter__(self):  return self
    def __exit__(self, *_): pass


def validate(model: nn.Module, loader: DataLoader, criterion,
             device: torch.device, is_regression: bool = False,
             multi_task: bool = False) -> dict:
    """Evaluate model and return metric dict."""
    model.eval()
    total_loss, n = 0.0, 0
    outputs_list, targets_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            if multi_task:
                # Evaluate only on primary horizon (1-day)
                key = f"h{PREDICTION_HORIZONS[0]}"
                out_val = out[key]
                y_val   = y[key].to(device)
            else:
                out_val = out
                y_val   = y.to(device)
            loss = criterion(out_val.view(-1), y_val.view(-1))
            total_loss += loss.item(); n += 1
            outputs_list.append(out_val.cpu().numpy())
            targets_list.append(y_val.cpu().numpy())
    preds   = np.concatenate(outputs_list).ravel()
    targets = np.concatenate(targets_list).ravel()
    metrics: dict = {"val_loss": total_loss / max(n, 1)}
    if is_regression:
        metrics["mse"]  = float(mean_squared_error(targets, preds))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"]   = float(r2_score(targets, preds))
    else:
        binary = (preds > 0.0).astype(int)
        metrics["accuracy"]  = float(accuracy_score(targets, binary))
        metrics["precision"] = float(precision_score(targets, binary, average="binary", zero_division=0))
        metrics["recall"]    = float(recall_score(targets, binary, average="binary", zero_division=0))
        metrics["f1"]        = float(f1_score(targets, binary, average="binary", zero_division=0))
        try:
            metrics["roc_auc"] = float(roc_auc_score(targets, preds))
        except Exception:
            metrics["roc_auc"] = 0.0
    return metrics


def prepare_data(X_train, X_test, y_train, y_test,
                 seq_length: int = 30, batch_size: int = 64,
                 multi_task: bool = False, data: pd.DataFrame = None):
    """Build DataLoader objects for train and test splits."""
    if multi_task and data is not None:
        target_cols = {f"h{h}": f"Target_Direction_{h}d" for h in PREDICTION_HORIZONS}
        available = {k: v for k, v in target_cols.items() if v in data.columns}
        train_sz = len(X_train)
        all_data = pd.concat([X_train, X_test], ignore_index=True) if hasattr(X_train, "index") else None
        if all_data is not None:
            tr_yd = {k: data[v].values[:train_sz] for k, v in available.items()}
            te_yd = {k: data[v].values[train_sz:] for k, v in available.items()}
            train_ds = MultiTaskStockDataset(X_train.values, tr_yd, seq_length)
            test_ds  = MultiTaskStockDataset(X_test.values,  te_yd, seq_length)
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
                DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
            )
    train_ds = StockDataset(X_train.values, y_train.values, seq_length)
    test_ds  = StockDataset(X_test.values,  y_test.values,  seq_length)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Feature importance
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(model: nn.Module, feature_names: list,
                                device: torch.device, seq_length: int = 30,
                                top_n: int = 25) -> pd.DataFrame:
    """Gradient-saliency feature importance.

    Feeds a ones-tensor through the model and accumulates |∂output/∂input|
    summed across the sequence dimension to rank features.
    """
    model.eval()
    D = len(feature_names)
    dummy = torch.ones((1, seq_length, D), requires_grad=True, device=device)
    try:
        out = model(dummy)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        out.mean().backward()
        if dummy.grad is not None:
            imp = dummy.grad.abs().squeeze(0).sum(0).cpu().numpy()
        else:
            imp = np.zeros(D)
    except Exception:
        imp = np.zeros(D)
    df = (
        pd.DataFrame({"Feature": feature_names, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df


def print_feature_importance(importance_df: pd.DataFrame) -> None:
    """Print a horizontal bar chart of feature importances to the terminal."""
    if importance_df.empty:
        return
    max_val = importance_df["Importance"].max()
    if RICH_AVAILABLE and console:
        table = Table(title="Top Feature Importances (Gradient Saliency)",
                      box=box.SIMPLE_HEAVY, header_style="bold cyan")
        table.add_column("Rank", style="dim", width=5)
        table.add_column("Feature", min_width=30)
        table.add_column("Importance", width=10, justify="right")
        table.add_column("", min_width=30)
        for i, row in importance_df.iterrows():
            bar_len = int(28 * row["Importance"] / max(max_val, 1e-9))
            bar = "[green]" + "▇" * bar_len + "[/green]"
            table.add_row(
                str(i + 1), row["Feature"],
                f"{row['Importance']:.4f}", bar,
            )
        console.print(table)
    else:
        print("\nTop Feature Importances:")
        for i, row in importance_df.iterrows():
            bar = "█" * int(28 * row["Importance"] / max(max_val, 1e-9))
            print(f"  {i+1:2d}. {row['Feature']:<35} {row['Importance']:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# Rich UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    if RICH_AVAILABLE and console:
        console.print(
            Panel(
                f"[bold yellow]{BANNER}[/bold yellow]\n"
                f"[bold white]Stock Price Prediction AI  ·  v{VERSION}[/bold white]",
                border_style="bright_blue", expand=False,
            )
        )
    else:
        print(BANNER)
        print(f"Warren — Stock Price Prediction AI  v{VERSION}\n")


def print_system_info(device: torch.device) -> None:
    if RICH_AVAILABLE and console:
        table = Table(title="System Information", box=box.SIMPLE, header_style="bold cyan")
        table.add_column("Property", style="bold")
        table.add_column("Value")
        table.add_row("Python",       sys.version.split()[0])
        table.add_row("PyTorch",      torch.__version__)
        table.add_row("Device",       str(device).upper())
        if device.type == "cuda":
            table.add_row("GPU Name",   torch.cuda.get_device_name(0))
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            table.add_row("GPU Memory", f"{gb:.1f} GB")
        table.add_row("Rich UI",      "✓ enabled" if RICH_AVAILABLE else "✗ disabled")
        table.add_row("Optuna",       "✓ available" if OPTUNA_AVAILABLE else "✗ not installed")
        console.print(table)
    else:
        print(f"Device: {device}  |  PyTorch: {torch.__version__}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")


def print_config_table(config: dict) -> None:
    if RICH_AVAILABLE and console:
        table = Table(title="Training Configuration", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Parameter", style="bold yellow", min_width=22)
        table.add_column("Value", min_width=20)
        sections = {
            "Model":    ["model_type", "hidden_dim", "num_layers", "num_heads",
                         "dropout", "bidirectional", "multi_task"],
            "Training": ["epochs", "batch_size", "learning_rate", "weight_decay",
                         "patience", "seq_length", "seed"],
            "Task":     ["is_regression", "walk_forward", "n_splits"],
            "Output":   ["model_dir", "save_results", "visualize"],
        }
        for section, keys in sections.items():
            table.add_row(f"[bold bright_blue]── {section} ──[/bold bright_blue]", "")
            for k in keys:
                if k in config:
                    table.add_row(f"  {k}", str(config[k]))
        console.print(table)
    else:
        print("\nConfiguration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print()


def _build_epoch_table(history: dict, best_epoch: int, patience: int,
                       patience_counter: int, config: dict) -> "Table":
    """Build the rich Table shown during live training."""
    is_reg  = config.get("is_regression", False)
    table   = Table(
        title=f"[bold]Epoch History[/bold]  ·  patience {_bar(patience_counter, patience)} {patience_counter}/{patience}",
        box=box.SIMPLE_HEAVY, header_style="bold cyan", show_edge=True,
    )
    table.add_column("Ep",        width=5,  justify="right")
    table.add_column("Train Loss", width=11, justify="right")
    table.add_column("Val Loss",   width=10, justify="right")
    table.add_column("Acc" if not is_reg else "RMSE",  width=8,  justify="right")
    table.add_column("F1"  if not is_reg else "R²",    width=8,  justify="right")
    table.add_column("AUC" if not is_reg else "—",     width=8,  justify="right")
    table.add_column("Time",       width=7,  justify="right")
    table.add_column("",           width=6)

    rows = list(zip(
        history["train_loss"], history["val_loss"],
        history["metrics"], history.get("epoch_times", [0] * len(history["train_loss"])),
    ))
    # Show last 15 rows max to keep table readable
    for i, (tl, vl, m, et) in enumerate(rows[-15:]):
        ep_idx  = len(rows) - min(15, len(rows)) + i
        is_best = ep_idx == best_epoch
        style   = "bold green" if is_best else ""
        star    = "⭐ BEST" if is_best else ""
        if is_reg:
            c1 = f"{m.get('rmse', 0):.4f}"
            c2 = f"{m.get('r2', 0):.4f}"
            c3 = "—"
        else:
            c1 = f"{m.get('accuracy', 0)*100:.1f}%"
            c2 = f"{m.get('f1', 0):.4f}"
            c3 = f"{m.get('roc_auc', 0):.4f}"
        table.add_row(
            str(ep_idx + 1), f"{tl:.4f}", f"{vl:.4f}",
            c1, c2, c3, f"{et:.1f}s", star,
            style=style,
        )
    return table


def print_final_results(results: dict) -> None:
    """Display final training summary."""
    m   = results["final_metrics"]
    cfg = results["config"]
    is_reg = cfg.get("is_regression", False)
    if RICH_AVAILABLE and console:
        console.print(Rule("[bold green]Training Complete[/bold green]"))
        table = Table(title="Final Evaluation Metrics", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Metric", style="bold yellow", min_width=18)
        table.add_column("Value",  min_width=12)
        table.add_row("Val Loss",  f"{m.get('val_loss', 0):.5f}")
        if is_reg:
            table.add_row("RMSE",  f"{m.get('rmse', 0):.5f}")
            table.add_row("R²",    f"{m.get('r2', 0):.5f}")
        else:
            table.add_row("Accuracy",  f"{m.get('accuracy', 0)*100:.2f}%")
            table.add_row("Precision", f"{m.get('precision', 0):.4f}")
            table.add_row("Recall",    f"{m.get('recall', 0):.4f}")
            table.add_row("F1 Score",  f"[bold green]{m.get('f1', 0):.4f}[/bold green]")
            table.add_row("ROC-AUC",   f"{m.get('roc_auc', 0):.4f}")
        table.add_row("Best Epoch", str(results["best_epoch"] + 1))
        table.add_row("Train Time", _fmt_time(results["training_time"]))
        console.print(table)
    else:
        print("\n=== Final Results ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Best epoch: {results['best_epoch'] + 1}")
        print(f"  Train time: {_fmt_time(results['training_time'])}")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive Configuration Wizard
# ══════════════════════════════════════════════════════════════════════════════

def interactive_config_wizard() -> dict:
    """Guide the user through an interactive configuration session.

    Returns a config dict that is merged into the CLI args defaults.
    """
    if not RICH_AVAILABLE:
        print("Rich is required for interactive mode. Install with: pip install rich")
        return {}

    console.print(Rule("[bold blue]⚙  Configuration Wizard[/bold blue]"))
    console.print("[dim]Press Enter to accept the default shown in brackets.[/dim]\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    console.print("[bold yellow]── Data Settings ──[/bold yellow]")
    symbol_choice = Prompt.ask(
        "Symbol set",
        choices=["default", "extended", "custom"],
        default="default",
    )
    symbols = None
    if symbol_choice == "extended":
        symbols = SYMBOLS_EXTENDED
        cprint(f"  Using {len(SYMBOLS_EXTENDED)} symbols.", "dim")
    elif symbol_choice == "custom":
        raw = Prompt.ask("Enter comma-separated tickers", default="AAPL,MSFT,GOOGL")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        cprint(f"  Using {len(symbols)} custom symbols: {', '.join(symbols[:5])}…", "dim")

    start_date = Prompt.ask("Start date (YYYY-MM-DD)", default="2018-01-01")
    use_cache  = Confirm.ask("Cache downloaded data to disk?", default=True)
    use_mktctx = Confirm.ask("Include market-context features (SPY, VIX, sector ETFs)?", default=True)

    # ── Model ────────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]── Model Settings ──[/bold yellow]")
    model_type  = Prompt.ask("Architecture", choices=["hybrid", "tft"], default="hybrid")
    hidden_dim  = IntPrompt.ask("Hidden dimension", default=128)
    num_layers  = IntPrompt.ask("Number of LSTM/encoder layers", default=2)
    num_heads   = IntPrompt.ask("Attention heads", default=4)
    dropout     = FloatPrompt.ask("Dropout rate", default=0.2)
    multi_task  = Confirm.ask("Multi-task learning (all forecast horizons)?", default=False)

    # ── Training ─────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]── Training Settings ──[/bold yellow]")
    epochs        = IntPrompt.ask("Max epochs", default=50)
    batch_size    = IntPrompt.ask("Batch size", default=64)
    learning_rate = FloatPrompt.ask("Learning rate", default=0.001)
    patience      = IntPrompt.ask("Early-stopping patience (epochs)", default=10)
    seq_length    = IntPrompt.ask("Sequence length (look-back window)", default=30)
    walk_forward  = Confirm.ask("Use walk-forward cross-validation?", default=False)
    n_splits      = IntPrompt.ask("Number of CV folds", default=5) if walk_forward else 5

    # ── Output ───────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]── Output Settings ──[/bold yellow]")
    model_dir    = Prompt.ask("Model save directory", default="models")
    save_results = Confirm.ask("Save model and metrics to disk?", default=True)
    visualize    = Confirm.ask("Generate training plots?", default=True)

    console.print("\n[bold green]Configuration complete![/bold green]\n")

    return dict(
        symbols=",".join(symbols) if symbols else None,
        start_date=start_date,
        cache_dir=".cache" if use_cache else None,
        include_market_context=use_mktctx,
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        multi_task=multi_task,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        seq_length=seq_length,
        walk_forward=walk_forward,
        n_splits=n_splits,
        model_dir=model_dir,
        save_results=save_results,
        visualize=visualize,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Optuna hyperparameter search
# ══════════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(X_train, X_test, y_train, y_test,
                          n_trials: int = 30, base_config: dict = None) -> dict:
    """Run Optuna TPE search and return the best hyperparameter config."""
    if not OPTUNA_AVAILABLE:
        cprint("[red]Optuna not installed. Run: pip install optuna[/red]")
        return base_config or {}

    base = base_config or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        cfg = dict(
            hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            num_heads=trial.suggest_categorical("num_heads", [2, 4, 8]),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            learning_rate=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            seq_length=trial.suggest_categorical("seq_length", [20, 30, 60]),
            bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
            epochs=min(base.get("epochs", 20), 20),
            patience=5, seed=42, is_regression=base.get("is_regression", False),
            model_dir="/tmp/optuna_trial", model_type=base.get("model_type", "hybrid"),
            multi_task=False,
        )
        try:
            res = train_model(X_train, X_test, y_train, y_test, cfg,
                              silent=True, device=device)
            m = res["final_metrics"]
            return m.get("rmse", 1.0) if cfg["is_regression"] else -m.get("f1", 0.0)
        except Exception:
            return 1e6

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    if RICH_AVAILABLE and console:
        with console.status(f"[cyan]Optuna: running {n_trials} trials…", spinner="dots"):
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    else:
        study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    cprint(f"\n[bold green]Optuna best params:[/bold green] {best}")
    logger.info(f"Optuna best params: {best}")
    # Map back to config keys
    param_map = {"lr": "learning_rate", "wd": "weight_decay"}
    merged = {**base, **{param_map.get(k, k): v for k, v in best.items()}}
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_cross_validate(data: pd.DataFrame, config: dict,
                                 n_splits: int = 5, test_size: float = 0.1) -> list:
    """Run walk-forward CV using expanding training windows.

    Returns list of per-fold metric dicts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = []
    splits = list(walk_forward_splits(data, n_splits=n_splits, test_size=test_size))
    if RICH_AVAILABLE and console:
        console.print(Rule(f"[bold cyan]Walk-Forward Cross-Validation  ({n_splits} folds)[/bold cyan]"))

    for fold_idx, (Xtr, Xte, ytr, yte) in enumerate(splits):
        cprint(f"\n[bold]Fold {fold_idx + 1}/{len(splits)}[/bold]  "
               f"train={len(Xtr):,}  test={len(Xte):,}", "cyan")
        results = train_model(Xtr, Xte, ytr, yte, config, silent=False, device=device)
        fold_results.append(results["final_metrics"])

    # Summary table
    if RICH_AVAILABLE and console and fold_results:
        is_reg = config.get("is_regression", False)
        metric = "rmse" if is_reg else "f1"
        vals   = [r.get(metric, 0) for r in fold_results]
        table  = Table(title="Walk-Forward CV Summary", box=box.SIMPLE_HEAVY,
                       header_style="bold cyan")
        table.add_column("Fold")
        table.add_column(metric.upper(), justify="right")
        for i, v in enumerate(vals):
            table.add_row(str(i + 1), f"{v:.4f}")
        table.add_row("[bold]Mean[/bold]", f"[bold]{np.mean(vals):.4f}[/bold]")
        table.add_row("[bold]Std[/bold]",  f"[bold]{np.std(vals):.4f}[/bold]")
        console.print(table)

    return fold_results


# ══════════════════════════════════════════════════════════════════════════════
# Core training function
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test, y_train, y_test, config: dict,
                silent: bool = False, device: torch.device = None) -> dict:
    """Train the stock prediction model with a rich live terminal display.

    Parameters
    ----------
    X_train, X_test : DataFrame  — feature matrices
    y_train, y_test : Series     — target labels
    config          : dict       — full configuration dict
    silent          : bool       — suppress terminal output (for tuning/CV)
    device          : torch.device or None (auto-detect)

    Returns
    -------
    dict with keys: model, history, config, final_metrics, best_epoch, training_time
    """
    set_seed(config.get("seed", 42))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    train_loader, test_loader = prepare_data(
        X_train, X_test, y_train, y_test,
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        multi_task=config.get("multi_task", False),
    )

    model = build_model(config.get("model_type", "hybrid"), input_dim, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model '{config.get('model_type','hybrid')}': {n_params:,} parameters on {device}")

    is_reg     = config.get("is_regression", False)
    multi_task = config.get("multi_task", False)
    criterion  = nn.MSELoss() if is_reg else nn.BCEWithLogitsLoss()
    optimizer  = optim.AdamW(model.parameters(),
                              lr=config["learning_rate"],
                              weight_decay=config.get("weight_decay", 1e-5))
    scheduler  = OneCycleLR(optimizer, max_lr=config["learning_rate"],
                             steps_per_epoch=len(train_loader),
                             epochs=config["epochs"],
                             pct_start=0.3, anneal_strategy="cos")
    scaler = _make_scaler()

    history: dict = {"train_loss": [], "val_loss": [], "metrics": [], "epoch_times": []}
    best_val_metric  = float("inf") if is_reg else 0.0
    best_epoch       = 0
    patience_counter = 0
    best_model_state = None
    os.makedirs(config["model_dir"], exist_ok=True)
    best_model_path  = os.path.join(config["model_dir"], "best_model.pth")
    patience         = config.get("patience", 10)
    epochs           = config["epochs"]
    start_time       = time.time()

    # ── Progress bar columns ──────────────────────────────────────────────────
    progress_cols = [
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    def _run_training_loop(progress=None, epoch_task=None):
        nonlocal best_val_metric, best_epoch, patience_counter, best_model_state
        for epoch in range(epochs):
            ep_start = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, scheduler, multi_task=multi_task,
            )
            val_metrics = validate(
                model, test_loader, criterion, device,
                is_regression=is_reg, multi_task=multi_task,
            )
            ep_time = time.time() - ep_start
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["val_loss"])
            history["metrics"].append(val_metrics)
            history["epoch_times"].append(ep_time)
            logger.info(
                f"Epoch {epoch+1}/{epochs} | TrLoss={train_loss:.4f} "
                f"ValLoss={val_metrics['val_loss']:.4f} "
                f"{'RMSE' if is_reg else 'F1'}="
                f"{val_metrics.get('rmse' if is_reg else 'f1', 0):.4f} "
                f"t={ep_time:.1f}s"
            )
            # Best model tracking
            monitor = val_metrics.get("rmse" if is_reg else "f1", 0.0)
            improved = (monitor < best_val_metric) if is_reg else (monitor > best_val_metric)
            if improved:
                best_val_metric  = monitor
                best_epoch       = epoch
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics, "config": config,
                }, best_model_path)
            else:
                patience_counter += 1

            if progress and epoch_task is not None:
                m_key = "rmse" if is_reg else "f1"
                mv    = val_metrics.get(m_key, 0)
                desc  = (
                    f"[cyan]Epoch {epoch+1}/{epochs}  "
                    f"loss={train_loss:.4f}  "
                    f"{m_key.upper()}={mv:.4f}  "
                    f"{'⭐' if improved else '  '}"
                )
                progress.update(epoch_task, advance=1, description=desc)

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

    # ── Main training block ───────────────────────────────────────────────────
    if not silent and RICH_AVAILABLE and console:
        with Progress(*progress_cols, console=console, refresh_per_second=4) as prog:
            epoch_task = prog.add_task(
                f"[cyan]Epoch 1/{epochs}", total=epochs
            )
            with Live(
                _build_epoch_table(history, best_epoch, patience, patience_counter, config),
                console=console, refresh_per_second=1, vertical_overflow="visible",
            ) as live:
                # Patch loop to update Live display each epoch
                _orig = train_epoch.__wrapped__ if hasattr(train_epoch, "__wrapped__") else None

                for epoch in range(epochs):
                    ep_start = time.time()
                    train_loss = train_epoch(
                        model, train_loader, optimizer, criterion,
                        device, scaler, scheduler, multi_task=multi_task,
                    )
                    val_metrics = validate(
                        model, test_loader, criterion, device,
                        is_regression=is_reg, multi_task=multi_task,
                    )
                    ep_time = time.time() - ep_start
                    history["train_loss"].append(train_loss)
                    history["val_loss"].append(val_metrics["val_loss"])
                    history["metrics"].append(val_metrics)
                    history["epoch_times"].append(ep_time)
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | TrLoss={train_loss:.4f} "
                        f"ValLoss={val_metrics['val_loss']:.4f} "
                        f"{'RMSE' if is_reg else 'F1'}="
                        f"{val_metrics.get('rmse' if is_reg else 'f1', 0):.4f} "
                        f"t={ep_time:.1f}s"
                    )
                    monitor  = val_metrics.get("rmse" if is_reg else "f1", 0.0)
                    improved = (monitor < best_val_metric) if is_reg else (monitor > best_val_metric)
                    if improved:
                        best_val_metric  = monitor
                        best_epoch       = epoch
                        patience_counter = 0
                        best_model_state = copy.deepcopy(model.state_dict())
                        torch.save({
                            "epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_metrics": val_metrics, "config": config,
                        }, best_model_path)
                    else:
                        patience_counter += 1

                    m_key = "rmse" if is_reg else "f1"
                    mv    = val_metrics.get(m_key, 0)
                    prog.update(
                        epoch_task, advance=1,
                        description=(
                            f"[cyan]Epoch {epoch+1}/{epochs}  "
                            f"loss={train_loss:.4f}  "
                            f"{m_key.upper()}={mv:.4f}  "
                            f"{'⭐ best' if improved else ''}"
                        ),
                    )
                    live.update(
                        _build_epoch_table(history, best_epoch, patience, patience_counter, config)
                    )
                    if patience_counter >= patience:
                        prog.update(epoch_task, description=f"[yellow]Early stopping at epoch {epoch+1}")
                        logger.info(f"Early stopping at epoch {epoch + 1}.")
                        break
    else:
        _run_training_loop()

    total_time = time.time() - start_time
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_metrics = validate(model, test_loader, criterion, device,
                             is_regression=is_reg, multi_task=multi_task)
    return {
        "model": model, "history": history, "config": config,
        "final_metrics": final_metrics, "best_epoch": best_epoch,
        "training_time": total_time,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def visualize_results(results: dict, save_dir: str = None) -> None:
    """Generate and optionally save training curve plots."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    history = results["history"]
    cfg     = results["config"]
    is_reg  = cfg.get("is_regression", False)
    best_ep = results["best_epoch"]
    sns.set_style("darkgrid")

    # ── Loss curves ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Warren — Training Results", fontsize=14, fontweight="bold")
    axes[0].plot(history["train_loss"], label="Train Loss",      color="#4FC3F7")
    axes[0].plot(history["val_loss"],   label="Validation Loss", color="#FF8A65")
    axes[0].axvline(best_ep, color="gold", linestyle="--", alpha=0.6, label=f"Best Epoch ({best_ep+1})")
    axes[0].set_title("Loss Curves"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    m_key = "rmse" if is_reg else "f1"
    mv    = [m.get(m_key, 0) for m in history["metrics"]]
    axes[1].plot(mv, color="#81C784", label=m_key.upper())
    axes[1].axvline(best_ep, color="gold", linestyle="--", alpha=0.6)
    axes[1].set_title(f"Validation {m_key.upper()}")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "training_curves.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        cprint(f"[dim]Saved: {path}[/dim]")
    plt.close(fig)

    # ── Classification metrics ────────────────────────────────────────────────
    if not is_reg:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Classification Metrics over Epochs", fontsize=13, fontweight="bold")
        metrics_to_plot = [
            ("accuracy", "#4FC3F7", axes[0][0]),
            ("precision", "#FF8A65", axes[0][1]),
            ("recall", "#81C784", axes[1][0]),
            ("f1", "#CE93D8", axes[1][1]),
        ]
        for metric, color, ax in metrics_to_plot:
            vals = [m.get(metric, 0) for m in history["metrics"]]
            ax.plot(vals, color=color, label=metric.capitalize())
            ax.axvline(best_ep, color="gold", linestyle="--", alpha=0.6)
            ax.set_title(metric.capitalize())
            ax.set_xlabel("Epoch"); ax.legend()
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "classification_metrics.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            cprint(f"[dim]Saved: {path}[/dim]")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.silent:
        print_banner()
        print_system_info(device)

    # ── Load JSON config (may be overridden by explicit CLI flags) ────────────
    json_cfg: dict = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as fh:
            raw = json.load(fh)
        for section in ("data_settings", "model_settings", "training_settings", "output_settings"):
            json_cfg.update(raw.get(section, {}))
        json_cfg["symbols_list"] = raw.get("stocks", None)
        cprint(f"[dim]Loaded config from {args.config}[/dim]")

    # ── Interactive wizard overrides CLI defaults ──────────────────────────────
    wizard_cfg: dict = {}
    if getattr(args, "interactive", False):
        wizard_cfg = interactive_config_wizard()

    # ── Resolve symbols ────────────────────────────────────────────────────────
    symbols = None
    if args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file) as fh:
            symbols = [ln.strip().upper() for ln in fh if ln.strip()]
    elif args.symbols or wizard_cfg.get("symbols"):
        raw = args.symbols or wizard_cfg["symbols"]
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    elif args.use_extended_symbols:
        symbols = SYMBOLS_EXTENDED
    elif json_cfg.get("symbols_list"):
        symbols = json_cfg["symbols_list"]

    # ── Data ─────────────────────────────────────────────────────────────────
    cache_dir = wizard_cfg.get("cache_dir") or getattr(args, "cache_dir", None)
    include_mkt = wizard_cfg.get("include_market_context", True)

    if args.data_path and os.path.exists(args.data_path):
        if not args.silent:
            cprint(f"[cyan]Loading data from {args.data_path}…[/cyan]")
        data = pd.read_csv(args.data_path)
        cprint(f"  Loaded {len(data):,} rows × {len(data.columns)} columns")
    else:
        if not args.silent:
            cprint("[cyan]Fetching & engineering features (this may take a few minutes)…[/cyan]")
        data = get_feature_engineered_stock_data(
            symbols=symbols,
            start_date=args.start_date or json_cfg.get("start_date") or wizard_cfg.get("start_date"),
            end_date=args.end_date   or json_cfg.get("end_date"),
            min_rows=args.min_rows   or json_cfg.get("min_rows", 10_000),
            cache_dir=cache_dir,
            n_jobs=getattr(args, "n_jobs", 4),
            include_market_context=include_mkt,
            include_fundamentals=getattr(args, "include_fundamentals", False),
        )
        if args.save_data:
            out_path = args.data_path or "stock_data_features.csv"
            data.to_csv(out_path, index=False)
            cprint(f"  [dim]Data saved to {out_path}[/dim]")

    # ── Build config dict (JSON → wizard → CLI precedence) ────────────────────
    _iv = lambda cli, wizard_k, json_k, default: (
        cli if cli is not None else
        wizard_cfg.get(wizard_k, json_cfg.get(json_k, default))
    )
    config = {
        "seed":           int(args.seed),
        "batch_size":     int(_iv(args.batch_size,    "batch_size",    "batch_size",    64)),
        "epochs":         int(_iv(args.epochs,         "epochs",        "epochs",        50)),
        "learning_rate":  float(_iv(args.learning_rate, "learning_rate", "learning_rate", 0.001)),
        "weight_decay":   float(_iv(args.weight_decay,  "weight_decay",  "weight_decay",  1e-5)),
        "hidden_dim":     int(_iv(args.hidden_dim,    "hidden_dim",    "hidden_dim",    128)),
        "num_layers":     int(_iv(args.num_layers,    "num_layers",    "num_layers",    2)),
        "num_heads":      int(_iv(args.num_heads,     "num_heads",     "num_heads",     4)),
        "dropout":        float(_iv(args.dropout,     "dropout",       "dropout",       0.2)),
        "bidirectional":  bool(json_cfg.get("bidirectional", True)),
        "seq_length":     int(_iv(args.seq_length,    "seq_length",    "seq_length",    30)),
        "patience":       int(_iv(args.patience,      "patience",      "patience",      10)),
        "is_regression":  bool(args.is_regression),
        "model_dir":      str(_iv(args.model_dir,     "model_dir",     "model_dir",     "models")),
        "model_type":     str(wizard_cfg.get("model_type") or getattr(args, "model_type", "hybrid")),
        "multi_task":     bool(wizard_cfg.get("multi_task") or getattr(args, "multi_task", False)),
        "walk_forward":   bool(wizard_cfg.get("walk_forward") or getattr(args, "walk_forward", False)),
        "n_splits":       int(wizard_cfg.get("n_splits") or getattr(args, "n_splits", 5)),
        "save_results":   bool(wizard_cfg.get("save_results", args.save_results)),
        "visualize":      bool(wizard_cfg.get("visualize",    args.visualize)),
    }

    if not args.silent:
        print_config_table(config)

    # ── Normalise + split ──────────────────────────────────────────────────────
    if not args.silent:
        cprint("[cyan]Normalising features…[/cyan]")
    normalized_data = normalize_features(data, scaler_type="robust")
    X_train, X_test, y_train, y_test = split_train_test(
        normalized_data, test_size=args.test_size, time_based=True
    )
    if not args.silent:
        cprint(f"  Train: [bold]{len(X_train):,}[/bold]  Test: [bold]{len(X_test):,}[/bold]  "
               f"Features: [bold]{X_train.shape[1]}[/bold]")

    # ── Optuna tuning ──────────────────────────────────────────────────────────
    if getattr(args, "tune", False):
        n_trials = getattr(args, "n_trials", 30)
        cprint(f"\n[bold cyan]Running Optuna hyperparameter search ({n_trials} trials)…[/bold cyan]")
        config = tune_hyperparameters(X_train, X_test, y_train, y_test,
                                       n_trials=n_trials, base_config=config)
        if not args.silent:
            print_config_table(config)

    # ── Walk-forward CV ────────────────────────────────────────────────────────
    if config["walk_forward"]:
        return walk_forward_cross_validate(
            normalized_data, config,
            n_splits=config["n_splits"],
            test_size=args.test_size,
        )

    # ── Standard train ─────────────────────────────────────────────────────────
    if not args.silent:
        cprint("\n[bold cyan]Starting training…[/bold cyan]\n")
    results = train_model(X_train, X_test, y_train, y_test, config,
                           silent=args.silent, device=device)

    # ── Feature importance ─────────────────────────────────────────────────────
    if not args.silent:
        cprint("\n[cyan]Computing feature importance…[/cyan]")
        imp_df = compute_feature_importance(
            results["model"], list(X_train.columns), device,
            seq_length=config["seq_length"],
        )
        print_feature_importance(imp_df)
        if config.get("save_results"):
            imp_path = os.path.join(config["model_dir"], "feature_importance.csv")
            imp_df.to_csv(imp_path, index=False)
            cprint(f"[dim]Saved feature importance → {imp_path}[/dim]")

    # ── Final results ──────────────────────────────────────────────────────────
    if not args.silent:
        print_final_results(results)

    # ── Save ───────────────────────────────────────────────────────────────────
    if config.get("save_results"):
        os.makedirs(config["model_dir"], exist_ok=True)
        torch.save({
            "model_state_dict": results["model"].state_dict(),
            "config": config,
            "final_metrics": results["final_metrics"],
            "feature_names": list(X_train.columns),
        }, os.path.join(config["model_dir"], "final_model.pth"))
        with open(os.path.join(config["model_dir"], "training_history.json"), "w") as fh:
            json.dump({
                "train_loss": results["history"]["train_loss"],
                "val_loss":   results["history"]["val_loss"],
                "metrics":    results["history"]["metrics"],
            }, fh, indent=2, default=str)
        cprint(f"[green]Model and artefacts saved to: {config['model_dir']}[/green]")
        logger.info(f"Saved to {config['model_dir']}")

    # ── Visualise ──────────────────────────────────────────────────────────────
    if config.get("visualize"):
        if not args.silent:
            cprint("[cyan]Generating plots…[/cyan]")
        visualize_results(results, save_dir=config["model_dir"])

    if not args.silent:
        cprint("\n[bold green]Done! ✓[/bold green]")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Warren — Advanced Stock Price Prediction Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_stock_model.py --interactive\n"
            "  python train_stock_model.py --config training_config.json\n"
            "  python train_stock_model.py --tune --n_trials 50\n"
            "  python train_stock_model.py --walk_forward --n_splits 5\n"
            "  python train_stock_model.py --model_type tft --multi_task --epochs 80\n"
            "  python train_stock_model.py --use_extended_symbols --save_results --visualize\n"
        ),
    )

    # ── Mode flags ────────────────────────────────────────────────────────────
    ap.add_argument("--interactive", action="store_true",
                    help="Launch interactive configuration wizard")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to JSON config file (e.g. training_config.json)")
    ap.add_argument("--silent", action="store_true",
                    help="Suppress terminal UI (useful for scripts/notebooks)")

    # ── Data ─────────────────────────────────────────────────────────────────
    ap.add_argument("--data_path",           type=str,   default=None)
    ap.add_argument("--save_data",           action="store_true")
    ap.add_argument("--min_rows",            type=int,   default=10_000)
    ap.add_argument("--test_size",           type=float, default=0.2)
    ap.add_argument("--symbols",             type=str,   default=None,
                    help="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL")
    ap.add_argument("--symbols_file",        type=str,   default=None)
    ap.add_argument("--use_extended_symbols",action="store_true")
    ap.add_argument("--start_date",          type=str,   default=None)
    ap.add_argument("--end_date",            type=str,   default=None)
    ap.add_argument("--cache_dir",           type=str,   default=None,
                    help="Directory for data caching (None = disabled)")
    ap.add_argument("--n_jobs",              type=int,   default=4,
                    help="Parallel workers for data download")
    ap.add_argument("--include_fundamentals",action="store_true",
                    help="Fetch and include fundamental data (P/E, beta, etc.)")

    # ── Model ─────────────────────────────────────────────────────────────────
    ap.add_argument("--model_type",   type=str,   default="hybrid",
                    choices=["hybrid", "tft"],
                    help="Model architecture: hybrid (LSTM+Attn+TCN) or tft")
    ap.add_argument("--hidden_dim",   type=int,   default=128)
    ap.add_argument("--num_layers",   type=int,   default=2)
    ap.add_argument("--num_heads",    type=int,   default=4)
    ap.add_argument("--dropout",      type=float, default=0.2)
    ap.add_argument("--seq_length",   type=int,   default=30)
    ap.add_argument("--is_regression",action="store_true",
                    help="Regression (predict return) instead of classification")
    ap.add_argument("--multi_task",   action="store_true",
                    help="Multi-task learning across all forecast horizons")

    # ── Training ──────────────────────────────────────────────────────────────
    ap.add_argument("--batch_size",    type=int,   default=64)
    ap.add_argument("--epochs",        type=int,   default=50)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--weight_decay",  type=float, default=1e-5)
    ap.add_argument("--patience",      type=int,   default=10)
    ap.add_argument("--seed",          type=int,   default=42)

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    ap.add_argument("--walk_forward", action="store_true",
                    help="Use walk-forward cross-validation instead of a single split")
    ap.add_argument("--n_splits",     type=int,   default=5,
                    help="Number of walk-forward folds")

    # ── Optuna ────────────────────────────────────────────────────────────────
    ap.add_argument("--tune",      action="store_true",
                    help="Run Optuna hyperparameter search before final training")
    ap.add_argument("--n_trials",  type=int,   default=30,
                    help="Number of Optuna trials")

    # ── Output ────────────────────────────────────────────────────────────────
    ap.add_argument("--model_dir",    type=str, default="models")
    ap.add_argument("--save_results", action="store_true")
    ap.add_argument("--visualize",    action="store_true")

    args = ap.parse_args()
    main(args)
