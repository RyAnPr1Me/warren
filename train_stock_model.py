"""
Advanced Stock Price Prediction Model Trainer

This script implements temperature-aware training with GPU acceleration and mixed precision
for stock prediction models. It uses a hybrid architecture combining LSTM, Transformer, and
TCN components for optimal performance on time series financial data.

Features:
- Mixed precision training for faster computation
- Learning rate scheduling with warmup
- Temperature scaling for calibrated predictions
- Automatic GPU/CPU detection and optimization
- Multi-head attention for capturing complex temporal dependencies
- Feature importance analysis
- Early stopping with model checkpointing
- Performance metrics visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import argparse
import json
from pathlib import Path
import time

# Extended symbols list
SYMBOLS_EXTENDED = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','INTC','AMD','TSLA','ORCL','CSCO','IBM','ADBE','CRM','NFLX','PYPL','QCOM','AVGO','TXN','MU',
    'JPM','BAC','WFC','GS','MS','C','BLK','AXP','V','MA','PNC','SCHW','CME','CB','MMC','TFC','USB','ALL','AIG','BK',
    'JNJ','PFE','MRK','ABBV','LLY','ABT','UNH','TMO','DHR','BMY','AMGN','MDT','ISRG','GILD','CVS','VRTX','ZTS','REGN','HUM','BIIB',
    'PG','KO','PEP','WMT','COST','MCD','SBUX','NKE','DIS','HD','LOW','TGT','MDLZ','CL','EL','ROST','TJX','YUM','MAR','CMG',
    'GE','HON','MMM','CAT','DE','BA','LMT','RTX','UPS','FDX','UNP','CSX','ETN','EMR','ITW','PH','GD','NSC','CARR','PCAR',
    'XOM','CVX','COP','EOG','PSX','PXD','VLO','SLB','MPC','OXY','T','VZ','TMUS','CMCSA','NEE','DUK','SO','D','AEP','AMT','PLD','CCI','SPG','EQIX'
]

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

# Metrics and utilities
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Local imports
from stock_data_generator import get_feature_engineered_stock_data, normalize_features, split_train_test

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

# Custom dataset for time series data
class StockDataset(Dataset):
    def __init__(self, features, targets, seq_length=30):
        """
        Custom dataset for stock time series data
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature matrix
        targets : numpy.ndarray
            Target values
        seq_length : int
            Length of sequence for each sample
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
        
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx+self.seq_length]
        # Get target (next day's direction)
        y = self.targets[idx+self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Transform input through query, key, and value projections
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention weights to values
        out = torch.matmul(attention, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape and combine heads
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, hidden_dim)
        
        # Final linear layer
        out = self.fc_out(out)
        
        return out

# Temporal Convolutional Network Block
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        # Use padding that preserves sequence length so residual add works
        same_padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=same_padding, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        # If any minor length mismatch due to padding/dilation, center crop
        if out.size(-1) != residual.size(-1):
            min_len = min(out.size(-1), residual.size(-1))
            out = out[..., -min_len:]
            residual = residual[..., -min_len:]
        out = out + residual
        out = out.transpose(1, 2)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)
        return out

# Hybrid Model: LSTM + Transformer + TCN
class StockPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=4, 
                 dropout=0.2, bidirectional=True, num_classes=1):
        super(StockPredictionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust hidden dimension if using bidirectional LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Self-Attention mechanism
        self.attention = SelfAttention(lstm_output_dim, num_heads=num_heads, dropout=dropout)
        
        # TCN layers
        self.tcn1 = TCNBlock(lstm_output_dim, lstm_output_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock(lstm_output_dim, lstm_output_dim, kernel_size=3, dilation=2, dropout=dropout)
        
        # Feature extraction layers
        self.feat_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temperature scaling parameter (for calibration)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Fully connected output layers
        self.fc1 = nn.Linear(lstm_output_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(lstm_output_dim)
        self.layer_norm2 = nn.LayerNorm(lstm_output_dim + hidden_dim)
        
    def forward(self, x, x_static=None, return_attention=False):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        attention_out = self.attention(lstm_out)
        attention_out = self.layer_norm1(attention_out)
        
        # Pass through TCN (need to transpose for Conv1d)
        tcn_input = attention_out.transpose(1, 2)  # (batch_size, lstm_output_dim, seq_len)
        tcn_out = self.tcn1(tcn_input)
        tcn_out = self.tcn2(tcn_out)
        tcn_out = tcn_out.transpose(1, 2)  # (batch_size, seq_len, lstm_output_dim)
        
        # Get last sequence element
        final_lstm_out = tcn_out[:, -1, :]
        
        # Process static features if provided
        if x_static is not None:
            static_features = self.feat_extractor(x_static)
            # Concatenate with LSTM output
            combined = torch.cat((final_lstm_out, static_features), dim=1)
        else:
            # Use the last time step data as static features
            static_features = self.feat_extractor(x[:, -1, :])
            combined = torch.cat((final_lstm_out, static_features), dim=1)
        
        # Apply layer normalization
        combined = self.layer_norm2(combined)
        
        # Final output layers
        out = self.fc1(combined)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Apply temperature scaling for better calibration
        out = out / self.temperature
        
        if self.num_classes == 1:
            # For regression or binary classification
            return out
        else:
            # For multi-class classification
            return torch.softmax(out, dim=1)

# Train function with mixed precision
def train_epoch(model, dataloader, optimizer, criterion, device, scaler, scheduler=None):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        use_amp = (device.type == 'cuda')
        if use_amp:
            with autocast():
                output = model(data)
                loss = criterion(output.view(-1), target.view(-1))
        else:
            output = model(data)
            loss = criterion(output.view(-1), target.view(-1))
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
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, criterion, device, is_regression=False):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Collect outputs and targets for metrics
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Concatenate batches
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    metrics = {}
    if is_regression:
        # Regression metrics
        metrics['mse'] = mean_squared_error(all_targets, all_outputs)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(all_targets, all_outputs)
    else:
        # Classification metrics (threshold at 0.5 for binary classification)
        all_outputs = (all_outputs > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(all_targets, all_outputs)
        metrics['precision'] = precision_score(all_targets, all_outputs, average='binary', zero_division=0)
        metrics['recall'] = recall_score(all_targets, all_outputs, average='binary', zero_division=0)
        metrics['f1'] = f1_score(all_targets, all_outputs, average='binary', zero_division=0)
    
    # Add validation loss
    metrics['val_loss'] = total_loss / len(dataloader)
    
    return metrics

# Helper function to prepare data
def prepare_data(X_train, X_test, y_train, y_test, seq_length=30, batch_size=64):
    # Create datasets
    train_dataset = StockDataset(X_train.values, y_train.values, seq_length=seq_length)
    test_dataset = StockDataset(X_test.values, y_test.values, seq_length=seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Main training function
def train_model(X_train, X_test, y_train, y_test, config):
    """
    Train the stock prediction model with mixed precision and temperature scaling
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : pandas.DataFrame
        Training and test data
    config : dict
        Configuration parameters
    
    Returns:
    --------
    dict
        Training results including model, metrics, and training history
    """
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    # Determine device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get data dimensions
    input_dim = X_train.shape[1]
    
    # Prepare data loaders
    train_loader, test_loader = prepare_data(
        X_train, X_test, y_train, y_test, 
        seq_length=config['seq_length'],
        batch_size=config['batch_size']
    )
    
    # Create model
    model = StockPredictionModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        num_classes=1  # Binary classification task
    )
    model = model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss() if not config['is_regression'] else nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': []
    }
    
    # Best model tracking
    best_val_metric = float('inf') if config['is_regression'] else 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(config['model_dir'], 'best_model.pth')
    
    # Create model directory if it doesn't exist
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler
        )
        
        # Validate
        val_metrics = validate(
            model, test_loader, criterion, device, is_regression=config['is_regression']
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['metrics'].append(val_metrics)
        
        # Log progress
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_metrics['val_loss']:.4f}, "
                   f"{'RMSE' if config['is_regression'] else 'F1'}: "
                   f"{val_metrics['rmse'] if config['is_regression'] else val_metrics['f1']:.4f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Check if this is the best model
        monitor_metric = val_metrics['rmse'] if config['is_regression'] else -val_metrics['f1']
        if (config['is_regression'] and monitor_metric < best_val_metric) or \
           (not config['is_regression'] and monitor_metric > best_val_metric):
            best_val_metric = monitor_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            
            logger.info(f"New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    logger.info(f"Best model at epoch {best_epoch+1} with "
               f"{'RMSE' if config['is_regression'] else 'F1'}: "
               f"{best_val_metric if config['is_regression'] else -best_val_metric:.4f}")
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_metrics = validate(
        model, test_loader, criterion, device, is_regression=config['is_regression']
    )
    logger.info(f"Final model metrics: {final_metrics}")
    
    # Return results
    results = {
        'model': model,
        'history': history,
        'config': config,
        'final_metrics': final_metrics,
        'best_epoch': best_epoch,
        'training_time': total_time
    }
    
    return results

# Function to visualize results
def visualize_results(results, save_dir=None):
    """
    Visualize training results
    
    Parameters:
    -----------
    results : dict
        Training results from train_model function
    save_dir : str, optional
        Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    history = results['history']
    config = results['config']
    is_regression = config['is_regression']
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot performance metric
    plt.subplot(1, 2, 2)
    metric_name = 'rmse' if is_regression else 'f1'
    metric_values = [m[metric_name] for m in history['metrics']]
    plt.plot(metric_values, label=metric_name.upper())
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.upper())
    plt.title(f'{metric_name.upper()} over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.show()
    
    # Plot additional metrics
    if not is_regression:
        plt.figure(figsize=(12, 5))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            values = [m[metric] for m in history['metrics']]
            plt.plot(values, label=metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} over Epochs')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'classification_metrics.png'), dpi=300)
        plt.show()

# Feature importance analysis
def analyze_feature_importance(model, feature_names, device):
    """Gradient-based simple feature importance (not currently used)."""
    model.eval()
    input_dim = len(feature_names)
    dummy_input = torch.ones((1, 1, input_dim), requires_grad=True).to(device)
    output = model(dummy_input)
    output.mean().backward()
    if dummy_input.grad is not None:
        importance = dummy_input.grad.abs().view(-1).cpu().numpy()
    else:
        importance = np.zeros(input_dim)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    return importance_df

# Main function
def main(args):
    logger.info("Loading or generating stock data...")
    symbols = None
    if args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from file")
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
        logger.info(f"Using {len(symbols)} symbols from --symbols argument")
    elif args.use_extended_symbols:
        symbols = SYMBOLS_EXTENDED
        logger.info(f"Using extended symbols list ({len(symbols)})")
    # Data load/generation
    if args.data_path and os.path.exists(args.data_path):
        data = pd.read_csv(args.data_path)
        logger.info(f"Loaded data from {args.data_path} with {len(data)} rows")
    else:
        data = get_feature_engineered_stock_data(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            min_rows=args.min_rows
        )
        logger.info(f"Generated new data with {len(data)} rows")
        if args.save_data:
            out_path = args.data_path or 'stock_data_features.csv'
            data.to_csv(out_path, index=False)
            logger.info(f"Saved data to {out_path}")
    
    # Define model configuration
    config = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional,
        'seq_length': args.seq_length,
        'patience': args.patience,
        'is_regression': args.is_regression,
        'model_dir': args.model_dir
    }
    
    # Prepare data
    logger.info("Preparing data for training...")
    
    # Normalize features
    normalized_data = normalize_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_test(
        normalized_data, test_size=args.test_size, time_based=True
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    logger.info("Starting model training...")
    results = train_model(X_train, X_test, y_train, y_test, config)
    
    # Save results
    if args.save_results:
        # Save model and configuration
        model_path = os.path.join(args.model_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'config': config,
            'final_metrics': results['final_metrics']
        }, model_path)
        
        # Save metrics history
        history_path = os.path.join(args.model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_loss': results['history']['train_loss'],
                'val_loss': results['history']['val_loss'],
                'metrics': results['history']['metrics']
            }, f, indent=4)
        
        logger.info(f"Saved model and results to {args.model_dir}")
    
    # Visualize results
    if args.visualize:
        logger.info("Visualizing results...")
        visualize_results(results, save_dir=args.model_dir)
    
    logger.info("Done!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Stock Price Prediction Model Trainer")
    # Data parameters
    parser.add_argument("--data_path", type=str, default=None, help="Path to input data CSV file")
    parser.add_argument("--save_data", action="store_true", help="Whether to save generated data")
    parser.add_argument("--min_rows", type=int, default=10000, help="Minimum number of rows to generate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (proportion)")
    parser.add_argument("--use_extended_symbols", action="store_true", help="Use extended predefined symbols list")
    parser.add_argument("--symbols_file", type=str, default=None, help="Path to file containing symbols (one per line)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols list overrides others")
    parser.add_argument("--start_date", type=str, default=None, help="Historical data start date YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default=None, help="Historical data end date YYYY-MM-DD")
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--bidirectional", type=bool, default=True, help="Whether to use bidirectional LSTM")
    parser.add_argument("--seq_length", type=int, default=30, help="Sequence length for time series")
    parser.add_argument("--is_regression", action="store_true", help="Whether it's a regression task")
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Output parameters
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--save_results", action="store_true", help="Whether to save results")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize results")
    args = parser.parse_args()
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU for training.")
    # Load/generate symbols before calling main
    # Pass through via global for main modifications below
    _cli_args = args
    main(args)
