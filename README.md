# Warren — Advanced Stock Price Prediction AI

A high-performance, feature-rich AI pipeline for stock price prediction.  
Warren combines a **180+ feature engineering engine**, a **hybrid deep-learning model**, and a **professional interactive terminal UI** into one cohesive toolkit.

---

## ✨ Highlights

| Category | What's included |
|---|---|
| **Data Engine** | 180+ TA indicators • Parallel download • Disk caching • Multi-timeframe • Market context (SPY/VIX/Sector ETFs) • Fundamental data • Multi-horizon targets |
| **Models** | Hybrid BiLSTM + Multi-Head Attention + TCN (3 dilated layers) • Temporal Fusion Transformer (TFT) variant • Multi-task learning across 1/5/10/21-day horizons |
| **Training** | AMP mixed-precision • Gradient clipping • OneCycleLR + warm-up • Focal Loss & label smoothing • Early stopping • Auto-resume from checkpoint • Walk-forward cross-validation • Optuna hyperparameter search |
| **System-aware** | Auto-detects CPU cores, RAM, GPU VRAM → tunes `batch_size`, `hidden_dim`, `num_workers`, `pin_memory` automatically |
| **Terminal UI** | Rich live epoch table • Progress bars • Interactive wizard • Coloured metrics • Feature importance bar chart |
| **Inference** | `predict.py` — one command real-time predictions with confidence bars |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch interactive wizard (recommended for first run)
python train_stock_model.py --interactive

# 3. Or use defaults on 30 large-cap stocks
python train_stock_model.py --save_results --visualize
```

---

## 📦 Installation

```bash
# TA-Lib C library (required first)
# macOS:   brew install ta-lib
# Ubuntu:  sudo apt-get install libta-lib-dev
# Windows: download from https://www.ta-lib.org

pip install -r requirements.txt
```

**Dependencies:** `torch`, `numpy`, `pandas`, `yfinance`, `ta-lib`, `scikit-learn`, `matplotlib`, `seaborn`, `rich`, `optuna`, `scipy`

---

## 🎮 Training Script — `train_stock_model.py`

### Interactive Mode (wizard)

```bash
python train_stock_model.py --interactive
```

Steps through symbol selection, date range, architecture choice, and hyperparameters with prompts, defaults, and confirmation.

### CLI Examples

```bash
# Load a JSON config file
python train_stock_model.py --config training_config.json

# Use extended 100-stock universe with market context caching
python train_stock_model.py --use_extended_symbols \
    --cache_dir .cache --save_results --visualize

# Temporal Fusion Transformer with multi-task learning
python train_stock_model.py --model_type tft --multi_task --epochs 100

# Walk-forward cross-validation (5 folds)
python train_stock_model.py --walk_forward --n_splits 5

# Optuna hyperparameter search (50 trials) + final train
python train_stock_model.py --tune --n_trials 50 --save_results

# Regression mode: predict actual return instead of direction
python train_stock_model.py --is_regression --hidden_dim 256 --num_layers 3

# Custom symbols + larger model
python train_stock_model.py --symbols AAPL,MSFT,NVDA,TSLA,META \
    --hidden_dim 256 --num_heads 8 --dropout 0.15 \
    --batch_size 128 --epochs 100 --patience 15

# Focal loss + gradient clipping for better accuracy
python train_stock_model.py --focal_loss --grad_clip 1.0

# Label smoothing (helps generalisation)
python train_stock_model.py --label_smoothing 0.05

# Resume an interrupted training run
python train_stock_model.py --resume models/training_checkpoint.pth
# (or simply re-run — Warren auto-detects the checkpoint)
python train_stock_model.py --model_dir models
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--interactive` | off | Launch configuration wizard |
| `--config` | None | Path to JSON config file |
| `--model_type` | `hybrid` | `hybrid` (BiLSTM+Attn+TCN) or `tft` |
| `--multi_task` | off | Predict all 4 forecast horizons simultaneously |
| `--walk_forward` | off | Walk-forward cross-validation |
| `--tune` | off | Optuna hyperparameter optimisation |
| `--n_trials` | 30 | Optuna trial count |
| `--n_splits` | 5 | Walk-forward folds |
| `--cache_dir` | None | Enable per-symbol data caching |
| `--n_jobs` | 4 | Parallel download workers |
| `--include_fundamentals` | off | Fetch P/E, beta, market cap, etc. |
| `--silent` | off | Suppress rich terminal UI (for scripts) |
| **Accuracy** | | |
| `--focal_loss` | off | Focal Loss instead of BCE (better for imbalanced targets) |
| `--focal_alpha` | 0.25 | Focal Loss positive-class weight |
| `--focal_gamma` | 2.0 | Focal Loss focusing exponent |
| `--label_smoothing` | 0.0 | Label smoothing ε (e.g. 0.05 helps generalisation) |
| `--grad_clip` | 1.0 | Max gradient norm (0 = disabled) |
| **Resume** | | |
| `--resume` | auto | Path to `training_checkpoint.pth`; auto-detected if omitted |

---

## 🔮 Inference — `predict.py`

```bash
# Predict next-day direction for AAPL
python predict.py --symbol AAPL --model_dir models

# Multi-symbol, 5-day horizon, save CSV
python predict.py --symbol AAPL MSFT NVDA TSLA --horizon 5 --output preds.csv

# Read symbols from file, show feature importance
python predict.py --symbols_file watchlist.txt --show_features

# All options
python predict.py --symbol GOOGL \
    --model_dir models \
    --horizon 10 \
    --lookback_days 500 \
    --cache_dir .cache \
    --top_features 20 \
    --output predictions.csv
```

**Output columns:** `Symbol`, `Date`, `Close`, `Horizon_Days`, `Pred_Prob`, `Direction`, `Confidence`

---

## 📊 Feature Engineering — `stock_data_generator.py`

### 180+ Features across 10 categories

| Category | Examples |
|---|---|
| **Core** | Returns, log-returns, price ranges, gaps, volume changes |
| **Moving Averages** | SMA/EMA (5,10,20,50,100,200), DEMA, TEMA, KAMA, Hull MA, WMA |
| **Momentum** | RSI (7/14/21), MACD, Stochastic K/D, StochRSI, CCI, Williams %R, ROC, Aroon, ULTOSC, PPO, APO, Fisher Transform, Elder Ray |
| **Volatility** | ATR, Bollinger Bands, Keltner Channels, Donchian Channels, Parkinson vol, Yang-Zhang vol, realized vol (5/10/20/30/60d) |
| **Trend** | ADX, Parabolic SAR, Ichimoku Cloud (Tenkan/Kijun/SpanA/SpanB/Chikou), 52-week high/low |
| **Volume** | OBV, MFI, Chaikin Money Flow, VWAP, ADOSC, volume ratio, volume surge |
| **Multi-timeframe** | Weekly RSI, weekly MA 4/12, weekly momentum, monthly MA 3/12, monthly range |
| **Market context** | SPY/QQQ/IWM correlation (20d), beta to SPY (60d), alpha vs SPY, VIX level, sector-ETF correlation |
| **Regime** | Hurst exponent (63d/126d), mean-reversion flag, vol regime, price Z-score |
| **Seasonality** | Day-of-week dummies, month dummies, quarter, is_month_end, is_quarter_end |

### Multi-horizon targets

| Target column | Description |
|---|---|
| `Target_Return_1d` / `Target_Direction_1d` | Next 1 trading day |
| `Target_Return_5d` / `Target_Direction_5d` | Next 5 trading days (≈1 week) |
| `Target_Return_10d` / `Target_Direction_10d` | Next 10 trading days (≈2 weeks) |
| `Target_Return_21d` / `Target_Direction_21d` | Next 21 trading days (≈1 month) |

### Python API

```python
from stock_data_generator import (
    get_feature_engineered_stock_data,
    normalize_features,
    split_train_test,
    walk_forward_splits,
)

# Generate full featured dataset with caching and market context
data = get_feature_engineered_stock_data(
    symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
    start_date="2018-01-01",
    cache_dir=".cache",          # speeds up repeated runs
    n_jobs=4,                    # parallel downloads
    include_market_context=True, # SPY, VIX, sector ETFs
    include_fundamentals=True,   # P/E, beta, market cap
    multi_horizon_targets=True,  # all 4 horizons
)

# Robust normalisation (median/IQR)
norm = normalize_features(data, scaler_type="robust")

# Time-based train/test split
X_train, X_test, y_train, y_test = split_train_test(norm, target_horizon=1)

# Walk-forward splits for proper CV
for X_tr, X_te, y_tr, y_te in walk_forward_splits(norm, n_splits=5):
    # train and evaluate each fold...
    pass
```

---

## 🧠 Model Architectures

### `hybrid` (default) — BiLSTM + Multi-Head Attention + TCN

```
Input → BiLSTM (bidirectional, N layers)
      → Multi-Head Self-Attention
      → LayerNorm
      → 3× Dilated TCN Blocks (dilation 1, 2, 4)
      → Static feature projection (last timestep)
      → Fused representation
      → FC → GELU → Dropout
      → Output head(s)  [single or multi-task]
      → Temperature scaling
```

### `tft` — Temporal Fusion Transformer

```
Input → Variable Selection Network (per-feature GRN + softmax weights)
      → LSTM encoder
      → Gated Residual Network
      → Interpretable Multi-Head Attention
      → Post-attention GRN + residual
      → Feed-forward GRN
      → LayerNorm
      → Output head(s)  [single or multi-task]
      → Temperature scaling
```

---

## 📁 Project Structure

```
warren/
├── stock_data_generator.py   # Feature engineering pipeline
├── train_stock_model.py      # Model trainer with rich terminal UI
├── predict.py                # Real-time inference script
├── training_config.json      # Example JSON config for training
├── requirements.txt
└── README.md
```

---

## 🔧 JSON Config

`training_config.json` is fully supported via `--config`:

```json
{
  "stocks": ["AAPL", "MSFT", "NVDA"],
  "data_settings": {
    "start_date": "2018-01-01",
    "min_rows": 20000
  },
  "model_settings": {
    "hidden_dim": 256,
    "num_layers": 3,
    "num_heads": 8,
    "dropout": 0.15,
    "bidirectional": true
  },
  "training_settings": {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.0005,
    "patience": 15
  }
}
```

---

## ⚠️ Disclaimer

This software is provided **for educational and research purposes only**.  
Stock price predictions are inherently uncertain. **Do not use these predictions for real trading decisions.** Past model performance does not guarantee future results. Always consult a qualified financial advisor.

---

## 📄 License

MIT License — see `LICENSE` for details.
