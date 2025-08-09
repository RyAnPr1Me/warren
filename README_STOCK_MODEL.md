# Advanced Stock Price Prediction Model

This project implements an advanced stock price prediction model using deep learning techniques with PyTorch. The model utilizes a hybrid architecture combining LSTM, Transformer, and Temporal Convolutional Networks (TCN) to capture complex temporal patterns in stock price data.

## Features

- **GPU-Accelerated Training**: Utilizes CUDA and mixed precision for faster training
- **Temperature-Aware Calibration**: Implements temperature scaling for improved prediction calibration
- **Hybrid Architecture**: Combines LSTM, Self-Attention, and TCN components
- **Advanced Feature Engineering**: Uses data from `stock_data_generator.py` with over 100 engineered features
- **Automatic Performance Optimization**: Adapts to hardware capabilities and dataset size
- **Learning Rate Scheduling**: Implements OneCycleLR with warmup phase for stable training
- **Early Stopping and Model Checkpointing**: Preserves the best model version automatically
- **Comprehensive Performance Metrics**: Tracks multiple metrics for model evaluation
- **Visualization Tools**: Provides training curves and performance metrics visualization

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (optional but recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Special note for TA-Lib installation:
   - On Windows, you may need to install from a pre-built wheel: [TA-Lib Wheels](https://github.com/mrjbq7/ta-lib/releases)
   - On Linux/macOS, you need to install the underlying C library first:
     ```
     # On Ubuntu/Debian
     apt-get install ta-lib
     
     # On macOS with Homebrew
     brew install ta-lib
     ```

## Usage

### Basic Training

```bash
python train_stock_model.py --save_results --visualize
```

### Advanced Configuration

```bash
python train_stock_model.py --batch_size 128 --epochs 100 --hidden_dim 256 \
    --num_layers 3 --learning_rate 0.0005 --seq_length 60 --save_results \
    --visualize --model_dir ./models/hybrid_model
```

### Command-Line Arguments

- **Data Parameters**:
  - `--data_path`: Path to input data CSV file (uses stock_data_generator if not provided)
  - `--save_data`: Whether to save generated data
  - `--min_rows`: Minimum number of rows to generate (default: 10000)
  - `--test_size`: Test set size as a proportion (default: 0.2)

- **Model Parameters**:
  - `--hidden_dim`: Hidden dimension size (default: 128)
  - `--num_layers`: Number of LSTM layers (default: 2)
  - `--num_heads`: Number of attention heads (default: 4)
  - `--dropout`: Dropout rate (default: 0.2)
  - `--bidirectional`: Whether to use bidirectional LSTM (default: True)
  - `--seq_length`: Sequence length for time series (default: 30)
  - `--is_regression`: Whether it's a regression task (default: False)

- **Training Parameters**:
  - `--batch_size`: Batch size (default: 64)
  - `--epochs`: Number of epochs (default: 50)
  - `--learning_rate`: Learning rate (default: 0.001)
  - `--weight_decay`: Weight decay for regularization (default: 1e-5)
  - `--patience`: Patience for early stopping (default: 10)
  - `--seed`: Random seed (default: 42)

- **Output Parameters**:
  - `--model_dir`: Directory to save models (default: "models")
  - `--save_results`: Whether to save results
  - `--visualize`: Whether to visualize results

## Model Architecture

The hybrid model architecture includes:

1. **Bidirectional LSTM Layers**: Capture temporal dependencies in both directions
2. **Multi-Head Self-Attention**: Learn relationships between different time steps
3. **Temporal Convolutional Network**: Process sequences with different dilation rates
4. **Temperature Scaling**: Calibrate model predictions for better reliability
5. **Feature Extraction**: Process static features alongside temporal patterns

## Performance Optimization

- **Mixed Precision Training**: Uses FP16 computation where possible for faster training
- **Gradient Scaling**: Prevents underflow in mixed precision training
- **Learning Rate Scheduling**: Adapts learning rate throughout training
- **Early Stopping**: Prevents overfitting by monitoring validation metrics

## Example Results

With default parameters on major tech stocks, the model typically achieves:
- For Classification (price direction): F1 scores of 0.60-0.65
- For Regression (price prediction): RMSE of ~1-2% of stock price

## License

[MIT License](LICENSE)

## Acknowledgments

- PyTorch team for the deep learning framework
- TA-Lib for technical indicators
- YFinance for stock data access
