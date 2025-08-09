# Stock Data Generator for AI Models

This project generates feature-engineered stock data that can be used for training AI models for stock price prediction.

## Features

The stock data generator includes:

- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price patterns and candlestick patterns
- Volatility measures
- Moving averages and their derivatives
- Momentum indicators
- Volume-based indicators
- Trend analysis features
- Seasonality features (day of week, month)
- Lagged variables for time series analysis
- Market regime indicators

## Requirements

To install the required dependencies:

```
pip install -r requirements.txt
```

Note: TA-Lib might require additional installation steps depending on your platform. 
See [TA-Lib installation instructions](https://github.com/mrjbq7/ta-lib#dependencies) for details.

## Usage

### Basic Usage

```python
from stock_data_generator import get_feature_engineered_stock_data

# Generate ~10,000 rows of feature-engineered stock data
stock_data = get_feature_engineered_stock_data()

# Save to CSV
stock_data.to_csv('stock_data_features.csv', index=False)
```

### Advanced Usage

```python
from stock_data_generator import (
    get_feature_engineered_stock_data, 
    normalize_features, 
    split_train_test
)

# Generate data with custom parameters
stock_data = get_feature_engineered_stock_data(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    start_date='2018-01-01',
    end_date='2023-01-01',
    min_rows=5000
)

# Normalize the features
normalized_data = normalize_features(stock_data)

# Split into training and test sets
X_train, X_test, y_train, y_test = split_train_test(normalized_data, test_size=0.2)
```

## Data Description

The generated dataset includes the following categories of features:

1. **Basic Price Data**
   - Open, High, Low, Close prices
   - Trading volume
   - Returns and log returns

2. **Technical Indicators**
   - Moving Averages (Simple and Exponential)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Stochastic Oscillator
   - CCI (Commodity Channel Index)
   - ADX (Average Directional Index)
   - ATR (Average True Range)

3. **Volume Indicators**
   - OBV (On-Balance Volume)
   - Volume changes and trends

4. **Volatility Measures**
   - Historical volatility over different timeframes
   - Price range and relative price range

5. **Trend Indicators**
   - Price relative to moving averages
   - Moving average slopes
   - Bull/bear market indicators

6. **Pattern Recognition**
   - Several candlestick patterns (Doji, Hammer, Engulfing, etc.)

7. **Seasonality Features**
   - Day of week
   - Month
   - Year

8. **Lagged Features**
   - Past returns and volume changes over various timeframes
   - Rolling statistics (mean, std, min, max)

## Target Variables

The dataset includes two types of target variables for supervised learning:

1. `Target_Next_Day_Return`: The actual return for the next trading day
2. `Target_Next_Day_Direction`: Binary indicator (1 if next day's return is positive, 0 otherwise)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
