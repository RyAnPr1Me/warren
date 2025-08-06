# Stock AI System - Phase 2 Implementation

A powerful AI-driven stock analysis and prediction system with advanced LSTM neural networks.

## 🚀 Features (Phase 2)

- **🧠 AI Price Prediction**: Advanced LSTM neural networks for 60-day price forecasting
- **🎯 AI-Powered Recommendations**: Smart buy/sell recommendations with confidence scoring
- **📊 Multi-Feature Analysis**: Technical indicators, volume, volatility as model inputs
- **🔮 Confidence Intervals**: Prediction uncertainty quantification
- **📈 Model Performance Tracking**: Comprehensive metrics and quality assessment
- **Real-time Data Collection**: Fetch stock data from multiple financial APIs
- **Technical Analysis**: Calculate RSI, MACD, Bollinger Bands, and more
- **Command Line Interface**: Easy-to-use CLI for both basic and AI analysis
- **Data Caching**: Intelligent caching for improved performance
- **Comprehensive Configuration**: Flexible configuration system

## 📋 Prerequisites

- Python 3.9 or higher
- pip package manager

## 🛠️ Installation

1. **Clone or download the project**
```bash
# If you have git
git clone <repository-url>
cd stock-ai

# Or extract downloaded files to stock-ai directory
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys** (Optional but recommended)
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your API keys
# Get free API keys from:
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key
# - Financial Modeling Prep: https://financialmodelingprep.com/developer/docs
```

## 🎯 Quick Start

### Phase 1: Basic Analysis
```bash
# Get comprehensive analysis for Apple
python src/cli.py analyze AAPL

# Get current price for Google
python src/cli.py price GOOGL

# Get technical analysis for Tesla
python src/cli.py technical TSLA
```

### Phase 2: AI-Powered Analysis
```bash
# Train AI model for Apple (first time setup)
python src/cli.py train AAPL

# Get AI price prediction for 60 days
python src/cli.py predict AAPL --days 60

# Get AI-powered investment recommendation
python src/cli.py ai-recommend AAPL --detailed

# Check AI model performance
python src/cli.py model-info AAPL
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| **Basic Analysis** |
| `analyze` | Comprehensive stock analysis | `python src/cli.py analyze AAPL` |
| `price` | Current or real-time price | `python src/cli.py price MSFT --real-time` |
| `technical` | Technical indicators & signals | `python src/cli.py technical NVDA --support-resistance` |
| `recommend` | Basic buy/sell recommendation | `python src/cli.py recommend TSLA` |
| **AI-Powered Features** |
| `train` | Train LSTM prediction model | `python src/cli.py train AAPL --period 3y` |
| `predict` | AI price prediction | `python src/cli.py predict AAPL --days 30` |
| `ai-recommend` | AI investment recommendation | `python src/cli.py ai-recommend TSLA --detailed` |
| `model-info` | Model performance metrics | `python src/cli.py model-info AAPL` |
| **Data Management** |
| `data` | Download & save stock data | `python src/cli.py data AMZN --save` |
| `config` | Show configuration | `python src/cli.py config --validate` |

### Examples

```bash
# Phase 1: Basic Analysis
python src/cli.py analyze AAPL
python src/cli.py technical MSFT --support-resistance
python src/cli.py price NVDA --real-time

# Phase 2: AI-Powered Analysis
# Step 1: Train AI model (do this once)
python src/cli.py train AAPL --period 3y

# Step 2: Get AI predictions
python src/cli.py predict AAPL --days 60 --save-report
python src/cli.py ai-recommend AAPL --detailed

# Step 3: Monitor model performance
python src/cli.py model-info AAPL

# Train models for multiple stocks
python src/cli.py train GOOGL
python src/cli.py train TSLA
python src/cli.py train MSFT

# Compare AI vs technical analysis
python src/cli.py recommend AAPL        # Technical analysis
python src/cli.py ai-recommend AAPL     # AI-powered analysis

# Download and save historical data
python src/cli.py data SPY --period 5y --save

# Check system configuration
python src/cli.py config --validate
```

## 📊 What You'll Get

### Basic Stock Analysis Output
```
📊 Analyzing AAPL...

🏢 AAPL - Apple Inc.
💰 Current Price: $175.84
📈 Market Cap: $2.8T
📊 P/E Ratio: 28.5
📉 52 Week Range: $124.17 - $198.23

📊 Technical Indicators:
   SMA 20: $172.45
   SMA 50: $168.92
   RSI: 65.3
   Volatility: 22.1%
```

### AI-Powered Prediction Output
```
� AI Price Prediction for AAPL
� Forecasting 60 days ahead...

📊 AI Prediction Results:
   Current Price: $175.84
   Predicted Price (60 days): $192.45
   Expected Change: +9.45%
   Confidence: Medium-High
   95% Confidence Range: $185.20 - $199.70

🎯 Model Performance:
   R² Score: 0.742
   RMSE: $8.34
   Features Used: 13
```

### AI Investment Recommendation
```
🤖 AI Investment Recommendation for AAPL

🎯 AI Recommendation: Buy
💡 Reasoning: Model predicts 9.5% gain with Medium-High confidence
📈 Expected Return: +9.45%
🎲 Risk Level: Medium
⚡ Confidence: Medium-High
📅 Prediction Horizon: 60 days

📊 Detailed Analysis:
   Current Price: $175.84
   Target Price: $192.45

🧠 AI Model Details:
   Model R² Score: 0.742
   Model RMSE: $8.34
   Training Date: 2025-08-05
   Features Used: 13
```

## 🔧 Configuration

The system uses a flexible configuration system. You can:

1. **Use default settings** - Works out of the box with yfinance
2. **Add API keys** - For enhanced data and higher rate limits
3. **Customize settings** - Modify `src/config.py` for advanced configuration

### Environment Variables (.env file)
```bash
# API Keys (optional but recommended)
ALPHA_VANTAGE_API_KEY=your_key_here
FMP_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Caching
CACHE_EXPIRY_MINUTES=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/stock_ai.log
```

## 📁 Project Structure

```
stock-ai/
├── src/
│   ├── data/
│   │   └── collector.py      # Data collection from APIs
│   ├── analysis/
│   │   └── technical.py      # Technical analysis engine
│   ├── models/
│   │   └── lstm_predictor.py # LSTM predictor (Phase 2)
│   ├── utils/
│   │   └── helpers.py        # Utility functions
│   ├── config.py             # Configuration management
│   ├── cli.py               # Command line interface
│   └── __init__.py
├── data/                    # Data storage
├── logs/                    # Log files
├── requirements.txt         # Dependencies
├── pyproject.toml          # Project configuration
├── .env.example            # Environment template
└── README.md
```

## 🧪 Phase 2 Capabilities

✅ **Implemented:**
- **🧠 LSTM Neural Networks**: Advanced multi-layer LSTM architecture for price prediction
- **📊 Multi-Feature Input**: Uses OHLCV data + 8 technical indicators as model features
- **🎯 60-Day Forecasting**: Accurate price predictions up to 60 days ahead
- **🔮 Confidence Intervals**: Statistical uncertainty quantification for predictions
- **⚡ Smart Recommendations**: AI-powered buy/sell signals with confidence scoring
- **📈 Model Performance Tracking**: R², RMSE, MAE metrics with quality assessment
- **🏗️ Enhanced Architecture**: Batch normalization, dropout, learning rate scheduling
- **💾 Model Persistence**: Save/load trained models with full state preservation
- **📋 Comprehensive Reports**: Detailed prediction reports with visualizations
- Real-time and historical data fetching
- Technical indicators (RSI, MACD, Bollinger Bands, SMA)
- Command line interface with AI commands
- Data caching and persistence
- Configuration management
- Logging system

🔄 **Coming in Phase 3:**
- News sentiment analysis integration
- Social media sentiment analysis
- Multi-modal AI combining news + technical + sentiment
- Real-time trading signals
- Portfolio optimization
- Web interface

🔮 **Planned for Phase 4:**
- Real-time trading integration
- Portfolio management
- Risk assessment tools
- Backtesting framework
- Advanced visualization dashboard

## 🚨 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're in the project directory
cd stock-ai

# Run commands from the project root
python src/cli.py analyze AAPL
```

**No Data Found:**
```bash
# Check if symbol is valid
python src/cli.py analyze INVALID_SYMBOL

# Try a well-known symbol
python src/cli.py analyze AAPL
```

**API Rate Limits:**
```bash
# Add API keys to .env file for higher limits
# Or wait a few minutes between requests
```

### Getting Help
```bash
# Show all available commands
python src/cli.py --help

# Get help for specific command
python src/cli.py analyze --help
```

## 📈 Usage Tips

### Phase 1: Getting Started
1. **Start with popular stocks**: AAPL, GOOGL, MSFT, TSLA for reliable data
2. **Check configuration**: Run `python src/cli.py config --validate` first
3. **Use technical analysis**: Combine multiple indicators for better insights
4. **Save important data**: Use `--save` flag to keep historical data locally

### Phase 2: AI-Powered Analysis
1. **Train models first**: Always run `train` command before `predict` or `ai-recommend`
2. **Use 3+ years of data**: `--period 3y` or `--period 5y` for better model performance
3. **Check model quality**: Use `model-info` to verify R² score > 0.6 for reliable predictions
4. **Compare with technical analysis**: Use both `recommend` and `ai-recommend` for validation
5. **Monitor confidence**: Higher confidence scores indicate more reliable predictions
6. **Save prediction reports**: Use `--save-report` for detailed analysis and charts

### Model Training Best Practices
- **Use sufficient data**: Minimum 2 years, recommended 3-5 years for training
- **Retrain periodically**: Markets change, retrain models monthly for best results
- **Quality threshold**: R² score > 0.6 indicates good model performance
- **Feature importance**: Models use 13 features including technical indicators

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Always do your own research and consult with qualified financial professionals before making investment decisions.

## 🗺️ Roadmap

- **Phase 1** ✅: Basic data collection and technical analysis
- **Phase 2** ✅: AI prediction models, LSTM networks, and 60-day forecasting
- **Phase 3** �: Advanced AI features with sentiment analysis and multi-modal approach
- **Phase 4** 🚀: Real-time trading integration and portfolio management

## 📞 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review the configuration with `python src/cli.py config --validate`
3. Check the logs in `logs/stock_ai.log`

---

Built with ❤️ for the trading community. Happy analyzing! 📊🚀
