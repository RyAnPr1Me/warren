# Stock AI System - Phase 1 Implementation

A powerful AI-driven stock analysis and prediction system built with Python.

## 🚀 Features (Phase 1)

- **Real-time Data Collection**: Fetch stock data from multiple financial APIs
- **Technical Analysis**: Calculate RSI, MACD, Bollinger Bands, and more
- **Command Line Interface**: Easy-to-use CLI for stock analysis
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

### Basic Stock Analysis
```bash
# Get comprehensive analysis for Apple
python src/cli.py analyze AAPL

# Get current price for Google
python src/cli.py price GOOGL

# Get technical analysis for Tesla
python src/cli.py technical TSLA
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `analyze` | Comprehensive stock analysis | `python src/cli.py analyze AAPL` |
| `price` | Current or real-time price | `python src/cli.py price MSFT --real-time` |
| `technical` | Technical indicators & signals | `python src/cli.py technical NVDA --support-resistance` |
| `recommend` | Buy/sell recommendation | `python src/cli.py recommend TSLA` |
| `data` | Download & save stock data | `python src/cli.py data AMZN --save` |
| `config` | Show configuration | `python src/cli.py config --validate` |

### Examples

```bash
# Analyze multiple stocks
python src/cli.py analyze AAPL
python src/cli.py analyze GOOGL
python src/cli.py analyze TSLA

# Get technical analysis with support/resistance
python src/cli.py technical MSFT --support-resistance

# Download and save historical data
python src/cli.py data SPY --period 5y --save

# Get real-time price updates
python src/cli.py price NVDA --real-time

# Check system configuration
python src/cli.py config --validate
```

## 📊 What You'll Get

### Stock Analysis Output
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

### Technical Analysis
```
📈 Technical Analysis for AAPL...

📊 Technical Indicators:
   RSI: 65.3
   MACD: 2.841
   BB Position: 0.73
   SMA 20: $172.45
   SMA 50: $168.92
   Volume Ratio: 1.2x

🎯 Signals:
   RSI: Neutral
   MACD: Buy
   BOLLINGER: Neutral
   TREND: Buy
   VOLUME: Moderate

🎯 Overall Signal: Buy
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

## 🧪 Phase 1 Capabilities

✅ **Implemented:**
- Real-time and historical data fetching
- Technical indicators (RSI, MACD, Bollinger Bands, SMA)
- Basic buy/sell signals based on technical analysis
- Command line interface
- Data caching and persistence
- Configuration management
- Logging system

🔄 **Coming in Phase 2:**
- LSTM neural network for price prediction
- Advanced AI models for buy/sell recommendations
- 60-day price forecasting
- Model training and evaluation
- Enhanced prediction accuracy

🔮 **Planned for Phase 3:**
- News sentiment analysis
- Social media sentiment integration
- Multi-modal AI analysis
- Real-time trading signals
- Web interface

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

1. **Start with popular stocks**: AAPL, GOOGL, MSFT, TSLA for reliable data
2. **Check configuration**: Run `python src/cli.py config --validate` first
3. **Use technical analysis**: Combine multiple indicators for better insights
4. **Save important data**: Use `--save` flag to keep historical data locally
5. **Monitor volatility**: High volatility stocks may need more careful analysis

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Always do your own research and consult with qualified financial professionals before making investment decisions.

## 🗺️ Roadmap

- **Phase 1** ✅: Basic data collection and technical analysis
- **Phase 2** 🔄: AI prediction models and forecasting
- **Phase 3** 🔮: Advanced AI features and web interface
- **Phase 4** 🚀: Real-time trading integration and portfolio management

## 📞 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review the configuration with `python src/cli.py config --validate`
3. Check the logs in `logs/stock_ai.log`

---

Built with ❤️ for the trading community. Happy analyzing! 📊🚀
