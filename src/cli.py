"""
Command Line Interface for Stock AI System
Simple CLI for Phase 1 functionality
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.data.collector import data_collector
from src.analysis.technical import technical_analyzer

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class StockAICLI:
    """Command line interface for Stock AI system"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI"""
        parser = argparse.ArgumentParser(
            description="Stock AI - AI-powered stock analysis and prediction system",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python cli.py analyze AAPL                    # Get basic analysis for Apple
  python cli.py price GOOGL                     # Get current price for Google
  python cli.py recommend TSLA                  # Get buy/sell recommendation for Tesla
  python cli.py technical MSFT                  # Get technical analysis for Microsoft
  python cli.py data AMZN --save               # Download and save Amazon data
  python cli.py train NVDA                     # Train prediction model for NVIDIA
  python cli.py predict SPY --days 30          # Predict SPY price for 30 days
            """
        )
        
        # Main command
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Get comprehensive stock analysis')
        analyze_parser.add_argument('symbol', help='Stock ticker symbol (e.g., AAPL)')
        
        # Price command
        price_parser = subparsers.add_parser('price', help='Get current stock price')
        price_parser.add_argument('symbol', help='Stock ticker symbol')
        price_parser.add_argument('--real-time', action='store_true', help='Get real-time price')
        
        # Recommendation command
        recommend_parser = subparsers.add_parser('recommend', help='Get buy/sell recommendation')
        recommend_parser.add_argument('symbol', help='Stock ticker symbol')
        
        # Technical analysis command
        tech_parser = subparsers.add_parser('technical', help='Get technical analysis')
        tech_parser.add_argument('symbol', help='Stock ticker symbol')
        tech_parser.add_argument('--support-resistance', action='store_true', help='Include support/resistance levels')
        
        # Data command
        data_parser = subparsers.add_parser('data', help='Download stock data')
        data_parser.add_argument('symbol', help='Stock ticker symbol')
        data_parser.add_argument('--period', default='2y', help='Time period (1y, 2y, 5y, max)')
        data_parser.add_argument('--save', action='store_true', help='Save data to CSV file')
        
        # Train command (placeholder for Phase 1)
        train_parser = subparsers.add_parser('train', help='Train prediction model')
        train_parser.add_argument('symbol', help='Stock ticker symbol')
        train_parser.add_argument('--period', default='2y', help='Training data period')
        
        # Predict command (placeholder for Phase 1)
        predict_parser = subparsers.add_parser('predict', help='Predict stock price')
        predict_parser.add_argument('symbol', help='Stock ticker symbol')
        predict_parser.add_argument('--days', type=int, default=60, help='Days to predict ahead')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Show configuration')
        config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
        
        return parser
    
    def run(self, args: Optional[list] = None):
        """Run the CLI with given arguments"""
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        try:
            # Route to appropriate handler
            if parsed_args.command == 'analyze':
                self._handle_analyze(parsed_args)
            elif parsed_args.command == 'price':
                self._handle_price(parsed_args)
            elif parsed_args.command == 'recommend':
                self._handle_recommend(parsed_args)
            elif parsed_args.command == 'technical':
                self._handle_technical(parsed_args)
            elif parsed_args.command == 'data':
                self._handle_data(parsed_args)
            elif parsed_args.command == 'train':
                self._handle_train(parsed_args)
            elif parsed_args.command == 'predict':
                self._handle_predict(parsed_args)
            elif parsed_args.command == 'config':
                self._handle_config(parsed_args)
                
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    def _handle_analyze(self, args):
        """Handle analyze command"""
        symbol = args.symbol.upper()
        print(f"\n📊 Analyzing {symbol}...")
        
        # Get basic stock data
        stock_data = data_collector.get_stock_data(symbol)
        current_price = stock_data.prices['Close'].iloc[-1]
        
        # Get market data with technical indicators
        market_data = data_collector.get_market_data(symbol)
        
        # Display results
        print(f"\n🏢 {symbol} - {stock_data.info.get('longName', 'N/A')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"📈 Market Cap: {self._format_market_cap(stock_data.info.get('marketCap', 0))}")
        print(f"📊 P/E Ratio: {stock_data.info.get('trailingPE', 'N/A')}")
        print(f"📉 52 Week Range: ${stock_data.info.get('fiftyTwoWeekLow', 'N/A')} - ${stock_data.info.get('fiftyTwoWeekHigh', 'N/A')}")
        
        print("\n📊 Technical Indicators:")
        if market_data.get('sma_20'):
            print(f"   SMA 20: ${market_data['sma_20']:.2f}")
        if market_data.get('sma_50'):
            print(f"   SMA 50: ${market_data['sma_50']:.2f}")
        if market_data.get('rsi'):
            print(f"   RSI: {market_data['rsi']:.1f}")
        if market_data.get('volatility'):
            print(f"   Volatility: {market_data['volatility']:.1f}%")
    
    def _handle_price(self, args):
        """Handle price command"""
        symbol = args.symbol.upper()
        
        if args.real_time:
            print(f"\n💰 Getting real-time price for {symbol}...")
            price_data = data_collector.get_real_time_price(symbol)
            
            print(f"\n{symbol} Real-time Data:")
            print(f"   Current Price: ${price_data['current_price']:.2f}")
            print(f"   Change: ${price_data['change']:.2f} ({price_data['change_percent']:.2f}%)")
            print(f"   Open: ${price_data['open']:.2f}")
            print(f"   High: ${price_data['high']:.2f}")
            print(f"   Low: ${price_data['low']:.2f}")
            print(f"   Volume: {price_data['volume']:,}")
        else:
            print(f"\n💰 Getting latest price for {symbol}...")
            stock_data = data_collector.get_stock_data(symbol)
            latest_price = stock_data.prices['Close'].iloc[-1]
            latest_date = stock_data.prices.index[-1].strftime('%Y-%m-%d')
            
            print(f"\n{symbol} Latest Price ({latest_date}): ${latest_price:.2f}")
    
    def _handle_recommend(self, args):
        """Handle recommend command - placeholder for Phase 1"""
        symbol = args.symbol.upper()
        print(f"\n🤖 Getting recommendation for {symbol}...")
        
        # Get technical analysis
        tech_signals = technical_analyzer.get_technical_signals(symbol)
        
        print(f"\n📊 Technical Analysis for {symbol}:")
        print(f"   Overall Signal: {tech_signals['overall_signal']}")
        print(f"   RSI Signal: {tech_signals['signals']['rsi']}")
        print(f"   MACD Signal: {tech_signals['signals']['macd']}")
        print(f"   Trend Signal: {tech_signals['signals']['trend']}")
        print(f"   Volume: {tech_signals['signals']['volume']}")
        
        print(f"\n⚠️  Note: This is a basic technical analysis. Full AI prediction model coming in Phase 2!")
    
    def _handle_technical(self, args):
        """Handle technical analysis command"""
        symbol = args.symbol.upper()
        print(f"\n📈 Technical Analysis for {symbol}...")
        
        # Get technical signals
        tech_data = technical_analyzer.get_technical_signals(symbol)
        
        print(f"\n📊 Technical Indicators:")
        indicators = tech_data['indicators']
        print(f"   RSI: {indicators['rsi']:.1f}")
        print(f"   MACD: {indicators['macd']:.3f}")
        print(f"   BB Position: {indicators['bb_position']:.2f}")
        print(f"   SMA 20: ${indicators['sma_20']:.2f}")
        print(f"   SMA 50: ${indicators['sma_50']:.2f}")
        print(f"   Volume Ratio: {indicators['volume_ratio']:.2f}x")
        
        print(f"\n🎯 Signals:")
        for indicator, signal in tech_data['signals'].items():
            print(f"   {indicator.upper()}: {signal}")
        
        print(f"\n🎯 Overall Signal: {tech_data['overall_signal']}")
        
        if args.support_resistance:
            print(f"\n📊 Support & Resistance Analysis...")
            sr_data = technical_analyzer.get_support_resistance(symbol)
            print(f"   Nearest Support: ${sr_data['nearest_support']:.2f} ({sr_data['distance_to_support']:.1f}% below)")
            print(f"   Nearest Resistance: ${sr_data['nearest_resistance']:.2f} ({sr_data['distance_to_resistance']:.1f}% above)")
    
    def _handle_data(self, args):
        """Handle data download command"""
        symbol = args.symbol.upper()
        period = args.period
        
        print(f"\n📥 Downloading {period} of data for {symbol}...")
        
        stock_data = data_collector.get_stock_data(symbol, period)
        
        print(f"   Data points: {len(stock_data.prices)}")
        print(f"   Date range: {stock_data.prices.index[0].strftime('%Y-%m-%d')} to {stock_data.prices.index[-1].strftime('%Y-%m-%d')}")
        
        if args.save:
            data_collector.save_data(symbol)
            print(f"   ✅ Data saved to data/{symbol}_data.csv")
        
        print("   ✅ Data download complete!")
    
    def _handle_train(self, args):
        """Handle model training command - placeholder for Phase 1"""
        symbol = args.symbol.upper()
        print(f"\n🧠 Training prediction model for {symbol}...")
        print(f"⚠️  Model training functionality coming in Phase 2!")
        print(f"   For now, using technical analysis for recommendations.")
    
    def _handle_predict(self, args):
        """Handle price prediction command - placeholder for Phase 1"""
        symbol = args.symbol.upper()
        days = args.days
        
        print(f"\n🔮 Predicting {symbol} price for {days} days ahead...")
        print(f"⚠️  AI prediction model coming in Phase 2!")
        print(f"   For now, see technical analysis with: python cli.py technical {symbol}")
    
    def _handle_config(self, args):
        """Handle configuration command"""
        print("\n⚙️  Stock AI Configuration:")
        print(f"   Log Level: {config.log_level}")
        print(f"   Log File: {config.log_file}")
        print(f"   Model Data Path: {config.model.data_path}")
        print(f"   Cache Expiry: {config.cache.expiry_minutes} minutes")
        
        if args.validate:
            print("\n🔍 Validating configuration...")
            issues = config.validate()
            
            if issues:
                print("   ❌ Configuration issues found:")
                for issue in issues:
                    print(f"      - {issue}")
            else:
                print("   ✅ Configuration is valid!")
        
        print(f"\n🔑 API Keys Status:")
        print(f"   Alpha Vantage: {'✅ Configured' if config.api.alpha_vantage_key else '❌ Not configured'}")
        print(f"   FMP: {'✅ Configured' if config.api.fmp_key else '❌ Not configured'}")
        print(f"   Finnhub: {'✅ Configured' if config.api.finnhub_key else '❌ Not configured'}")
        
        if not config.api.alpha_vantage_key:
            print("\n💡 To configure API keys:")
            print("   1. Copy .env.example to .env")
            print("   2. Add your API keys to .env file")
    
    def _format_market_cap(self, market_cap: int) -> str:
        """Format market cap in human readable form"""
        if market_cap == 0:
            return "N/A"
        elif market_cap >= 1e12:
            return f"${market_cap/1e12:.1f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.1f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.1f}M"
        else:
            return f"${market_cap:,.0f}"


def main():
    """Main entry point for CLI"""
    cli = StockAICLI()
    cli.run()


if __name__ == "__main__":
    main()
