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
from src.models.lstm_predictor import create_enhanced_predictor
from src.utils.helpers import format_currency, format_percentage

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
  # Basic Analysis (Phase 1)
  python cli.py analyze AAPL                    # Get comprehensive analysis for Apple
  python cli.py price GOOGL --real-time         # Get real-time price for Google
  python cli.py technical TSLA --support-resistance  # Get technical analysis for Tesla
  python cli.py recommend MSFT                  # Get basic recommendation for Microsoft
  
  # AI-Powered Features (Phase 2)
  python cli.py train AAPL                      # Train AI model for Apple
  python cli.py predict AAPL --days 30          # AI price prediction for 30 days
  python cli.py ai-recommend TSLA --detailed    # AI-powered investment recommendation
  python cli.py model-info NVDA                 # Show AI model performance metrics
  
  # Data Management
  python cli.py data AMZN --save                # Download and save Amazon data
  python cli.py config --validate               # Validate system configuration
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
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train LSTM prediction model')
        train_parser.add_argument('symbol', help='Stock ticker symbol')
        train_parser.add_argument('--period', default='3y', help='Training data period (1y, 2y, 3y, 5y)')
        train_parser.add_argument('--force', action='store_true', help='Force retrain even if model exists')
        
        # Predict command
        predict_parser = subparsers.add_parser('predict', help='AI-powered stock price prediction')
        predict_parser.add_argument('symbol', help='Stock ticker symbol')
        predict_parser.add_argument('--days', type=int, default=60, help='Days to predict ahead')
        predict_parser.add_argument('--save-report', action='store_true', help='Save detailed prediction report')
        
        # AI Recommend command (new Phase 2 feature)
        ai_recommend_parser = subparsers.add_parser('ai-recommend', help='AI-powered buy/sell recommendation')
        ai_recommend_parser.add_argument('symbol', help='Stock ticker symbol')
        ai_recommend_parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
        
        # Model info command
        model_parser = subparsers.add_parser('model-info', help='Show model information and performance')
        model_parser.add_argument('symbol', help='Stock ticker symbol')
        
        # Cleanup command
        cleanup_parser = subparsers.add_parser('cleanup', help='Clean up model files')
        cleanup_parser.add_argument('symbol', nargs='?', help='Stock ticker symbol (optional, cleans all if not provided)')
        cleanup_parser.add_argument('--all', action='store_true', help='Clean up all model files')
        
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
            elif parsed_args.command == 'ai-recommend':
                self._handle_ai_recommend(parsed_args)
            elif parsed_args.command == 'model-info':
                self._handle_model_info(parsed_args)
            elif parsed_args.command == 'cleanup':
                self._handle_cleanup(parsed_args)
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
        """Handle model training command with quality validation"""
        symbol = args.symbol.upper()
        period = args.period
        
        print(f"\n🧠 Training Enhanced LSTM model for {symbol}...")
        print(f"📊 Using {period} of historical data")
        
        try:
            # Create predictor
            predictor = create_enhanced_predictor(symbol)
            
            # Check if model already exists (but still allow retraining since we have quality validation now)
            if predictor.model_path.exists() and not args.force:
                print(f"⚠️  Model already exists for {symbol}")
                print(f"   Use --force to retrain or check model performance with: python cli.py model-info {symbol}")
                return
            
            # Start training with quality validation
            print(f"🔄 Training in progress... This may take several minutes.")
            print(f"🔍 Quality validation enabled - poor models will be automatically deleted")
            
            result = predictor.train_enhanced_model(period)
            
            # Handle validation results
            if 'error' in result:
                if result.get('validation_failed'):
                    print(f"\n❌ Model Quality Validation Failed!")
                    print(f"Reason: {result['error']}")
                    if 'r2_score' in result:
                        print(f"R² Score: {result['r2_score']:.3f} (minimum required: 0.3)")
                    if 'recommendation' in result:
                        print(f"💡 {result['recommendation']}")
                    print(f"\n🗑️ Poor quality model automatically deleted")
                else:
                    print(f"\n❌ Training failed: {result['error']}")
                return
            
            # Success - display comprehensive results
            print(f"\n✅ Model training completed successfully!")
            print(f"📈 Model Performance:")
            print(f"   Validation R² Score: {result.get('val_r2', 0):.4f}")
            print(f"   Test R² Score: {result.get('test_r2', 0):.4f}")
            print(f"   Validation RMSE: ${result.get('val_rmse', 0):.2f}")
            print(f"   Validation MAE: ${result.get('val_mae', 0):.2f}")
            print(f"   Training Time: {result.get('training_time', 0):.1f}s")
            print(f"   Epochs Trained: {result.get('epochs_trained', 0)}")
            print(f"   Total Parameters: {result.get('total_parameters', 0):,}")
            
            # Enhanced model quality assessment
            r2_score = result.get('val_r2', 0)
            if r2_score >= 0.8:
                quality = "Excellent 🎯"
            elif r2_score >= 0.6:
                quality = "Good 👍"
            elif r2_score >= 0.3:  # Updated threshold to match validation
                quality = "Fair 👌"
            else:
                quality = "Poor ❌ (should not reach here due to validation)"
            
            print(f"\n🎯 Model Quality: {quality}")
            print(f"📋 Model saved to: {predictor.model_path}")
            print(f"\n✨ Model is ready for predictions and AI recommendations!")
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {str(e)}")
            print(f"❌ Training failed: {str(e)}")
    
    def _handle_predict(self, args):
        """Handle AI price prediction command"""
        symbol = args.symbol.upper()
        days = args.days
        
        print(f"\n🔮 AI Price Prediction for {symbol}")
        print(f"📅 Forecasting {days} days ahead...")
        
        try:
            # Create predictor
            predictor = create_enhanced_predictor(symbol)
            
            # Check if model exists
            if not predictor.load_model():
                print(f"❌ No trained model found for {symbol}")
                print(f"💡 Train a model first: python cli.py train {symbol}")
                return
            
            # Make prediction
            prediction = predictor.predict_price(days)
            
            print(f"\n📊 AI Prediction Results:")
            print(f"   Current Price: {format_currency(prediction['current_price'])}")
            print(f"   Predicted Price ({days} days): {format_currency(prediction['predicted_price'])}")
            print(f"   Expected Change: {format_percentage(prediction['percent_change'])}")
            print(f"   Confidence: {prediction['confidence_score']['score']}")
            
            # Show confidence intervals if available
            if 'confidence_intervals' in prediction and prediction['confidence_intervals']:
                last_ci = prediction['confidence_intervals'][-1]
                print(f"   95% Confidence Range: {format_currency(last_ci['lower'])} - {format_currency(last_ci['upper'])}")
            
            # Model performance
            if 'model_performance' in prediction:
                perf = prediction['model_performance']
                print(f"\n🎯 Model Performance:")
                print(f"   R² Score: {perf.get('validation_r2', 0):.3f}")
                print(f"   RMSE: ${perf.get('validation_rmse', 0):.2f}")
                print(f"   Features Used: {perf.get('num_features', 0)}")
            
            # Save detailed report if requested
            if args.save_report:
                report_path = Path(f"reports/{symbol}_prediction_report.png")
                report_path.parent.mkdir(exist_ok=True)
                
                report = predictor.create_prediction_report(report_path)
                print(f"\n📋 Detailed report saved to: {report_path}")
                
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {str(e)}")
            print(f"❌ Prediction failed: {str(e)}")
    
    def _handle_ai_recommend(self, args):
        """Handle AI recommendation command"""
        symbol = args.symbol.upper()
        
        print(f"\n🤖 AI Investment Recommendation for {symbol}")
        
        try:
            # Create predictor
            predictor = create_enhanced_predictor(symbol)
            
            # Check if model exists
            if not predictor.load_model():
                print(f"❌ No trained AI model found for {symbol}")
                print(f"💡 Train a model first: python cli.py train {symbol}")
                print(f"📊 Using technical analysis instead...")
                self._handle_recommend(args)
                return
            
            # Get AI recommendation
            recommendation = predictor.get_enhanced_recommendation()
            
            print(f"\n🎯 AI Recommendation: {recommendation['recommendation']}")
            print(f"💡 Reasoning: {recommendation['reasoning']}")
            print(f"📈 Expected Return: {format_percentage(recommendation['predicted_change_percent'])}")
            print(f"🎲 Risk Level: {recommendation['risk_level']}")
            print(f"⚡ Confidence: {recommendation['confidence']['score']}")
            print(f"📅 Prediction Horizon: {recommendation['prediction_horizon_days']} days")
            
            if args.detailed:
                print(f"\n📊 Detailed Analysis:")
                print(f"   Current Price: {format_currency(recommendation['current_price'])}")
                print(f"   Target Price: {format_currency(recommendation['target_price'])}")
                
                # Model performance details
                if 'model_performance' in recommendation:
                    perf = recommendation['model_performance']
                    print(f"\n🧠 AI Model Details:")
                    print(f"   Model R² Score: {perf.get('validation_r2', 0):.3f}")
                    print(f"   Model RMSE: ${perf.get('validation_rmse', 0):.2f}")
                    print(f"   Training Date: {perf.get('training_date', 'Unknown')[:10]}")
                    print(f"   Features Used: {perf.get('num_features', 0)}")
            
        except Exception as e:
            logger.error(f"AI recommendation failed for {symbol}: {str(e)}")
            print(f"❌ AI recommendation failed: {str(e)}")
    
    def _handle_model_info(self, args):
        """Handle model information command"""
        symbol = args.symbol.upper()
        
        print(f"\n🧠 AI Model Information for {symbol}")
        
        try:
            # Create predictor
            predictor = create_enhanced_predictor(symbol)
            
            # Check if model exists
            if not predictor.load_model():
                print(f"❌ No trained model found for {symbol}")
                print(f"� Train a model first: python cli.py train {symbol}")
                return
            
            # Display model information
            if predictor.training_metrics:
                metrics = predictor.training_metrics
                
                print(f"\n📊 Model Performance Metrics:")
                print(f"   Training R² Score: {metrics.get('train_r2', 0):.4f}")
                print(f"   Validation R² Score: {metrics.get('val_r2', 0):.4f}")
                print(f"   Test R² Score: {metrics.get('test_r2', 0):.4f}")
                print(f"   Validation RMSE: ${metrics.get('val_rmse', 0):.2f}")
                print(f"   Validation MAE: ${metrics.get('val_mae', 0):.2f}")
                
                print(f"\n🏗️ Model Architecture:")
                print(f"   Sequence Length: {metrics.get('sequence_length', 0)} days")
                print(f"   Prediction Horizon: {metrics.get('prediction_days', 0)} days")
                print(f"   Number of Features: {metrics.get('num_features', 0)}")
                print(f"   Total Parameters: {metrics.get('total_parameters', 0):,}")
                print(f"   Epochs Trained: {metrics.get('epochs_trained', 0)}")
                
                print(f"\n📅 Training Information:")
                training_date = metrics.get('training_date', 'Unknown')
                if training_date != 'Unknown':
                    print(f"   Training Date: {training_date[:10]}")
                print(f"   Best Validation Loss: {metrics.get('best_val_loss', 0):.6f}")
                
                # Model quality assessment
                r2_score = metrics.get('val_r2', 0)
                if r2_score >= 0.8:
                    quality = "Excellent 🎯 - High accuracy predictions"
                elif r2_score >= 0.6:
                    quality = "Good 👍 - Reliable for medium-term trends"
                elif r2_score >= 0.4:
                    quality = "Fair 👌 - Use with caution"
                else:
                    quality = "Poor 📈 - Consider retraining with more data"
                
                print(f"\n🎯 Overall Model Quality: {quality}")
                
                # File information
                print(f"\n📁 Model Files:")
                print(f"   Model: {predictor.model_path}")
                print(f"   Scaler: {predictor.scaler_path}")
                print(f"   Feature Scaler: {predictor.feature_scaler_path}")
                print(f"   Metrics: {predictor.metrics_path}")
            else:
                print(f"⚠️  Model loaded but no training metrics available")
                
        except Exception as e:
            logger.error(f"Model info failed for {symbol}: {str(e)}")
            print(f"❌ Failed to get model info: {str(e)}")
    
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
    
    def _handle_cleanup(self, args):
        """Handle model cleanup command"""
        try:
            if args.all or not args.symbol:
                # Clean up all models
                import glob
                from pathlib import Path
                
                models_dir = Path("data/models")
                if models_dir.exists():
                    model_files = list(models_dir.glob("*"))
                    if model_files:
                        for file_path in model_files:
                            file_path.unlink()
                        print(f"✅ Cleaned up {len(model_files)} model files")
                    else:
                        print("ℹ️ No model files found to clean up")
                else:
                    print("ℹ️ No models directory found")
            else:
                # Clean up specific symbol
                symbol = args.symbol.upper()
                predictor = create_enhanced_predictor(symbol)
                predictor._cleanup_existing_models()
                print(f"✅ Cleaned up model files for {symbol}")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            print(f"❌ Cleanup failed: {str(e)}")


def main():
    """Main entry point for CLI"""
    cli = StockAICLI()
    cli.run()


if __name__ == "__main__":
    main()
