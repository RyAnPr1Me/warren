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
        train_parser = subparsers.add_parser('train', help='Train LSTM and ensemble prediction models')
        train_parser.add_argument('symbol', help='Stock ticker symbol')
        train_parser.add_argument('--period', default='3y', help='Training data period (1y, 2y, 3y, 5y, max)')
        train_parser.add_argument('--force', action='store_true', help='Force retrain even if model exists')
        train_parser.add_argument('--lstm-only', action='store_true', help='Train only LSTM model (skip ensemble)')
        train_parser.add_argument('--ensemble-only', action='store_true', help='Train only ensemble models (requires existing LSTM)')
        train_parser.add_argument('--no-ensemble', action='store_true', help='Skip ensemble training (same as --lstm-only)')
        train_parser.add_argument('--mega-data', action='store_true', help='Use maximum data from ALL APIs for training')
        
        # Predict command
        predict_parser = subparsers.add_parser('predict', help='AI-powered stock price prediction')
        predict_parser.add_argument('symbol', help='Stock ticker symbol')
        predict_parser.add_argument('--days', type=int, default=15, help='Days to predict ahead (default: 15 = 3 weeks)')
        predict_parser.add_argument('--save-report', action='store_true', help='Save detailed prediction report')
        predict_parser.add_argument('--rating', action='store_true', help='Include buy/sell rating')
        
        # AI Recommend command (new Phase 2 feature)
        ai_recommend_parser = subparsers.add_parser('ai-recommend', help='AI-powered buy/sell recommendation')
        ai_recommend_parser.add_argument('symbol', help='Stock ticker symbol')
        ai_recommend_parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
        
        # Phase 3: Sentiment Analysis command
        sentiment_parser = subparsers.add_parser('sentiment', help='Analyze news and social sentiment')
        sentiment_parser.add_argument('symbol', help='Stock ticker symbol')
        sentiment_parser.add_argument('--timeframe', default='7d', help='Analysis timeframe (7d, 30d)')
        
        # Phase 3: Ensemble Prediction command
        ensemble_parser = subparsers.add_parser('ensemble', help='Train and use ensemble prediction models')
        ensemble_parser.add_argument('symbol', help='Stock ticker symbol')
        ensemble_parser.add_argument('--train', action='store_true', help='Train ensemble models')
        ensemble_parser.add_argument('--predict', action='store_true', help='Make ensemble prediction')
        ensemble_parser.add_argument('--days', type=int, default=15, help='Days to predict ahead')
        
        # Phase 3: Advanced Analysis command
        advanced_parser = subparsers.add_parser('advanced', help='Comprehensive AI analysis with all Phase 3 features')
        advanced_parser.add_argument('symbol', help='Stock ticker symbol')
        advanced_parser.add_argument('--include-sentiment', action='store_true', help='Include sentiment analysis')
        advanced_parser.add_argument('--use-ensemble', action='store_true', help='Use ensemble predictions')
        
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
            elif parsed_args.command == 'sentiment':
                self._handle_sentiment(parsed_args)
            elif parsed_args.command == 'ensemble':
                self._handle_ensemble(parsed_args)
            elif parsed_args.command == 'advanced':
                self._handle_advanced(parsed_args)
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
        """Handle model training command with quality validation and ensemble training"""
        symbol = args.symbol.upper()
        period = args.period
        use_mega_data = args.mega_data
        
        # Determine training mode
        train_lstm = not args.ensemble_only
        train_ensemble = not (args.lstm_only or args.no_ensemble or args.ensemble_only)
        train_ensemble_only = args.ensemble_only
        
        if use_mega_data:
            period = "max"  # Force maximum data when mega-data is requested
            print(f"\n🚀 MEGA DATA MODE: Collecting maximum historical data from ALL APIs!")
            print(f"📊 This will gather comprehensive data from Alpha Vantage, FMP, Finnhub")
            print(f"💾 Expect 10-20+ years of data with enhanced features")
        
        if train_ensemble_only:
            print(f"\n🤖 Training Ensemble Models only for {symbol}...")
        elif train_lstm and train_ensemble:
            print(f"\n🧠 Training Enhanced LSTM + Ensemble models for {symbol}...")
        elif train_lstm:
            print(f"\n🧠 Training Enhanced LSTM model only for {symbol}...")
        
        print(f"📊 Using {period} of historical data")
        if use_mega_data:
            print(f"🔥 MEGA DATA: Enhanced with multi-API comprehensive dataset")
        print(f"🎯 Training for 3-week (15 day) predictions")
        
        lstm_success = False
        
        try:
            # === LSTM TRAINING PHASE ===
            if train_lstm:
                # Create predictor with default 15-day prediction
                predictor = create_enhanced_predictor(symbol)
                
                # Check if model already exists (but still allow retraining since we have quality validation now)
                if predictor.model_path.exists() and not args.force:
                    print(f"⚠️  LSTM Model already exists for {symbol}")
                    print(f"   Use --force to retrain or check model performance with: python cli.py model-info {symbol}")
                    if not train_ensemble_only:
                        return
                    lstm_success = True  # Assume existing model is good for ensemble training
                else:
                    # Start training with quality validation
                    print(f"🔄 LSTM Training in progress... This may take several minutes.")
                    print(f"🔍 Quality validation enabled - poor models will be automatically deleted")
                    
                    result = predictor.train_enhanced_model(period, mega_data=use_mega_data)
                    
                    # Handle validation results
                    if 'error' in result:
                        if result.get('validation_failed'):
                            print(f"\n❌ LSTM Model Quality Validation Failed!")
                            print(f"Reason: {result['error']}")
                            if 'r2_score' in result:
                                print(f"R² Score: {result['r2_score']:.3f} (minimum required: 0.3)")
                            if 'recommendation' in result:
                                print(f"💡 {result['recommendation']}")
                            print(f"\n🗑️ Poor quality model automatically deleted")
                        else:
                            print(f"\n❌ LSTM Training failed: {result['error']}")
                        
                        if train_ensemble:
                            print(f"⚠️  Skipping ensemble training due to LSTM failure")
                        return
                    
                    # Success - display comprehensive results
                    print(f"\n✅ LSTM Model training completed successfully!")
                    print(f"📈 LSTM Model Performance:")
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
                    
                    print(f"\n🎯 LSTM Model Quality: {quality}")
                    print(f"📋 LSTM Model saved to: {predictor.model_path}")
                    lstm_success = True
            
            elif train_ensemble_only:
                # Check if LSTM model exists for ensemble training
                predictor = create_enhanced_predictor(symbol)
                if not predictor.model_path.exists():
                    print(f"❌ No LSTM model found for {symbol}")
                    print(f"💡 Train LSTM first: python cli.py train {symbol}")
                    return
                lstm_success = True
            
            # === ENSEMBLE TRAINING PHASE ===
            if (train_ensemble or train_ensemble_only) and lstm_success:
                print(f"\n" + "="*60)
                print(f"🤖 Training Ensemble Models for {symbol}...")
                print(f"📊 Using same {period} of historical data as LSTM")
                print(f"🔗 Integrating with LSTM predictor features")
                
                try:
                    from src.models.ensemble import create_ensemble_predictor
                    
                    ensemble = create_ensemble_predictor(symbol)
                    ensemble_metrics = ensemble.train_ensemble(period, mega_data=use_mega_data)
                    
                    if 'error' in ensemble_metrics:
                        print(f"⚠️  Ensemble training failed: {ensemble_metrics['error']}")
                        if train_lstm:
                            print(f"📝 LSTM model is still available for predictions")
                    else:
                        print(f"\n✅ Ensemble training completed successfully!")
                        print(f"📈 Ensemble Performance:")
                        print(f"   Ensemble Test R²: {ensemble_metrics['ensemble_test_r2']:.4f}")
                        print(f"   Ensemble Test RMSE: {ensemble_metrics['ensemble_test_rmse']:.4f}")
                        print(f"   Features Used: {ensemble_metrics['feature_count']}")
                        print(f"   Training Samples: {ensemble_metrics['training_samples']:,}")
                        
                        print(f"\n🔧 Individual Model Performance:")
                        for model_name, metrics in ensemble_metrics['individual_models'].items():
                            print(f"   {model_name}: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.4f}")
                        
                        print(f"\n⚖️  Model Weights:")
                        for model_name, weight in ensemble_metrics['model_weights'].items():
                            print(f"   {model_name}: {weight:.3f}")
                        
                        # Ensemble quality assessment
                        ensemble_r2 = ensemble_metrics['ensemble_test_r2']
                        if ensemble_r2 >= 0.8:
                            ensemble_quality = "Excellent 🎯"
                        elif ensemble_r2 >= 0.6:
                            ensemble_quality = "Good 👍"
                        elif ensemble_r2 >= 0.3:
                            ensemble_quality = "Fair 👌"
                        else:
                            ensemble_quality = "Poor ❌"
                        
                        print(f"\n🎯 Ensemble Model Quality: {ensemble_quality}")
                        
                except ImportError:
                    print(f"⚠️  Ensemble models not available (missing dependencies)")
                except Exception as e:
                    print(f"⚠️  Ensemble training failed: {str(e)}")
                    if train_lstm:
                        print(f"📝 LSTM model is still available for predictions")
            
            # === COMPLETION MESSAGE ===
            print(f"\n" + "="*60)
            if train_lstm and train_ensemble:
                print(f"✨ Training Complete! Both LSTM and Ensemble models ready!")
                print(f"🔮 Use 'python cli.py predict {symbol}' for LSTM predictions")
                print(f"🤖 Use 'python cli.py ensemble {symbol} --predict' for ensemble predictions")
                print(f"🚀 Use 'python cli.py advanced {symbol} --use-ensemble' for comprehensive analysis")
            elif train_lstm:
                print(f"✨ LSTM Training Complete!")
                print(f"🔮 Use 'python cli.py predict {symbol}' for predictions")
                print(f"💡 Add ensemble models: python cli.py train {symbol} --ensemble-only")
            elif train_ensemble_only:
                print(f"✨ Ensemble Training Complete!")
                print(f"🤖 Use 'python cli.py ensemble {symbol} --predict' for ensemble predictions")
                print(f"🚀 Use 'python cli.py advanced {symbol} --use-ensemble' for comprehensive analysis")
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {str(e)}")
            print(f"❌ Training failed: {str(e)}")
    
    def _handle_predict(self, args):
        """Handle AI price prediction command"""
        symbol = args.symbol.upper()
        days = args.days
        show_rating = getattr(args, 'rating', False)
        
        print(f"\n🔮 AI Price Prediction for {symbol}")
        print(f"📅 Forecasting {days} days ahead ({days//5:.1f} weeks)...")
        
        try:
            # Create predictor (always use model trained for default horizon)
            predictor = create_enhanced_predictor(symbol)
            
            # Check if model exists
            if not predictor.load_model():
                print(f"❌ No trained model found for {symbol}")
                print(f"💡 Train a model first: python cli.py train {symbol}")
                return
            
            # Make prediction for requested time horizon
            prediction = predictor.predict_price(days)
            
            print(f"\n📊 AI Prediction Results:")
            print(f"   Current Price: {format_currency(prediction['current_price'])}")
            print(f"   Predicted Price ({days} days): {format_currency(prediction['predicted_price'])}")
            print(f"   Expected Change: {format_percentage(prediction['percent_change'])}")
            print(f"   Confidence: {prediction['confidence_score']['score']}")
            
            # Show buy/sell rating if requested (or always show for better UX)
            if show_rating or True:  # Always show rating for better UX
                rating = predictor.get_buy_sell_rating(prediction['percent_change'])
                print(f"\n💡 Investment Rating: {rating['color']} {rating['rating']}")
                print(f"   {rating['reasoning']}")
            
            # Show confidence intervals if available
            if 'confidence_intervals' in prediction and prediction['confidence_intervals']:
                last_ci = prediction['confidence_intervals'][-1]
                print(f"\n📈 Confidence Range:")
                print(f"   95% Range: {format_currency(last_ci['lower'])} - {format_currency(last_ci['upper'])}")
            
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
    
    def _handle_sentiment(self, args):
        """Handle sentiment analysis command - Phase 3"""
        symbol = args.symbol.upper()
        timeframe = args.timeframe
        
        print(f"\n🧠 Analyzing sentiment for {symbol} ({timeframe})...")
        
        try:
            if not config.features.enable_sentiment_analysis:
                print("❌ Sentiment analysis is not enabled in configuration")
                return
            
            from src.analysis.sentiment import sentiment_engine
            
            # Get sentiment analysis with improved output
            print("\n" + "="*60)
            
            # Show the simple summary that anyone can understand
            simple_summary = sentiment_engine.get_simple_sentiment_summary(symbol)
            print(simple_summary)
            
            # Also show detailed metrics for advanced users
            print("\n� DETAILED METRICS (for advanced users):")
            print("-" * 50)
            sentiment_metrics = sentiment_engine.analyze_sentiment(symbol, timeframe)
            print(sentiment_metrics)
            
        except Exception as e:
            print(f"❌ Error analyzing sentiment: {e}")
            print("This might be due to API limits, network issues, or missing dependencies.")
    
    def _handle_ensemble(self, args):
        """Handle ensemble model command - Phase 3"""
        symbol = args.symbol.upper()
        
        try:
            if not config.features.enable_ensemble_models:
                print("❌ Ensemble models are not enabled in configuration")
                return
            
            from src.models.ensemble import create_ensemble_predictor
            
            ensemble = create_ensemble_predictor(symbol)
            
            if args.train:
                print(f"\n🤖 Training ensemble models for {symbol}...")
                metrics = ensemble.train_ensemble()
                
                if 'error' in metrics:
                    print(f"❌ Training failed: {metrics['error']}")
                else:
                    print(f"\n✅ Ensemble training completed!")
                    print(f"Test R²: {metrics['ensemble_test_r2']:.4f}")
                    print(f"Test RMSE: {metrics['ensemble_test_rmse']:.4f}")
                    print(f"Features used: {metrics['feature_count']}")
                    
                    print("\n📊 Individual Model Performance:")
                    for model_name, model_metrics in metrics['individual_models'].items():
                        print(f"   {model_name}: R² = {model_metrics['test_r2']:.4f}")
            
            elif args.predict:
                print(f"\n🔮 Making ensemble prediction for {symbol}...")
                prediction = ensemble.predict_with_ensemble(args.days)
                
                if 'error' in prediction:
                    print(f"❌ Prediction failed: {prediction['error']}")
                else:
                    current_price = prediction['current_price']
                    predicted_price = prediction['predicted_price']
                    change_pct = prediction['predicted_change_pct']
                    confidence = prediction['confidence']
                    
                    print(f"\n📈 Ensemble Prediction Results:")
                    print(f"Current Price: ${current_price:.2f}")
                    print(f"Predicted Price: ${predicted_price:.2f}")
                    print(f"Expected Change: {change_pct:+.2f}%")
                    print(f"Confidence: {confidence:.1%}")
                    
                    print(f"\n🤖 Model Contributions:")
                    for model_name, contrib in prediction['model_contributions'].items():
                        print(f"   {model_name}: {contrib['weighted_contribution']:+.4f} (weight: {contrib['weight']:.3f})")
            else:
                print("Please specify --train or --predict")
                
        except Exception as e:
            print(f"❌ Error with ensemble: {e}")
    
    def _handle_advanced(self, args):
        """Handle advanced analysis command - Phase 3"""
        symbol = args.symbol.upper()
        
        print(f"\n🚀 Advanced AI Analysis for {symbol}...")
        print("=" * 50)
        
        try:
            # 1. Basic market data
            stock_data = data_collector.get_stock_data(symbol)
            current_price = stock_data.prices['Close'].iloc[-1]
            
            print(f"\n💰 Current Price: ${current_price:.2f}")
            
            # 2. Technical analysis
            print(f"\n📊 Technical Analysis:")
            market_data = data_collector.get_market_data(symbol)
            if market_data.get('rsi'):
                print(f"   RSI: {market_data['rsi']:.1f}")
            if market_data.get('sma_20'):
                print(f"   20-day SMA: ${market_data['sma_20']:.2f}")
            
            # 3. Sentiment analysis (if enabled)
            if args.include_sentiment and config.features.enable_sentiment_analysis:
                print(f"\n🧠 Sentiment Analysis:")
                try:
                    from src.analysis.sentiment import sentiment_engine
                    sentiment_metrics = sentiment_engine.analyze_sentiment(symbol)
                    print(f"   Overall Sentiment: {sentiment_metrics.overall_sentiment:.3f}")
                    print(f"   Confidence: {sentiment_metrics.confidence:.1%}")
                    print(f"   News Count: {sentiment_metrics.news_count}")
                except Exception as e:
                    print(f"   ⚠️ Sentiment analysis failed: {e}")
            
            # 4. AI Predictions
            print(f"\n🤖 AI Predictions:")
            
            # LSTM Prediction
            try:
                predictor = create_enhanced_predictor(symbol)
                if predictor.load_model():
                    lstm_prediction = predictor.predict_price()
                    print(f"   LSTM Model: {lstm_prediction['predicted_change_pct']:+.2f}% ({lstm_prediction['confidence']:.1%} confidence)")
                else:
                    print(f"   LSTM Model: Not trained (run 'python cli.py train {symbol}')")
            except Exception as e:
                print(f"   LSTM Model: Error - {e}")
            
            # Ensemble Prediction (if enabled)
            if args.use_ensemble and config.features.enable_ensemble_models:
                try:
                    from src.models.ensemble import create_ensemble_predictor
                    ensemble = create_ensemble_predictor(symbol)
                    ensemble_prediction = ensemble.predict_with_ensemble()
                    print(f"   Ensemble Model: {ensemble_prediction['predicted_change_pct']:+.2f}% ({ensemble_prediction['confidence']:.1%} confidence)")
                except Exception as e:
                    print(f"   Ensemble Model: Error - {e}")
            
            # 5. Final recommendation
            print(f"\n🎯 AI Recommendation:")
            try:
                predictor = create_enhanced_predictor(symbol)
                if predictor.load_model():
                    prediction = predictor.predict_price()
                    rating = predictor.get_buy_sell_rating(prediction['predicted_change_pct'] / 100)
                    print(f"   {rating['color']} {rating['rating']}")
                    print(f"   Reasoning: {rating['reasoning']}")
                else:
                    print(f"   ⚠️ No trained model available")
            except Exception as e:
                print(f"   ❌ Error generating recommendation: {e}")
                
        except Exception as e:
            print(f"❌ Error in advanced analysis: {e}")
    
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
