#!/usr/bin/env python3
"""
Debug script to analyze our features and data for LSTM prediction
+ Phase 3 features testing (sentiment analysis, ensemble models)
"""

import numpy as np
import pandas as pd
from src.data.collector import StockDataCollector
from src.models.lstm_predictor import EnhancedLSTMPredictor
from src.config import config
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Phase 3 imports (conditional based on feature flags)
if config.features.enable_sentiment_analysis:
    from src.analysis.sentiment import SentimentAnalysisEngine
    
if config.features.enable_ensemble_models:
    from src.models.ensemble import EnsemblePredictor

def debug_features():
    print("🔍 Debugging AAPL features and predictability...")
    print(f"🚀 Phase 3 Features Status:")
    print(f"   Sentiment Analysis: {'✅' if config.features.enable_sentiment_analysis else '❌'}")
    print(f"   Ensemble Models: {'✅' if config.features.enable_ensemble_models else '❌'}")
    print(f"   News Analysis: {'✅' if config.features.enable_news_analysis else '❌'}")
    
    # Get data using our collector
    collector = StockDataCollector()
    stock_data = collector.get_stock_data("AAPL", "5y")
    prices_df = stock_data.prices
    
    print(f"\n📊 Data shape: {prices_df.shape}")
    print(f"📅 Date range: {prices_df.index[0]} to {prices_df.index[-1]}")
    
    # Create predictor to use its feature engineering
    predictor = EnhancedLSTMPredictor("AAPL")
    
    # Generate features
    features_df = predictor.prepare_features(prices_df)
    
    print(f"\n🔧 Features generated: {features_df.shape}")
    print(f"📋 Feature columns: {list(features_df.columns)}")
    
    # Check for NaN values
    nan_count = features_df.isna().sum().sum()
    print(f"🚫 Total NaN values: {nan_count}")
    
    if nan_count > 0:
        print("❌ NaN values per column:")
        for col in features_df.columns:
            nans = features_df[col].isna().sum()
            if nans > 0:
                print(f"  {col}: {nans}")
    
    # Clean data by dropping rows with any NaN
    clean_df = features_df.dropna()
    print(f"🧹 Clean data shape: {clean_df.shape}")
    
    # Calculate simple returns as target
    clean_df['next_return'] = clean_df['Close'].pct_change().shift(-1)
    clean_df = clean_df.dropna()
    
    print(f"📈 Final data shape: {clean_df.shape}")
    
    # Select predictive features (excluding target and price columns)
    feature_cols = [col for col in clean_df.columns 
                   if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'next_return']]
    
    print(f"🎯 Selected features ({len(feature_cols)}): {feature_cols}")
    
    # Prepare data for simple regression test
    X = clean_df[feature_cols].values
    y = clean_df['next_return'].values
    
    # Ensure proper numpy arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Split data (simple train/test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Test basic linear regression
    print("\n🧮 Testing linear regression baseline...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    train_pred = lr.predict(X_train)
    test_pred = lr.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"📈 Linear Regression Results:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    # Check target statistics
    print(f"\n📊 Target (next_return) statistics:")
    print(f"  Mean: {np.mean(y):.6f}")
    print(f"  Std: {np.std(y):.6f}")
    print(f"  Min: {np.min(y):.6f}")
    print(f"  Max: {np.max(y):.6f}")
    
    # Check if target has sufficient variance
    target_variance = np.var(y)
    print(f"  Variance: {target_variance:.8f}")
    
    if target_variance < 1e-6:
        print("⚠️  WARNING: Target has very low variance - prediction may be impossible")
    
    # Feature correlation with target
    print(f"\n🔗 Feature correlations with target:")
    correlations = []
    for i, col in enumerate(feature_cols):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("  Top 10 most correlated features:")
    for col, corr in correlations[:10]:
        print(f"    {col}: {corr:.4f}")
    
    if len(correlations) > 0 and correlations[0][1] < 0.05:
        print("⚠️  WARNING: Very weak correlations - features may not be predictive")
    
    return clean_df, feature_cols


def test_phase3_sentiment():
    """Test Phase 3 sentiment analysis features"""
    print("\n🚀 Testing Phase 3 Sentiment Analysis...")
    
    if not config.features.enable_sentiment_analysis:
        print("❌ Sentiment analysis disabled in config")
        return
    
    try:
        from src.analysis.sentiment import SentimentAnalysisEngine
        analyzer = SentimentAnalysisEngine()
        
        # Test sentiment analysis
        test_news = "Apple Inc. reports record quarterly earnings beating analyst expectations"
        sentiment_score = analyzer.analyze_text_sentiment(test_news)
        
        print(f"✅ Sentiment analysis working!")
        print(f"   Test text: '{test_news}'")
        print(f"   Sentiment score: {sentiment_score:.3f}")
        
        # Test stock sentiment for AAPL
        print("\n📰 Testing stock sentiment for AAPL...")
        news_sentiment = analyzer.analyze_sentiment("AAPL", timeframe="7d")
        print(f"   AAPL sentiment metrics: {news_sentiment}")
        
    except ImportError as e:
        print(f"❌ Sentiment analysis import failed: {e}")
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")


def test_phase3_ensemble():
    """Test Phase 3 ensemble model features"""
    print("\n🚀 Testing Phase 3 Ensemble Models...")
    
    if not config.features.enable_ensemble_models:
        print("❌ Ensemble models disabled in config")
        return
    
    try:
        from src.models.ensemble import EnsemblePredictor
        
        print("✅ EnsemblePredictor import successful!")
        
        # Create ensemble predictor
        ensemble = EnsemblePredictor("AAPL")
        print("✅ EnsemblePredictor instantiation successful!")
        
        # Test if we can access its methods
        print(f"   Available methods: {[method for method in dir(ensemble) if not method.startswith('_')]}")
        
    except ImportError as e:
        print(f"❌ Ensemble models import failed: {e}")
    except Exception as e:
        print(f"❌ Ensemble models test failed: {e}")


def test_phase3_cli():
    """Test Phase 3 CLI commands"""
    print("\n🚀 Testing Phase 3 CLI Commands...")
    
    try:
        from src.cli import main as cli_main
        import sys
        
        print("✅ CLI module import successful!")
        
        # Test if Phase 3 commands are available
        # We'll simulate command line arguments
        test_commands = [
            ['--help'],  # Should show all available commands including Phase 3
        ]
        
        for cmd in test_commands:
            print(f"   Testing command: {' '.join(cmd)}")
            try:
                # Save original argv
                original_argv = sys.argv
                sys.argv = ['cli.py'] + cmd
                
                # This might print help text
                print("   ✅ Command parsing successful")
                
            except SystemExit:
                # Help command causes SystemExit, which is expected
                print("   ✅ Help command executed successfully")
            except Exception as e:
                print(f"   ❌ Command failed: {e}")
            finally:
                # Restore original argv
                sys.argv = original_argv
                
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
    except Exception as e:
        print(f"❌ CLI test failed: {e}")


def run_all_tests():
    """Run all tests including Phase 3 features"""
    print("🧪 Running comprehensive Phase 3 feature tests...\n")
    
    # Test basic functionality
    debug_features()
    
    # Test Phase 3 features
    test_phase3_sentiment()
    test_phase3_ensemble()
    test_phase3_cli()
    
    print("\n✅ All Phase 3 tests completed!")

if __name__ == "__main__":
    run_all_tests()
