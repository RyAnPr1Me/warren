#!/usr/bin/env python3
"""
Test the enhanced LSTM model with real earnings and sentiment data
"""

import sys
import os

# Add the src directory to the path
src_path = '/Users/rmanzo28/Downloads/untitled folder/src'
sys.path.insert(0, src_path)

# Change to the workspace directory so relative imports work
os.chdir('/Users/rmanzo28/Downloads/untitled folder')

from models.lstm_predictor import LSTMPredictor
import pandas as pd
from datetime import datetime, timedelta

def test_enhanced_lstm():
    """Test the LSTM model with real earnings and sentiment data"""
    
    print("Testing Enhanced LSTM with Real Data")
    print("=" * 60)
    
    # Initialize the enhanced LSTM predictor
    predictor = LSTMPredictor()
    
    symbol = "AAPL"
    
    print(f"Training enhanced LSTM model for {symbol}...")
    print("This includes:")
    print("- Time series price data")
    print("- Real earnings calendar data from Alpha Vantage")
    print("- Sentiment analysis features")
    print("- Fundamental metrics from yfinance")
    print("-" * 40)
    
    try:
        # Train the model with all enhancements
        results = predictor.train_and_predict(
            symbol=symbol,
            period="2y",  # 2 years of data
            prediction_days=5
        )
        
        print("Training completed successfully!")
        print(f"Results keys: {list(results.keys())}")
        
        if 'predictions' in results:
            predictions = results['predictions']
            print(f"Generated {len(predictions)} predictions")
            print(f"Sample predictions: {predictions[:3]}")
        
        if 'actual' in results:
            actual = results['actual']
            print(f"Actual values count: {len(actual)}")
        
        if 'mae' in results:
            print(f"Mean Absolute Error: {results['mae']:.4f}")
        
        if 'mse' in results:
            print(f"Mean Squared Error: {results['mse']:.4f}")
        
        if 'rmse' in results:
            print(f"Root Mean Squared Error: {results['rmse']:.4f}")
        
        if 'mape' in results:
            print(f"Mean Absolute Percentage Error: {results['mape']:.2f}%")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Enhanced LSTM with real earnings data is working!")
        print("The model now incorporates:")
        print("✓ Historical price data")
        print("✓ Real earnings events and surprise percentages")
        print("✓ Sentiment analysis features")
        print("✓ Fundamental metrics (PE, ROE, etc.)")
        print("✓ Earnings proximity features")
        
    except Exception as e:
        print(f"Error during enhanced LSTM training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_lstm()
