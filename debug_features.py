#!/usr/bin/env python3
"""
Debug script to analyze our features and data for LSTM prediction
"""

import numpy as np
import pandas as pd
from src.data.collector import StockDataCollector
from src.models.lstm_predictor import EnhancedLSTMPredictor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def debug_features():
    print("🔍 Debugging AAPL features and predictability...")
    
    # Get data using our collector
    collector = StockDataCollector()
    stock_data = collector.get_stock_data("AAPL", "5y")
    prices_df = stock_data.prices
    
    print(f"📊 Data shape: {prices_df.shape}")
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

if __name__ == "__main__":
    debug_features()
