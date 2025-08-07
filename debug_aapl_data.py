#!/usr/bin/env python3
"""
Debug AAPL data collection to understand why we're getting so few rows
"""

import sys
sys.path.insert(0, '.')

from src.data.collector import StockDataCollector
import yfinance as yf
from datetime import datetime, timedelta

def debug_aapl_data_collection():
    print("🔍 Debugging AAPL Data Collection")
    print("=" * 50)
    
    # Test direct yfinance first
    print("1️⃣ Testing Direct yfinance Access:")
    print("-" * 30)
    
    try:
        ticker = yf.Ticker("AAPL")
        
        # Test different periods
        periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
        
        for period in periods:
            try:
                hist = ticker.history(period=period)
                print(f"   {period:4} period: {len(hist)} rows")
                if len(hist) > 0:
                    print(f"        Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
                else:
                    print("        No data returned")
            except Exception as e:
                print(f"   {period:4} period: ERROR - {e}")
        
        print()
        
    except Exception as e:
        print(f"❌ Direct yfinance test failed: {e}")
    
    # Test our StockDataCollector
    print("2️⃣ Testing Our StockDataCollector:")
    print("-" * 30)
    
    try:
        collector = StockDataCollector()
        
        for period in periods:
            try:
                stock_data = collector.get_stock_data("AAPL", period)
                prices_df = stock_data.prices
                print(f"   {period:4} period: {len(prices_df)} rows")
                if len(prices_df) > 0:
                    print(f"        Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
                    print(f"        Columns: {list(prices_df.columns)}")
                else:
                    print("        No data returned")
            except Exception as e:
                print(f"   {period:4} period: ERROR - {e}")
        
        print()
        
    except Exception as e:
        print(f"❌ StockDataCollector test failed: {e}")
    
    # Check what's happening in data validation
    print("3️⃣ Testing Data Validation:")
    print("-" * 30)
    
    try:
        from src.data.validator import DataValidator
        
        # Get raw data first
        ticker = yf.Ticker("AAPL")
        raw_data = ticker.history(period="1y")
        print(f"Raw yfinance data: {len(raw_data)} rows")
        
        if len(raw_data) > 0:
            print(f"Raw data date range: {raw_data.index[0].date()} to {raw_data.index[-1].date()}")
            print(f"Raw data columns: {list(raw_data.columns)}")
            
            # Test validation
            validator = DataValidator()
            validation_result = validator.validate_stock_data(raw_data, "AAPL")
            
            print(f"Validation passed: {validation_result.is_valid}")
            if not validation_result.is_valid:
                print(f"Validation issues: {validation_result.issues}")
            
            # Test cleaning
            cleaned_data = validator.clean_stock_data(raw_data)
            print(f"Cleaned data: {len(cleaned_data)} rows")
            
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check market hours and current date
    print("4️⃣ Checking Market Context:")
    print("-" * 30)
    
    now = datetime.now()
    print(f"Current time: {now}")
    print(f"Current date: {now.date()}")
    print(f"Day of week: {now.strftime('%A')}")
    
    # Check if markets are open (rough estimate)
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        print("⚠️  Market is closed (weekend)")
    elif now.hour < 9 or now.hour >= 16:  # Rough market hours
        print("⚠️  Market might be closed (outside trading hours)")
    else:
        print("✅ Market should be open")
    
    # Check recent trading days
    print("\n5️⃣ Checking Recent Trading Days:")
    print("-" * 30)
    
    try:
        # Get last 10 days to see pattern
        ticker = yf.Ticker("AAPL")
        recent_data = ticker.history(period="10d")
        
        if len(recent_data) > 0:
            print("Recent trading days:")
            for date, row in recent_data.iterrows():
                print(f"   {date.date()}: Close=${row['Close']:.2f}, Volume={row['Volume']:,}")
        else:
            print("No recent data available")
            
    except Exception as e:
        print(f"❌ Recent data check failed: {e}")

if __name__ == "__main__":
    debug_aapl_data_collection()
