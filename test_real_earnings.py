#!/usr/bin/env python3
"""
Test the updated fundamentals analyzer with real earnings data
"""

import sys
import os
sys.path.append('/Users/rmanzo28/Downloads/untitled folder/src')

from analysis.fundamentals import FundamentalAnalyzer

def test_real_earnings():
    """Test the updated fundamentals analyzer with real earnings data"""
    
    analyzer = FundamentalAnalyzer()
    symbol = "AAPL"
    
    print(f"Testing earnings calendar for {symbol}...")
    print("=" * 60)
    
    # Test earnings calendar
    earnings = analyzer.get_earnings_calendar(symbol, periods=5)
    
    print(f"Found {len(earnings)} earnings events:")
    for i, earning in enumerate(earnings):
        print(f"\nEarnings {i+1}:")
        print(f"  Date: {earning.date}")
        print(f"  EPS Estimate: {earning.eps_estimate}")
        print(f"  EPS Actual: {earning.eps_actual}")
        print(f"  Revenue Estimate: {earning.revenue_estimate}")
        print(f"  Revenue Actual: {earning.revenue_actual}")
        print(f"  Surprise %: {earning.surprise_percent}")
        print(f"  Days Until: {earning.days_until}")
    
    print("\n" + "=" * 60)
    print("Testing fundamental metrics...")
    
    # Test fundamental metrics
    fundamental_metrics = analyzer.get_fundamental_metrics(symbol)
    
    print("Fundamental Metrics:")
    print(f"Available attributes: {[attr for attr in dir(fundamental_metrics) if not attr.startswith('_')]}")
    print(f"  PE Ratio: {getattr(fundamental_metrics, 'pe_ratio', 'N/A')}")
    print(f"  Forward PE: {getattr(fundamental_metrics, 'forward_pe', 'N/A')}")
    print(f"  PB Ratio: {getattr(fundamental_metrics, 'pb_ratio', 'N/A')}")
    print(f"  ROE: {getattr(fundamental_metrics, 'roe', 'N/A')}")
    print(f"  Revenue Growth: {getattr(fundamental_metrics, 'revenue_growth', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Testing earnings proximity features...")
    
    # Test earnings proximity features
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    proximity_features = analyzer.get_earnings_proximity_features(symbol, date_range)
    
    print(f"Generated {len(proximity_features)} proximity features")
    if len(proximity_features) > 0:
        print("Sample proximity features:")
        print(f"  Min days to earnings: {min(proximity_features)}")
        print(f"  Max days to earnings: {max(proximity_features)}")
        print(f"  Mean days to earnings: {sum(proximity_features) / len(proximity_features):.2f}")

if __name__ == "__main__":
    test_real_earnings()
