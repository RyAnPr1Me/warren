#!/usr/bin/env python3
"""
Test earnings API integration with Financial Modeling Prep
"""

import sys
sys.path.insert(0, '.')

from src.analysis.fundamentals import FundamentalAnalyzer
import json

def test_earnings_api():
    print("🔍 Testing Real Earnings API Integration")
    print("=" * 50)
    
    analyzer = FundamentalAnalyzer()
    
    # Test AAPL earnings calendar
    print("1️⃣ Testing AAPL Earnings Calendar:")
    print("-" * 30)
    
    try:
        earnings_events = analyzer.get_earnings_calendar("AAPL", periods=4)
        
        print(f"Retrieved {len(earnings_events)} earnings events:")
        for i, event in enumerate(earnings_events):
            print(f"   Event {i+1}:")
            print(f"      Date: {event.date.strftime('%Y-%m-%d')}")
            print(f"      Days until/since: {event.days_until}")
            print(f"      EPS Estimate: {event.eps_estimate}")
            print(f"      EPS Actual: {event.eps_actual}")
            print(f"      Surprise %: {event.surprise_percent}")
            print()
            
    except Exception as e:
        print(f"❌ Error testing earnings calendar: {e}")
        import traceback
        traceback.print_exc()
    
    # Test fundamental metrics
    print("2️⃣ Testing AAPL Fundamental Metrics:")
    print("-" * 30)
    
    try:
        metrics = analyzer.get_fundamental_metrics("AAPL")
        
        print("Fundamental Metrics:")
        print(f"   PE Ratio: {metrics.pe_ratio}")
        print(f"   Forward PE: {metrics.forward_pe}")
        print(f"   PEG Ratio: {metrics.peg_ratio}")
        print(f"   Price to Book: {metrics.price_to_book}")
        print(f"   Price to Sales: {metrics.price_to_sales}")
        print(f"   Debt to Equity: {metrics.debt_to_equity}")
        print(f"   ROE: {metrics.roe}")
        print(f"   ROA: {metrics.roa}")
        print(f"   Profit Margin: {metrics.profit_margin}")
        print(f"   Revenue Growth: {metrics.revenue_growth}")
        print(f"   Earnings Growth: {metrics.earnings_growth}")
        print()
        
    except Exception as e:
        print(f"❌ Error testing fundamental metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # Test earnings proximity features
    print("3️⃣ Testing Earnings Proximity Features:")
    print("-" * 30)
    
    try:
        from src.data.collector import StockDataCollector
        
        collector = StockDataCollector()
        stock_data = collector.get_stock_data("AAPL", "6mo")
        
        earnings_features = analyzer.calculate_earnings_proximity_features(
            stock_data.prices, earnings_events
        )
        
        print(f"Generated earnings features for {len(earnings_features)} trading days")
        print("Sample of recent earnings features:")
        print(earnings_features.tail(10))
        
    except Exception as e:
        print(f"❌ Error testing earnings features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_earnings_api()
