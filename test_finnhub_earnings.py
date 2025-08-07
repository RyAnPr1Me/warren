#!/usr/bin/env python3
"""
Test script for Finnhub earnings API integration
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_finnhub_earnings():
    """Test Finnhub earnings calendar API"""
    
    finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    if not finnhub_api_key:
        print("Error: FINNHUB_API_KEY not found in environment variables")
        return
    
    symbol = "AAPL"
    current_date = datetime.now()
    
    # Get earnings calendar for the past year and future
    from_date = (current_date - timedelta(days=365)).strftime('%Y-%m-%d')
    to_date = (current_date + timedelta(days=180)).strftime('%Y-%m-%d')
    
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {
        'from': from_date,
        'to': to_date,
        'symbol': symbol.upper(),
        'token': finnhub_api_key
    }
    
    print(f"Testing Finnhub earnings API for {symbol}...")
    print(f"Date range: {from_date} to {to_date}")
    print(f"URL: {url}")
    print(f"Params: {params}")
    print("-" * 50)
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            earnings_list = data.get('earningsCalendar', [])
            
            print(f"Found {len(earnings_list)} earnings events")
            
            for i, earning in enumerate(earnings_list[:5]):  # Show first 5
                print(f"\nEarnings Event {i+1}:")
                print(f"  Date: {earning.get('date', 'N/A')}")
                print(f"  EPS Estimate: {earning.get('epsEstimate', 'N/A')}")
                print(f"  EPS Actual: {earning.get('epsActual', 'N/A')}")
                print(f"  Revenue Estimate: {earning.get('revenueEstimate', 'N/A')}")
                print(f"  Revenue Actual: {earning.get('revenueActual', 'N/A')}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

def test_alpha_vantage_earnings():
    """Test Alpha Vantage earnings API as backup"""
    
    alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not alpha_vantage_api_key:
        print("Error: ALPHA_VANTAGE_API_KEY not found in environment variables")
        return
    
    symbol = "AAPL"
    
    # Alpha Vantage earnings endpoint
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'EARNINGS',
        'symbol': symbol,
        'apikey': alpha_vantage_api_key
    }
    
    print(f"\nTesting Alpha Vantage earnings API for {symbol}...")
    print(f"URL: {url}")
    print("-" * 50)
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'Error Message' in data:
                print(f"API Error: {data['Error Message']}")
            elif 'Note' in data:
                print(f"API Note: {data['Note']}")
            else:
                quarterly_earnings = data.get('quarterlyEarnings', [])
                print(f"Found {len(quarterly_earnings)} quarterly earnings")
                
                for i, earning in enumerate(quarterly_earnings[:3]):  # Show first 3
                    print(f"\nQuarterly Earnings {i+1}:")
                    print(f"  Fiscal Date Ending: {earning.get('fiscalDateEnding', 'N/A')}")
                    print(f"  Reported Date: {earning.get('reportedDate', 'N/A')}")
                    print(f"  Reported EPS: {earning.get('reportedEPS', 'N/A')}")
                    print(f"  Estimated EPS: {earning.get('estimatedEPS', 'N/A')}")
                    print(f"  Surprise: {earning.get('surprise', 'N/A')}")
                    print(f"  Surprise Percentage: {earning.get('surprisePercentage', 'N/A')}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_finnhub_earnings()
    test_alpha_vantage_earnings()
