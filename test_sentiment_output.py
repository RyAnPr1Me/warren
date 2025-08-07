#!/usr/bin/env python3
"""
Test the improved sentiment analysis output for easy understanding
"""

import sys
sys.path.insert(0, '.')

from src.analysis.sentiment import SentimentAnalysisEngine

def test_sentiment_output():
    print("🧪 Testing Improved Sentiment Analysis Output")
    print("=" * 60)
    
    analyzer = SentimentAnalysisEngine()
    
    # Test simple text sentiment first
    print("\n1️⃣ Testing Simple Text Sentiment:")
    test_texts = [
        "Apple reports record quarterly earnings beating expectations!",
        "Apple stock crashes due to disappointing earnings and guidance cuts",
        "Apple maintains steady performance with mixed quarterly results"
    ]
    
    for text in test_texts:
        score = analyzer.analyze_text_sentiment(text)
        print(f"📝 Text: '{text}'")
        print(f"💯 Score: {score:.3f} {'📈 POSITIVE' if score > 0.1 else '📉 NEGATIVE' if score < -0.1 else '😐 NEUTRAL'}")
        print()
    
    # Test full sentiment analysis for AAPL
    print("2️⃣ Testing Full AAPL Sentiment Analysis:")
    print("-" * 40)
    
    try:
        # Get detailed sentiment metrics
        sentiment_result = analyzer.analyze_sentiment("AAPL")
        
        # Print the detailed formatted output
        print(sentiment_result)
        
        print("\n" + "=" * 60)
        
        # Print the simple summary
        print("3️⃣ Simple Summary for Beginners:")
        print("-" * 40)
        simple_summary = analyzer.get_simple_sentiment_summary("AAPL")
        print(simple_summary)
        
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        print("This might be due to API limits or network issues.")

if __name__ == "__main__":
    test_sentiment_output()
