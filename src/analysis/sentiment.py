"""
Sentiment Analysis Engine - Phase 3 Implementation
News and social media sentiment analysis for stock prediction
"""

import logging
import re
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from textblob import TextBlob
import yfinance as yf

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Data class for news items"""
    title: str
    summary: str
    url: str
    publish_time: datetime
    source: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None


@dataclass
class SentimentMetrics:
    """Data class for sentiment analysis results"""
    overall_sentiment: float  # -1 to 1
    confidence: float        # 0 to 1
    bullish_ratio: float     # 0 to 1
    bearish_ratio: float     # 0 to 1
    neutral_ratio: float     # 0 to 1
    news_count: int
    timeframe: str


class NewsCollector:
    """Collect news articles for sentiment analysis"""
    
    def __init__(self):
        self.alpha_vantage_key = config.api.alpha_vantage_key
        
    def get_news_sentiment(self, symbol: str, timeframe: str = "7d") -> Dict:
        """
        Get news sentiment data from Alpha Vantage
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Time frame for news (7d, 30d, etc.)
            
        Returns:
            Dictionary with news sentiment data
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured for news sentiment")
            return self._get_mock_sentiment()
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key,
                'time_from': (datetime.now() - timedelta(days=7)).strftime('%Y%m%dT%H%M'),
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' in data:
                return self._process_news_data(data['feed'], symbol)
            else:
                logger.warning(f"No news data returned for {symbol}")
                return self._get_mock_sentiment()
                
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            return self._get_mock_sentiment()
    
    def _process_news_data(self, news_feed: List[Dict], symbol: str) -> Dict:
        """Process raw news data into sentiment metrics"""
        news_items = []
        
        for item in news_feed:
            try:
                # Extract ticker-specific sentiment if available
                ticker_sentiment = None
                if 'ticker_sentiment' in item:
                    for ticker_data in item['ticker_sentiment']:
                        if ticker_data.get('ticker') == symbol:
                            ticker_sentiment = float(ticker_data.get('relevance_score', 0))
                            break
                
                news_item = NewsItem(
                    title=item.get('title', ''),
                    summary=item.get('summary', ''),
                    url=item.get('url', ''),
                    publish_time=self._parse_time(item.get('time_published', '')),
                    source=item.get('source', ''),
                    sentiment_score=float(item.get('overall_sentiment_score', 0)),
                    relevance_score=ticker_sentiment
                )
                
                news_items.append(news_item)
                
            except Exception as e:
                logger.warning(f"Error processing news item: {e}")
                continue
        
        return self._calculate_sentiment_metrics(news_items)
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse Alpha Vantage time format"""
        try:
            return datetime.strptime(time_str, '%Y%m%dT%H%M%S')
        except:
            return datetime.now()
    
    def _calculate_sentiment_metrics(self, news_items: List[NewsItem]) -> Dict:
        """Calculate aggregate sentiment metrics"""
        if not news_items:
            return self._get_mock_sentiment()
        
        # Filter out items with no sentiment score
        valid_items = [item for item in news_items if item.sentiment_score is not None]
        
        if not valid_items:
            return self._get_mock_sentiment()
        
        sentiments = [item.sentiment_score for item in valid_items]
        
        # Calculate metrics
        overall_sentiment = np.mean(sentiments)
        confidence = min(len(valid_items) / 20.0, 1.0)  # More news = higher confidence
        
        # Categorize sentiments
        bullish_count = sum(1 for s in sentiments if s > 0.1)
        bearish_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - bullish_count - bearish_count
        
        total_count = len(sentiments)
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': confidence,
            'bullish_ratio': bullish_count / total_count if total_count > 0 else 0,
            'bearish_ratio': bearish_count / total_count if total_count > 0 else 0,
            'neutral_ratio': neutral_count / total_count if total_count > 0 else 0,
            'news_count': total_count,
            'sentiment_trend': self._calculate_trend(valid_items),
            'recent_sentiment': self._get_recent_sentiment(valid_items),
        }
    
    def _calculate_trend(self, news_items: List[NewsItem]) -> str:
        """Calculate sentiment trend over time"""
        if len(news_items) < 5:
            return "insufficient_data"
        
        # Sort by time
        sorted_items = sorted(news_items, key=lambda x: x.publish_time)
        
        # Split into early and recent periods
        mid_point = len(sorted_items) // 2
        early_sentiment = np.mean([item.sentiment_score for item in sorted_items[:mid_point]])
        recent_sentiment = np.mean([item.sentiment_score for item in sorted_items[mid_point:]])
        
        diff = recent_sentiment - early_sentiment
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def _get_recent_sentiment(self, news_items: List[NewsItem], hours: int = 24) -> float:
        """Get sentiment for most recent news"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_items = [item for item in news_items if item.publish_time > cutoff_time]
        
        if not recent_items:
            return 0.0
        
        return np.mean([item.sentiment_score for item in recent_items])
    
    def _get_mock_sentiment(self) -> Dict:
        """Return mock sentiment data when API is unavailable"""
        return {
            'overall_sentiment': 0.0,
            'confidence': 0.1,
            'bullish_ratio': 0.33,
            'bearish_ratio': 0.33,
            'neutral_ratio': 0.34,
            'news_count': 0,
            'sentiment_trend': "unknown",
            'recent_sentiment': 0.0,
        }


class SocialSentimentAnalyzer:
    """Analyze social media sentiment (Reddit, Twitter-like sources)"""
    
    def __init__(self):
        pass
    
    def get_reddit_sentiment(self, symbol: str) -> Dict:
        """
        Get Reddit sentiment for a stock symbol
        Note: This is a placeholder - real implementation would use Reddit API
        """
        # Mock implementation for Phase 3
        # In production, this would scrape r/investing, r/stocks, r/SecurityAnalysis
        return {
            'reddit_sentiment': 0.0,
            'reddit_posts': 0,
            'reddit_engagement': 0.0,
            'subreddit_breakdown': {
                'investing': 0.0,
                'stocks': 0.0,
                'SecurityAnalysis': 0.0
            }
        }
    
    def get_social_buzz(self, symbol: str) -> Dict:
        """
        Get overall social media buzz metrics
        """
        # Mock implementation - would integrate multiple social platforms
        return {
            'buzz_score': 0.5,  # 0-1 scale
            'mention_volume': 0,
            'sentiment_distribution': {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34
            }
        }


class SentimentAnalysisEngine:
    """Main sentiment analysis engine combining news and social data"""
    
    def __init__(self):
        self.news_collector = NewsCollector()
        self.social_analyzer = SocialSentimentAnalyzer()
    
    def analyze_sentiment(self, symbol: str, timeframe: str = "7d") -> SentimentMetrics:
        """
        Comprehensive sentiment analysis combining news and social data
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Analysis timeframe
            
        Returns:
            SentimentMetrics object with comprehensive analysis
        """
        logger.info(f"Analyzing sentiment for {symbol} over {timeframe}")
        
        # Collect news sentiment
        news_data = self.news_collector.get_news_sentiment(symbol, timeframe)
        
        # Collect social sentiment
        social_data = self.social_analyzer.get_reddit_sentiment(symbol)
        buzz_data = self.social_analyzer.get_social_buzz(symbol)
        
        # Combine and weight the different sources
        combined_sentiment = self._combine_sentiment_sources(news_data, social_data, buzz_data)
        
        return SentimentMetrics(
            overall_sentiment=combined_sentiment['overall_sentiment'],
            confidence=combined_sentiment['confidence'],
            bullish_ratio=news_data.get('bullish_ratio', 0.33),
            bearish_ratio=news_data.get('bearish_ratio', 0.33),
            neutral_ratio=news_data.get('neutral_ratio', 0.34),
            news_count=news_data.get('news_count', 0),
            timeframe=timeframe
        )
    
    def _combine_sentiment_sources(self, news_data: Dict, social_data: Dict, buzz_data: Dict) -> Dict:
        """Combine different sentiment sources with appropriate weighting"""
        
        # Weight news sentiment more heavily (70%) than social (30%)
        news_weight = 0.7
        social_weight = 0.3
        
        news_sentiment = news_data.get('overall_sentiment', 0.0)
        social_sentiment = social_data.get('reddit_sentiment', 0.0)
        
        # Combine weighted sentiments
        overall_sentiment = (news_sentiment * news_weight + social_sentiment * social_weight)
        
        # Confidence based on data availability
        news_confidence = news_data.get('confidence', 0.1)
        social_confidence = min(social_data.get('reddit_posts', 0) / 10.0, 1.0)
        
        combined_confidence = (news_confidence * news_weight + social_confidence * social_weight)
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': combined_confidence,
            'news_component': news_sentiment,
            'social_component': social_sentiment
        }
    
    def get_sentiment_features(self, symbol: str) -> Dict:
        """
        Get sentiment features for ML model integration
        
        Returns:
            Dictionary of sentiment-based features for model training
        """
        sentiment_metrics = self.analyze_sentiment(symbol)
        
        return {
            'sentiment_overall': sentiment_metrics.overall_sentiment,
            'sentiment_confidence': sentiment_metrics.confidence,
            'sentiment_bullish_ratio': sentiment_metrics.bullish_ratio,
            'sentiment_bearish_ratio': sentiment_metrics.bearish_ratio,
            'sentiment_news_volume': min(sentiment_metrics.news_count / 20.0, 1.0),  # Normalized
        }


# Create global instance
sentiment_engine = SentimentAnalysisEngine()
