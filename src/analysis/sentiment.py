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
    """Data class for sentiment analysis results with human-readable formatting"""
    overall_sentiment: float  # -1 to 1
    confidence: float        # 0 to 1
    bullish_ratio: float     # 0 to 1
    bearish_ratio: float     # 0 to 1
    neutral_ratio: float     # 0 to 1
    news_count: int
    timeframe: str
    
    def __str__(self) -> str:
        """Human-readable sentiment summary"""
        sentiment_label = self.get_sentiment_label()
        confidence_label = self.get_confidence_label()
        
        return f"""
📊 SENTIMENT ANALYSIS RESULTS
═══════════════════════════════════════

🎯 OVERALL SENTIMENT: {sentiment_label}
   Score: {self.overall_sentiment:.2f} (Range: -1.0 to +1.0)
   {self.get_sentiment_description()}

📈 CONFIDENCE LEVEL: {confidence_label}
   Score: {self.confidence:.0%}
   Based on {self.news_count} news articles analyzed

📰 SENTIMENT BREAKDOWN:
   🟢 Bullish (Positive): {self.bullish_ratio:.0%}
   🔴 Bearish (Negative): {self.bearish_ratio:.0%}
   ⚪ Neutral: {self.neutral_ratio:.0%}

⏰ Analysis Period: {self.timeframe}

{self.get_investment_recommendation()}
        """
    
    def get_sentiment_label(self) -> str:
        """Get simple sentiment label"""
        if self.overall_sentiment >= 0.3:
            return "🚀 VERY BULLISH"
        elif self.overall_sentiment >= 0.1:
            return "📈 BULLISH"
        elif self.overall_sentiment >= -0.1:
            return "😐 NEUTRAL"
        elif self.overall_sentiment >= -0.3:
            return "📉 BEARISH"
        else:
            return "💥 VERY BEARISH"
    
    def get_confidence_label(self) -> str:
        """Get confidence level description"""
        if self.confidence >= 0.8:
            return "🔥 VERY HIGH"
        elif self.confidence >= 0.6:
            return "✅ HIGH"
        elif self.confidence >= 0.4:
            return "⚠️ MODERATE"
        elif self.confidence >= 0.2:
            return "⚡ LOW"
        else:
            return "❌ VERY LOW"
    
    def get_sentiment_description(self) -> str:
        """Get detailed sentiment description"""
        if self.overall_sentiment >= 0.3:
            return "   💡 The news is overwhelmingly positive! Great earnings, good news, bullish outlook."
        elif self.overall_sentiment >= 0.1:
            return "   💡 Most news is positive. Good signs for potential upward movement."
        elif self.overall_sentiment >= -0.1:
            return "   💡 Mixed sentiment. No clear positive or negative bias in the news."
        elif self.overall_sentiment >= -0.3:
            return "   💡 Most news is negative. Concerns about company performance or market conditions."
        else:
            return "   💡 Very negative news! Major concerns, bad earnings, or serious problems."
    
    def get_investment_recommendation(self) -> str:
        """Get simple investment guidance based on sentiment"""
        if self.confidence < 0.3:
            return "⚠️  WARNING: Low confidence due to limited news data. Be cautious with decisions."
        
        if self.overall_sentiment >= 0.2 and self.confidence >= 0.5:
            return "💰 POSITIVE SIGNAL: Strong bullish sentiment with good confidence."
        elif self.overall_sentiment <= -0.2 and self.confidence >= 0.5:
            return "🛑 NEGATIVE SIGNAL: Strong bearish sentiment with good confidence."
        else:
            return "🤔 NEUTRAL SIGNAL: Mixed or weak sentiment. Consider other factors."


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
            # Try alternative news sources
            return self._get_alternative_news_sentiment(symbol)
        
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
                # Try alternative sources
                return self._get_alternative_news_sentiment(symbol)
                
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            # Try alternative sources
            return self._get_alternative_news_sentiment(symbol)
    
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
            return self._get_minimal_sentiment()
        
        # Filter out items with no sentiment score
        valid_items = [item for item in news_items if item.sentiment_score is not None]
        
        if not valid_items:
            return self._get_minimal_sentiment()
        
        sentiments = [item.sentiment_score for item in valid_items if item.sentiment_score is not None]
        
        # Calculate metrics
        overall_sentiment = float(np.mean(sentiments))
        confidence = min(len(valid_items) / 20.0, 1.0)  # More news = higher confidence
        
        # Categorize sentiments
        bullish_count = sum(1 for s in sentiments if s is not None and s > 0.1)
        bearish_count = sum(1 for s in sentiments if s is not None and s < -0.1)
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
        early_scores = [item.sentiment_score for item in sorted_items[:mid_point] if item.sentiment_score is not None]
        recent_scores = [item.sentiment_score for item in sorted_items[mid_point:] if item.sentiment_score is not None]
        
        if not early_scores or not recent_scores:
            return "insufficient_data"
        
        early_sentiment = float(np.mean(early_scores))
        recent_sentiment = float(np.mean(recent_scores))
        
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
        
        valid_scores = [item.sentiment_score for item in recent_items if item.sentiment_score is not None]
        if not valid_scores:
            return 0.0
            
        return float(np.mean(valid_scores))
    
    def _get_alternative_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment using yfinance and other free sources"""
        try:
            # Use yfinance to get news
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.warning(f"No news found for {symbol}")
                return self._get_minimal_sentiment()
            
            # Analyze sentiment using TextBlob
            news_items = []
            for item in news[:20]:  # Limit to recent 20 items
                try:
                    title = item.get('title', '')
                    summary = item.get('summary', item.get('description', ''))
                    
                    # Combine title and summary for sentiment analysis
                    text = f"{title}. {summary}"
                    
                    # Use TextBlob for sentiment
                    sentiment_score = self._analyze_text_sentiment(text)
                    
                    news_item = NewsItem(
                        title=title,
                        summary=summary,
                        url=item.get('link', ''),
                        publish_time=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                        source=item.get('publisher', ''),
                        sentiment_score=sentiment_score,
                        relevance_score=0.8  # Assume high relevance since it's ticker-specific
                    )
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"Error processing news item: {e}")
                    continue
            
            if news_items:
                return self._calculate_sentiment_metrics(news_items)
            else:
                return self._get_minimal_sentiment()
                
        except Exception as e:
            logger.error(f"Error getting alternative news sentiment: {e}")
            return self._get_minimal_sentiment()
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            # Access sentiment polarity safely
            sentiment_obj = blob.sentiment
            polarity = getattr(sentiment_obj, 'polarity', 0.0)
            return float(polarity)
        except Exception:
            return 0.0
    
    def _get_minimal_sentiment(self) -> Dict:
        """Return minimal sentiment data when no real data is available"""
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
    """Analyze social media sentiment from Reddit and other sources"""
    
    def __init__(self):
        self.reddit_base_url = "https://www.reddit.com"
    
    def get_reddit_sentiment(self, symbol: str) -> Dict:
        """
        Get Reddit sentiment for a stock symbol
        Uses Reddit's JSON API to scrape recent posts
        """
        try:
            # Search multiple investment subreddits
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets']
            all_posts = []
            
            for subreddit in subreddits:
                posts = self._get_reddit_posts(subreddit, symbol)
                all_posts.extend(posts)
            
            if not all_posts:
                logger.warning(f"No Reddit posts found for {symbol}")
                return self._get_empty_social_data()
            
            # Analyze sentiment of collected posts
            sentiment_scores = []
            for post in all_posts:
                text = f"{post.get('title', '')} {post.get('selftext', '')}"
                if text.strip():
                    sentiment_score = self._analyze_text_sentiment(text)
                    sentiment_scores.append(sentiment_score)
            
            if not sentiment_scores:
                return self._get_empty_social_data()
            
            avg_sentiment = float(np.mean(sentiment_scores))
            engagement = min(len(all_posts) / 20.0, 1.0)  # Normalize engagement
            
            return {
                'reddit_sentiment': avg_sentiment,
                'reddit_posts': len(all_posts),
                'reddit_engagement': engagement,
                'subreddit_breakdown': self._calculate_subreddit_breakdown(all_posts, sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return self._get_empty_social_data()
    
    def _get_reddit_posts(self, subreddit: str, symbol: str) -> List[Dict]:
        """Get Reddit posts mentioning the symbol from a specific subreddit"""
        try:
            # Use Reddit's JSON API
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': symbol,
                'restrict_sr': 'true',
                'sort': 'new',
                'limit': 10,
                't': 'week'  # Last week
            }
            
            headers = {
                'User-Agent': 'StockAI/1.0'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = []
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    
                    # Filter posts that actually mention the symbol
                    title = post_data.get('title', '').upper()
                    selftext = post_data.get('selftext', '').upper()
                    
                    if symbol.upper() in title or symbol.upper() in selftext:
                        posts.append({
                            'title': post_data.get('title', ''),
                            'selftext': post_data.get('selftext', ''),
                            'score': post_data.get('score', 0),
                            'subreddit': subreddit,
                            'created_utc': post_data.get('created_utc', 0)
                        })
                
                return posts[:5]  # Limit to 5 most recent relevant posts
            else:
                logger.warning(f"Reddit API returned status {response.status_code} for {subreddit}")
                return []
                
        except Exception as e:
            logger.warning(f"Error fetching Reddit posts from {subreddit}: {e}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment using TextBlob"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            # Access sentiment polarity safely
            sentiment_obj = blob.sentiment
            polarity = getattr(sentiment_obj, 'polarity', 0.0)
            return float(polarity)
        except Exception:
            return 0.0
    
    def _calculate_subreddit_breakdown(self, posts: List[Dict], sentiments: List[float]) -> Dict:
        """Calculate sentiment breakdown by subreddit"""
        subreddit_sentiments = {}
        
        for i, post in enumerate(posts):
            if i < len(sentiments):
                subreddit = post.get('subreddit', 'unknown')
                if subreddit not in subreddit_sentiments:
                    subreddit_sentiments[subreddit] = []
                subreddit_sentiments[subreddit].append(sentiments[i])
        
        # Calculate average sentiment per subreddit
        breakdown = {}
        for subreddit, scores in subreddit_sentiments.items():
            if scores:
                breakdown[subreddit] = float(np.mean(scores))
            else:
                breakdown[subreddit] = 0.0
        
        return breakdown
    
    def _get_empty_social_data(self) -> Dict:
        """Return empty social data structure"""
        return {
            'reddit_sentiment': 0.0,
            'reddit_posts': 0,
            'reddit_engagement': 0.0,
            'subreddit_breakdown': {}
        }
    
    def get_social_buzz(self, symbol: str) -> Dict:
        """
        Get overall social media buzz metrics
        """
        try:
            reddit_data = self.get_reddit_sentiment(symbol)
            
            # Calculate buzz score based on post volume and engagement
            post_count = reddit_data.get('reddit_posts', 0)
            engagement = reddit_data.get('reddit_engagement', 0.0)
            sentiment = reddit_data.get('reddit_sentiment', 0.0)
            
            # Buzz score combines volume and sentiment strength
            buzz_score = min((post_count / 10.0) * (abs(sentiment) + 0.1), 1.0)
            
            # Calculate sentiment distribution
            if sentiment > 0.1:
                positive = 0.6
                negative = 0.2
                neutral = 0.2
            elif sentiment < -0.1:
                positive = 0.2
                negative = 0.6
                neutral = 0.2
            else:
                positive = 0.33
                negative = 0.33
                neutral = 0.34
            
            return {
                'buzz_score': buzz_score,
                'mention_volume': post_count,
                'sentiment_distribution': {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating social buzz for {symbol}: {e}")
            return {
                'buzz_score': 0.0,
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
    
    def analyze_text_sentiment(self, text: str) -> float:
        """
        Simple text sentiment analysis for testing
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            # Access sentiment polarity safely
            sentiment_obj = blob.sentiment
            polarity = getattr(sentiment_obj, 'polarity', 0.0)
            return float(polarity)
        except Exception as e:
            logger.warning(f"Text sentiment analysis failed: {e}")
            return 0.0
    
    def get_simple_sentiment_summary(self, symbol: str) -> str:
        """
        Get a simple, easy-to-understand sentiment summary
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Simple text summary that anyone can understand
        """
        try:
            sentiment_metrics = self.analyze_sentiment(symbol)
            
            # Get the main verdict
            if sentiment_metrics.overall_sentiment >= 0.2:
                verdict = "👍 POSITIVE"
                action = "This looks GOOD for the stock!"
            elif sentiment_metrics.overall_sentiment <= -0.2:
                verdict = "👎 NEGATIVE" 
                action = "This looks BAD for the stock!"
            else:
                verdict = "🤷 NEUTRAL"
                action = "News is mixed - no clear direction."
            
            # Confidence explanation
            if sentiment_metrics.confidence >= 0.6:
                confidence_text = "We're pretty confident about this"
            elif sentiment_metrics.confidence >= 0.3:
                confidence_text = "We're somewhat confident about this"
            else:
                confidence_text = "We're not very confident (limited news)"
            
            return f"""
🔍 SENTIMENT ANALYSIS FOR {symbol.upper()}
{verdict} - {action}
{confidence_text} (based on {sentiment_metrics.news_count} news articles)

📊 What people are saying:
• {sentiment_metrics.bullish_ratio:.0%} of news is POSITIVE 📈
• {sentiment_metrics.bearish_ratio:.0%} of news is NEGATIVE 📉  
• {sentiment_metrics.neutral_ratio:.0%} of news is NEUTRAL 😐

Bottom Line: {sentiment_metrics.get_investment_recommendation()}
            """
            
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {e}")
            return f"❌ Could not analyze sentiment for {symbol.upper()}"


# Create global instance
sentiment_engine = SentimentAnalysisEngine()
