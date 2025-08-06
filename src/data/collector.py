"""
Stock Data Collector
Handles fetching and caching of financial market data from various APIs
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import requests
import time
from dataclasses import dataclass

from ..config import config
from .validator import StockDataValidator, DataPipeline

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    """Container for stock market data"""
    symbol: str
    prices: pd.DataFrame
    info: Dict
    financials: Optional[pd.DataFrame] = None
    news: Optional[List[Dict]] = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class StockDataCollector:
    """
    Main class for collecting stock market data from multiple sources
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = config.cache.expiry_minutes
        # Use dynamic minimum based on period - shorter periods need fewer points
        self.validator = StockDataValidator(min_data_points=50)  # More reasonable minimum
        self.pipeline = DataPipeline(self.validator)
        
    def get_stock_data(self, symbol: str, period: str = "2y", validate: bool = True) -> StockData:
        """
        Get comprehensive stock data for a given symbol
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period for historical data ('1y', '2y', '5y', 'max')
            validate: Whether to run data validation pipeline
            
        Returns:
            StockData object containing price history and company info
        """
        symbol = symbol.upper()
        cache_key = f"{symbol}_{period}"
        
        # Check cache first
        if self._is_cached(cache_key):
            logger.info(f"Returning cached data for {symbol} ({period})")
            return self.cache[cache_key]
        
        logger.info(f"Fetching fresh data for {symbol} ({period})")
        
        try:
            # Get data from yfinance
            ticker = yf.Ticker(symbol)
            
            # Fetch historical price data
            hist_data = ticker.history(period=period)
            if hist_data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Data validation and cleaning pipeline
            if validate:
                logger.info(f"Running data validation pipeline for {symbol}")
                try:
                    clean_data, pipeline_report = self.pipeline.prepare_training_data(hist_data, symbol)
                    
                    # Log validation results
                    if pipeline_report['warnings']:
                        logger.warning(f"Data warnings for {symbol}: {'; '.join(pipeline_report['warnings'][:3])}")
                    
                    logger.info(f"Data pipeline: {pipeline_report['input_rows']} -> {pipeline_report['final_rows']} rows")
                    hist_data = clean_data
                    
                except Exception as validation_error:
                    logger.error(f"Data validation failed for {symbol}: {validation_error}")
                    logger.warning(f"Proceeding with raw data for {symbol}")
                    # Continue with raw data if validation fails
            
            # Get company information
            info = ticker.info
            
            # Create StockData object
            stock_data = StockData(
                symbol=symbol,
                prices=hist_data,
                info=info,
                last_updated=datetime.now()
            )
            
            # Cache the data
            self.cache[cache_key] = stock_data
            
            logger.info(f"Successfully fetched data for {symbol}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_real_time_price(self, symbol: str) -> Dict:
        """
        Get real-time price data for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with current price information
        """
        symbol = symbol.upper()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get real-time data
            data = ticker.history(period="1d", interval="1m")
            if data.empty:
                raise ValueError(f"No real-time data available for {symbol}")
            
            latest = data.iloc[-1]
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'previous_close': float(info.get('previousClose', 0)),
                'change': float(latest['Close'] - info.get('previousClose', latest['Close'])),
                'change_percent': float((latest['Close'] - info.get('previousClose', latest['Close'])) / info.get('previousClose', latest['Close']) * 100),
                'timestamp': latest.name
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            raise
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, StockData]:
        """
        Get data for multiple stocks efficiently
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period for historical data
            
        Returns:
            Dictionary mapping symbols to StockData objects
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(symbol, period)
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get comprehensive market data including technical indicators
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with market data and basic technical indicators
        """
        stock_data = self.get_stock_data(symbol)
        prices = stock_data.prices
        
        # Calculate basic technical indicators
        prices['SMA_20'] = prices['Close'].rolling(window=20).mean()
        prices['SMA_50'] = prices['Close'].rolling(window=50).mean()
        prices['EMA_12'] = prices['Close'].ewm(span=12).mean()
        prices['EMA_26'] = prices['Close'].ewm(span=26).mean()
        
        # MACD
        prices['MACD'] = prices['EMA_12'] - prices['EMA_26']
        prices['MACD_Signal'] = prices['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = prices['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        prices['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        prices['BB_Middle'] = prices['Close'].rolling(window=20).mean()
        bb_std = prices['Close'].rolling(window=20).std()
        prices['BB_Upper'] = prices['BB_Middle'] + (bb_std * 2)
        prices['BB_Lower'] = prices['BB_Middle'] - (bb_std * 2)
        
        return {
            'symbol': symbol,
            'current_price': float(prices['Close'].iloc[-1]),
            'sma_20': float(prices['SMA_20'].iloc[-1]) if not pd.isna(prices['SMA_20'].iloc[-1]) else None,
            'sma_50': float(prices['SMA_50'].iloc[-1]) if not pd.isna(prices['SMA_50'].iloc[-1]) else None,
            'rsi': float(prices['RSI'].iloc[-1]) if not pd.isna(prices['RSI'].iloc[-1]) else None,
            'macd': float(prices['MACD'].iloc[-1]) if not pd.isna(prices['MACD'].iloc[-1]) else None,
            'bb_position': self._calculate_bb_position(prices),
            'volume_avg': float(prices['Volume'].rolling(window=20).mean().iloc[-1]),
            'volatility': float(prices['Close'].pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252)),
            'price_data': prices
        }
    
    def _calculate_bb_position(self, prices: pd.DataFrame) -> float:
        """Calculate where current price sits within Bollinger Bands"""
        try:
            current_price = prices['Close'].iloc[-1]
            bb_upper = prices['BB_Upper'].iloc[-1]
            bb_lower = prices['BB_Lower'].iloc[-1]
            
            if pd.isna(bb_upper) or pd.isna(bb_lower):
                return 0.5
            
            # Return position as percentage (0 = lower band, 1 = upper band)
            return (current_price - bb_lower) / (bb_upper - bb_lower)
        except:
            return 0.5
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if symbol data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        time_diff = datetime.now() - cached_data.last_updated
        
        return time_diff.total_seconds() < (self.cache_expiry * 60)
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cleared data cache")
    
    def save_data(self, symbol: str, filepath: Optional[Path] = None):
        """Save stock data to CSV file"""
        if symbol not in self.cache:
            raise ValueError(f"No cached data for {symbol}")
        
        if filepath is None:
            filepath = Path(f"data/{symbol}_data.csv")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        stock_data = self.cache[symbol]
        stock_data.prices.to_csv(filepath)
        
        logger.info(f"Saved {symbol} data to {filepath}")


# Global instance
data_collector = StockDataCollector()
