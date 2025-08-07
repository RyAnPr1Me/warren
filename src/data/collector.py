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
    last_updated: Optional[datetime] = None
    
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
        
    def get_stock_data(self, symbol: str, period: str = "2y", validate: bool = True, mega_data: bool = False) -> StockData:
        """
        Get comprehensive stock data for a given symbol
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period for historical data ('1y', '2y', '5y', 'max')
            validate: Whether to run data validation pipeline
            mega_data: If True, collect maximum data from ALL available APIs
            
        Returns:
            StockData object containing price history and company info
        """
        symbol = symbol.upper()
        cache_key = f"{symbol}_{period}_{'mega' if mega_data else 'std'}"
        
        # Check cache first
        if self._is_cached(cache_key):
            logger.info(f"Returning cached data for {symbol} ({period})")
            return self.cache[cache_key]
        
        if mega_data:
            logger.info(f"🚀 MEGA DATA MODE: Fetching comprehensive data from ALL APIs for {symbol}")
            return self._get_mega_data(symbol, period, validate)
        
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
        
        # RSI - Fix pandas series comparison with proper numeric handling
        delta = prices['Close'].astype(float).diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
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
    
    def _get_mega_data(self, symbol: str, period: str, validate: bool = True) -> StockData:
        """
        Comprehensive data collection from ALL available APIs for maximum training effectiveness
        
        This method fetches data from:
        - Yahoo Finance (base historical data)
        - Alpha Vantage (enhanced historical + fundamentals)
        - FMP (financial metrics + earnings)
        - Finnhub (news sentiment + market data)
        
        Args:
            symbol: Stock ticker symbol
            period: Time period ('max' for maximum available data)
            validate: Whether to run validation pipeline
            
        Returns:
            Enhanced StockData with comprehensive features from all APIs
        """
        logger.info(f"🚀 MEGA DATA: Starting comprehensive data collection for {symbol}")
        
        # Force maximum period for mega data
        if period != 'max':
            period = 'max'
            logger.info(f"🔥 MEGA DATA: Forcing period to 'max' for maximum historical coverage")
        
        # 1. Start with Yahoo Finance as the base (most reliable historical data)
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                raise ValueError(f"No base data found for symbol {symbol}")
            
            logger.info(f"📊 Yahoo Finance: Retrieved {len(hist_data)} days of base data")
            
            # Get company info
            info = ticker.info
            
        except Exception as e:
            logger.error(f"Failed to get base data from Yahoo Finance: {e}")
            raise
        
        # 2. Enhance with Alpha Vantage data (if API key available)
        try:
            alpha_data = self._get_alpha_vantage_data(symbol)
            if alpha_data is not None:
                hist_data = self._merge_alpha_vantage_data(hist_data, alpha_data)
                logger.info(f"🔗 Alpha Vantage: Enhanced with additional data")
        except Exception as e:
            logger.warning(f"Alpha Vantage data unavailable: {e}")
        
        # 3. Enhance with FMP financial data (if API key available)
        try:
            fmp_data = self._get_fmp_data(symbol)
            if fmp_data is not None:
                hist_data = self._merge_fmp_data(hist_data, fmp_data)
                logger.info(f"💰 FMP: Enhanced with financial metrics")
        except Exception as e:
            logger.warning(f"FMP data unavailable: {e}")
        
        # 4. Enhance with Finnhub news/sentiment (if API key available)
        try:
            finnhub_data = self._get_finnhub_data(symbol)
            if finnhub_data is not None:
                hist_data = self._merge_finnhub_data(hist_data, finnhub_data)
                logger.info(f"📰 Finnhub: Enhanced with sentiment data")
        except Exception as e:
            logger.warning(f"Finnhub data unavailable: {e}")
        
        # 5. Run comprehensive feature engineering for mega data
        hist_data = self._enhance_mega_features(hist_data, symbol)
        
        # 6. Data validation and cleaning
        if validate:
            logger.info(f"🔍 Running enhanced validation pipeline for mega data")
            try:
                clean_data, pipeline_report = self.pipeline.prepare_training_data(hist_data, symbol)
                
                if pipeline_report['warnings']:
                    logger.warning(f"Mega data warnings: {'; '.join(pipeline_report['warnings'][:3])}")
                
                logger.info(f"📈 Mega data pipeline: {pipeline_report['input_rows']} -> {pipeline_report['final_rows']} rows")
                hist_data = clean_data
                
            except Exception as validation_error:
                logger.error(f"Mega data validation failed: {validation_error}")
                logger.warning(f"Proceeding with raw mega data")
        
        # Create enhanced StockData object
        stock_data = StockData(
            symbol=symbol,
            prices=hist_data,
            info=info,
            last_updated=datetime.now()
        )
        
        # Cache with mega flag
        cache_key = f"{symbol}_{period}_mega"
        self.cache[cache_key] = stock_data
        
        logger.info(f"✅ MEGA DATA: Successfully collected comprehensive dataset with {len(hist_data)} rows and {len(hist_data.columns)} features")
        
        return stock_data
    
    def _get_alpha_vantage_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from Alpha Vantage API (placeholder - implement if API key available)"""
        if not config.api.alpha_vantage_key:
            return None
        
        # TODO: Implement Alpha Vantage API calls for additional historical data
        logger.info(f"Alpha Vantage API not yet implemented")
        return None
    
    def _get_fmp_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get financial data from FMP API (placeholder - implement if API key available)"""
        if not config.api.fmp_key:
            return None
        
        # TODO: Implement FMP API calls for financial metrics
        logger.info(f"FMP API not yet implemented")
        return None
    
    def _get_finnhub_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get news/sentiment data from Finnhub API (placeholder - implement if API key available)"""
        if not config.api.finnhub_key:
            return None
        
        # TODO: Implement Finnhub API calls for news sentiment
        logger.info(f"Finnhub API not yet implemented")
        return None
    
    def _merge_alpha_vantage_data(self, base_data: pd.DataFrame, alpha_data: pd.DataFrame) -> pd.DataFrame:
        """Merge Alpha Vantage data with base dataset"""
        # TODO: Implement smart merging of Alpha Vantage data
        return base_data
    
    def _merge_fmp_data(self, base_data: pd.DataFrame, fmp_data: pd.DataFrame) -> pd.DataFrame:
        """Merge FMP financial data with base dataset"""
        # TODO: Implement smart merging of FMP data
        return base_data
    
    def _merge_finnhub_data(self, base_data: pd.DataFrame, finnhub_data: pd.DataFrame) -> pd.DataFrame:
        """Merge Finnhub sentiment data with base dataset"""
        # TODO: Implement smart merging of Finnhub data
        return base_data
    
    def _enhance_mega_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply comprehensive feature engineering for mega data collection"""
        logger.info(f"🧠 Applying mega feature engineering...")
        
        # Advanced technical indicators
        data = self._add_advanced_technical_indicators(data)
        
        # Market regime detection
        data = self._add_market_regime_features(data)
        
        # Volatility clustering features
        data = self._add_volatility_features(data)
        
        # Cycle and seasonality features
        data = self._add_temporal_features(data)
        
        logger.info(f"✨ Enhanced with {len(data.columns)} total features")
        return data
    
    def _add_advanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators for mega data"""
        # Bollinger Bands
        window = 20
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        data['bb_upper'] = rolling_mean + (rolling_std * 2)
        data['bb_lower'] = rolling_mean - (rolling_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / rolling_mean
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Williams %R
        data['williams_r'] = ((data['High'].rolling(window=14).max() - data['Close']) / 
                             (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())) * -100
        
        return data
    
    def _add_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        # Trend strength
        data['trend_strength'] = data['Close'].rolling(window=20).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        # Market state (bull/bear market indicator)
        data['sma_50'] = data['Close'].rolling(window=50).mean()
        data['sma_200'] = data['Close'].rolling(window=200).mean()
        data['bull_market'] = (data['sma_50'] > data['sma_200']).astype(int)
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility clustering and regime features"""
        # Calculate returns
        data['returns'] = data['Close'].pct_change()
        
        # Realized volatility
        data['realized_vol'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # GARCH-like volatility
        data['vol_regime'] = data['realized_vol'].rolling(window=50).quantile(0.8)
        data['high_vol_regime'] = (data['realized_vol'] > data['vol_regime']).astype(int)
        
        return data
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based and seasonal features"""
        data.index = pd.to_datetime(data.index)
        
        # Day of week effect
        data['day_of_week'] = data.index.dayofweek
        data['is_monday'] = (data['day_of_week'] == 0).astype(int)
        data['is_friday'] = (data['day_of_week'] == 4).astype(int)
        
        # Month effects
        data['month'] = data.index.month
        data['is_january'] = (data['month'] == 1).astype(int)
        data['is_december'] = (data['month'] == 12).astype(int)
        
        # Quarter effects
        data['quarter'] = data.index.quarter
        data['is_earnings_season'] = data['quarter'].isin([1, 4]).astype(int)
        
        return data
    
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
