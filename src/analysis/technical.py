"""
Technical Analysis Engine
Provides technical indicators and analysis for stock market data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta

from ..data.collector import StockDataCollector

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Technical analysis engine for stock market data
    Provides various technical indicators and signals
    """
    
    def __init__(self):
        self.data_collector = StockDataCollector()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Series of closing prices
            period: Period for RSI calculation (default: 14)
            
        Returns:
            Series with RSI values
        """
        delta = prices.astype(float).diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            Dictionary with MACD, Signal, and Histogram series
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Series of closing prices
            period: Period for moving average (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    def calculate_moving_averages(self, prices: pd.Series, periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calculate multiple Simple Moving Averages
        
        Args:
            prices: Series of closing prices
            periods: List of periods for SMA calculation
            
        Returns:
            Dictionary with SMA series for each period
        """
        smas = {}
        for period in periods:
            smas[f'sma_{period}'] = prices.rolling(window=period).mean()
        
        return smas
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume indicators
        """
        volume = data['Volume']
        close = data['Close']
        
        # Volume Moving Average
        volume_ma = volume.rolling(window=20).mean()
        
        # Volume Rate of Change
        volume_roc = volume.pct_change(periods=10) * 100
        
        # On-Balance Volume (OBV) - Fix pandas comparison
        close_diff = close.astype(float).diff()
        obv = (volume * ((close_diff > 0).astype(int) - (close_diff < 0).astype(int))).cumsum()
        
        return {
            'volume_ma': volume_ma,
            'volume_roc': volume_roc,
            'obv': obv
        }
    
    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> Dict[str, float]:
        """
        Calculate volatility metrics
        
        Args:
            prices: Series of closing prices
            period: Period for volatility calculation
            
        Returns:
            Dictionary with volatility metrics
        """
        returns = prices.pct_change().dropna()
        
        # Historical volatility (annualized)
        historical_vol = returns.rolling(window=period).std().iloc[-1] * np.sqrt(252) * 100
        
        # Average True Range approximation
        price_changes = prices.diff().abs()
        atr_approx = price_changes.rolling(window=period).mean().iloc[-1]
        
        return {
            'historical_volatility': float(historical_vol) if not pd.isna(historical_vol) else 0.0,
            'atr_approximation': float(atr_approx) if not pd.isna(atr_approx) else 0.0,
            'recent_volatility': float(returns.tail(period).std() * np.sqrt(252) * 100)
        }
    
    def get_technical_signals(self, symbol: str) -> Dict:
        """
        Generate technical analysis signals for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with technical signals and recommendations
        """
        # Get stock data
        stock_data = self.data_collector.get_stock_data(symbol, "1y")
        data = stock_data.prices
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate indicators
        close_prices = data['Close']
        
        # RSI
        rsi = self.calculate_rsi(close_prices)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        macd_data = self.calculate_macd(close_prices)
        current_macd = macd_data['macd'].iloc[-1] if not pd.isna(macd_data['macd'].iloc[-1]) else 0
        current_signal = macd_data['signal'].iloc[-1] if not pd.isna(macd_data['signal'].iloc[-1]) else 0
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(close_prices)
        current_price = close_prices.iloc[-1]
        bb_upper = bb_data['upper'].iloc[-1] if not pd.isna(bb_data['upper'].iloc[-1]) else current_price
        bb_lower = bb_data['lower'].iloc[-1] if not pd.isna(bb_data['lower'].iloc[-1]) else current_price
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # Moving Averages
        sma_data = self.calculate_moving_averages(close_prices)
        sma_20 = sma_data['sma_20'].iloc[-1] if not pd.isna(sma_data['sma_20'].iloc[-1]) else current_price
        sma_50 = sma_data['sma_50'].iloc[-1] if not pd.isna(sma_data['sma_50'].iloc[-1]) else current_price
        
        # Volume indicators
        volume_data = self.calculate_volume_indicators(data)
        current_volume = data['Volume'].iloc[-1]
        avg_volume = volume_data['volume_ma'].iloc[-1] if not pd.isna(volume_data['volume_ma'].iloc[-1]) else current_volume
        
        # Volatility
        volatility = self.calculate_volatility(close_prices)
        
        # Generate signals
        signals = self._generate_signals(
            current_price, current_rsi, current_macd, current_signal,
            bb_position, sma_20, sma_50, current_volume, avg_volume
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'indicators': {
                'rsi': float(current_rsi),
                'macd': float(current_macd),
                'macd_signal': float(current_signal),
                'bb_position': float(bb_position),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'volume_ratio': float(current_volume / avg_volume) if avg_volume > 0 else 1.0
            },
            'volatility': volatility,
            'signals': signals,
            'overall_signal': self._calculate_overall_signal(signals)
        }
    
    def _generate_signals(self, price: float, rsi: float, macd: float, macd_signal: float,
                         bb_position: float, sma_20: float, sma_50: float,
                         volume: float, avg_volume: float) -> Dict[str, str]:
        """Generate individual technical signals"""
        
        signals = {}
        
        # RSI signals
        if rsi > 70:
            signals['rsi'] = 'Sell'
        elif rsi < 30:
            signals['rsi'] = 'Buy'
        else:
            signals['rsi'] = 'Neutral'
        
        # MACD signals
        if macd > macd_signal:
            signals['macd'] = 'Buy'
        elif macd < macd_signal:
            signals['macd'] = 'Sell'
        else:
            signals['macd'] = 'Neutral'
        
        # Bollinger Bands signals
        if bb_position > 0.8:
            signals['bollinger'] = 'Sell'
        elif bb_position < 0.2:
            signals['bollinger'] = 'Buy'
        else:
            signals['bollinger'] = 'Neutral'
        
        # Moving Average signals
        if price > sma_20 > sma_50:
            signals['trend'] = 'Buy'
        elif price < sma_20 < sma_50:
            signals['trend'] = 'Sell'
        else:
            signals['trend'] = 'Neutral'
        
        # Volume signals
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        if volume_ratio > 1.5:
            signals['volume'] = 'Strong'
        elif volume_ratio > 1.2:
            signals['volume'] = 'Moderate'
        else:
            signals['volume'] = 'Weak'
        
        return signals
    
    def _calculate_overall_signal(self, signals: Dict[str, str]) -> str:
        """Calculate overall signal from individual signals"""
        
        buy_signals = sum(1 for signal in signals.values() if signal == 'Buy')
        sell_signals = sum(1 for signal in signals.values() if signal == 'Sell')
        
        # Weight the signals
        if buy_signals >= 3:
            return 'Strong Buy'
        elif buy_signals >= 2:
            return 'Buy'
        elif sell_signals >= 3:
            return 'Strong Sell'
        elif sell_signals >= 2:
            return 'Sell'
        else:
            return 'Hold'
    
    def get_support_resistance(self, symbol: str, period: str = "6m") -> Dict:
        """
        Identify support and resistance levels
        
        Args:
            symbol: Stock ticker symbol
            period: Period for analysis
            
        Returns:
            Dictionary with support and resistance levels
        """
        stock_data = self.data_collector.get_stock_data(symbol, period)
        data = stock_data.prices
        
        high_prices = data['High']
        low_prices = data['Low']
        close_prices = data['Close']
        
        # Simple support/resistance identification
        # Find local maxima and minima
        recent_highs = high_prices.rolling(window=20, center=True).max()
        recent_lows = low_prices.rolling(window=20, center=True).min()
        
        # Resistance levels (recent highs)
        resistance_levels = []
        for i in range(len(high_prices)):
            if high_prices.iloc[i] == recent_highs.iloc[i]:
                resistance_levels.append(float(high_prices.iloc[i]))
        
        # Support levels (recent lows)
        support_levels = []
        for i in range(len(low_prices)):
            if low_prices.iloc[i] == recent_lows.iloc[i]:
                support_levels.append(float(low_prices.iloc[i]))
        
        # Get most significant levels
        current_price = float(close_prices.iloc[-1])
        
        # Find nearest support and resistance
        support_below = [level for level in support_levels if level < current_price]
        resistance_above = [level for level in resistance_levels if level > current_price]
        
        nearest_support = max(support_below) if support_below else min(support_levels) if support_levels else current_price * 0.95
        nearest_resistance = min(resistance_above) if resistance_above else max(resistance_levels) if resistance_levels else current_price * 1.05
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'all_support_levels': sorted(list(set(support_levels)), reverse=True)[:5],
            'all_resistance_levels': sorted(list(set(resistance_levels)))[:5],
            'distance_to_support': ((current_price - nearest_support) / current_price) * 100,
            'distance_to_resistance': ((nearest_resistance - current_price) / current_price) * 100
        }


# Global instance
technical_analyzer = TechnicalAnalyzer()
