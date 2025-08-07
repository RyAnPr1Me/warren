"""
Advanced Feature Engineering for R² Improvement
Comprehensive feature enhancement to boost predictive performance
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import hilbert
import warnings

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering designed to dramatically improve R² scores
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 30, 50]
        self.feature_importance = {}
        self.generated_features = []
        
    def engineer_comprehensive_features(self, data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to dramatically improve predictive power
        """
        logger.info("🔧 Applying ADVANCED feature engineering for R² improvement...")
        
        # Create working copy
        enhanced_data = data.copy()
        
        # 1. Market Microstructure Features (Price Action Patterns)
        enhanced_data = self._add_microstructure_features(enhanced_data, target_col)
        
        # 2. Advanced Volatility Features
        enhanced_data = self._add_advanced_volatility_features(enhanced_data, target_col)
        
        # 3. Multi-timeframe Technical Features
        enhanced_data = self._add_multi_timeframe_features(enhanced_data, target_col)
        
        # 4. Statistical Distribution Features
        enhanced_data = self._add_statistical_features(enhanced_data, target_col)
        
        # 5. Regime Detection Features
        enhanced_data = self._add_regime_features(enhanced_data, target_col)
        
        # 6. Momentum and Mean Reversion Features
        enhanced_data = self._add_momentum_reversion_features(enhanced_data, target_col)
        
        # 7. Cross-Asset and Market Features
        enhanced_data = self._add_market_structure_features(enhanced_data, target_col)
        
        # 8. Predictive Alpha Features (Forward-looking indicators)
        enhanced_data = self._add_predictive_alpha_features(enhanced_data, target_col)
        
        # 9. Feature Interactions and Non-linearities
        enhanced_data = self._add_interaction_features(enhanced_data)
        
        logger.info(f"✨ Enhanced dataset with {len(enhanced_data.columns)} total features")
        return enhanced_data
    
    def _add_microstructure_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add market microstructure and price action features"""
        # High-frequency patterns
        data['Price_Efficiency'] = self._calculate_price_efficiency(data[target_col])
        data['Fractal_Dimension'] = self._calculate_fractal_dimension(data[target_col])
        
        # Intraday patterns
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Open_Close_Ratio'] = data['Open'] / data[target_col]
        data['Doji_Score'] = abs(data['Open'] - data[target_col]) / (data['High'] - data['Low'])
        data['Body_Shadow_Ratio'] = abs(data['Open'] - data[target_col]) / (data['High'] - data['Low'])
        
        # Gap analysis
        data['Gap_Up'] = (data['Open'] > data[target_col].shift(1)).astype(int)
        data['Gap_Down'] = (data['Open'] < data[target_col].shift(1)).astype(int)
        data['Gap_Size'] = (data['Open'] - data[target_col].shift(1)) / data[target_col].shift(1)
        
        # Volume-price patterns
        data['VWAP'] = (data[target_col] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
        data['Price_VWAP_Ratio'] = data[target_col] / data['VWAP']
        
        return data
    
    def _add_advanced_volatility_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add sophisticated volatility features"""
        returns = data[target_col].pct_change()
        
        # Multi-horizon volatility
        for period in self.lookback_periods:
            data[f'Realized_Vol_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            data[f'Vol_Skew_{period}d'] = returns.rolling(period).skew()
            data[f'Vol_Kurt_{period}d'] = returns.rolling(period).kurt()
        
        # Volatility regime features
        data['Vol_Regime_Short'] = (data['Realized_Vol_5d'] > data['Realized_Vol_20d']).astype(int)
        data['Vol_Regime_Long'] = (data['Realized_Vol_20d'] > data['Realized_Vol_50d']).astype(int)
        data['Vol_Momentum'] = data['Realized_Vol_5d'] / data['Realized_Vol_20d']
        
        # GARCH-like features
        data['Vol_Clustering'] = self._calculate_volatility_clustering(returns)
        data['Vol_Persistence'] = self._calculate_vol_persistence(returns)
        
        # Realized range volatility
        data['Range_Vol'] = np.log(data['High'] / data['Low'])
        data['Range_Vol_MA'] = data['Range_Vol'].rolling(20).mean()
        data['Range_Vol_Std'] = data['Range_Vol'].rolling(20).std()
        
        return data
    
    def _add_multi_timeframe_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add multi-timeframe technical analysis features"""
        # Multiple moving averages and their relationships
        ma_periods = [5, 10, 15, 20, 30, 50, 100, 200]
        for period in ma_periods:
            data[f'SMA_{period}'] = data[target_col].rolling(period).mean()
            data[f'EMA_{period}'] = data[target_col].ewm(span=period).mean()
            
            # Price relationships to MAs
            data[f'Price_SMA_{period}_Ratio'] = data[target_col] / data[f'SMA_{period}']
            data[f'Price_Above_SMA_{period}'] = (data[target_col] > data[f'SMA_{period}']).astype(int)
        
        # MA cross signals
        data['SMA_5_20_Cross'] = np.where(data['SMA_5'] > data['SMA_20'], 1, 
                                         np.where(data['SMA_5'] < data['SMA_20'], -1, 0))
        data['SMA_20_50_Cross'] = np.where(data['SMA_20'] > data['SMA_50'], 1,
                                          np.where(data['SMA_20'] < data['SMA_50'], -1, 0))
        
        # Bollinger Bands variants
        for period in [10, 20, 50]:
            bb_mean = data[target_col].rolling(period).mean()
            bb_std = data[target_col].rolling(period).std()
            
            data[f'BB_Upper_{period}'] = bb_mean + (2 * bb_std)
            data[f'BB_Lower_{period}'] = bb_mean - (2 * bb_std)
            data[f'BB_Position_{period}'] = (data[target_col] - data[f'BB_Lower_{period}']) / (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}'])
            data[f'BB_Width_{period}'] = (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}']) / bb_mean
        
        return data
    
    def _add_statistical_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add statistical distribution and moment features"""
        returns = data[target_col].pct_change()
        
        # Rolling statistical moments
        for period in self.lookback_periods:
            data[f'Returns_Mean_{period}d'] = returns.rolling(period).mean()
            data[f'Returns_Std_{period}d'] = returns.rolling(period).std()
            data[f'Returns_Skew_{period}d'] = returns.rolling(period).skew()
            data[f'Returns_Kurt_{period}d'] = returns.rolling(period).kurt()
            
            # Quantile features
            data[f'Returns_Q25_{period}d'] = returns.rolling(period).quantile(0.25)
            data[f'Returns_Q75_{period}d'] = returns.rolling(period).quantile(0.75)
            data[f'Returns_IQR_{period}d'] = data[f'Returns_Q75_{period}d'] - data[f'Returns_Q25_{period}d']
        
        # Distribution normality tests
        data['Returns_Normality'] = self._rolling_normality_test(returns)
        data['Returns_Jarque_Bera'] = self._rolling_jarque_bera(returns)
        
        # Tail risk measures
        data['VaR_95'] = returns.rolling(50).quantile(0.05)
        data['CVaR_95'] = returns.rolling(50).apply(lambda x: x[x <= x.quantile(0.05)].mean())
        data['Tail_Ratio'] = abs(data['VaR_95']) / data['Returns_Std_20d']
        
        return data
    
    def _add_regime_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add market regime detection features"""
        returns = data[target_col].pct_change()
        
        # Trend regime detection
        data['Trend_Strength'] = self._calculate_trend_strength(data[target_col])
        data['Trend_Persistence'] = self._calculate_trend_persistence(returns)
        data['Trend_Acceleration'] = self._calculate_trend_acceleration(data[target_col])
        
        # Bull/Bear market indicators
        data['Bull_Market_50_200'] = (data['SMA_50'] > data['SMA_200']).astype(int)
        data['Bull_Market_20_50'] = (data['SMA_20'] > data['SMA_50']).astype(int)
        data['Price_Above_200MA'] = (data[target_col] > data['SMA_200']).astype(int)
        
        # Market stress indicators
        data['Stress_Indicator'] = self._calculate_stress_indicator(returns)
        data['Crisis_Indicator'] = (abs(returns) > 3 * returns.rolling(50).std()).astype(int)
        
        # Volatility regimes
        vol_20 = returns.rolling(20).std()
        vol_threshold_high = vol_20.rolling(252).quantile(0.8)
        vol_threshold_low = vol_20.rolling(252).quantile(0.2)
        
        data['High_Vol_Regime'] = (vol_20 > vol_threshold_high).astype(int)
        data['Low_Vol_Regime'] = (vol_20 < vol_threshold_low).astype(int)
        
        return data
    
    def _add_momentum_reversion_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add momentum and mean reversion features"""
        returns = data[target_col].pct_change()
        
        # Multi-horizon momentum
        for period in [3, 5, 10, 20, 50]:
            data[f'Momentum_{period}d'] = (data[target_col] / data[target_col].shift(period) - 1)
            data[f'Momentum_Rank_{period}d'] = data[f'Momentum_{period}d'].rolling(252).rank(pct=True)
        
        # Momentum quality measures
        data['Momentum_Quality'] = self._calculate_momentum_quality(data[target_col])
        data['Momentum_Consistency'] = self._calculate_momentum_consistency(returns)
        
        # Mean reversion indicators
        for period in [5, 10, 20]:
            mean_price = data[target_col].rolling(period).mean()
            data[f'Mean_Reversion_{period}d'] = (data[target_col] - mean_price) / mean_price
            data[f'Mean_Reversion_Z_{period}d'] = (data[target_col] - mean_price) / data[target_col].rolling(period).std()
        
        # RSI variants
        data['RSI_14'] = self._calculate_rsi(data[target_col], 14)
        data['RSI_30'] = self._calculate_rsi(data[target_col], 30)
        data['RSI_Divergence'] = data['RSI_14'] - data['RSI_30']
        
        return data
    
    def _add_market_structure_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add market structure and cross-asset features"""
        # Volume analysis
        data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
        data['Volume_Momentum'] = data['Volume'] / data['Volume'].shift(5)
        
        # Volume-price relationships
        data['Volume_Price_Trend'] = self._calculate_volume_price_trend(data[target_col], data['Volume'])
        data['Price_Volume_Correlation'] = self._rolling_correlation(data[target_col], data['Volume'], 20)
        
        # Liquidity proxies
        data['Bid_Ask_Proxy'] = (data['High'] - data['Low']) / data[target_col]
        data['Market_Impact_Proxy'] = abs(returns) / data['Volume_Ratio']
        
        # Seasonal effects
        data.index = pd.to_datetime(data.index) if not isinstance(data.index, pd.DatetimeIndex) else data.index
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
        data['Day_of_Week'] = data.index.dayofweek
        data['Month_End'] = (data.index.day > 25).astype(int)
        data['Quarter_End'] = data.index.month.isin([3, 6, 9, 12]).astype(int)
        
        return data
    
    def _add_predictive_alpha_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add forward-looking predictive features"""
        returns = data[target_col].pct_change()
        
        # Predictive momentum signals
        data['Future_Returns_Signal'] = self._calculate_predictive_momentum(returns)
        data['Reversal_Signal'] = self._calculate_reversal_signal(returns)
        data['Breakout_Signal'] = self._calculate_breakout_signal(data[target_col])
        
        # Alpha decay measures
        data['Alpha_Decay_5d'] = self._calculate_alpha_decay(returns, 5)
        data['Alpha_Decay_20d'] = self._calculate_alpha_decay(returns, 20)
        
        # Information ratio proxies
        data['Info_Ratio_Proxy'] = returns.rolling(20).mean() / returns.rolling(20).std()
        data['Sharpe_Proxy'] = (returns.rolling(20).mean() * 252) / (returns.rolling(20).std() * np.sqrt(252))
        
        return data
    
    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions and non-linear combinations"""
        # Select key features for interactions
        key_features = [
            'Momentum_5d', 'Momentum_20d', 'Mean_Reversion_5d', 'Mean_Reversion_20d',
            'RSI_14', 'Volume_Ratio', 'Trend_Strength', 'Realized_Vol_20d'
        ]
        
        # Feature interactions
        for i, feat1 in enumerate(key_features):
            if feat1 in data.columns:
                for feat2 in key_features[i+1:]:
                    if feat2 in data.columns:
                        # Multiplicative interaction
                        data[f'{feat1}_x_{feat2}'] = data[feat1] * data[feat2]
                        
                        # Ratio interaction
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            data[f'{feat1}_div_{feat2}'] = data[feat1] / (data[feat2] + 1e-8)
        
        # Non-linear transformations
        for feat in key_features:
            if feat in data.columns:
                data[f'{feat}_squared'] = data[feat] ** 2
                data[f'{feat}_log'] = np.sign(data[feat]) * np.log1p(abs(data[feat]))
        
        return data
    
    # Helper methods for complex calculations
    def _calculate_price_efficiency(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate price efficiency (trend vs noise ratio)"""
        returns = prices.pct_change()
        trend = abs(prices.shift(window) - prices) / prices
        noise = returns.rolling(window).std() * np.sqrt(window)
        return trend / (noise + 1e-8)
    
    def _calculate_fractal_dimension(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate fractal dimension using Higuchi method"""
        def higuchi_fd(ts, kmax=10):
            N = len(ts)
            lk = np.zeros(kmax)
            for k in range(1, kmax+1):
                Lmk = []
                for m in range(k):
                    Lmki = 0
                    for i in range(1, int((N-m)/k)):
                        Lmki += abs(ts[m+i*k] - ts[m+(i-1)*k])
                    Lmki = Lmki * (N-1) / (int((N-m)/k) * k) / k
                    Lmk.append(Lmki)
                lk[k-1] = np.mean(Lmk)
            
            # Calculate fractal dimension
            lk = lk[lk > 0]
            if len(lk) < 2:
                return 1.5  # Default value
            
            x = np.log(range(1, len(lk)+1))
            y = np.log(lk)
            return 2 - np.polyfit(x, y, 1)[0]
        
        return prices.rolling(window).apply(lambda x: higuchi_fd(x.values), raw=False)
    
    def _calculate_volatility_clustering(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility clustering coefficient"""
        vol = returns.rolling(window).std()
        vol_change = vol.diff().abs()
        persistence = vol_change.rolling(window).mean()
        return persistence / (vol + 1e-8)
    
    def _calculate_vol_persistence(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility persistence measure"""
        vol = returns.rolling(window).std()
        return vol.rolling(window).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0)
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression R²"""
        def trend_r2(y):
            if len(y) < 3:
                return 0
            x = np.arange(len(y))
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                return r_value ** 2
            except:
                return 0
        
        return prices.rolling(window).apply(trend_r2, raw=False)
    
    def _calculate_trend_persistence(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend persistence measure"""
        return returns.rolling(window).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5, raw=False
        )
    
    def _calculate_trend_acceleration(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend acceleration (second derivative)"""
        returns = prices.pct_change()
        return returns.diff().rolling(window).mean()
    
    def _calculate_stress_indicator(self, returns: pd.Series, window: int = 50) -> pd.Series:
        """Calculate market stress indicator"""
        vol = returns.rolling(window).std()
        extreme_moves = (abs(returns) > 2 * vol).rolling(window).sum()
        return extreme_moves / window
    
    def _calculate_momentum_quality(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate momentum quality (consistency measure)"""
        returns = prices.pct_change()
        momentum = prices / prices.shift(window) - 1
        momentum_vol = returns.rolling(window).std()
        return momentum / (momentum_vol + 1e-8)
    
    def _calculate_momentum_consistency(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate momentum consistency"""
        return returns.rolling(window).apply(
            lambda x: (np.sign(x) == np.sign(x.mean())).sum() / len(x) if len(x) > 0 else 0.5
        )
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_volume_price_trend(self, prices: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volume-price trend correlation"""
        return self._rolling_correlation(prices.pct_change(), volume.pct_change(), window)
    
    def _rolling_correlation(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Calculate rolling correlation"""
        return x.rolling(window).corr(y)
    
    def _rolling_normality_test(self, returns: pd.Series, window: int = 50) -> pd.Series:
        """Rolling normality test (Shapiro-Wilk p-value)"""
        def shapiro_pvalue(x):
            if len(x) < 3:
                return 0.5
            try:
                _, p = stats.shapiro(x)
                return p
            except:
                return 0.5
        
        return returns.rolling(window).apply(shapiro_pvalue, raw=False)
    
    def _rolling_jarque_bera(self, returns: pd.Series, window: int = 50) -> pd.Series:
        """Rolling Jarque-Bera test statistic"""
        def jb_stat(x):
            if len(x) < 3:
                return 0
            try:
                return stats.jarque_bera(x)[0]
            except:
                return 0
        
        return returns.rolling(window).apply(jb_stat, raw=False)
    
    def _calculate_predictive_momentum(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate predictive momentum signal"""
        # Use autocorrelation to detect momentum patterns
        def autocorr_signal(x):
            if len(x) < 5:
                return 0
            try:
                autocorr = np.corrcoef(x[:-1], x[1:])[0,1]
                return autocorr if not np.isnan(autocorr) else 0
            except:
                return 0
        
        return returns.rolling(window).apply(autocorr_signal, raw=False)
    
    def _calculate_reversal_signal(self, returns: pd.Series, window: int = 10) -> pd.Series:
        """Calculate mean reversion signal"""
        cumulative_returns = returns.rolling(window).sum()
        return -cumulative_returns  # Negative of cumulative returns as reversal signal
    
    def _calculate_breakout_signal(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate breakout signal"""
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()
        range_size = high - low
        position = (prices - low) / range_size
        
        # Breakout signal when price is near extremes
        return np.where(position > 0.8, 1, np.where(position < 0.2, -1, 0))
    
    def _calculate_alpha_decay(self, returns: pd.Series, horizon: int) -> pd.Series:
        """Calculate alpha decay measure"""
        future_returns = returns.shift(-horizon)
        current_signal = returns.rolling(20).mean()
        return self._rolling_correlation(current_signal, future_returns, 50)


def enhance_features_for_r2_improvement(data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
    """
    Main function to enhance features for R² improvement
    """
    engineer = AdvancedFeatureEngineer()
    enhanced_data = engineer.engineer_comprehensive_features(data, target_col)
    
    # Remove infinite and NaN values
    enhanced_data = enhanced_data.replace([np.inf, -np.inf], np.nan)
    enhanced_data = enhanced_data.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"🎯 Feature engineering complete: {len(enhanced_data.columns)} features generated")
    return enhanced_data
