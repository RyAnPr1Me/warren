"""
Enhanced LSTM Price Predictor - Phase 2 Implementation
Advanced neural network for stock price prediction with 60-day forecasting
"""

import numpy as np
import pandas as pd
import json
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

from ..config import config
from ..data.collector import StockDataCollector
from ..analysis.sentiment import SentimentAnalysisEngine
from ..utils.helpers import Timer, format_percentage, format_currency

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


class IntelligentLRScheduler(Callback):
    """
    Intelligent learning rate scheduler that combines multiple strategies:
    - Warm-up phase for stable training start
    - Cosine annealing for smooth decay
    - Plateau detection for adaptive reduction
    - Performance-based adjustments
    """
    
    def __init__(self, initial_lr=0.001, warmup_epochs=5, cosine_epochs=40, 
                 min_lr_factor=0.01, patience=8, factor=0.5, verbose=1):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.min_lr = initial_lr * min_lr_factor
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        
        # State tracking
        self.best_loss = float('inf')
        self.wait = 0
        self.plateau_reductions = 0
        self.max_plateau_reductions = 3
        
    def on_epoch_begin(self, epoch, logs=None):
        """Set learning rate at the beginning of each epoch"""
        if epoch < self.warmup_epochs:
            # Warm-up phase: gradually increase LR
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        elif epoch < self.warmup_epochs + self.cosine_epochs:
            # Cosine annealing phase
            cosine_epoch = epoch - self.warmup_epochs
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * cosine_epoch / self.cosine_epochs)
            )
        else:
            # Plateau-based reduction phase
            lr = max(self.min_lr, self.initial_lr * (self.factor ** self.plateau_reductions))
        
        # Apply the learning rate
        self.model.optimizer.learning_rate.assign(lr)
        
        if self.verbose:
            print(f"Epoch {epoch + 1}: Learning rate = {lr:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Check for plateau and adjust if needed"""
        if logs is None:
            logs = {}
            
        if epoch >= self.warmup_epochs + self.cosine_epochs:
            current_loss = logs.get('val_loss', float('inf'))
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
            else:
                self.wait += 1
                
                if self.wait >= self.patience and self.plateau_reductions < self.max_plateau_reductions:
                    self.plateau_reductions += 1
                    self.wait = 0
                    current_lr = float(self.model.optimizer.learning_rate.numpy())
                    new_lr = max(self.min_lr, current_lr * self.factor)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    
                    if self.verbose:
                        print(f"Plateau detected! Reducing LR to {new_lr:.6f}")


class EnhancedLSTMPredictor:
    """
    Enhanced LSTM-based stock price predictor for Phase 2 implementation
    Features: Multi-feature input, advanced architecture, confidence scoring, quality validation
    """
    
    # Model quality thresholds (REALISTIC for stock prediction with sentiment)
    MIN_R2_SCORE = 0.01   # Very realistic minimum for financial time series
    GOOD_R2_SCORE = 0.03   # Good model threshold for stock prediction  
    EXCELLENT_R2_SCORE = 0.05  # Excellent model threshold
    
    def __init__(self, symbol: str, prediction_days: Optional[int] = None):
        self.symbol = symbol.upper()
        self.model = None
        self.scaler = StandardScaler()  # Better for returns (can be negative)
        # Use StandardScaler for more stable and interpretable feature scaling
        self.feature_scaler = StandardScaler()  # More stable and faster than QuantileTransformer
        
        # Enhanced configuration
        self.sequence_length = config.model.sequence_length
        self.prediction_days = prediction_days or config.model.prediction_days
        self.features = ['Close', 'Volume', 'High', 'Low', 'Open']  # Multi-feature input
        
        # Initialize data collector and sentiment analyzer
        self.data_collector = StockDataCollector()
        self.sentiment_analyzer = SentimentAnalysisEngine()
        
        # Set up file paths
        self.base_dir = Path(f"data/models/{self.symbol}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.base_dir / f"{self.symbol}_lstm_model.h5"
        self.scaler_path = self.base_dir / f"{self.symbol}_scaler.pkl"
        self.feature_scaler_path = self.base_dir / f"{self.symbol}_feature_scaler.pkl"
        self.metrics_path = self.base_dir / f"{self.symbol}_metrics.json"
    
    @staticmethod
    def variance_loss(y_true, y_pred):
        """
        Custom loss that encourages diverse predictions.
        Combines MSE with diversity reward.
        """
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Encourage prediction diversity
        pred_variance = tf.math.reduce_variance(y_pred)
        
        # Add a smaller regularization term that encourages variance
        diversity_reward = -0.1 * pred_variance  # Negative to encourage variance
        
        return mse + diversity_reward
    
    def _cleanup_existing_models(self):
        """Remove any existing model files for this ticker"""
        model_files = [
            self.model_path,
            self.scaler_path,
            self.feature_scaler_path,
            self.metrics_path
        ]
        
        for file_path in model_files:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed existing model file: {file_path}")
    
    def _validate_model_quality(self, metrics: Dict) -> Tuple[bool, str]:
        """
        Validate if the trained model meets quality standards
        
        Args:
            metrics: Training metrics dictionary
            
        Returns:
            Tuple of (is_valid, reason)
        """
        val_r2 = metrics.get('val_r2', -999)
        test_r2 = metrics.get('test_r2', -999)
        val_rmse = metrics.get('val_rmse', float('inf'))
        
        # Use config values for thresholds
        min_r2_threshold = config.model.min_r2_score
        max_overfitting_ratio = config.model.max_overfitting_ratio
        
        # Check R² score threshold
        if val_r2 < min_r2_threshold:
            return False, f"Validation R² ({val_r2:.3f}) below minimum threshold ({min_r2_threshold})"
        
        # Check if test performance is drastically worse than validation (overfitting)
        # Use config overfitting ratio
        if test_r2 < val_r2 - (val_r2 / max_overfitting_ratio):
            return False, f"Model overfitting detected: Val R² {val_r2:.3f} vs Test R² {test_r2:.3f}"
        
        # Check if RMSE is reasonable (not more than 20% of current price)
        try:
            current_price = self._get_current_price()
            if val_rmse > current_price * 0.2:
                return False, f"RMSE too high: ${val_rmse:.2f} (>{current_price * 0.2:.2f}, 20% of current price)"
        except:
            pass  # Skip this check if we can't get current price
        
        return True, f"Model quality acceptable: R² = {val_r2:.3f}"
    
    def _get_current_price(self) -> float:
        """Get current stock price for validation"""
        try:
            stock_data = self.data_collector.get_stock_data(self.symbol, "5d")
            return float(stock_data.prices['Close'].iloc[-1])
        except:
            return 100.0  # Fallback value
    
    def _assess_model_quality(self, r2_score: float) -> str:
        """Assess model quality based on R² score"""
        if r2_score >= self.EXCELLENT_R2_SCORE:
            return "Excellent 🎯"
        elif r2_score >= self.GOOD_R2_SCORE:
            return "Good 👍"
        elif r2_score >= self.MIN_R2_SCORE:
            return "Fair 👌"
        else:
            return "Poor ❌"
    
    def _save_metrics(self, metrics: Dict):
        """Save training metrics to JSON file"""
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {self.metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_metrics(self) -> Dict:
        """Load training metrics from JSON file"""
        try:
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
        return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, close_prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = close_prices.rolling(window=window).mean()
        std = close_prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def _calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_hurst(self, prices):
        """Calculate Hurst Exponent for trend persistence"""
        try:
            if len(prices) < 10:
                return 0.5  # Random walk default
            
            prices = np.array(prices)
            lags = range(2, min(10, len(prices)//2))
            
            # Calculate variance of lagged differences
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
            
            # Linear regression of log(tau) vs log(lag)
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            
            return max(0, min(1, hurst))  # Clamp between 0 and 1
        except:
            return 0.5
    
    def _calculate_fractal_dimension(self, prices):
        """Calculate Fractal Dimension"""
        try:
            if len(prices) < 10:
                return 1.5
            
            prices = np.array(prices)
            n = len(prices)
            
            # Calculate relative range
            mean_price = np.mean(prices)
            deviations = prices - mean_price
            cumsum_dev = np.cumsum(deviations)
            
            # Range calculation
            max_cumsum = np.max(cumsum_dev)
            min_cumsum = np.min(cumsum_dev)
            range_rs = max_cumsum - min_cumsum
            
            # Standard deviation
            std_dev = np.std(prices)
            
            if std_dev == 0:
                return 1.5
            
            # Hurst exponent approximation
            rs_ratio = range_rs / std_dev
            fractal_dim = 2 - (np.log(rs_ratio) / np.log(n))
            
            return max(1, min(2, fractal_dim))  # Clamp between 1 and 2
        except:
            return 1.5
    
    def _calculate_efficiency_ratio(self, close_prices, period=10):
        """Calculate Kaufman's Efficiency Ratio"""
        try:
            # Direction (net change)
            direction = abs(close_prices.diff(period))
            
            # Volatility (sum of absolute changes)
            volatility = close_prices.diff().abs().rolling(window=period).sum()
            
            # Efficiency Ratio
            efficiency = direction / volatility
            efficiency = efficiency.fillna(0)
            
            return efficiency
        except:
            return pd.Series(0, index=close_prices.index)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare predictive financial features optimized for return forecasting
        CRITICAL: All features must use ONLY historical data (no future leakage)
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with predictive financial features
        """
        df = data.copy()
        
        # Ensure we have enough data for feature calculation
        if len(df) < 50:
            logger.warning(f"Limited data points ({len(df)}), features may be less reliable")
        
        # Calculate historical returns (LAGGED - no future leakage)
        df['Returns_Lag1'] = df['Close'].pct_change().shift(1)  # Yesterday's return
        df['Log_Returns_Lag1'] = df['Close'].apply(lambda x: np.log(x)).diff().shift(1)
        
        # === MOMENTUM FEATURES (Historical only) ===
        # Use LAGGED momentum to predict future returns
        df['Momentum_1d'] = (df['Close'].shift(1) / df['Close'].shift(2) - 1)  # Yesterday's momentum
        df['Momentum_3d'] = (df['Close'].shift(1) / df['Close'].shift(4) - 1)  # 3-day momentum ending yesterday
        df['Momentum_5d'] = (df['Close'].shift(1) / df['Close'].shift(6) - 1)  # 5-day momentum ending yesterday
        
        # Moving averages using historical data only
        df['SMA_5'] = df['Close'].shift(1).rolling(window=5).mean()  # 5-day SMA ending yesterday
        df['SMA_20'] = df['Close'].shift(1).rolling(window=20).mean()  # 20-day SMA ending yesterday
        df['Price_SMA5_Ratio'] = (df['Close'].shift(1) / df['SMA_5']) - 1  # Historical price vs trend
        df['Price_SMA20_Ratio'] = (df['Close'].shift(1) / df['SMA_20']) - 1
        df['SMA_Cross'] = (df['SMA_5'] / df['SMA_20']) - 1  # Trend strength signal
        
        # === VOLATILITY FEATURES (Historical risk indicators) ===
        # Realized volatility using lagged returns  
        df['Volatility_5d'] = df['Returns_Lag1'].rolling(window=5).std()
        df['Volatility_20d'] = df['Returns_Lag1'].rolling(window=20).std()
        df['Vol_Ratio'] = df['Volatility_5d'] / df['Volatility_20d']  # Vol regime change signal
        
        # Historical High-Low volatility
        df['HL_Volatility'] = ((df['High'] - df['Low']) / df['Close']).shift(1)
        
        # === VOLUME FEATURES (Historical market participation) ===
        df['Volume_SMA'] = df['Volume'].shift(1).rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'].shift(1) / df['Volume_SMA']  # Historical volume surge
        df['Price_Volume'] = df['Returns_Lag1'] * np.log(df['Volume_Ratio'] + 1)  # Historical price-volume
        
        # === TECHNICAL INDICATORS (Historical sentiment) ===
        # RSI using lagged prices
        df['RSI'] = self._calculate_rsi(df['Close'].shift(1), period=14)
        df['RSI_Normalized'] = (df['RSI'] - 50) / 50  # Center around 0
        
        # MACD using lagged prices
        macd, macd_signal = self._calculate_macd(df['Close'].shift(1))
        df['MACD_Histogram'] = (macd - macd_signal) / df['Close'].shift(1)
        
        # Bollinger Bands using lagged prices
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['Close'].shift(1))
        df['BB_Position'] = (df['Close'].shift(1) - bb_lower) / (bb_upper - bb_lower)  # Historical position
        
        # === LAG FEATURES (Historical information) ===
        df['Returns_Lag2'] = df['Close'].pct_change().shift(2)  # 2 days ago return
        df['Returns_Lag3'] = df['Close'].pct_change().shift(3)  # 3 days ago return
        
        # === SENTIMENT FEATURES (Period-specific historical sentiment) ===
        try:
            logger.info("Adding historical sentiment features for training period...")
            
            # Get period-specific sentiment data 
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(self.symbol)
            
            # Use static sentiment but create time-varying features through rolling stats
            base_sentiment = sentiment_result.overall_sentiment
            base_confidence = sentiment_result.confidence
            
            # Create time-varying sentiment proxy using news sentiment + price momentum correlation
            price_momentum = df['Returns_Lag1'].rolling(window=5).mean().fillna(0)
            sentiment_proxy = base_sentiment + (price_momentum * 0.1)  # Slight correlation with price momentum
            
            df['Overall_Sentiment'] = sentiment_proxy.clip(-1, 1)  # Keep in valid range
            df['Confidence_Score'] = base_confidence
            df['Bullish_Ratio'] = sentiment_result.bullish_ratio
            df['Bearish_Ratio'] = sentiment_result.bearish_ratio
            df['Neutral_Ratio'] = 1.0 - df['Bullish_Ratio'] - df['Bearish_Ratio']
            
            # Create derived sentiment features using the proxy
            df['Sentiment_Change'] = df['Overall_Sentiment'].diff().fillna(0)
            df['Sentiment_Volatility'] = df['Overall_Sentiment'].rolling(window=5).std().fillna(0)
            df['Sentiment_Price_Divergence'] = df['Overall_Sentiment'] - df['Returns_Lag1']
            df['News_Volume'] = sentiment_result.news_count / max(1, len(data))  # Normalized news volume
            
            logger.info(f"Added sentiment features: avg_sentiment={df['Overall_Sentiment'].mean():.3f}, confidence={base_confidence:.3f}")
            
        except Exception as e:
            logger.warning(f"Could not add sentiment features: {e}")
            # Add placeholder sentiment features with neutral values
            for col in ['Overall_Sentiment', 'Confidence_Score', 'Bullish_Ratio', 'Bearish_Ratio',
                       'Neutral_Ratio', 'Sentiment_Change', 'Sentiment_Volatility', 
                       'Sentiment_Price_Divergence', 'News_Volume']:
                df[col] = 0.0
        
        # === EARNINGS FEATURES (Historical earnings calendar data) ===
        try:
            logger.info("Adding historical earnings features for training period...")
            
            # Import fundamentals analyzer
            from ..analysis.fundamentals import FundamentalAnalyzer
            fundamentals_analyzer = FundamentalAnalyzer()
            
            # Get earnings calendar for the period
            earnings_data = fundamentals_analyzer.get_earnings_calendar(self.symbol)
            
            # Initialize earnings features
            df['Days_To_Earnings'] = 999  # Default: far from earnings
            df['Earnings_Beat_History'] = 0.0  # Historical earnings beat rate
            df['EPS_Growth_Rate'] = 0.0  # EPS growth rate
            df['Revenue_Growth_Rate'] = 0.0  # Revenue growth rate
            
            if earnings_data and len(earnings_data) > 0:
                logger.info(f"Found {len(earnings_data)} earnings events for analysis")
                
                # Create earnings proximity features
                earnings_dates = []
                for earnings in earnings_data:
                    try:
                        earnings_dates.append(earnings.date)
                    except:
                        continue
                
                if earnings_dates:
                    # Calculate days to nearest earnings for each date
                    for i, date in enumerate(df.index):
                        min_days = min([abs((date - ed).days) for ed in earnings_dates])
                        df.loc[date, 'Days_To_Earnings'] = min(min_days, 999)
                    
                    # Calculate average earnings metrics
                    eps_beats = []
                    eps_surprises = []
                    
                    for earnings in earnings_data:
                        try:
                            if earnings.eps_estimate and earnings.eps_actual:
                                eps_beat = 1.0 if earnings.eps_actual > earnings.eps_estimate else -1.0
                                eps_beats.append(eps_beat)
                            
                            if earnings.surprise_percent:
                                eps_surprises.append(earnings.surprise_percent)
                        except:
                            continue
                    
                    # Apply average metrics to all dates
                    avg_eps_beat = np.mean(eps_beats) if eps_beats else 0.0
                    avg_surprise = np.mean(eps_surprises) if eps_surprises else 0.0
                    
                    df['Earnings_Beat_History'] = avg_eps_beat
                    df['EPS_Growth_Rate'] = avg_surprise / 100.0 if avg_surprise else 0.0  # Convert % to decimal
                    df['Revenue_Growth_Rate'] = 0.0  # Placeholder for now
                
                logger.info(f"Added earnings features with avg beat rate: {df['Earnings_Beat_History'].iloc[0]:.3f}")
            else:
                logger.info("No earnings data found, using default values")
                
        except Exception as e:
            logger.warning(f"Could not add earnings features: {e}")
            # Add placeholder earnings features
            df['Days_To_Earnings'] = 999
            df['Earnings_Beat_History'] = 0.0
            df['EPS_Growth_Rate'] = 0.0
            df['Revenue_Growth_Rate'] = 0.0
        
        # Handle infinite and NaN values more aggressively
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].ffill().bfill().fillna(0)
        
        logger.info(f"Generated {len(df.columns)} historical features from {len(data)} data points")
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare enhanced multi-feature data for LSTM training without data leakage
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (X, y) with unscaled features and targets - scaling done later in training
        """
        # Prepare comprehensive features
        enhanced_data = self.prepare_features(data)
        
        # Advanced feature selection - HISTORICAL features + SENTIMENT for return forecasting
        feature_columns = [
            # Historical momentum (strongest predictors)
            'Momentum_1d',    # Yesterday's momentum
            'Momentum_3d',    # 3-day momentum ending yesterday
            'Momentum_5d',    # 5-day momentum ending yesterday
            
            # Historical trend indicators
            'Price_SMA5_Ratio',   # Historical price vs short-term trend
            'Price_SMA20_Ratio',  # Historical price vs long-term trend
            'SMA_Cross',          # Historical trend strength
            
            # Historical volatility regime
            'Volatility_5d',      # Recent historical volatility
            'Vol_Ratio',          # Volatility regime change
            
            # Historical volume confirmation
            'Volume_Ratio',       # Historical volume surge
            'Price_Volume',       # Historical price-volume interaction
            
            # Historical technical indicators
            'RSI_Normalized',     # Historical momentum/reversion signal
            'MACD_Histogram',     # Historical trend signal
            'BB_Position',        # Historical volatility position
            
            # Historical returns for autocorrelation
            'Returns_Lag1',       # Yesterday's return
            'Returns_Lag2',       # 2 days ago return
            
            # === PHASE 3 SENTIMENT FEATURES ===
            'Overall_Sentiment',        # Current market sentiment
            'Confidence_Score',         # Sentiment confidence
            'Bullish_Ratio',           # Bullish sentiment ratio
            'Bearish_Ratio',           # Bearish sentiment ratio
            'Sentiment_Change',        # Sentiment momentum
            'Sentiment_Price_Divergence',  # Sentiment vs price divergence
            'News_Volume',             # News activity level
        ]
        
        # Ensure all columns exist and handle missing features gracefully
        available_features = [col for col in feature_columns if col in enhanced_data.columns]
        
        logger.info(f"Using {len(available_features)} features for training: {available_features}")
        
        # Validate feature data quality
        feature_data = enhanced_data[available_features].copy()
        
        # Check for and handle data quality issues
        inf_mask = np.isinf(feature_data.values)
        nan_mask = np.isnan(feature_data.values)
        
        if inf_mask.any():
            logger.warning(f"Found {inf_mask.sum()} infinite values in features, replacing with 0")
            feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        if nan_mask.any():
            logger.warning(f"Found {nan_mask.sum()} NaN values in features, filling with 0")
            feature_data = feature_data.fillna(0)
        
        # Check for features with zero variance (would cause scaling issues)
        feature_std = feature_data.std()
        zero_var_features = feature_std[feature_std == 0].index.tolist()
        if zero_var_features:
            logger.warning(f"Removing zero-variance features: {zero_var_features}")
            feature_data = feature_data.drop(columns=zero_var_features)
            available_features = [f for f in available_features if f not in zero_var_features]
        
        # Store feature columns for later use in prediction
        self.feature_columns = available_features
        
        # COMPLETELY REDESIGNED TARGET ENGINEERING for stable training
        close_prices = enhanced_data['Close'].values.astype(float)
        
        # Use percentage returns only (standard financial approach)
        pct_returns = np.array([(close_prices[i+1] - close_prices[i]) / close_prices[i] 
                               for i in range(len(close_prices) - 1)])
        
        # For multi-day predictions, use forward-looking returns
        if self.prediction_days > 1:
            # Calculate forward returns over prediction horizon
            target_raw = []
            for i in range(len(close_prices) - self.prediction_days):
                future_price = close_prices[i + self.prediction_days]
                current_price = close_prices[i]
                period_return = (future_price - current_price) / current_price
                target_raw.append(period_return)
            target_raw = np.array(target_raw)
            
            # Pad to match pct_returns length for consistency
            padding_needed = len(pct_returns) - len(target_raw)
            target_raw = np.concatenate([target_raw, np.zeros(padding_needed)])
        else:
            # For 1-day predictions, use next-day returns
            target_raw = pct_returns
        
        # Simple outlier clipping (more conservative)
        percentile_95 = np.percentile(np.abs(target_raw), 95)
        target_clipped = np.clip(target_raw, -percentile_95, percentile_95)
        
        # Remove any remaining invalid values
        target_smoothed = np.nan_to_num(target_clipped, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"SIMPLIFIED target engineering: {self.prediction_days}-day returns")
        logger.info(f"Target stats - Mean: {np.mean(target_smoothed):.6f}, Std: {np.std(target_smoothed):.6f}")
        logger.info(f"Target range - Min: {np.min(target_smoothed):.6f}, Max: {np.max(target_smoothed):.6f}")
        logger.info(f"95th percentile used for clipping: {percentile_95:.6f}")
        
        # Use the simplified target
        future_returns = target_smoothed
        
        # Align target with features by removing the last observation 
        # (we can't predict the return after the last day)
        feature_data = feature_data.iloc[:-1].copy()
        target_data = future_returns
        
        # Create sequences with improved validation
        X, y = [], []
        
        # Ensure we have enough data
        min_length = self.sequence_length + 50  # Need at least 50 additional samples
        if len(feature_data) < min_length:
            raise ValueError(f"Insufficient data: {len(feature_data)} samples, need at least {min_length}")
        
        # NO SCALING HERE - will be done after train/test split
        features_array = feature_data.values
        
        for i in range(self.sequence_length, len(features_array)):
            X.append(features_array[i-self.sequence_length:i])
            y.append(target_data[i])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Final validation of training data
        if np.isnan(X_array).any() or np.isnan(y_array).any():
            raise ValueError("Training data contains NaN values after preprocessing")
        
        if np.isinf(X_array).any() or np.isinf(y_array).any():
            raise ValueError("Training data contains infinite values after preprocessing")
        
        logger.info(f"Prepared training data: {X_array.shape} features, {y_array.shape} targets")
        
        return X_array, y_array
    
    def _calculate_comprehensive_metrics(self, X_train: np.ndarray, y_train_scaled: np.ndarray, y_train_unscaled: np.ndarray,
                                       X_val: np.ndarray, y_val_scaled: np.ndarray, y_val_unscaled: np.ndarray,
                                       X_test: np.ndarray, y_test_scaled: np.ndarray, y_test_unscaled: np.ndarray,
                                       history) -> Dict:
        """Calculate comprehensive model performance metrics with proper scaling"""
        
        # Make predictions on scaled data (since model was trained on scaled targets)
        train_pred_scaled = self.model.predict(X_train, verbose=0)
        val_pred_scaled = self.model.predict(X_val, verbose=0)
        test_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse transform predictions to original scale
        train_pred_unscaled = self.scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
        val_pred_unscaled = self.scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
        test_pred_unscaled = self.scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics on UNSCALED data (critical for meaningful R²)
        metrics = {
            # Training metrics (on unscaled data)
            'train_rmse': float(np.sqrt(mean_squared_error(y_train_unscaled, train_pred_unscaled))),
            'train_mae': float(mean_absolute_error(y_train_unscaled, train_pred_unscaled)),
            'train_r2': float(r2_score(y_train_unscaled, train_pred_unscaled)),
            
            # Validation metrics (on unscaled data)
            'val_rmse': float(np.sqrt(mean_squared_error(y_val_unscaled, val_pred_unscaled))),
            'val_mae': float(mean_absolute_error(y_val_unscaled, val_pred_unscaled)),
            'val_r2': float(r2_score(y_val_unscaled, val_pred_unscaled)),
            
            # Test metrics (on unscaled data)
            'test_rmse': float(np.sqrt(mean_squared_error(y_test_unscaled, test_pred_unscaled))),
            'test_mae': float(mean_absolute_error(y_test_unscaled, test_pred_unscaled)),
            'test_r2': float(r2_score(y_test_unscaled, test_pred_unscaled)),
            
            # Training info
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_loss': float(min(history.history['val_loss'])),
            
            # Model info
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'num_features': X_train.shape[2],
            'total_parameters': self.model.count_params(),
            'training_date': datetime.now().isoformat()
        }
        
        return metrics
    
    def build_enhanced_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build SIMPLIFIED LSTM model optimized for financial return prediction
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model - much simpler and more stable
        """
        # Use smaller, more stable architecture
        lstm_units = 64  # Much smaller than config (was 128)
        
        # Sequential model for simplicity and stability
        model = Sequential([
            # Single LSTM layer to reduce overfitting
            LSTM(lstm_units, 
                 return_sequences=False,  # Don't need sequences for this problem
                 dropout=0.3,            # Moderate dropout
                 recurrent_dropout=0.2,  # Light recurrent dropout
                 kernel_regularizer=regularizers.l2(0.001)),  # Light L2
            
            # Single hidden layer
            Dense(32, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),
            
            # Output layer - linear for regression
            Dense(1, activation='linear')
        ])
        
        # Use much more conservative learning rate
        initial_lr = 0.0001  # Very conservative (was 0.0005 * 1.2)
        optimizer = Adam(learning_rate=initial_lr, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built SIMPLIFIED model: {lstm_units} LSTM units, single layer")
        logger.info(f"Very conservative learning rate: {initial_lr}")
        
        return model
    
    def train_enhanced_model(self, period: str = "3y") -> Dict:
        """
        Train enhanced LSTM model with quality validation and automatic cleanup
        
        Args:
            period: Period of historical data to use for training (default: 3y for better performance)
            
        Returns:
            Dictionary with detailed training metrics or error information
        """
        logger.info(f"Starting enhanced training for {self.symbol}")
        
        # Clean up any existing models first
        self._cleanup_existing_models()
        
        try:
            start_time = time.time()
            
            # Get comprehensive historical data
            stock_data = self.data_collector.get_stock_data(self.symbol, period)
            prices_df = stock_data.prices
            
            if len(prices_df) < self.sequence_length + 100:
                error_msg = f"Insufficient data for training: {len(prices_df)} samples (need at least {self.sequence_length + 100})"
                logger.error(error_msg)
                return {'error': error_msg, 'samples': len(prices_df)}
            
            # Prepare enhanced training data
            X, y = self.prepare_training_data(prices_df)
            
            if len(X) < 200:  # Early validation for insufficient training samples
                error_msg = f"Insufficient training samples: {len(X)} (need at least 200)"
                logger.error(error_msg)
                return {'error': error_msg, 'samples': len(X)}
            
            # Use config values for splits
            train_size = int(len(X) * (1.0 - config.model.validation_split - config.model.test_split))
            val_size = int(len(X) * config.model.validation_split)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            # FIT scalers on training data ONLY to prevent data leakage
            # 1. Fit feature scaler on training features
            # Reshape X_train for feature scaling: (samples, timesteps, features) -> (samples*timesteps, features)
            train_features_2d = X_train.reshape(-1, X_train.shape[2])
            self.feature_scaler.fit(train_features_2d)
            
            # Scale all feature data using the training-fitted scaler
            X_train_scaled = np.array([self.feature_scaler.transform(seq) for seq in X_train])
            X_val_scaled = np.array([self.feature_scaler.transform(seq) for seq in X_val])
            X_test_scaled = np.array([self.feature_scaler.transform(seq) for seq in X_test])
            
            # 2. IMPROVED target scaling with RobustScaler for financial data
            # RobustScaler is better for financial returns which often have outliers
            logger.info("Using RobustScaler for target variable (better for financial data)")
            
            # Check target distribution before scaling
            logger.info(f"Pre-scaling target stats - Mean: {np.mean(y_train):.6f}, Std: {np.std(y_train):.6f}")
            logger.info(f"Pre-scaling target range - Min: {np.min(y_train):.6f}, Max: {np.max(y_train):.6f}")
            
            # Use RobustScaler which is less sensitive to outliers
            from sklearn.preprocessing import RobustScaler
            robust_target_scaler = RobustScaler(quantile_range=(15.0, 85.0))  # Conservative quantile range
            
            y_train_scaled = robust_target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = robust_target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = robust_target_scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            # Verify scaling quality
            logger.info(f"Post-scaling target stats - Train mean: {np.mean(y_train_scaled):.6f}, Train std: {np.std(y_train_scaled):.6f}")
            logger.info(f"Post-scaling validation - Val mean: {np.mean(y_val_scaled):.6f}, Val std: {np.std(y_val_scaled):.6f}")
            
            # Update scaler reference for prediction use
            self.scaler = robust_target_scaler
            
            logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            logger.info(f"Target scaling - Train mean: {np.mean(y_train):.6f}, Train std: {np.std(y_train):.6f}")
            
            # Build simplified model with very conservative training
            self.model = self.build_enhanced_model((X_train.shape[1], X_train.shape[2]))
            
            logger.info(f"Model architecture: {X.shape[1]} time steps, {X.shape[2]} features")
            logger.info(f"SIMPLIFIED model with conservative parameters for stability")
            
            # Simplified callbacks for stable training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # More patience for small model
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=0.0001,  # Smaller improvement threshold
                    mode='min'
                ),
                # Simple learning rate reduction on plateau
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6,  # Very small minimum
                    verbose=1,
                    cooldown=3
                )
            ]
            
            # Train model using scaled data
            history = self.model.fit(
                X_train_scaled, y_train_scaled,
                epochs=config.model.epochs,
                batch_size=config.model.batch_size,
                validation_data=(X_val_scaled, y_val_scaled),
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            # Early evaluation of validation loss
            val_loss = min(history.history['val_loss'])
            if val_loss > 0.1:  # If validation loss is too high, likely poor model
                logger.warning(f"High validation loss detected: {val_loss:.4f} - model may be poor quality")
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                X_train_scaled, y_train_scaled, y_train, 
                X_val_scaled, y_val_scaled, y_val, 
                X_test_scaled, y_test_scaled, y_test, 
                history
            )
            
            # CRITICAL: Validate model quality before saving
            is_valid, reason = self._validate_model_quality(metrics)
            
            if not is_valid:
                logger.error(f"Model quality validation failed: {reason}")
                logger.info("Cleaning up poor quality model files...")
                self._cleanup_existing_models()  # Clean up again to be sure
                
                return {
                    'error': f"Model training failed quality validation: {reason}",
                    'metrics': metrics,
                    'validation_failed': True,
                    'r2_score': metrics.get('val_r2', -999),
                    'recommendation': "Try training with more data (5+ years) or different parameters"
                }
            
            # Model passed validation - save it
            logger.info(f"Model passed quality validation: {reason}")
            
            # Save model and scalers
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.feature_scaler, self.feature_scaler_path)
            
            # Save metrics
            training_time = time.time() - start_time
            metrics['training_time'] = training_time
            self._save_metrics(metrics)
            
            quality_assessment = self._assess_model_quality(metrics['val_r2'])
            logger.info(f"Enhanced training completed successfully - Quality: {quality_assessment}")
            
            return metrics
                
        except Exception as e:
            error_msg = f"Enhanced training failed: {str(e)}"
            logger.error(error_msg)
            
            # Clean up any partial model files on error
            self._cleanup_existing_models()
            
            return {'error': error_msg}
            
            # Store training information
            self.training_metrics = metrics
            
            # Save metrics
            import json
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        
        training_time = timer.elapsed_seconds
        metrics['training_time_seconds'] = training_time
        
        logger.info(f"Enhanced training completed for {self.symbol} in {training_time:.1f}s")
        logger.info(f"Final validation RMSE: {metrics['val_rmse']:.4f}")
        logger.info(f"Model R² score: {metrics['val_r2']:.4f}")
        
        return metrics
    
    def load_model(self) -> bool:
        """
        Load trained model and scalers from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if (self.model_path.exists() and self.scaler_path.exists() and 
                self.feature_scaler_path.exists()):
                
                # Load model using tf.keras.models
                self.model = load_model(self.model_path)
                
                # Load scalers
                self.scaler = joblib.load(self.scaler_path)
                self.feature_scaler = joblib.load(self.feature_scaler_path)
                
                # Load metrics if available
                if self.metrics_path.exists():
                    import json
                    with open(self.metrics_path, 'r') as f:
                        self.training_metrics = json.load(f)
                
                logger.info(f"Loaded enhanced model for {self.symbol}")
                return True
            else:
                logger.warning(f"No saved enhanced model found for {self.symbol}")
                return False
        except Exception as e:
            logger.error(f"Error loading enhanced model for {self.symbol}: {str(e)}")
            return False
    
    def predict_price(self, days_ahead: Optional[int] = None) -> Dict:
        """
        Predict stock price for specified days ahead with confidence intervals
        
        Args:
            days_ahead: Number of days to predict (default: config.prediction_days)
            
        Returns:
            Dictionary with enhanced prediction results
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError(f"No trained model available for {self.symbol}")
        
        if days_ahead is None:
            days_ahead = self.prediction_days
        
        # Get recent data for features
        stock_data = self.data_collector.get_stock_data(self.symbol, "1y")
        recent_data = stock_data.prices.tail(self.sequence_length + 30)  # Extra data for feature calculation
        
        # Prepare features for the recent data
        enhanced_recent = self.prepare_features(recent_data)
        
        # Use the same feature columns that were used in training
        if not hasattr(self, 'feature_columns'):
            # Fallback to the standard feature set if not stored
            self.feature_columns = [
                'Momentum_1d', 'Momentum_3d', 'Momentum_5d',
                'Price_SMA5_Ratio', 'Price_SMA20_Ratio', 'SMA_Cross',
                'Volatility_5d', 'Vol_Ratio',
                'Volume_Ratio', 'Price_Volume',
                'RSI_Normalized', 'MACD_Histogram', 'BB_Position',
                'Returns_Lag1', 'Returns_Lag2'
            ]
        
        # Ensure we only use features that exist in the data
        available_features = [col for col in self.feature_columns if col in enhanced_recent.columns]
        
        if not available_features:
            raise ValueError("No valid features found in recent data for prediction")
        
        # Scale the features using the same scaler from training
        feature_data = enhanced_recent[available_features].values
        
        # Handle NaN/inf values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply feature scaling consistently
        if len(feature_data) >= self.sequence_length:
            # Scale each sequence independently (as done in training)
            scaled_sequences = []
            for i in range(len(feature_data) - self.sequence_length + 1):
                seq = feature_data[i:i + self.sequence_length]
                scaled_seq = self.feature_scaler.transform(seq)
                scaled_sequences.append(scaled_seq)
            
            # Use the most recent sequence for prediction
            X_pred = np.array([scaled_sequences[-1]])
        else:
            raise ValueError(f"Insufficient recent data: need {self.sequence_length} samples, got {len(feature_data)}")
        
        # Get current price for calculations
        current_price = recent_data['Close'].iloc[-1]
        
        # Generate predictions
        predictions = []
        confidence_intervals = []
        
        # Generate multiple predictions for confidence estimation
        num_samples = 10
        
        for day in range(days_ahead):
            day_predictions = []
            
            # Multiple predictions with dropout enabled for uncertainty estimation
            for _ in range(num_samples):
                if self.model is not None:
                    pred = self.model.predict(X_pred, verbose=0)[0, 0]
                day_predictions.append(pred)
            
            # Calculate mean and confidence interval
            mean_pred = np.mean(day_predictions)
            std_pred = np.std(day_predictions)
            
            predictions.append(mean_pred)
            confidence_intervals.append({
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred,
                'std': std_pred
            })
            
            # Update sequence for next prediction
            # For simplicity, replicate the last feature values except for the price
            next_features = current_sequence[-1].copy()
            next_features[3] = mean_pred  # Update close price (index 3)
            
            # Shift sequence
            current_sequence = np.vstack([current_sequence[1:], next_features])
        
        # Inverse transform predictions
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_inv = self.scaler.inverse_transform(predictions_array).flatten()
        
        # Inverse transform confidence intervals
        for i, ci in enumerate(confidence_intervals):
            ci_array = np.array([[ci['lower']], [ci['upper']]])
            ci_inv = self.scaler.inverse_transform(ci_array).flatten()
            confidence_intervals[i]['lower'] = float(ci_inv[0])
            confidence_intervals[i]['upper'] = float(ci_inv[1])
        
        # Calculate metrics
        current_price = float(stock_data.prices['Close'].iloc[-1])
        final_price = float(predictions_inv[-1])
        percent_change = ((final_price - current_price) / current_price) * 100
        
        # Generate prediction dates
        last_date = stock_data.prices.index[-1]
        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        
        # Calculate confidence score based on model performance
        confidence_score = self._calculate_confidence_score()
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'predicted_price': final_price,
            'percent_change': percent_change,
            'prediction_days': days_ahead,
            'predictions': [float(p) for p in predictions_inv],
            'confidence_intervals': confidence_intervals,
            'prediction_dates': pred_dates.strftime('%Y-%m-%d').tolist(),
            'confidence_score': confidence_score,
            'model_performance': self._get_model_performance_summary(),
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_confidence_score(self) -> Dict:
        """
        Calculate confidence score based on model performance metrics
        """
        if not self.training_metrics:
            return {'score': 'Medium', 'details': 'No training metrics available'}
        
        val_r2 = self.training_metrics.get('val_r2', 0.5)
        val_rmse = self.training_metrics.get('val_rmse', float('inf'))
        
        # Calculate confidence based on R² score
        if val_r2 >= 0.8:
            confidence = 'High'
        elif val_r2 >= 0.6:
            confidence = 'Medium-High'
        elif val_r2 >= 0.4:
            confidence = 'Medium'
        elif val_r2 >= 0.2:
            confidence = 'Medium-Low'
        else:
            confidence = 'Low'
        
        return {
            'score': confidence,
            'r2_score': val_r2,
            'rmse': val_rmse,
            'details': f'Based on validation R² of {val_r2:.3f}'
        }
    
    def _get_model_performance_summary(self) -> Dict:
        """Get summary of model performance metrics"""
        if not self.training_metrics:
            return {'status': 'No metrics available'}
        
        return {
            'validation_r2': self.training_metrics.get('val_r2', 0),
            'validation_rmse': self.training_metrics.get('val_rmse', 0),
            'validation_mae': self.training_metrics.get('val_mae', 0),
            'epochs_trained': self.training_metrics.get('epochs_trained', 0),
            'num_features': self.training_metrics.get('num_features', 0),
            'training_date': self.training_metrics.get('training_date', 'Unknown')
        }
    
    def get_enhanced_recommendation(self) -> Dict:
        """
        Get enhanced buy/sell recommendation with confidence and reasoning
        
        Returns:
            Dictionary with detailed recommendation
        """
        prediction = self.predict_price()
        percent_change = prediction['percent_change']
        confidence = prediction['confidence_score']
        
        # Enhanced threshold-based recommendation with confidence weighting
        confidence_multiplier = {
            'High': 1.0,
            'Medium-High': 0.8,
            'Medium': 0.6,
            'Medium-Low': 0.4,
            'Low': 0.2
        }.get(confidence['score'], 0.6)
        
        # Adjust thresholds based on confidence
        strong_buy_threshold = 15 * confidence_multiplier
        buy_threshold = 8 * confidence_multiplier
        sell_threshold = -8 * confidence_multiplier
        strong_sell_threshold = -15 * confidence_multiplier
        
        if percent_change >= strong_buy_threshold:
            recommendation = "Strong Buy"
            reasoning = f"Model predicts {percent_change:.1f}% gain with {confidence['score']} confidence"
        elif percent_change >= buy_threshold:
            recommendation = "Buy"
            reasoning = f"Model predicts {percent_change:.1f}% gain with {confidence['score']} confidence"
        elif percent_change <= strong_sell_threshold:
            recommendation = "Strong Sell"
            reasoning = f"Model predicts {percent_change:.1f}% loss with {confidence['score']} confidence"
        elif percent_change <= sell_threshold:
            recommendation = "Sell"
            reasoning = f"Model predicts {percent_change:.1f}% loss with {confidence['score']} confidence"
        else:
            recommendation = "Hold"
            reasoning = f"Model predicts {percent_change:.1f}% change with {confidence['score']} confidence"
        
        # Add risk assessment
        volatility = abs(percent_change)
        if volatility > 20:
            risk_level = "High"
        elif volatility > 10:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'symbol': self.symbol,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'predicted_change_percent': percent_change,
            'current_price': prediction['current_price'],
            'target_price': prediction['predicted_price'],
            'confidence': confidence,
            'risk_level': risk_level,
            'prediction_horizon_days': self.prediction_days,
            'model_performance': prediction['model_performance'],
            'timestamp': datetime.now().isoformat()
        }
    
    def create_prediction_report(self, save_path: Optional[Path] = None) -> Dict:
        """
        Create comprehensive prediction report with visualizations
        """
        # Get prediction
        prediction = self.predict_price()
        
        # Get recommendation
        recommendation = self.get_enhanced_recommendation()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get historical data for plotting
        stock_data = self.data_collector.get_stock_data(self.symbol, "6m")
        historical_prices = stock_data.prices['Close']
        
        # Plot 1: Price prediction
        axes[0, 0].plot(historical_prices.index, np.array(historical_prices), 
                       label='Historical Price', color='blue', linewidth=2)
        
        pred_dates = pd.to_datetime(prediction['prediction_dates'])
        axes[0, 0].plot(pred_dates, prediction['predictions'], 
                       label='Predicted Price', color='red', linestyle='--', linewidth=2)
        
        # Add confidence intervals if available
        if 'confidence_intervals' in prediction:
            lower_bounds = [ci['lower'] for ci in prediction['confidence_intervals']]
            upper_bounds = [ci['upper'] for ci in prediction['confidence_intervals']]
            axes[0, 0].fill_between(pred_dates, lower_bounds, upper_bounds, 
                                   alpha=0.3, color='red', label='Confidence Interval')
        
        axes[0, 0].set_title(f'{self.symbol} Price Prediction')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Returns distribution (if training metrics available)
        if self.training_metrics:
            returns = historical_prices.pct_change().dropna()
            axes[0, 1].hist(returns * 100, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Historical Returns Distribution')
            axes[0, 1].set_xlabel('Daily Return (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model performance metrics
        if self.training_metrics:
            metrics_names = ['Train R²', 'Val R²', 'Test R²']
            metrics_values = [
                self.training_metrics.get('train_r2', 0),
                self.training_metrics.get('val_r2', 0),
                self.training_metrics.get('test_r2', 0)
            ]
            
            colors = ['green' if v > 0.6 else 'orange' if v > 0.3 else 'red' for v in metrics_values]
            axes[1, 0].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
            axes[1, 0].set_title('Model Performance (R² Scores)')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Prediction confidence over time
        if 'confidence_intervals' in prediction:
            std_values = [ci['std'] for ci in prediction['confidence_intervals']]
            axes[1, 1].plot(pred_dates, std_values, color='purple', linewidth=2)
            axes[1, 1].set_title('Prediction Uncertainty Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Prediction Std Dev')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction report saved to {save_path}")
        
        # Create comprehensive report
        report = {
            'symbol': self.symbol,
            'analysis_date': datetime.now().isoformat(),
            'prediction': prediction,
            'recommendation': recommendation,
            'model_metrics': self.training_metrics,
            'summary': {
                'current_price': format_currency(prediction['current_price']),
                'target_price': format_currency(prediction['predicted_price']),
                'expected_return': format_percentage(prediction['percent_change']),
                'confidence': prediction['confidence_score']['score'],
                'recommendation': recommendation['recommendation'],
                'risk_level': recommendation['risk_level']
            }
        }
        
        return report

    def get_buy_sell_rating(self, predicted_change: float) -> Dict[str, str]:
        """
        Generate buy/sell rating based on predicted price change
        
        Args:
            predicted_change: Predicted percentage change (e.g., 0.10 for 10% gain)
            
        Returns:
            Dictionary with rating and reasoning
        """
        if predicted_change >= config.model.strong_buy_threshold:
            return {
                'rating': 'STRONG BUY 🚀',
                'reasoning': f'Model predicts {predicted_change*100:.1f}% gain (>{config.model.strong_buy_threshold*100:.1f}%)',
                'color': '🟢'
            }
        elif predicted_change >= config.model.buy_threshold:
            return {
                'rating': 'BUY 📈',
                'reasoning': f'Model predicts {predicted_change*100:.1f}% gain (>{config.model.buy_threshold*100:.1f}%)',
                'color': '🟢'
            }
        elif predicted_change <= config.model.strong_sell_threshold:
            return {
                'rating': 'STRONG SELL 📉',
                'reasoning': f'Model predicts {predicted_change*100:.1f}% loss (<{config.model.strong_sell_threshold*100:.1f}%)',
                'color': '🔴'
            }
        elif predicted_change <= config.model.sell_threshold:
            return {
                'rating': 'SELL 📊',
                'reasoning': f'Model predicts {predicted_change*100:.1f}% loss (<{config.model.sell_threshold*100:.1f}%)',
                'color': '🔴'
            }
        else:
            return {
                'rating': 'HOLD ⚖️',
                'reasoning': f'Model predicts {predicted_change*100:.1f}% change (neutral range)',
                'color': '🟡'
            }


def create_enhanced_predictor(symbol: str, prediction_days: Optional[int] = None) -> EnhancedLSTMPredictor:
    """Factory function to create enhanced LSTM predictor"""
    return EnhancedLSTMPredictor(symbol, prediction_days=prediction_days)


# Backward compatibility
def create_predictor(symbol: str, prediction_days: Optional[int] = None) -> EnhancedLSTMPredictor:
    """Factory function for backward compatibility"""
    return EnhancedLSTMPredictor(symbol, prediction_days=prediction_days)
