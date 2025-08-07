"""
Advanced ML Models and Ensemble System - Phase 3 Implementation
Combines LSTM with other models for improved prediction accuracy
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from ..config import config
from ..data.collector import StockDataCollector
from ..analysis.sentiment import sentiment_engine

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Advanced ensemble predictor combining LSTM with traditional ML models
    Phase 3 implementation with sentiment integration
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.data_collector = StockDataCollector()
        
        # Model paths
        self.model_dir = Path(f"data/models/{self.symbol}/ensemble")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the ensemble of models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def prepare_enhanced_features(self, data: pd.DataFrame, include_sentiment: bool = True) -> pd.DataFrame:
        """
        Prepare comprehensive features including sentiment data
        
        Args:
            data: Raw OHLCV data
            include_sentiment: Whether to include sentiment features
            
        Returns:
            DataFrame with enhanced features
        """
        # Start with technical features (reuse from LSTM predictor)
        from ..models.lstm_predictor import EnhancedLSTMPredictor
        
        lstm_predictor = EnhancedLSTMPredictor(self.symbol)
        enhanced_data = lstm_predictor.prepare_features(data)
        
        # Add additional ensemble-specific features
        self._add_statistical_features(enhanced_data)
        
        # Add sentiment features if enabled and available
        if include_sentiment and config.features.enable_sentiment_analysis:
            try:
                sentiment_features = sentiment_engine.get_sentiment_features(self.symbol)
                
                # Add sentiment features as constant columns (they don't vary by time in this simple implementation)
                for feature_name, value in sentiment_features.items():
                    enhanced_data[feature_name] = value
                    
                logger.info(f"Added {len(sentiment_features)} sentiment features")
            except Exception as e:
                logger.warning(f"Could not add sentiment features: {e}")
        
        # Select and clean features
        self.feature_columns = self._select_features(enhanced_data)
        feature_data = enhanced_data[self.feature_columns].copy()
        
        # Handle missing values
        # Handle missing values in feature data
        feature_data = feature_data.ffill().bfill().fillna(0)
        
        return feature_data
    
    def _add_statistical_features(self, data: pd.DataFrame):
        """Add statistical and derived features"""
        close_prices = data['Close']
        
        # Price percentiles and rankings
        data['Price_Percentile_20d'] = close_prices.rolling(20).rank(pct=True)
        data['Price_Percentile_60d'] = close_prices.rolling(60).rank(pct=True)
        
        # Relative strength vs market (simplified)
        data['Relative_Strength'] = close_prices.pct_change() - close_prices.pct_change().rolling(20).mean()
        
        # Volatility metrics
        returns = close_prices.pct_change()
        data['Skewness_20d'] = returns.rolling(20).skew()
        data['Kurtosis_20d'] = returns.rolling(20).kurt()
        
        # Price acceleration (second derivative)
        data['Price_Acceleration'] = close_prices.pct_change().diff()
        
        # Volume-price relationship
        if 'Volume' in data.columns:
            data['Volume_Price_Corr'] = data['Volume'].rolling(20).corr(close_prices)
            data['Volume_Weighted_Price'] = (close_prices * data['Volume']).rolling(5).sum() / data['Volume'].rolling(5).sum()
    
    def _select_features(self, data: pd.DataFrame) -> List[str]:
        """Select the best features for ensemble models"""
        # Core technical features
        core_features = [
            'Momentum_1d', 'Momentum_3d', 'Momentum_5d',
            'Price_SMA5_Ratio', 'Price_SMA20_Ratio', 'SMA_Cross',
            'Volatility_5d', 'Vol_Ratio',
            'Volume_Ratio', 'Price_Volume',
            'RSI_Normalized', 'MACD_Histogram', 'BB_Position',
            'Returns_Lag1', 'Returns_Lag2'
        ]
        
        # Statistical features
        statistical_features = [
            'Price_Percentile_20d', 'Price_Percentile_60d',
            'Relative_Strength', 'Price_Acceleration',
            'Volume_Price_Corr', 'Volume_Weighted_Price'
        ]
        
        # Sentiment features (if enabled)
        sentiment_features = []
        if config.features.enable_sentiment_analysis:
            sentiment_features = [
                'sentiment_overall', 'sentiment_confidence',
                'sentiment_bullish_ratio', 'sentiment_bearish_ratio',
                'sentiment_news_volume'
            ]
        
        # Combine and filter existing columns
        all_features = core_features + statistical_features + sentiment_features
        available_features = [f for f in all_features if f in data.columns]
        
        logger.info(f"Selected {len(available_features)} features for ensemble models")
        return available_features
    
    def train_ensemble(self, period: str = "3y", mega_data: bool = False) -> Dict:
        """
        Train the ensemble of models
        
        Args:
            period: Historical data period
            mega_data: If True, use comprehensive data from all APIs
            
        Returns:
            Training metrics
        """
        logger.info(f"Training ensemble models for {self.symbol}")
        
        try:
            # Get historical data with mega_data option
            stock_data = self.data_collector.get_stock_data(self.symbol, period, mega_data=mega_data)
            prices_df = stock_data.prices
            
            # Prepare features
            feature_data = self.prepare_enhanced_features(prices_df)
            
            # Create target (future returns)
            # Calculate future returns for targets
            close_prices = np.array(prices_df['Close'].values, dtype=np.float64)
            future_returns = np.log(close_prices[1:] / close_prices[:-1])
            
            # Align features and targets
            feature_data = feature_data.iloc[:-1]  # Remove last row
            target_data = future_returns
            
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_data.values).any(axis=1) | np.isnan(target_data))
            X = feature_data.values[valid_mask]
            y = target_data[valid_mask]
            
            if len(X) < 100:
                raise ValueError(f"Insufficient data for ensemble training: {len(X)} samples")
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train each model
            model_metrics = {}
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                model_metrics[model_name] = {
                    'train_r2': r2_score(y_train, train_pred),
                    'test_r2': r2_score(y_test, test_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
                }
                
                logger.info(f"{model_name} - Test R²: {model_metrics[model_name]['test_r2']:.4f}")
            
            # Train ensemble weights
            ensemble_weights = self._train_ensemble_weights(X_test, y_test)
            
            # Save models
            self._save_ensemble(ensemble_weights)
            
            # Calculate ensemble metrics
            ensemble_pred = self._predict_ensemble(X_test, ensemble_weights)
            ensemble_metrics = {
                'ensemble_test_r2': r2_score(y_test, ensemble_pred),
                'ensemble_test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                'model_weights': ensemble_weights,
                'individual_models': model_metrics,
                'feature_count': len(self.feature_columns),
                'training_samples': len(X_train)
            }
            
            logger.info(f"Ensemble training completed - Test R²: {ensemble_metrics['ensemble_test_r2']:.4f}")
            return ensemble_metrics
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {'error': str(e)}
    
    def _train_ensemble_weights(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Train optimal ensemble weights using validation data"""
        
        # Get predictions from each model
        model_predictions = {}
        for model_name, model in self.models.items():
            X_test_scaled = self.scalers[model_name].transform(X_test)
            model_predictions[model_name] = model.predict(X_test_scaled)
        
        # Simple equal weighting for now (could be optimized)
        weights = {model_name: 1.0 / len(self.models) for model_name in self.models.keys()}
        
        # Could implement more sophisticated weight optimization here
        # For example, using scipy.optimize to minimize ensemble error
        
        return weights
    
    def _predict_ensemble(self, X: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Make ensemble prediction using weighted average"""
        predictions = []
        
        for model_name, model in self.models.items():
            X_scaled = self.scalers[model_name].transform(X)
            pred = model.predict(X_scaled)
            predictions.append(pred * weights[model_name])
        
        return np.sum(predictions, axis=0)
    
    def _save_ensemble(self, weights: Dict[str, float]):
        """Save trained ensemble models and weights"""
        try:
            # Save each model and scaler
            for model_name, model in self.models.items():
                model_path = self.model_dir / f"{model_name}.pkl"
                scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
            
            # Save ensemble weights and metadata
            ensemble_metadata = {
                'weights': weights,
                'feature_columns': self.feature_columns,
                'model_names': list(self.models.keys())
            }
            
            metadata_path = self.model_dir / "ensemble_metadata.pkl"
            joblib.dump(ensemble_metadata, metadata_path)
            
            logger.info(f"Ensemble models saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
    
    def load_ensemble(self) -> bool:
        """Load trained ensemble models"""
        try:
            # Load metadata
            metadata_path = self.model_dir / "ensemble_metadata.pkl"
            if not metadata_path.exists():
                return False
            
            metadata = joblib.load(metadata_path)
            self.ensemble_weights = metadata['weights']
            self.feature_columns = metadata['feature_columns']
            
            # Load models and scalers
            for model_name in metadata['model_names']:
                model_path = self.model_dir / f"{model_name}.pkl"
                scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                else:
                    logger.warning(f"Missing files for {model_name}")
                    return False
            
            logger.info(f"Ensemble models loaded for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False
    
    def predict_with_ensemble(self, days_ahead: int = 15) -> Dict:
        """
        Make predictions using the ensemble approach
        
        Args:
            days_ahead: Number of days to predict
            
        Returns:
            Ensemble prediction results
        """
        if not hasattr(self, 'ensemble_weights') and not self.load_ensemble():
            raise ValueError(f"No trained ensemble available for {self.symbol}")
        
        try:
            # Get recent data
            stock_data = self.data_collector.get_stock_data(self.symbol, "1y")
            recent_data = stock_data.prices.tail(100)  # Use more data for feature calculation
            
            # Prepare features
            feature_data = self.prepare_enhanced_features(recent_data)
            
            # Get most recent feature vector
            latest_features = feature_data.iloc[-1:].values
            
            # Make ensemble prediction
            prediction = self._predict_ensemble(latest_features, self.ensemble_weights)[0]
            
            # Convert to price prediction
            current_price = recent_data['Close'].iloc[-1]
            predicted_price = current_price * np.exp(prediction)
            predicted_change = prediction
            
            return {
                'predicted_return': float(prediction),
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'predicted_change_pct': float(predicted_change * 100),
                'confidence': self._calculate_ensemble_confidence(latest_features),
                'model_contributions': self._get_model_contributions(latest_features),
                'days_ahead': days_ahead,
                'prediction_type': 'ensemble'
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {'error': str(e)}
    
    def _calculate_ensemble_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence based on model agreement"""
        try:
            # Get individual model predictions
            individual_predictions = []
            for model_name, model in self.models.items():
                X_scaled = self.scalers[model_name].transform(features)
                pred = model.predict(X_scaled)[0]
                individual_predictions.append(pred)
            
            # Calculate standard deviation of predictions
            pred_std = np.std(individual_predictions)
            
            # Convert to confidence (lower std = higher confidence)
            confidence = max(0.1, 1.0 - min(pred_std * 10, 0.9))
            
            return float(confidence)
            
        except Exception as e:
            logger.warning(f"Could not calculate ensemble confidence: {e}")
            return 0.5
    
    def _get_model_contributions(self, features: np.ndarray) -> Dict[str, float]:
        """Get individual model contributions to ensemble prediction"""
        contributions = {}
        
        try:
            for model_name, model in self.models.items():
                X_scaled = self.scalers[model_name].transform(features)
                pred = model.predict(X_scaled)[0]
                weight = self.ensemble_weights[model_name]
                contributions[model_name] = {
                    'prediction': float(pred),
                    'weight': float(weight),
                    'weighted_contribution': float(pred * weight)
                }
        except Exception as e:
            logger.warning(f"Could not get model contributions: {e}")
        
        return contributions


def create_ensemble_predictor(symbol: str) -> EnsemblePredictor:
    """Factory function to create ensemble predictor"""
    return EnsemblePredictor(symbol)
