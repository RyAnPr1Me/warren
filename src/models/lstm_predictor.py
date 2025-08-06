"""
Basic LSTM Price Predictor
Simple neural network for stock price prediction using historical data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

from ..config import config
from ..data.collector import StockDataCollector

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    LSTM-based stock price predictor for Phase 1 implementation
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = config.model.sequence_length
        self.prediction_days = config.model.prediction_days
        self.model_path = config.model.data_path / f"{self.symbol}_lstm_model.h5"
        self.scaler_path = config.model.data_path / f"{self.symbol}_scaler.pkl"
        
        # Create model directory
        config.model.data_path.mkdir(parents=True, exist_ok=True)
        
        self.data_collector = StockDataCollector()
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with stock price data
            target_column: Column to predict (default: 'Close')
            
        Returns:
            Tuple of (X_train, y_train, scaled_data)
        """
        # Get the target column
        prices = data[target_column].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaled_data
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config.model.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def train(self, period: str = "2y") -> Dict:
        """
        Train the LSTM model on historical data
        
        Args:
            period: Period of historical data to use for training
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting training for {self.symbol}")
        
        # Get historical data
        stock_data = self.data_collector.get_stock_data(self.symbol, period)
        prices_df = stock_data.prices
        
        if len(prices_df) < self.sequence_length + 50:
            raise ValueError(f"Insufficient data for training. Need at least {self.sequence_length + 50} days")
        
        # Prepare training data
        X, y, scaled_data = self.prepare_data(prices_df)
        
        # Reshape X for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split into train/validation
        split_idx = int(len(X) * (1 - config.model.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model((X.shape[1], 1))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Inverse transform for metrics calculation
        train_pred_inv = self.scaler.inverse_transform(train_pred)
        val_pred_inv = self.scaler.inverse_transform(val_pred)
        y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_val_inv = self.scaler.inverse_transform(y_val.reshape(-1, 1))
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_inv, train_pred_inv)),
            'val_rmse': np.sqrt(mean_squared_error(y_val_inv, val_pred_inv)),
            'train_mae': mean_absolute_error(y_train_inv, train_pred_inv),
            'val_mae': mean_absolute_error(y_val_inv, val_pred_inv),
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        logger.info(f"Training completed for {self.symbol}")
        logger.info(f"Validation RMSE: {metrics['val_rmse']:.2f}")
        logger.info(f"Validation MAE: {metrics['val_mae']:.2f}")
        
        return metrics
    
    def load_model(self) -> bool:
        """
        Load trained model and scaler from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded model for {self.symbol}")
                return True
            else:
                logger.warning(f"No saved model found for {self.symbol}")
                return False
        except Exception as e:
            logger.error(f"Error loading model for {self.symbol}: {str(e)}")
            return False
    
    def predict_price(self, days_ahead: int = None) -> Dict:
        """
        Predict stock price for specified days ahead
        
        Args:
            days_ahead: Number of days to predict (default: config.prediction_days)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError(f"No trained model available for {self.symbol}")
        
        if days_ahead is None:
            days_ahead = self.prediction_days
        
        # Get recent data
        stock_data = self.data_collector.get_stock_data(self.symbol, "1y")
        recent_prices = stock_data.prices['Close'].values[-self.sequence_length:].reshape(-1, 1)
        
        # Scale recent data
        recent_scaled = self.scaler.transform(recent_prices)
        
        # Generate predictions
        predictions = []
        current_sequence = recent_scaled.flatten()
        
        for _ in range(days_ahead):
            # Prepare input for prediction
            X_pred = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Make prediction
            next_pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence, next_pred)
        
        # Inverse transform predictions
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_inv = self.scaler.inverse_transform(predictions_array).flatten()
        
        # Calculate percentage change
        current_price = stock_data.prices['Close'].iloc[-1]
        final_price = predictions_inv[-1]
        percent_change = ((final_price - current_price) / current_price) * 100
        
        # Generate prediction dates
        last_date = stock_data.prices.index[-1]
        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        
        return {
            'symbol': self.symbol,
            'current_price': float(current_price),
            'predicted_price': float(final_price),
            'percent_change': float(percent_change),
            'prediction_days': days_ahead,
            'predictions': predictions_inv.tolist(),
            'prediction_dates': pred_dates.strftime('%Y-%m-%d').tolist(),
            'confidence': self._calculate_confidence()
        }
    
    def _calculate_confidence(self) -> str:
        """
        Calculate prediction confidence based on model performance
        Simple heuristic for Phase 1
        """
        # This is a simplified confidence calculation
        # In later phases, this would be more sophisticated
        return "Medium"  # Placeholder for Phase 1
    
    def get_recommendation(self) -> Dict:
        """
        Get buy/sell recommendation based on prediction
        
        Returns:
            Dictionary with recommendation and reasoning
        """
        prediction = self.predict_price()
        percent_change = prediction['percent_change']
        
        # Simple threshold-based recommendation for Phase 1
        if percent_change > 10:
            recommendation = "Strong Buy"
            reasoning = f"Model predicts {percent_change:.1f}% gain over {self.prediction_days} days"
        elif percent_change > 5:
            recommendation = "Buy"
            reasoning = f"Model predicts {percent_change:.1f}% gain over {self.prediction_days} days"
        elif percent_change > -5:
            recommendation = "Hold"
            reasoning = f"Model predicts {percent_change:.1f}% change over {self.prediction_days} days"
        elif percent_change > -10:
            recommendation = "Sell"
            reasoning = f"Model predicts {percent_change:.1f}% loss over {self.prediction_days} days"
        else:
            recommendation = "Strong Sell"
            reasoning = f"Model predicts {percent_change:.1f}% loss over {self.prediction_days} days"
        
        return {
            'symbol': self.symbol,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'predicted_change_percent': percent_change,
            'current_price': prediction['current_price'],
            'target_price': prediction['predicted_price'],
            'confidence': prediction['confidence'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def plot_prediction(self, save_path: Optional[Path] = None):
        """
        Create a plot showing historical prices and predictions
        
        Args:
            save_path: Path to save the plot (optional)
        """
        # Get historical data
        stock_data = self.data_collector.get_stock_data(self.symbol, "6m")
        historical_prices = stock_data.prices['Close']
        
        # Get prediction
        prediction = self.predict_price()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_prices.index, historical_prices.values, label='Historical Price', color='blue')
        
        # Plot predictions
        pred_dates = pd.to_datetime(prediction['prediction_dates'])
        plt.plot(pred_dates, prediction['predictions'], label='Predicted Price', color='red', linestyle='--')
        
        plt.title(f'{self.symbol} Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def create_predictor(symbol: str) -> LSTMPredictor:
    """Factory function to create LSTM predictor"""
    return LSTMPredictor(symbol)
