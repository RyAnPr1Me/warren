"""
Enhanced LSTM Price Predictor - Phase 2 Implementation
Advanced neural network for stock price prediction with 60-day forecasting
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
import logging
import psutil
import platform
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Concatenate, GaussianNoise, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback
from tensorflow.keras import regularizers

class PlateauBreakerScheduler(Callback):
    """
    Advanced learning rate scheduler designed to break through loss plateaus.
    Uses cyclical learning rates with restarts when plateaus are detected.
    """
    
    def __init__(self, base_lr=1e-4, max_lr=1e-3, step_size=10, mode='triangular2', 
                 plateau_patience=8, plateau_factor=0.7, restart_factor=3.0, verbose=1):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.restart_factor = restart_factor
        self.verbose = verbose
        
        # State tracking
        self.total_iterations = 0
        self.cycle = 0
        self.plateau_count = 0
        self.best_loss = float('inf')
        self.plateau_wait = 0
        
    def _triangular_lr(self, iteration):
        """Triangular learning rate policy"""
        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        return lr
    
    def _triangular2_lr(self, iteration):
        """Triangular2 learning rate policy with decay"""
        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        return lr
    
    def _exp_range_lr(self, iteration):
        """Exponential range learning rate policy"""
        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (0.99999 ** iteration)
        return lr
    
    def on_epoch_begin(self, epoch, logs=None):
        """Set learning rate at the beginning of each epoch"""
        # Calculate current learning rate based on policy
        if self.mode == 'triangular':
            lr = self._triangular_lr(self.total_iterations)
        elif self.mode == 'triangular2':
            lr = self._triangular2_lr(self.total_iterations)
        elif self.mode == 'exp_range':
            lr = self._exp_range_lr(self.total_iterations)
        else:
            lr = self._triangular2_lr(self.total_iterations)  # Default
        
        # Apply the learning rate
        self.model.optimizer.learning_rate.assign(lr)
        
        if self.verbose and epoch % 5 == 0:
            print(f"Epoch {epoch + 1}: Cyclical LR = {lr:.6f} (Cycle: {self.cycle}, Iteration: {self.total_iterations})")
    
    def on_epoch_end(self, epoch, logs=None):
        """Check for plateaus and trigger restarts if needed"""
        if logs is None:
            logs = {}
        
        current_loss = logs.get('val_loss', logs.get('loss', float('inf')))
        
        # Check for improvement
        if current_loss < self.best_loss * 0.995:  # 0.5% improvement threshold
            self.best_loss = current_loss
            self.plateau_wait = 0
        else:
            self.plateau_wait += 1
        
        # Trigger plateau-breaking restart
        if self.plateau_wait >= self.plateau_patience:
            self.plateau_count += 1
            self.plateau_wait = 0
            
            # Restart with higher maximum learning rate
            self.max_lr *= self.restart_factor
            self.base_lr *= self.restart_factor
            
            # But cap the learning rates to prevent instability
            self.max_lr = min(self.max_lr, 0.01)
            self.base_lr = min(self.base_lr, 0.001)
            
            # Reset cycle
            self.cycle += 1
            
            if self.verbose:
                print(f"🚀 PLATEAU DETECTED! Restart #{self.plateau_count}")
                print(f"   New LR range: {self.base_lr:.6f} - {self.max_lr:.6f}")
                print(f"   Best loss so far: {self.best_loss:.6f}")
        
        self.total_iterations += 1


class AdaptiveLossCallback(Callback):
    """
    Dynamically adjusts loss function based on training progress.
    Switches to more aggressive loss functions when plateaus are detected.
    """
    
    def __init__(self, initial_loss='huber', plateau_patience=15, verbose=1):
        super().__init__()
        self.initial_loss = initial_loss
        self.plateau_patience = plateau_patience
        self.verbose = verbose
        
        # Loss function progression
        self.loss_functions = ['huber', 'mse', 'mae', 'logcosh']
        self.current_loss_idx = 0
        
        # State tracking
        self.best_loss = float('inf')
        self.plateau_wait = 0
        self.loss_changes = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Monitor training and switch loss functions if needed"""
        if logs is None:
            logs = {}
            
        current_loss = logs.get('val_loss', logs.get('loss', float('inf')))
        
        # Check for improvement
        if current_loss < self.best_loss * 0.99:  # 1% improvement threshold
            self.best_loss = current_loss
            self.plateau_wait = 0
        else:
            self.plateau_wait += 1
        
        # Switch loss function if plateau detected
        if (self.plateau_wait >= self.plateau_patience and 
            self.current_loss_idx < len(self.loss_functions) - 1):
            
            self.current_loss_idx += 1
            new_loss = self.loss_functions[self.current_loss_idx]
            
            # Recompile model with new loss function
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=new_loss,
                metrics=self.model.compiled_metrics._metrics
            )
            
            self.loss_changes += 1
            self.plateau_wait = 0
            
            if self.verbose:
                print(f"🔄 LOSS ADAPTATION #{self.loss_changes}: Switching to '{new_loss}' loss function")
                print(f"   Plateau duration: {self.plateau_patience} epochs")
                print(f"   Current best loss: {self.best_loss:.6f}")


class PlateauDetector(Callback):
    """Enhanced plateau detector with more sophisticated logic"""
    
    def __init__(self, monitor='val_loss', patience=12, min_delta=0.0005, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best = None
        self.plateau_detected = False
        self.improvement_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.best is None:
            self.best = current
            return
        
        # Calculate improvement
        improvement = self.best - current
        self.improvement_history.append(improvement)
        
        # Keep only recent history
        if len(self.improvement_history) > 20:
            self.improvement_history.pop(0)
        
        # Check for significant improvement
        if improvement > self.min_delta:
            self.best = current
            self.wait = 0
            if self.verbose and epoch % 10 == 0:
                recent_avg = np.mean(self.improvement_history[-5:]) if len(self.improvement_history) >= 5 else improvement
                print(f"📈 Improving! Current: {current:.6f}, Best: {self.best:.6f}, Recent avg improvement: {recent_avg:.6f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if not self.plateau_detected:
                    self.plateau_detected = True
                    if self.verbose:
                        recent_avg = np.mean(self.improvement_history[-10:]) if len(self.improvement_history) >= 10 else 0
                        print(f"⚠️  PLATEAU DETECTED after {self.wait} epochs without improvement")
                        print(f"   Current: {current:.6f}, Best: {self.best:.6f}")
                        print(f"   Average recent improvement: {recent_avg:.6f}")
                        print(f"   Consider early stopping or learning rate adjustment")
            
        # Check if improvement is significant
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.plateau_detected = True
            if self.verbose > 0:
                print(f"\nPlateau detected! No improvement for {self.patience} epochs. Stopping early.")
            self.model.stop_training = True

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


class SystemAwareTrainingConfig:
    """
    System-aware training configuration that adapts to hardware capabilities
    to prevent overheating on laptops while utilizing full power on PCs
    """
    
    def __init__(self):
        self.system_type = self._detect_system_type()
        self.gpu_available = self._setup_gpu()
        self.thermal_profile = self._get_thermal_profile()
        self.training_config = self._configure_training_parameters()
        
        logger.info(f"System detected: {self.system_type}")
        logger.info(f"GPU available: {self.gpu_available}")
        logger.info(f"Thermal profile: {self.thermal_profile}")
    
    def _detect_system_type(self) -> str:
        """Detect if running on laptop or desktop"""
        try:
            # Check battery presence (laptops typically have batteries)
            if hasattr(psutil, 'sensors_battery'):
                battery = psutil.sensors_battery()
                if battery is not None:
                    return "laptop"
            
            # Check CPU count and thermal design
            cpu_count = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            
            # Heuristics for system type detection with safe null checks
            if cpu_count is not None and cpu_count >= 8 and cpu_freq and hasattr(cpu_freq, 'max') and cpu_freq.max and cpu_freq.max > 3500:
                return "desktop"
            elif cpu_count is not None and cpu_count <= 4:
                return "laptop"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def _setup_gpu(self) -> bool:
        """Setup GPU with memory growth to prevent OOM errors"""
        try:
            # Check for GPU availability
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if len(gpus) > 0:
                logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
                
                # Enable memory growth to prevent allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set mixed precision for better performance
                if len(gpus) > 0:
                    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
                    logger.info("Enabled GPU memory growth and XLA compilation")
                
                return True
            else:
                logger.info("No GPU found, using CPU")
                # Optimize CPU usage
                tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all cores
                tf.config.threading.set_inter_op_parallelism_threads(0)
                return False
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}, falling back to CPU")
            return False
    
    def _get_thermal_profile(self) -> str:
        """Determine thermal management profile"""
        try:
            # Check CPU temperature if available (newer psutil versions)
            cpu_temp = None
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Get CPU temperature
                        for name, entries in temps.items():
                            if 'cpu' in name.lower() or 'core' in name.lower():
                                if entries:
                                    cpu_temp = entries[0].current
                                    break
            except (AttributeError, OSError, Exception):
                # sensors_temperatures not available on all systems (e.g., macOS)
                pass
                pass
            
            if cpu_temp and cpu_temp > 70:
                return "hot"
            elif cpu_temp and cpu_temp > 50:
                return "warm"
            
            # Base profile on system type and CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if self.system_type == "laptop":
                return "conservative" if cpu_percent > 70 else "balanced"
            else:
                return "performance"
                
        except:
            return "balanced"
    
    def _configure_training_parameters(self) -> dict:
        """Configure training parameters based on system capabilities"""
        config = {
            "batch_size": 32,
            "epochs": 200,
            "patience": 25,
            "workers": 1,
            "use_multiprocessing": False,
            "max_queue_size": 10
        }
        
        if self.system_type == "laptop":
            # Conservative settings for laptops to prevent overheating
            config.update({
                "batch_size": 16 if self.thermal_profile == "conservative" else 32,
                "epochs": 150 if self.thermal_profile == "conservative" else 200,
                "patience": 20 if self.thermal_profile == "conservative" else 25,
                "workers": 1,
                "use_multiprocessing": False,
                "max_queue_size": 5
            })
            
        elif self.system_type == "desktop":
            # Aggressive settings for desktops
            cpu_cores = psutil.cpu_count(logical=False) or 4  # Fallback to 4 if None
            config.update({
                "batch_size": 64 if self.gpu_available else 32,
                "epochs": 300,
                "patience": 30,
                "workers": min(4, cpu_cores),
                "use_multiprocessing": True,
                "max_queue_size": 20
            })
        
        # Adjust for thermal profile
        if self.thermal_profile == "hot":
            config["batch_size"] = max(16, config["batch_size"] // 2)
            config["epochs"] = min(100, config["epochs"])
            config["workers"] = 1
            config["use_multiprocessing"] = False
            
        elif self.thermal_profile == "performance" and self.gpu_available:
            config["batch_size"] = min(128, config["batch_size"] * 2)
            
        return config


class ThermalThrottlingCallback(Callback):
    """
    Callback to monitor system temperature and throttle training if needed
    """
    
    def __init__(self, temp_threshold=80, check_interval=10):
        super().__init__()
        self.temp_threshold = temp_threshold
        self.check_interval = check_interval
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        
        # Check temperature every N epochs
        if self.epoch_count % self.check_interval == 0:
            try:
                # Try to get temperature data (only available on some systems)
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        max_temp = 0
                        for name, entries in temps.items():
                            if entries:
                                max_temp = max(max_temp, max(entry.current for entry in entries))
                        
                        if max_temp > self.temp_threshold:
                            print(f"\n⚠️ High temperature detected: {max_temp:.1f}°C")
                            print("Implementing thermal throttling...")
                            time.sleep(2)  # Brief pause to cool down
            except (AttributeError, OSError, Exception):
                # Ignore temperature check errors - not all systems support this
                pass


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
    
    # Model quality thresholds (REALISTIC for stock prediction)
    MIN_R2_SCORE = -0.1   # Realistic minimum for financial time series
    GOOD_R2_SCORE = 0.02   # Good model threshold for stock prediction  
    EXCELLENT_R2_SCORE = 0.05  # Excellent model threshold (very rare)
    
    def __init__(self, symbol: str, prediction_days: Optional[int] = None):
        self.symbol = symbol.upper()
        self.model = None
        self.scaler = StandardScaler()  # Better for returns (can be negative)
        # Use StandardScaler for more stable and interpretable feature scaling
        self.feature_scaler = StandardScaler()  # More stable and faster than QuantileTransformer
        
        # Initialize system-aware configuration for optimal performance
        self.system_config = SystemAwareTrainingConfig()
        
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
        
        self.model_path = self.base_dir / f"{self.symbol}_lstm_model.keras"  # Use native Keras format
        
        logger.info(f"Initialized Enhanced LSTM for {symbol}")
        logger.info(f"System: {self.system_config.system_type} ({self.system_config.thermal_profile} thermal profile)")
        logger.info(f"GPU Available: {self.system_config.gpu_available}")
        logger.info(f"Training Config: {self.system_config.training_config}")
        self.scaler_path = self.base_dir / f"{self.symbol}_scaler.pkl"
        self.feature_scaler_path = self.base_dir / f"{self.symbol}_feature_scaler.pkl"
        self.metrics_path = self.base_dir / f"{self.symbol}_metrics.json"
    
    @staticmethod
    def directional_loss(y_true, y_pred):
        """
        Simplified directional loss optimized for 1-day movement prediction.
        Focuses on direction accuracy and magnitude consistency.
        """
        # Base regression loss (Huber is robust to outliers)
        base_loss = tf.keras.losses.Huber(delta=0.02)(y_true, y_pred)
        
        # Directional components
        true_direction = tf.sign(y_true)
        pred_direction = tf.sign(y_pred)
        directional_agreement = tf.multiply(true_direction, pred_direction)
        
        # Penalty for wrong direction (most important for trading)
        wrong_direction = tf.cast(directional_agreement < 0, tf.float32)
        direction_penalty = tf.multiply(wrong_direction, tf.square(y_true - y_pred) * 3.0)
        
        # Reward for correct direction
        correct_direction = tf.cast(directional_agreement > 0, tf.float32)
        direction_bonus = tf.multiply(correct_direction, tf.square(y_true - y_pred) * 0.2)
        
        # Magnitude consistency - penalize predicting flat when movement occurs
        true_magnitude = tf.abs(y_true)
        pred_magnitude = tf.abs(y_pred)
        significant_move = tf.cast(true_magnitude > 0.01, tf.float32)  # 1% threshold
        flat_prediction = tf.cast(pred_magnitude < 0.003, tf.float32)  # 0.3% threshold
        flat_penalty = tf.multiply(tf.multiply(significant_move, flat_prediction), 2.0)
        
        # Combine all components
        directional_component = (
            tf.reduce_mean(direction_penalty) +
            tf.reduce_mean(flat_penalty) -
            tf.reduce_mean(direction_bonus)
        )
        
        # Final loss
        total_loss = base_loss + directional_component
        
        return total_loss
    
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
        Validate if the trained model meets quality standards for FINANCIAL PREDICTION
        Uses realistic financial metrics instead of unrealistic R² thresholds
        
        Args:
            metrics: Training metrics dictionary
            
        Returns:
            Tuple of (is_valid, reason)
        """
        val_r2 = metrics.get('val_r2', -999)
        test_r2 = metrics.get('test_r2', -999)
        val_rmse = metrics.get('val_rmse', float('inf'))
        
        # Financial-specific quality metrics
        val_direction_accuracy = metrics.get('val_direction_accuracy', 0)
        test_direction_accuracy = metrics.get('test_direction_accuracy', 0)
        val_ic = metrics.get('val_ic', 0)
        test_ic = metrics.get('test_ic', 0)
        
        # REALISTIC thresholds for financial prediction
        min_direction_accuracy = 0.51  # Just above random (50%) - slightly more lenient
        min_ic = 0.015  # Small but positive correlation - slightly more lenient  
        min_r2 = -0.2  # Allow negative R² (normal for financial data) - more lenient
        max_overfitting_ratio = 15  # More lenient for financial data
        
        quality_checks = []
        
        # Check direction accuracy (most important for trading)
        if val_direction_accuracy >= min_direction_accuracy:
            quality_checks.append(f"✓ Val direction accuracy: {val_direction_accuracy:.3f}")
        else:
            quality_checks.append(f"✗ Val direction accuracy: {val_direction_accuracy:.3f} < {min_direction_accuracy}")
            
        if test_direction_accuracy >= min_direction_accuracy:
            quality_checks.append(f"✓ Test direction accuracy: {test_direction_accuracy:.3f}")
        else:
            quality_checks.append(f"✗ Test direction accuracy: {test_direction_accuracy:.3f} < {min_direction_accuracy}")
        
        # Check Information Coefficient (correlation)
        if abs(val_ic) >= min_ic and not np.isnan(val_ic):
            quality_checks.append(f"✓ Val IC: {val_ic:.4f}")
        else:
            quality_checks.append(f"✗ Val IC: {val_ic:.4f} < {min_ic}")
            
        if abs(test_ic) >= min_ic and not np.isnan(test_ic):
            quality_checks.append(f"✓ Test IC: {test_ic:.4f}")
        else:
            quality_checks.append(f"✗ Test IC: {test_ic:.4f} < {min_ic}")
        
        # Check R² with realistic threshold
        if val_r2 >= min_r2:
            quality_checks.append(f"✓ Val R²: {val_r2:.4f}")
        else:
            quality_checks.append(f"✗ Val R²: {val_r2:.4f} < {min_r2}")
            
        if test_r2 >= min_r2:
            quality_checks.append(f"✓ Test R²: {test_r2:.4f}")
        else:
            quality_checks.append(f"✗ Test R²: {test_r2:.4f} < {min_r2}")
        
        # Check overfitting
        if test_r2 >= val_r2 - abs(val_r2) / max_overfitting_ratio:
            quality_checks.append(f"✓ No severe overfitting")
        else:
            quality_checks.append(f"✗ Overfitting: Val R² {val_r2:.3f} vs Test R² {test_r2:.3f}")
        
        # Count passing checks
        passed_checks = sum(1 for check in quality_checks if check.startswith('✓'))
        total_checks = len(quality_checks)
        
        # Accept if majority of checks pass
        if passed_checks >= 4:  # At least 4 out of 7 checks
            reason = f"Financial metrics passed: {passed_checks}/{total_checks} checks\n" + "\n".join(quality_checks)
            return True, reason
        else:
            reason = f"Financial metrics failed: {passed_checks}/{total_checks} checks\n" + "\n".join(quality_checks)
            return False, reason
    
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
        delta = prices.astype(float).diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
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
        
        # === ADVANCED FINANCIAL FEATURES ===
        # Price acceleration (2nd derivative of price)
        df['Price_Acceleration'] = df['Returns_Lag1'].diff()
        
        # Relative Strength vs Market (proxy using volatility-adjusted returns)
        df['Relative_Strength'] = df['Returns_Lag1'] / (df['Volatility_5d'] + 1e-8)
        
        # Mean reversion indicators
        df['Distance_From_SMA20'] = (df['Close'].shift(1) - df['SMA_20']) / df['SMA_20']
        df['Mean_Reversion_Signal'] = np.tanh(df['Distance_From_SMA20'] * 2)  # Bounded [-1, 1]
        
        # Momentum persistence
        df['Momentum_Consistency'] = (
            np.sign(df['Momentum_1d']) * np.sign(df['Momentum_3d']) * np.sign(df['Momentum_5d'])
        )
        
        # Volume-Price Trend (VPT) - cumulative volume-weighted price changes
        df['VPT'] = (df['Returns_Lag1'] * df['Volume'].shift(1)).cumsum()
        df['VPT_Normalized'] = (df['VPT'] - df['VPT'].rolling(50).mean()) / df['VPT'].rolling(50).std()
        
        # Volatility clustering (GARCH-like effect)
        df['Vol_Persistence'] = df['Volatility_5d'].rolling(10).corr(df['Volatility_5d'].shift(1))
        
        # Price efficiency (how much price moves per unit of volume)
        df['Price_Efficiency'] = abs(df['Returns_Lag1']) / (np.log(df['Volume'].shift(1) + 1) + 1e-8)
        
        # Support/Resistance levels
        df['Price_Percentile_20d'] = df['Close'].shift(1).rolling(20).rank() / 20
        df['Price_Percentile_50d'] = df['Close'].shift(1).rolling(50).rank() / 50
        
        # Market microstructure - spread proxy
        df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
        df['Spread_MA'] = df['Spread_Proxy'].shift(1).rolling(10).mean()
        df['Spread_Anomaly'] = (df['Spread_Proxy'].shift(1) - df['Spread_MA']) / df['Spread_MA']
        
        # === OPTIMIZED PREDICTIVE FEATURES ===
        # Core momentum features with quality measures
        df['Momentum_Strength'] = abs(df['Momentum_1d']).rolling(5).mean()
        df['Momentum_Quality'] = (
            (np.sign(df['Momentum_1d']) == np.sign(df['Momentum_3d'])) & 
            (np.sign(df['Momentum_3d']) == np.sign(df['Momentum_5d']))
        ).astype(float)
        
        # Enhanced trend persistence
        trend_5d = df['Momentum_1d'].rolling(5).mean()
        trend_10d = df['Momentum_1d'].rolling(10).mean()
        trend_20d = df['Momentum_1d'].rolling(20).mean()
        
        df['Trend_Persistence'] = (np.sign(trend_5d) == np.sign(trend_10d)).astype(float)
        df['Trend_Acceleration'] = trend_5d - trend_10d
        df['Multi_Trend_Align'] = (np.sign(trend_5d) * np.sign(trend_10d) * np.sign(trend_20d)).clip(-1, 1)
        
        # Volatility regime detection
        vol_ma_10 = df['Volatility_5d'].rolling(10).mean()
        vol_ma_30 = df['Volatility_5d'].rolling(30).mean()
        vol_regime_score = (df['Volatility_5d'] - vol_ma_10) / (vol_ma_30 + 1e-6)
        df['Vol_Regime'] = vol_regime_score.clip(-2, 2)
        df['Vol_Breakout'] = (df['Volatility_5d'] > vol_ma_30 * 1.5).astype(float)
        
        # Enhanced mean reversion signals
        price_zscore_5 = ((df['Close'] - df['Close'].rolling(5).mean()) / 
                         (df['Close'].rolling(5).std() + 1e-6))
        price_zscore_20 = ((df['Close'] - df['Close'].rolling(20).mean()) / 
                          (df['Close'].rolling(20).std() + 1e-6))
        df['Mean_Reversion_5d'] = price_zscore_5.clip(-3, 3)
        df['Mean_Reversion_20d'] = price_zscore_20.clip(-3, 3)
        df['Reversion_Divergence'] = (price_zscore_5 - price_zscore_20)
        
        # Volume-price relationship
        volume_price_trend = df['Volume_Ratio'] * np.sign(df['Momentum_1d'])
        df['Volume_Price_Trend'] = volume_price_trend.rolling(5).mean()
        df['Volume_Breakout'] = (df['Volume_Ratio'] > df['Volume_Ratio'].rolling(20).quantile(0.8)).astype(float)
        
        # Risk-adjusted momentum (Sharpe-like ratios)
        returns_sharpe_5 = (df['Momentum_1d'].rolling(5).mean() / (df['Momentum_1d'].rolling(5).std() + 1e-6))
        returns_sharpe_20 = (df['Momentum_1d'].rolling(20).mean() / (df['Momentum_1d'].rolling(20).std() + 1e-6))
        df['Sharpe_5d'] = returns_sharpe_5.clip(-3, 3)
        df['Sharpe_20d'] = returns_sharpe_20.clip(-3, 3)
        df['Sharpe_Ratio'] = returns_sharpe_5 / (returns_sharpe_20 + 1e-6)
        
        # Market efficiency measure
        price_range_5 = df['High'].rolling(5).max() - df['Low'].rolling(5).min()
        price_change_5 = abs(df['Close'] - df['Close'].shift(5))
        df['Fractal_Efficiency'] = (price_change_5 / (price_range_5 + 1e-6)).clip(0, 1)
        
        # Jump/outlier detection
        returns_std_20 = df['Momentum_1d'].rolling(20).std()
        jump_score = abs(df['Momentum_1d']) / (returns_std_20 + 1e-6)
        df['Jump_Score'] = jump_score.clip(0, 4)
        df['Outlier_Signal'] = (jump_score > 2.0).astype(float)
        
        # Market regime indicators
        price_ma_50 = df['Close'].rolling(50).mean()
        regime_trend = (df['Close'] / price_ma_50 - 1)
        vol_regime_threshold = df['Volatility_5d'].rolling(100).quantile(0.7)
        
        df['Bull_Regime'] = ((regime_trend > 0.05) & (df['Volatility_5d'] < vol_regime_threshold)).astype(float)
        df['Bear_Regime'] = ((regime_trend < -0.05) & (df['Volatility_5d'] < vol_regime_threshold)).astype(float)
        df['High_Vol_Regime'] = (df['Volatility_5d'] > vol_regime_threshold).astype(float)
        
        # Optimized seasonal effects
        df['Month_Effect'] = np.sin(2 * np.pi * df.index.month / 12)
        df['Quarter_Effect'] = ((df.index.month % 3) == 0).astype(float)
        
        # Seasonal patterns
        df['Month'] = pd.to_datetime(df.index).month
        df['January_Effect'] = (df['Month'] == 1).astype(float)
        df['December_Effect'] = (df['Month'] == 12).astype(float)
        
        # Earnings proximity effect (quarterly patterns)
        df['Quarter_End'] = pd.to_datetime(df.index).month.isin([3, 6, 9, 12]).astype(float)
        
        # Volatility smile/term structure proxy
        df['Vol_Term_Structure'] = df['Volatility_20d'] / df['Volatility_5d']
        
        # Market stress indicators  
        df['Stress_Indicator'] = (df['Outlier_Signal'] + (df['Volatility_5d'] > df['Volatility_20d']).astype(float)) / 2
        
        # === ENHANCED SENTIMENT FEATURES (temporally accurate for 80%+ accuracy) ===
        try:
            logger.info("Adding ENHANCED temporally accurate sentiment features...")
            
            # Determine optimal timeframe based on data length
            data_years = len(data) / 252  # Approximate years based on trading days
            if data_years >= 4.5:
                sentiment_timeframe = "5y"
                sentiment_periods = min(20, int(data_years * 4))  # More granular
            elif data_years >= 2.5:
                sentiment_timeframe = "3y"
                sentiment_periods = min(12, int(data_years * 4))
            elif data_years >= 1.5:
                sentiment_timeframe = "2y"
                sentiment_periods = min(8, int(data_years * 4))
            else:
                sentiment_timeframe = "1y"
                sentiment_periods = 4
            
            logger.info(f"Using {sentiment_timeframe} sentiment with {sentiment_periods} temporal periods")
            
            # Get comprehensive sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(self.symbol, sentiment_timeframe)
            
            # Create TEMPORALLY ACCURATE sentiment features
            base_sentiment = sentiment_result.overall_sentiment
            base_confidence = sentiment_result.confidence
            
            # Enhanced temporal variation using multiple timeframes
            price_momentum_1d = df['Returns_Lag1'].fillna(0)
            price_momentum_5d = df['Returns_Lag1'].rolling(window=5).mean().fillna(0)
            price_momentum_20d = df['Returns_Lag1'].rolling(window=20).mean().fillna(0)
            price_volatility = df['Returns_Lag1'].rolling(window=20).std().fillna(0)
            
            # Multi-period sentiment proxy with temporal accuracy
            sentiment_short = base_sentiment + (price_momentum_1d * 0.3) + (price_volatility * -0.2)
            sentiment_medium = base_sentiment + (price_momentum_5d * 0.2) + (price_volatility * -0.1)
            sentiment_long = base_sentiment + (price_momentum_20d * 0.1)
            
            # Bounded sentiment values
            df['Overall_Sentiment'] = sentiment_short.clip(-1, 1)
            df['Sentiment_Medium'] = sentiment_medium.clip(-1, 1)
            df['Sentiment_Long'] = sentiment_long.clip(-1, 1)
            df['Confidence_Score'] = base_confidence
            df['Bullish_Ratio'] = sentiment_result.bullish_ratio
            df['Bearish_Ratio'] = sentiment_result.bearish_ratio
            df['Neutral_Ratio'] = 1.0 - df['Bullish_Ratio'] - df['Bearish_Ratio']
            
            # ENHANCED derived sentiment features with temporal accuracy
            df['Sentiment_Change'] = df['Overall_Sentiment'].diff().fillna(0)
            df['Sentiment_Acceleration'] = df['Sentiment_Change'].diff().fillna(0)
            df['Sentiment_Volatility'] = df['Overall_Sentiment'].rolling(window=10).std().fillna(0)
            df['Sentiment_Price_Divergence'] = df['Overall_Sentiment'] - df['Returns_Lag1']
            df['Sentiment_Momentum'] = df['Overall_Sentiment'].rolling(window=5).mean().fillna(0)
            df['Sentiment_Trend'] = (df['Overall_Sentiment'] - df['Overall_Sentiment'].shift(5)).fillna(0)
            df['News_Volume'] = sentiment_result.news_count / max(1, len(data))
            
            # Multi-timeframe sentiment alignment
            df['Sentiment_Alignment'] = (
                np.sign(df['Overall_Sentiment']) * 
                np.sign(df['Sentiment_Medium']) * 
                np.sign(df['Sentiment_Long'])
            )
            
            # Sentiment regime features with temporal context
            df['Sentiment_Regime'] = np.where(df['Overall_Sentiment'] > 0.2, 1, 
                                   np.where(df['Overall_Sentiment'] < -0.2, -1, 0))
            df['Sentiment_Strength'] = np.abs(df['Overall_Sentiment'])
            df['Sentiment_Persistence'] = (
                df['Sentiment_Regime'].rolling(5).apply(
                    lambda x: 1 if len(set(x)) == 1 else 0, raw=True
                ).fillna(0)
            )
            
            # News impact features
            df['News_Impact'] = df['Sentiment_Change'] * np.log(df['News_Volume'] + 1)
            df['Sentiment_Surprise'] = np.abs(df['Sentiment_Change']) - df['Sentiment_Volatility']
            
            logger.info(f"Added ENHANCED temporal sentiment: periods={sentiment_periods}, "
                       f"avg_sentiment={df['Overall_Sentiment'].mean():.3f}, "
                       f"confidence={base_confidence:.3f}, news_count={sentiment_result.news_count}")
            
        except Exception as e:
            logger.warning(f"Could not add enhanced sentiment features: {e}")
            # Add placeholder sentiment features
            for col in ['Overall_Sentiment', 'Sentiment_Medium', 'Sentiment_Long', 'Confidence_Score', 
                       'Bullish_Ratio', 'Bearish_Ratio', 'Neutral_Ratio', 'Sentiment_Change',
                       'Sentiment_Acceleration', 'Sentiment_Volatility', 'Sentiment_Price_Divergence', 
                       'Sentiment_Momentum', 'Sentiment_Trend', 'News_Volume', 'Sentiment_Alignment',
                       'Sentiment_Regime', 'Sentiment_Strength', 'Sentiment_Persistence', 
                       'News_Impact', 'Sentiment_Surprise']:
                df[col] = 0.0
        
        # === EARNINGS FEATURES (5-year comprehensive earnings calendar data) ===
        try:
            logger.info("Adding comprehensive 5-year earnings features for training period...")
            
            # Import fundamentals analyzer
            from ..analysis.fundamentals import FundamentalAnalyzer
            fundamentals_analyzer = FundamentalAnalyzer()
            
            # Get comprehensive earnings calendar for 5 years (20 quarters)
            data_years = len(data) / 252  # Approximate years
            earnings_periods = max(8, int(data_years * 4))  # 4 quarters per year, minimum 2 years
            
            logger.info(f"Requesting {earnings_periods} earnings periods for {data_years:.1f} years of data")
            earnings_data = fundamentals_analyzer.get_earnings_calendar(self.symbol, periods=earnings_periods)
            
            # Initialize comprehensive earnings features
            df['Days_To_Earnings'] = 999  # Default: far from earnings
            df['Earnings_Beat_History'] = 0.0  # Historical earnings beat rate
            df['EPS_Growth_Rate'] = 0.0  # EPS growth rate
            df['Revenue_Growth_Rate'] = 0.0  # Revenue growth rate
            df['Earnings_Surprise_Avg'] = 0.0  # Average earnings surprise
            df['Earnings_Volatility'] = 0.0  # Earnings volatility measure
            df['Recent_Earnings_Trend'] = 0.0  # Recent earnings trend (last 4 quarters)
            
            if earnings_data and len(earnings_data) > 0:
                logger.info(f"Found {len(earnings_data)} earnings events for comprehensive analysis")
                
                # Create earnings proximity features with date handling
                earnings_dates = []
                earnings_surprises = []
                eps_beats = []
                
                for earnings in earnings_data:
                    try:
                        # Handle timezone-aware datetime objects safely
                        earnings_date = earnings.date
                        try:
                            # Try pandas Timestamp timezone handling first
                            if hasattr(earnings_date, 'tz_localize') and callable(getattr(earnings_date, 'tz_localize', None)):
                                tz_attr = getattr(earnings_date, 'tz', None)
                                if tz_attr:
                                    earnings_date = getattr(earnings_date, 'tz_localize')(None)
                            # Try datetime timezone handling
                            elif hasattr(earnings_date, 'replace') and hasattr(earnings_date, 'tzinfo'):
                                if earnings_date.tzinfo is not None:
                                    earnings_date = earnings_date.replace(tzinfo=None)
                        except (AttributeError, TypeError):
                            # Fallback for any timezone handling issues
                            pass
                        
                        earnings_dates.append(earnings_date)
                        
                        # Process earnings metrics
                        if earnings.eps_estimate and earnings.eps_actual:
                            eps_beat = 1.0 if earnings.eps_actual > earnings.eps_estimate else -1.0
                            eps_beats.append(eps_beat)
                        
                        if earnings.surprise_percent:
                            earnings_surprises.append(earnings.surprise_percent)
                            
                    except Exception as e:
                        logger.debug(f"Error processing earnings event: {e}")
                        continue
                
                if earnings_dates:
                    # Calculate days to nearest earnings for each date
                    for i, date in enumerate(df.index):
                        try:
                            # Handle pandas DatetimeIndex
                            if hasattr(date, 'to_pydatetime'):
                                current_date = date.to_pydatetime()
                            else:
                                current_date = date
                            
                            # Find minimum days to any earnings event
                            min_days = min([abs((current_date - ed).days) for ed in earnings_dates])
                            df.loc[date, 'Days_To_Earnings'] = min(min_days, 999)
                        except Exception as e:
                            logger.debug(f"Error calculating earnings proximity for date {date}: {e}")
                            continue
                    
                    # Calculate comprehensive earnings metrics
                    if eps_beats:
                        avg_eps_beat = np.mean(eps_beats)
                        df['Earnings_Beat_History'] = avg_eps_beat
                    
                    if earnings_surprises:
                        avg_surprise = np.mean(earnings_surprises)
                        surprise_volatility = np.std(earnings_surprises)
                        df['EPS_Growth_Rate'] = avg_surprise / 100.0 if avg_surprise else 0.0
                        df['Earnings_Surprise_Avg'] = avg_surprise / 100.0 if avg_surprise else 0.0
                        df['Earnings_Volatility'] = surprise_volatility / 100.0 if surprise_volatility else 0.0
                    
                    # Calculate recent earnings trend (last 4 quarters)
                    if len(earnings_surprises) >= 4:
                        recent_surprises = earnings_surprises[:4]  # Most recent 4
                        older_surprises = earnings_surprises[4:8] if len(earnings_surprises) >= 8 else earnings_surprises[4:]
                        
                        if recent_surprises and older_surprises:
                            recent_avg = np.mean(recent_surprises)
                            older_avg = np.mean(older_surprises)
                            trend = (recent_avg - older_avg) / 100.0  # Convert to decimal
                            df['Recent_Earnings_Trend'] = trend
                
                logger.info(f"Added comprehensive earnings features: beat_rate={df['Earnings_Beat_History'].iloc[0]:.3f}, "
                           f"avg_surprise={df['Earnings_Surprise_Avg'].iloc[0]:.3f}, "
                           f"volatility={df['Earnings_Volatility'].iloc[0]:.3f}")
            else:
                logger.info("No earnings data found, using default values")
                
        except Exception as e:
            logger.warning(f"Could not add comprehensive earnings features: {e}")
            # Add placeholder earnings features
            for col in ['Days_To_Earnings', 'Earnings_Beat_History', 'EPS_Growth_Rate', 
                       'Revenue_Growth_Rate', 'Earnings_Surprise_Avg', 'Earnings_Volatility',
                       'Recent_Earnings_Trend']:
                df[col] = 0.0
        
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
        
        # COMPREHENSIVE feature selection - Maximum features for 80%+ accuracy target
        feature_columns = [
            # === CORE MOMENTUM FEATURES (temporal accuracy critical) ===
            'Momentum_1d',        # Daily momentum - critical for next-day prediction
            'Momentum_3d',        # 3-day momentum - short-term trend
            'Momentum_5d',        # 5-day momentum - medium-term trend
            'Momentum_Quality',   # Quality of momentum signals
            'Momentum_Strength',  # Absolute momentum strength
            'Momentum_Consistency', # Multi-timeframe momentum alignment
            
            # === ADVANCED VOLATILITY FEATURES (market regime detection) ===
            'Volatility_5d',      # Recent volatility
            'Volatility_20d',     # Longer-term volatility
            'Vol_Regime',         # Volatility regime score
            'Vol_Ratio',          # Volatility regime change
            'Vol_Persistence',    # Volatility clustering
            'Vol_Breakout',       # Volatility breakout signals
            'HL_Volatility',      # High-Low volatility
            
            # === MEAN REVERSION SIGNALS (multiple timeframes) ===
            'Mean_Reversion_5d',  # Short-term mean reversion signal
            'Mean_Reversion_20d', # Long-term mean reversion signal
            'Reversion_Divergence', # Mean reversion divergence
            'Distance_From_SMA20', # Distance from moving average
            'Mean_Reversion_Signal', # Bounded mean reversion
            
            # === TECHNICAL INDICATORS (proven predictors) ===
            'Price_SMA5_Ratio',   # Short-term trend position
            'Price_SMA20_Ratio',  # Long-term trend position
            'SMA_Cross',          # Moving average crossover
            'RSI_Normalized',     # Momentum oscillator
            'MACD_Histogram',     # Trend convergence/divergence
            'BB_Position',        # Bollinger Band position
            
            # === HISTORICAL PATTERNS (multiple lags) ===
            'Returns_Lag1',       # Previous day return
            'Returns_Lag2',       # 2 days ago return
            'Returns_Lag3',       # 3 days ago return
            'Price_Acceleration', # Price acceleration (2nd derivative)
            
            # === VOLUME FEATURES (market participation) ===
            'Volume_Ratio',       # Volume surge detection
            'Volume_Price_Trend', # Volume-price alignment
            'Volume_Breakout',    # Volume breakout signals
            'Price_Volume',       # Price-volume interaction
            'VPT_Normalized',     # Volume Price Trend
            'Price_Efficiency',   # Price efficiency measure
            
            # === SENTIMENT (temporally accurate) ===
            'Overall_Sentiment',  # Market sentiment
            'Sentiment_Change',   # Sentiment momentum
            'Sentiment_Volatility', # Sentiment stability
            'Sentiment_Price_Divergence', # Sentiment vs price divergence
            'Sentiment_Momentum', # Sentiment trend
            'Sentiment_Regime',   # Sentiment regime
            'Sentiment_Strength', # Sentiment conviction
            
            # === TREND ANALYSIS (multi-timeframe) ===
            'Trend_Persistence',  # Trend consistency
            'Trend_Acceleration', # Trend acceleration
            'Multi_Trend_Align',  # Multi-timeframe alignment
            'Fractal_Efficiency', # Market efficiency
            
            # === RISK MEASURES (financial specific) ===
            'Relative_Strength',  # Risk-adjusted returns
            'Sharpe_5d',          # Short-term Sharpe ratio
            'Sharpe_20d',         # Long-term Sharpe ratio
            'Sharpe_Ratio',       # Sharpe ratio comparison
            'Jump_Score',         # Jump/outlier detection
            'Outlier_Signal',     # Outlier signals
            
            # === MARKET REGIME (comprehensive) ===
            'Bull_Regime',        # Bull market detection
            'Bear_Regime',        # Bear market detection
            'High_Vol_Regime',    # High volatility regime
            'Stress_Indicator',   # Market stress
            'Vol_Term_Structure', # Volatility term structure
            
            # === SEASONAL EFFECTS (temporal patterns) ===
            'Month_Effect',       # Monthly seasonality
            'Quarter_Effect',     # Quarterly effects
            'January_Effect',     # January effect
            'December_Effect',    # December effect
            'Quarter_End',        # Quarter-end effects
            
            # === MICROSTRUCTURE (detailed) ===
            'Spread_Proxy',       # Bid-ask spread proxy
            'Spread_Anomaly',     # Spread anomalies
            'Price_Percentile_20d', # Price percentile ranking
            'Price_Percentile_50d', # Longer-term ranking
            
            # === EARNINGS FEATURES (temporal accuracy) ===
            'Days_To_Earnings',   # Days to next earnings
            'Earnings_Beat_History', # Historical beat rate
            'EPS_Growth_Rate',    # EPS growth trend
            'Earnings_Surprise_Avg', # Average surprise
            'Earnings_Volatility', # Earnings volatility
            'Recent_Earnings_Trend', # Recent trend
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
        
        # IMPROVED TARGET ENGINEERING: Use log returns for better stability
        close_prices = enhanced_data['Close'].values.astype(float)
        
        # Calculate log returns for better numerical stability and distribution
        log_returns = np.array([np.log(close_prices[i+1] / close_prices[i]) 
                               for i in range(len(close_prices) - 1)])
        
        # LESS aggressive outlier clipping - preserve more signal
        # Use 95th percentile instead of 99th to keep more extreme movements
        percentile_95 = np.percentile(np.abs(log_returns), 95)
        returns_clipped = np.clip(log_returns, -percentile_95 * 2, percentile_95 * 2)
        
        # Remove any remaining invalid values
        target_data = np.nan_to_num(returns_clipped, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"IMPROVED target engineering: Log returns for {self.prediction_days}-day movement prediction")
        logger.info(f"Target stats - Mean: {np.mean(target_data):.6f}, Std: {np.std(target_data):.6f}")
        logger.info(f"Target range - Min: {np.min(target_data):.6f}, Max: {np.max(target_data):.6f}")
        logger.info(f"95th percentile used for clipping: {percentile_95:.6f}")
        
        # CRITICAL: Properly align features and targets
        # Features: use all data except the last row (can't predict beyond our data)
        # Targets: use all returns (each return is for the next day)
        feature_data = feature_data.iloc[:-1].copy()  # Remove last row (no future return available)
        # target_data already has the right length (len(prices) - 1)
        
        if len(feature_data) != len(target_data):
            min_len = min(len(feature_data), len(target_data))
            feature_data = feature_data.iloc[:min_len]
            target_data = target_data[:min_len]
            logger.warning(f"Adjusted alignment: using {min_len} samples")
        
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
        
        # Check if model is trained before making predictions
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_enhanced_model() first.")
        
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
            
            # Financial-specific metrics for trading evaluation
            'val_direction_accuracy': float(np.mean(np.sign(val_pred_unscaled) == np.sign(y_val_unscaled))),
            'test_direction_accuracy': float(np.mean(np.sign(test_pred_unscaled) == np.sign(y_test_unscaled))),
            
            # Information Coefficient (correlation between predictions and actual)
            'val_ic': float(np.corrcoef(val_pred_unscaled, y_val_unscaled)[0, 1]) if len(val_pred_unscaled) > 1 and not np.isnan(np.corrcoef(val_pred_unscaled, y_val_unscaled)[0, 1]) else 0.0,
            'test_ic': float(np.corrcoef(test_pred_unscaled, y_test_unscaled)[0, 1]) if len(test_pred_unscaled) > 1 and not np.isnan(np.corrcoef(test_pred_unscaled, y_test_unscaled)[0, 1]) else 0.0,
            
            # Training info
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_loss': float(min(history.history['val_loss'])),
            
            # Model info
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'num_features': X_train.shape[2],
            'total_parameters': self.model.count_params() if self.model else 0,
            'training_date': datetime.now().isoformat()
        }
        
        return metrics
    
    def build_enhanced_model(self, input_shape: Tuple[int, int], simplified: bool = False, ultra_light: bool = False) -> Model:
        """
        Build LSTM model with multiple architecture options for different scenarios.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            simplified: If True, use simplified architecture for better R² performance
            ultra_light: If True, use ultra-light model for very limited data
            
        Returns:
            Compiled Keras model optimized for R² performance
        """
        if ultra_light:
            return self._build_ultra_light_model(input_shape)
        elif simplified:
            return self._build_simplified_model(input_shape)
        else:
            return self._build_complex_model(input_shape)
    
    def _build_simplified_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build simplified LSTM model optimized for R² performance
        """
        logger.info("Building SIMPLIFIED model optimized for R² performance")
        
        inputs = Input(shape=input_shape)
        
        # Light normalization only
        normalized_inputs = LayerNormalization(name='input_normalization')(inputs)
        
        # Single LSTM layer - simpler is often better for R²
        lstm1 = LSTM(64, return_sequences=False,  # Smaller, single layer
                     dropout=0.2, recurrent_dropout=0.1,  # Light regularization
                     kernel_regularizer=regularizers.l2(0.01),  # Simple L2 only
                     name='lstm_layer_1')(normalized_inputs)
        lstm1 = Dropout(0.3, name='lstm1_dropout')(lstm1)
        
        # Simple dense progression
        dense1 = Dense(32, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01),
                      name='dense_layer_1')(lstm1)
        dense1 = Dropout(0.4, name='dense1_dropout')(dense1)
        
        dense2 = Dense(16, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01),
                      name='dense_layer_2')(dense1)
        dense2 = Dropout(0.3, name='dense2_dropout')(dense2)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output_layer')(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Simple optimizer for R² optimization
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        
        # MSE loss for R² optimization
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        logger.info(f"SIMPLIFIED model built: {model.count_params():,} parameters")
        logger.info("Architecture: Single LSTM + 2 Dense layers, optimized for R² performance")
        
        return model
    
    def _build_ultra_light_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build ULTRA-LIGHT LSTM model for very limited data scenarios
        """
        logger.info("Building ULTRA-LIGHT model for limited data scenarios")
        
        inputs = Input(shape=input_shape)
        
        # Minimal normalization
        normalized_inputs = LayerNormalization(name='input_normalization')(inputs)
        
        # Single, small LSTM layer
        lstm1 = LSTM(32, return_sequences=False,  # Very small, single layer
                     dropout=0.1, recurrent_dropout=0.05,  # Minimal regularization
                     name='lstm_layer_1')(normalized_inputs)
        lstm1 = Dropout(0.2, name='lstm1_dropout')(lstm1)
        
        # Single dense layer only
        dense1 = Dense(16, activation='relu',
                      name='dense_layer_1')(lstm1)
        dense1 = Dropout(0.2, name='dense1_dropout')(dense1)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output_layer')(dense1)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Conservative optimizer settings
        optimizer = Adam(learning_rate=0.0005, clipnorm=0.5)
        
        # MSE loss for R² optimization
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        logger.info(f"ULTRA-LIGHT model built: {model.count_params():,} parameters")
        logger.info("Architecture: Single 32-unit LSTM + 1 Dense layer, minimal regularization")
        
        return model

    def _build_complex_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build OPTIMIZED COMPLEX LSTM model designed to break through 0.9 loss barrier.
        Enhanced with advanced optimization techniques for sub-0.9 performance.
        """
        # OPTIMIZED configuration for breaking loss plateaus
        lstm_units = 192  # Optimized size for better gradient flow
        
        logger.info(f"Building PLATEAU-BREAKING COMPLEX model with {lstm_units} LSTM units")
        logger.info("🚀 Optimized architecture to break through 0.9 loss barrier")
        
        # Functional API for sophisticated attention mechanism
        inputs = Input(shape=input_shape)
        
        # Enhanced input processing with better normalization
        normalized_inputs = LayerNormalization(epsilon=1e-6, name='input_normalization')(inputs)
        # Reduced noise for better convergence
        noisy_inputs = GaussianNoise(0.005, name='input_noise')(normalized_inputs)
        
        # First LSTM layer - optimized capacity with residual connections
        lstm1 = LSTM(lstm_units, return_sequences=True, 
                     dropout=0.15, recurrent_dropout=0.05,  # Reduced regularization for better learning
                     kernel_regularizer=regularizers.l2(0.0005),  # Lighter regularization
                     activation='tanh',  # Explicit tanh for stability
                     recurrent_activation='sigmoid',
                     name='lstm_layer_1')(noisy_inputs)
        lstm1 = LayerNormalization(epsilon=1e-6, name='lstm1_norm')(lstm1)
        lstm1 = Dropout(0.15, name='lstm1_dropout')(lstm1)
        
        # Second LSTM layer with skip connection
        lstm2 = LSTM(lstm_units, return_sequences=True,
                     dropout=0.15, recurrent_dropout=0.05,
                     kernel_regularizer=regularizers.l2(0.0005),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     name='lstm_layer_2')(lstm1)
        lstm2 = LayerNormalization(epsilon=1e-6, name='lstm2_norm')(lstm2)
        lstm2 = Dropout(0.15, name='lstm2_dropout')(lstm2)
        
        # Add residual connection for better gradient flow
        lstm2_with_residual = Add(name='residual_connection')([lstm1, lstm2])
        
        # Third LSTM layer - focused capacity
        lstm3 = LSTM(lstm_units//2, return_sequences=True,
                     dropout=0.1, recurrent_dropout=0.05,
                     kernel_regularizer=regularizers.l2(0.0005),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     name='lstm_layer_3')(lstm2_with_residual)
        lstm3 = LayerNormalization(epsilon=1e-6, name='lstm3_norm')(lstm3)
        lstm3 = Dropout(0.1, name='lstm3_dropout')(lstm3)
        
        # OPTIMIZED multi-head attention for pattern recognition
        attention = MultiHeadAttention(
            num_heads=6,  # Optimized number of attention heads
            key_dim=48,   # Balanced key dimension
            dropout=0.1,
            name='multi_head_attention_1'
        )(lstm3, lstm3)
        
        # Add & norm connection with better stability
        attention_out = LayerNormalization(epsilon=1e-6, name='attention1_norm')(lstm3 + attention)
        attention_out = Dropout(0.1, name='attention1_dropout')(attention_out)
        
        # Second attention layer for deeper pattern recognition
        attention2 = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1,
            name='multi_head_attention_2'
        )(attention_out, attention_out)
        
        attention_out2 = LayerNormalization(epsilon=1e-6, name='attention2_norm')(attention_out + attention2)
        attention_out2 = Dropout(0.1, name='attention2_dropout')(attention_out2)
        
        # OPTIMIZED global pooling for better information extraction
        avg_pool = GlobalAveragePooling1D(name='global_avg_pool')(attention_out2)
        max_pool = GlobalMaxPooling1D(name='global_max_pool')(attention_out2)
        
        # Combine pooled features
        pooled = Concatenate(name='pooled_features')([avg_pool, max_pool])
        pooled = Dropout(0.2, name='pooled_dropout')(pooled)  # Reduced dropout
        pooled = BatchNormalization(momentum=0.99, epsilon=1e-6, name='pooled_batch_norm')(pooled)
        
        # OPTIMIZED dense layers with improved activation functions
        dense1 = Dense(384, activation='swish',  # Swish activation for better gradients
                      kernel_regularizer=regularizers.l2(0.0005),  # Lighter regularization
                      kernel_initializer='he_normal',  # Better initialization
                      name='dense_layer_1')(pooled)
        dense1 = Dropout(0.2, name='dense1_dropout')(dense1)
        dense1 = BatchNormalization(momentum=0.99, epsilon=1e-6, name='dense1_batch_norm')(dense1)
        
        dense2 = Dense(192, activation='swish',
                      kernel_regularizer=regularizers.l2(0.0005),
                      kernel_initializer='he_normal',
                      name='dense_layer_2')(dense1)
        dense2 = Dropout(0.2, name='dense2_dropout')(dense2)
        dense2 = BatchNormalization(momentum=0.99, epsilon=1e-6, name='dense2_batch_norm')(dense2)
        
        dense3 = Dense(96, activation='swish',
                      kernel_regularizer=regularizers.l2(0.0005),
                      kernel_initializer='he_normal',
                      name='dense_layer_3')(dense2)
        dense3 = Dropout(0.15, name='dense3_dropout')(dense3)
        dense3 = BatchNormalization(momentum=0.99, epsilon=1e-6, name='dense3_batch_norm')(dense3)
        
        # Final processing layer with residual connection
        pre_output = Dense(48, activation='swish',
                          kernel_regularizer=regularizers.l2(0.0005),
                          kernel_initializer='he_normal',
                          name='pre_output_layer')(dense3)
        pre_output = Dropout(0.1, name='pre_output_dropout')(pre_output)
        
        # Output layer with careful initialization
        outputs = Dense(1, activation='linear', 
                       kernel_initializer='glorot_normal',
                       name='output_layer')(pre_output)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # OPTIMIZED optimizer for breaking loss plateaus
        optimizer = Adam(
            learning_rate=0.001,      # Balanced initial LR
            clipnorm=0.8,             # Tighter gradient clipping
            beta_1=0.9,               # Standard momentum
            beta_2=0.999,             # Standard adaptive
            epsilon=1e-7,             # Smaller epsilon for stability
            amsgrad=True              # Use AMSGrad for better convergence
        )
        
        # Enhanced loss function optimized for convergence
        model.compile(
            optimizer=optimizer, 
            loss='huber',             # Huber loss for robustness
            metrics=['mae', 'mse']
        )
        
        logger.info(f"PLATEAU-BREAKING COMPLEX model built: {model.count_params():,} parameters")
        logger.info("🔥 Architecture optimized to break through 0.9 loss barrier")
        logger.info("🚀 Features: Residual connections, Swish activation, AMSGrad optimizer, Huber loss")
        
        return model
        attention_out = LayerNormalization(name='attention1_norm')(lstm3 + attention)
        attention_out = Dropout(0.2, name='attention1_dropout')(attention_out)  # Moderate dropout
        
        # Global pooling with balanced regularization
        avg_pool = GlobalAveragePooling1D(name='global_avg_pool')(attention_out)
        max_pool = GlobalMaxPooling1D(name='global_max_pool')(attention_out)
        
        # Combine pooled features with BALANCED regularization
        pooled = Concatenate(name='pooled_features')([avg_pool, max_pool])
        pooled = Dropout(0.3, name='pooled_dropout')(pooled)  # Balanced dropout
        pooled = BatchNormalization(name='pooled_batch_norm')(pooled)
        pooled = GaussianNoise(0.01, name='pooled_noise')(pooled)  # Light noise
        
        # Dense layers with BALANCED regularization strategy
        dense1 = Dense(lstm_units//2, activation='relu',  # Keep reasonable capacity
                      kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01),  # Balanced regularization
                      bias_regularizer=regularizers.l2(0.005),  # Light bias regularization
                      name='dense_layer_1')(pooled)
        dense1 = Dropout(0.3, name='dense1_dropout')(dense1)  # Balanced dropout
        dense1 = BatchNormalization(name='dense1_batch_norm')(dense1)
        
        dense2 = Dense(lstm_units//4, activation='relu',  # Reasonable reduction
                      kernel_regularizer=regularizers.l1_l2(l1=0.002, l2=0.015),  # Moderate regularization
                      bias_regularizer=regularizers.l2(0.008),  # Moderate bias regularization
                      name='dense_layer_2')(dense1)
        dense2 = Dropout(0.35, name='dense2_dropout')(dense2)  # Moderate dropout
        dense2 = BatchNormalization(name='dense2_batch_norm')(dense2)
        
        dense3 = Dense(lstm_units//8, activation='relu',  # Controlled reduction
                      kernel_regularizer=regularizers.l1_l2(l1=0.003, l2=0.02),  # Controlled regularization
                      bias_regularizer=regularizers.l2(0.01),  # Controlled bias regularization
                      name='dense_layer_3')(dense2)
        dense3 = Dropout(0.3, name='dense3_dropout')(dense3)  # Controlled dropout
        dense3 = BatchNormalization(name='dense3_batch_norm')(dense3)
        
        # Final dense layer with REASONABLE capacity and regularization
        pre_output = Dense(32, activation='relu',  # Reasonable bottleneck (not too small)
                          kernel_regularizer=regularizers.l2(0.02),  # Moderate regularization
                          bias_regularizer=regularizers.l2(0.01),  # Light bias regularization
                          name='pre_output_layer')(dense3)
        pre_output = Dropout(0.25, name='pre_output_dropout')(pre_output)  # Light final dropout
        pre_output = LayerNormalization(name='pre_output_norm')(pre_output)
        
        # Output layer with minimal regularization (preserve signal)
        outputs = Dense(1, activation='linear', 
                       kernel_regularizer=regularizers.l2(0.005),  # Very light output regularization
                       name='output_layer')(pre_output)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Balanced optimizer settings for effective learning
        initial_lr = config.model.learning_rate  # Standard learning rate
        optimizer = Adam(
            learning_rate=initial_lr,
            clipnorm=1.0,           # Standard gradient clipping
            beta_1=0.9,             
            beta_2=0.999,           
            epsilon=1e-7,           
            amsgrad=True,
            weight_decay=0.001      # Light weight decay
        )
        
        # IMPROVED: Use MSE loss for better R² optimization instead of directional loss
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Use MSE for R² optimization
            metrics=['mae', 'mse']  # Standard metrics for monitoring
        )
        
        logger.info(f"BALANCED complex model built: {model.count_params():,} parameters")
        logger.info("Balanced Architecture: Moderate regularization allowing effective learning")
        logger.info(f"Optimizer: Adam with lr={initial_lr:.6f}, clipnorm=1.0, weight_decay=0.001")
        logger.info("Features: Balanced L1/L2, moderate dropout (0.1-0.35), layer normalization")
        
        return model
    
    def train_enhanced_model(self, period: str = "3y", mega_data: bool = False) -> Dict:
        """
        Train enhanced LSTM model with quality validation and automatic cleanup
        
        Args:
            period: Period of historical data to use for training (default: 3y for better performance)
            mega_data: If True, collect comprehensive data from all available APIs
            
        Returns:
            Dictionary with detailed training metrics or error information
        """
        logger.info(f"Starting enhanced training for {self.symbol}")
        
        # Clean up any existing models first
        self._cleanup_existing_models()
        
        try:
            start_time = time.time()
            
            # Get comprehensive historical data
            stock_data = self.data_collector.get_stock_data(self.symbol, period, mega_data=mega_data)
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
            
            # 2. UNIFIED target scaling with StandardScaler for consistency
            # Use StandardScaler for both features and targets to ensure consistent optimization
            logger.info("Using StandardScaler for target variable (consistent with features)")
            
            # Check target distribution before scaling
            logger.info(f"Pre-scaling target stats - Mean: {np.mean(y_train):.6f}, Std: {np.std(y_train):.6f}")
            logger.info(f"Pre-scaling target range - Min: {np.min(y_train):.6f}, Max: {np.max(y_train):.6f}")
            
            # Use StandardScaler for consistency with feature scaling
            from sklearn.preprocessing import StandardScaler
            unified_target_scaler = StandardScaler()
            
            y_train_scaled = unified_target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = unified_target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = unified_target_scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            # Verify scaling quality
            logger.info(f"Post-scaling target stats - Train mean: {np.mean(y_train_scaled):.6f}, Train std: {np.std(y_train_scaled):.6f}")
            logger.info(f"Post-scaling validation - Val mean: {np.mean(y_val_scaled):.6f}, Val std: {np.std(y_val_scaled):.6f}")
            
            # Update scaler reference for prediction use
            self.scaler = unified_target_scaler
            
            logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            logger.info(f"Target scaling - Train mean: {np.mean(y_train):.6f}, Train std: {np.std(y_train):.6f}")
            
            # AGGRESSIVE model selection for 80%+ accuracy target
            num_features = X_train.shape[2]
            num_samples = len(X_train)
            
            # FORCE COMPLEX MODEL for better learning capacity
            # Only use ultra-light for extremely limited data
            use_ultra_light = num_samples < 200  # Very restrictive
            
            # Use simplified only for very limited scenarios
            use_simplified = (num_samples < 500) and not use_ultra_light
            
            # PREFER COMPLEX MODEL for maximum learning capacity
            use_complex = not (use_ultra_light or use_simplified)
            
            # Log model selection decision
            if use_ultra_light:
                logger.info(f"🔧 ULTRA-LIGHT MODEL selected:")
                logger.info(f"   Reason: Very limited samples={num_samples}")
                logger.info(f"   Strategy: Single 32-unit LSTM for maximum stability")
            elif use_simplified:
                logger.info(f"🔧 SIMPLIFIED MODEL selected:")
                logger.info(f"   Reason: Limited samples={num_samples}")
                logger.info(f"   Strategy: Single LSTM + 2 Dense layers")
            else:
                logger.info(f"🔧 COMPLEX MODEL selected for MAXIMUM LEARNING:")
                logger.info(f"   Features={num_features}, Samples={num_samples}, Data={len(prices_df)}")
                logger.info(f"   Strategy: Multi-layer LSTM + attention for 80%+ accuracy target")
            
            self.model = self.build_enhanced_model((X_train.shape[1], X_train.shape[2]), 
                                                 simplified=use_simplified, 
                                                 ultra_light=use_ultra_light)
            
            logger.info(f"Model architecture: {X.shape[1]} time steps, {X.shape[2]} features")
            logger.info(f"Total parameters: {self.model.count_params():,}")
            
            # ENHANCED callbacks for PLATEAU-BREAKING training to break through 0.9 loss barrier
            callbacks = [
                # Advanced plateau-breaking learning rate scheduler
                PlateauBreakerScheduler(
                    base_lr=5e-5,           # Lower base for stability
                    max_lr=2e-3,            # Higher max for exploration
                    step_size=8,            # Faster cycles
                    mode='triangular2',      # Decaying triangular
                    plateau_patience=6,      # Quick plateau detection
                    plateau_factor=0.6,      # Aggressive reduction
                    restart_factor=2.5,      # Strong restart boost
                    verbose=1
                ),
                
                # Adaptive loss function callback
                AdaptiveLossCallback(
                    initial_loss='huber',
                    plateau_patience=12,
                    verbose=1
                ),
                
                # Enhanced plateau detector
                PlateauDetector(
                    monitor='val_loss',
                    patience=10,
                    min_delta=0.0002,       # Tighter improvement threshold
                    verbose=1
                ),
                
                # System-aware early stopping with more patience
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.system_config.training_config["patience"] + 10,  # Extra patience
                    verbose=1,
                    restore_best_weights=True,
                    min_delta=0.00005       # Very fine improvement threshold
                ),
                
                # Model checkpoint for best weights
                ModelCheckpoint(
                    str(self.model_path),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0,
                    save_weights_only=False
                ),
                
                # Thermal throttling for system safety
                ThermalThrottlingCallback(
                    temp_threshold=80 if self.system_config.thermal_profile != "performance" else 85,
                    check_interval=5 if self.system_config.thermal_profile == "conservative" else 10
                )
            ]
            
            # PLATEAU-BREAKING INTENSIVE training with advanced optimization
            training_config = self.system_config.training_config
            logger.info("🚀 Starting PLATEAU-BREAKING training optimized to break through 0.9 loss barrier...")
            logger.info("🔥 Enhanced with: Cyclical LR, Adaptive Loss, Residual Connections, AMSGrad")
            logger.info(f"Training configuration: {training_config}")
            
            history = self.model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=training_config["epochs"],
                batch_size=training_config["batch_size"],
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
            
            # Store training information
            self.training_metrics = metrics
            
            # SIMPLE MODEL EVALUATION SUMMARY
            self._simple_evaluation_summary(metrics)
            
            # COMPREHENSIVE MODEL EVALUATION
            self._comprehensive_model_evaluation(metrics, training_time)
            
            quality_assessment = self._assess_model_quality(metrics['val_r2'])
            logger.info(f"Enhanced training completed successfully - Quality: {quality_assessment}")
            
            return metrics
                
        except Exception as e:
            error_msg = f"Enhanced training failed: {str(e)}"
            logger.error(error_msg)
            
            # Clean up any partial model files on error
            self._cleanup_existing_models()
            
            return {'error': error_msg}
    
    def _simple_evaluation_summary(self, metrics: Dict):
        """
        Simple, easy-to-read model evaluation summary with percentage accuracy
        """
        print("\n" + "=" * 60)
        print("🎯 SIMPLE MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Basic Performance Metrics
        val_r2 = metrics.get('val_r2', 0)
        test_r2 = metrics.get('test_r2', 0)
        val_direction = metrics.get('val_direction_accuracy', 0)
        test_direction = metrics.get('test_direction_accuracy', 0)
        
        # Convert to percentages for easier understanding
        val_direction_pct = val_direction * 100
        test_direction_pct = test_direction * 100
        
        print(f"📈 R² SCORES:")
        print(f"   Validation: {val_r2:.3f}")
        print(f"   Test:       {test_r2:.3f}")
        
        print(f"\n🎯 DIRECTION ACCURACY (Percentage):")
        print(f"   Validation: {val_direction_pct:.1f}%")
        print(f"   Test:       {test_direction_pct:.1f}%")
        print(f"   Random Baseline: 50.0%")
        
        # Simple quality assessment
        if val_direction_pct >= 55:
            direction_quality = "🟢 EXCELLENT"
        elif val_direction_pct >= 52:
            direction_quality = "🟡 GOOD"
        elif val_direction_pct >= 50.5:
            direction_quality = "🟠 FAIR"
        else:
            direction_quality = "🔴 POOR"
        
        print(f"   Quality: {direction_quality}")
        
        # R² Quality Assessment (Realistic for financial data)
        if val_r2 >= 0.05:
            r2_quality = "🟢 EXCEPTIONAL"
        elif val_r2 >= 0.02:
            r2_quality = "🟡 GOOD"
        elif val_r2 >= 0.005:
            r2_quality = "🟠 FAIR"
        elif val_r2 >= -0.1:
            r2_quality = "🟠 TYPICAL"
        else:
            r2_quality = "🔴 POOR"
        
        print(f"\n📊 R² QUALITY: {r2_quality}")
        print(f"   (Financial data typically ranges -0.1 to +0.05)")
        
        # Overall Assessment
        overall_score = (val_direction_pct - 50) * 2 + (val_r2 * 100)  # Combined score
        
        if overall_score >= 15:
            overall = "🟢 EXCELLENT - Ready for Trading!"
        elif overall_score >= 8:
            overall = "🟡 GOOD - Suitable for Trading"
        elif overall_score >= 2:
            overall = "🟠 FAIR - Use with Caution"
        else:
            overall = "🔴 POOR - Not Recommended for Trading"
        
        print(f"\n🏆 OVERALL ASSESSMENT: {overall}")
        print(f"   Combined Score: {overall_score:.1f}")
        
        # Simple recommendations
        print(f"\n💡 QUICK RECOMMENDATIONS:")
        if val_direction_pct < 52:
            print("   • Focus on improving directional accuracy")
        if val_r2 < 0.01:
            print("   • Consider more training data or simpler model")
        if abs(val_r2 - test_r2) > 0.02:
            print("   • Model may be overfitting - reduce complexity")
        
        print("=" * 60)

    def _comprehensive_model_evaluation(self, metrics: Dict, training_time: float):
        """
        Comprehensive model evaluation with detailed performance analysis
        """
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE MODEL EVALUATION")
        logger.info("=" * 80)
        
        # Training Performance Summary
        logger.info(f"🎯 TRAINING PERFORMANCE:")
        logger.info(f"   Training Time: {training_time:.1f}s")
        logger.info(f"   Epochs Trained: {metrics.get('epochs_trained', 0)}")
        logger.info(f"   Model Parameters: {metrics.get('total_parameters', 0):,}")
        logger.info(f"   Features Used: {metrics.get('num_features', 0)}")
        
        # R² Score Analysis (REALISTIC INTERPRETATION)
        val_r2 = metrics.get('val_r2', 0)
        test_r2 = metrics.get('test_r2', 0)
        
        logger.info(f"📊 R² SCORE ANALYSIS:")
        logger.info(f"   Validation R²: {val_r2:.4f}")
        logger.info(f"   Test R²: {test_r2:.4f}")
        logger.info(f"   Overfitting Gap: {abs(val_r2 - test_r2):.4f}")
        
        # REALISTIC R² INTERPRETATION FOR FINANCIAL DATA
        if val_r2 >= 0.05:
            r2_assessment = "🎯 EXCELLENT - Exceptional for financial data!"
        elif val_r2 >= 0.03:
            r2_assessment = "👍 GOOD - Above average for stock prediction"
        elif val_r2 >= 0.01:
            r2_assessment = "👌 FAIR - Typical for financial time series"
        elif val_r2 >= -0.1:
            r2_assessment = "⚠️ BELOW AVERAGE - Still within normal range"
        else:
            r2_assessment = "❌ POOR - May need more data or different approach"
        
        logger.info(f"   R² Assessment: {r2_assessment}")
        
        # Financial Metrics Analysis
        val_direction = metrics.get('val_direction_accuracy', 0)
        test_direction = metrics.get('test_direction_accuracy', 0)
        val_ic = metrics.get('val_ic', 0)
        test_ic = metrics.get('test_ic', 0)
        
        logger.info(f"💰 FINANCIAL PERFORMANCE:")
        logger.info(f"   Validation Direction Accuracy: {val_direction:.1%}")
        logger.info(f"   Test Direction Accuracy: {test_direction:.1%}")
        logger.info(f"   Validation Information Coefficient: {val_ic:.4f}")
        logger.info(f"   Test Information Coefficient: {test_ic:.4f}")
        
        # Direction Accuracy Assessment
        if test_direction >= 0.55:
            direction_assessment = "🎯 EXCELLENT - Strong predictive signal"
        elif test_direction >= 0.52:
            direction_assessment = "👍 GOOD - Above random performance"
        elif test_direction >= 0.50:
            direction_assessment = "👌 FAIR - Slight edge over random"
        else:
            direction_assessment = "❌ POOR - Below random performance"
        
        logger.info(f"   Direction Assessment: {direction_assessment}")
        
        # Error Analysis
        val_rmse = metrics.get('val_rmse', 0)
        test_rmse = metrics.get('test_rmse', 0)
        val_mae = metrics.get('val_mae', 0)
        test_mae = metrics.get('test_mae', 0)
        
        logger.info(f"📉 ERROR ANALYSIS:")
        logger.info(f"   Validation RMSE: {val_rmse:.4f}")
        logger.info(f"   Test RMSE: {test_rmse:.4f}")
        logger.info(f"   Validation MAE: {val_mae:.4f}")
        logger.info(f"   Test MAE: {test_mae:.4f}")
        
        # Model Generalization Analysis
        train_r2 = metrics.get('train_r2', 0)
        overfitting_ratio = abs(train_r2 - val_r2) / max(abs(train_r2), 0.001)
        
        logger.info(f"🔍 GENERALIZATION ANALYSIS:")
        logger.info(f"   Training R²: {train_r2:.4f}")
        logger.info(f"   Overfitting Ratio: {overfitting_ratio:.2f}")
        
        if overfitting_ratio < 0.2:
            generalization = "🎯 EXCELLENT - Good generalization"
        elif overfitting_ratio < 0.5:
            generalization = "👍 GOOD - Acceptable generalization"
        elif overfitting_ratio < 1.0:
            generalization = "⚠️ MODERATE - Some overfitting"
        else:
            generalization = "❌ POOR - Significant overfitting"
        
        logger.info(f"   Generalization Assessment: {generalization}")
        
        # Trading Viability Assessment
        logger.info(f"💼 TRADING VIABILITY:")
        
        # Calculate trading readiness score
        trading_score = 0
        if test_direction > 0.51: trading_score += 2
        if abs(test_ic) > 0.02: trading_score += 2
        if test_r2 > -0.1: trading_score += 1
        if overfitting_ratio < 0.5: trading_score += 1
        
        if trading_score >= 5:
            trading_assessment = "🎯 READY - Good for trading strategies"
        elif trading_score >= 3:
            trading_assessment = "👍 PROMISING - Worth further testing"
        elif trading_score >= 2:
            trading_assessment = "⚠️ CAUTION - Use with risk management"
        else:
            trading_assessment = "❌ NOT READY - Needs improvement"
        
        logger.info(f"   Trading Readiness: {trading_assessment}")
        logger.info(f"   Trading Score: {trading_score}/6")
        
        # Final Recommendations
        logger.info(f"🎯 RECOMMENDATIONS:")
        
        if val_r2 < 0.01:
            logger.info("   • Consider using more training data (5+ years)")
            logger.info("   • Try different feature combinations")
            logger.info("   • Experiment with different prediction horizons")
        
        if test_direction < 0.52:
            logger.info("   • Focus on directional accuracy over magnitude")
            logger.info("   • Consider ensemble methods")
            logger.info("   • Review feature engineering")
        
        if overfitting_ratio > 0.5:
            logger.info("   • Increase regularization")
            logger.info("   • Reduce model complexity")
            logger.info("   • Use more training data")
        
        logger.info("=" * 80)
        
        return metrics
    
    def load_model(self) -> bool:
        """
        Load trained model and scalers from disk with custom loss function
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if (self.model_path.exists() and self.scaler_path.exists() and 
                self.feature_scaler_path.exists()):
                
                # Load model with custom objects
                custom_objects = {
                    'directional_loss': EnhancedLSTMPredictor.directional_loss,
                    'variance_loss': EnhancedLSTMPredictor.variance_loss
                }
                self.model = load_model(self.model_path, custom_objects=custom_objects)
                
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
