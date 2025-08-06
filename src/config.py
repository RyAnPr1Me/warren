"""
Stock AI System Configuration
Centralized configuration management for the entire system
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class APIConfig:
    """API configuration settings"""
    alpha_vantage_key: Optional[str] = None
    fmp_key: Optional[str] = None
    finnhub_key: Optional[str] = None
    
    def __post_init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///stock_ai.db"
    
    def __post_init__(self):
        self.url = os.getenv('DATABASE_URL', self.url)


@dataclass
class ModelConfig:
    """Configuration for LSTM model training and prediction"""
    
    # Data settings
    model_data_path: Path = Path("data")  # Base path for model data storage
    
    # Model architecture - Balanced size with attention and overfitting prevention
    sequence_length: int = 30          # Shorter context for less overfitting
    lstm_units: int = 128              # Reduced capacity to prevent overfitting
    dropout_rate: float = 0.5          # Higher dropout for regularization
    
    # Training parameters - Anti-overfitting focus
    batch_size: int = 64               # Larger batches for more stable gradients
    epochs: int = 100                  # Reasonable epochs with early stopping
    learning_rate: float = 0.0005       # Lower learning rate for stability
    patience: int = 20                 # More patience for proper convergence
    validation_split: float = 0.15     # Validation data split
    test_split: float = 0.15           # Test data split
    
    # Quality thresholds - REALISTIC for stock prediction
    min_r2_score: float = 0.02         # Realistic minimum R² for stock prediction
    
    # Prediction settings
    prediction_days: int = 15          # Default: 3 weeks (15 trading days) 
    confidence_level: float = 0.95     # Confidence level for intervals
    
    # Buy/Sell rating thresholds (percentage changes)
    strong_buy_threshold: float = 0.10   # 10%+ predicted gain = Strong Buy
    buy_threshold: float = 0.05         # 5%+ predicted gain = Buy  
    sell_threshold: float = -0.05       # 5%+ predicted loss = Sell
    strong_sell_threshold: float = -0.10 # 10%+ predicted loss = Strong Sell
    
    # Data storage
    data_path: Path = Path("data/models")  # Path for model storage
    
    def __post_init__(self):
        # Allow override from environment
        if model_data_path := os.getenv('MODEL_DATA_PATH'):
            self.data_path = Path(model_data_path)


@dataclass
class CacheConfig:
    """Caching configuration"""
    redis_url: str = "redis://localhost:6379/0"
    expiry_minutes: int = 30
    
    def __post_init__(self):
        self.redis_url = os.getenv('REDIS_URL', self.redis_url)
        self.expiry_minutes = int(os.getenv('CACHE_EXPIRY_MINUTES', str(self.expiry_minutes)))


@dataclass
class FeatureFlags:
    """Feature flags for enabling/disabling functionality"""
    enable_real_time_data: bool = True
    enable_sentiment_analysis: bool = False  # Phase 3
    enable_news_analysis: bool = False       # Phase 3
    
    def __post_init__(self):
        self.enable_real_time_data = os.getenv('ENABLE_REAL_TIME_DATA', 'true').lower() == 'true'
        self.enable_sentiment_analysis = os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'false').lower() == 'true'
        self.enable_news_analysis = os.getenv('ENABLE_NEWS_ANALYSIS', 'false').lower() == 'true'


class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Load environment variables from .env file if it exists
        self._load_env_file()
        
        # Initialize configuration sections
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.cache = CacheConfig()
        self.features = FeatureFlags()
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = Path(os.getenv('LOG_FILE', 'logs/stock_ai.log'))
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # API server configuration
        self.api_host = os.getenv('API_HOST', 'localhost')
        self.api_port = int(os.getenv('API_PORT', '8000'))
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check for required API keys for Phase 1
        if not self.api.alpha_vantage_key:
            issues.append("Alpha Vantage API key not configured")
        
        # Validate model configuration
        if self.model.sequence_length <= 0:
            issues.append("Model sequence length must be positive")
        
        if self.model.prediction_days <= 0:
            issues.append("Model prediction days must be positive")
        
        return issues


# Global configuration instance
config = Config()
