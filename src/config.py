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
    """Machine learning model configuration"""
    data_path: Path = Path("data/models/")
    sequence_length: int = 60  # Days of historical data for LSTM
    prediction_days: int = 60  # Days to predict forward
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    def __post_init__(self):
        self.data_path = Path(os.getenv('MODEL_DATA_PATH', str(self.data_path)))
        self.data_path.mkdir(parents=True, exist_ok=True)


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
