"""
Utility functions for the Stock AI system
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Normalized symbol (uppercase)
        
    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Basic validation - symbols are typically 1-5 characters
    if not symbol.isalpha() or len(symbol) < 1 or len(symbol) > 5:
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    return symbol


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount for display
    
    Args:
        amount: Currency amount
        currency: Currency code (default: USD)
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage for display
    
    Args:
        value: Percentage value
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def format_large_number(number: Union[int, float]) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B, T)
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    if abs(number) >= 1e12:
        return f"{number/1e12:.1f}T"
    elif abs(number) >= 1e9:
        return f"{number/1e9:.1f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.1f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:.0f}"


def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of trading days between two dates
    Approximation: excludes weekends but not holidays
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of trading days
    """
    total_days = (end_date - start_date).days
    weeks = total_days // 7
    remaining_days = total_days % 7
    
    # Count weekdays in remaining days
    weekdays = 0
    current_date = start_date + timedelta(days=weeks * 7)
    
    for i in range(remaining_days):
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            weekdays += 1
        current_date += timedelta(days=1)
    
    return weeks * 5 + weekdays


def calculate_volatility_score(volatility: float) -> str:
    """
    Convert volatility percentage to descriptive score
    
    Args:
        volatility: Volatility percentage
        
    Returns:
        Volatility score (Low, Medium, High, Extreme)
    """
    if volatility < 15:
        return "Low"
    elif volatility < 25:
        return "Medium"
    elif volatility < 40:
        return "High"
    else:
        return "Extreme"


def calculate_rsi_signal(rsi: float) -> str:
    """
    Convert RSI value to signal
    
    Args:
        rsi: RSI value (0-100)
        
    Returns:
        RSI signal description
    """
    if rsi >= 70:
        return "Overbought"
    elif rsi <= 30:
        return "Oversold"
    elif rsi >= 60:
        return "Strong"
    elif rsi <= 40:
        return "Weak"
    else:
        return "Neutral"


def save_json_data(data: Dict, file_path: Path) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Data saved to {file_path}")


def load_json_data(file_path: Path) -> Optional[Dict]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data or None if file doesn't exist
    """
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Data loaded from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None


def ensure_directory(directory: Path) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
    """
    directory.mkdir(parents=True, exist_ok=True)


def get_cache_key(symbol: str, data_type: str, **kwargs) -> str:
    """
    Generate cache key for data storage
    
    Args:
        symbol: Stock symbol
        data_type: Type of data (prices, technical, etc.)
        **kwargs: Additional parameters
        
    Returns:
        Cache key string
    """
    key_parts = [symbol.upper(), data_type]
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    
    return "_".join(key_parts)


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    
    return ((new_value - old_value) / old_value) * 100


def is_market_hours() -> bool:
    """
    Check if current time is during market hours (approximate)
    US market: 9:30 AM - 4:00 PM ET, Monday-Friday
    
    Returns:
        True if during market hours
    """
    now = datetime.now()
    
    # Check if weekday (Monday = 0, Friday = 4)
    if now.weekday() > 4:
        return False
    
    # Simple approximation - doesn't account for timezone or holidays
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def get_business_days_ahead(days: int) -> datetime:
    """
    Get date that is specified business days ahead
    
    Args:
        days: Number of business days ahead
        
    Returns:
        Future business date
    """
    current_date = datetime.now()
    business_days_added = 0
    
    while business_days_added < days:
        current_date += timedelta(days=1)
        # Skip weekends
        if current_date.weekday() < 5:
            business_days_added += 1
    
    return current_date


def round_to_significant_figures(number: float, sig_figs: int = 3) -> float:
    """
    Round number to specified significant figures
    
    Args:
        number: Number to round
        sig_figs: Number of significant figures
        
    Returns:
        Rounded number
    """
    if number == 0:
        return 0
    
    import math
    
    # Calculate the order of magnitude
    magnitude = math.floor(math.log10(abs(number)))
    
    # Calculate the factor to multiply by
    factor = 10 ** (sig_figs - 1 - magnitude)
    
    # Round and return
    return round(number * factor) / factor


class Timer:
    """Simple timer utility for performance measurement"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return timedelta(0)
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed_time.total_seconds()


def create_error_response(error_message: str, error_code: str = "UNKNOWN") -> Dict:
    """
    Create standardized error response
    
    Args:
        error_message: Error message
        error_code: Error code
        
    Returns:
        Error response dictionary
    """
    return {
        'success': False,
        'error': {
            'code': error_code,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
    }


def create_success_response(data: Dict, message: str = "Success") -> Dict:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        Success response dictionary
    """
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
