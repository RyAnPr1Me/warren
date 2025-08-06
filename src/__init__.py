"""
Package initialization for Stock AI system
"""

from .data.collector import data_collector
from .analysis.technical import technical_analyzer
from .config import config

__version__ = "0.1.0"
__author__ = "Stock AI Team"

# Export main components
__all__ = [
    "data_collector",
    "technical_analyzer", 
    "config"
]
