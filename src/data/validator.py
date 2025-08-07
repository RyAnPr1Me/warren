"""
Data Validation Module for Stock Market Data

This module provides comprehensive validation for stock data to ensure
quality before training ML models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    stats: Dict[str, float]
    cleaned_data: Optional[pd.DataFrame] = None

class StockDataValidator:
    """Comprehensive stock data validation and cleaning"""
    
    def __init__(self, min_data_points: int = 1250):  # ~5 years of trading days (252 * 5)
        self.min_data_points = min_data_points
        
    def validate_and_clean(self, data: pd.DataFrame, symbol: str) -> ValidationResult:
        """
        Comprehensive validation and cleaning of stock data
        
        Args:
            data: Raw stock data with OHLCV columns
            symbol: Stock symbol for logging
            
        Returns:
            ValidationResult with validation status and cleaned data
        """
        issues = []
        warnings = []
        stats = {}
        
        logger.info(f"Starting validation for {symbol} with {len(data)} rows")
        
        # 1. Basic structure validation
        if not self._validate_structure(data):
            issues.append("Invalid data structure - missing required columns")
            return ValidationResult(False, issues, warnings, stats)
        
        # 2. Data sufficiency check
        if len(data) < self.min_data_points:
            issues.append(f"Insufficient data: {len(data)} rows < {self.min_data_points} minimum")
            return ValidationResult(False, issues, warnings, stats)
        
        # 3. Clean the data step by step
        cleaned_data = data.copy()
        
        # Remove duplicates
        initial_count = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        if len(cleaned_data) < initial_count:
            warnings.append(f"Removed {initial_count - len(cleaned_data)} duplicate rows")
        
        # 4. Handle missing values
        missing_before = cleaned_data.isnull().sum().sum()
        cleaned_data = self._handle_missing_values(cleaned_data)
        missing_after = cleaned_data.isnull().sum().sum()
        if missing_before > 0:
            warnings.append(f"Handled {missing_before} missing values, {missing_after} remaining")
        
        # 5. Validate and fix price data
        price_issues = self._validate_prices(cleaned_data)
        if price_issues:
            warnings.extend(price_issues)
            cleaned_data = self._fix_price_anomalies(cleaned_data)
        
        # 6. Validate volume data
        volume_issues = self._validate_volume(cleaned_data)
        if volume_issues:
            warnings.extend(volume_issues)
            cleaned_data = self._fix_volume_anomalies(cleaned_data)
        
        # 7. Check for data continuity
        continuity_issues = self._check_continuity(cleaned_data)
        if continuity_issues:
            warnings.extend(continuity_issues)
        
        # 8. Calculate statistics
        stats = self._calculate_stats(cleaned_data)
        
        # 9. Final validation
        if len(cleaned_data) < self.min_data_points:
            issues.append(f"After cleaning: {len(cleaned_data)} rows < {self.min_data_points} minimum")
            return ValidationResult(False, issues, warnings, stats)
        
        # 10. Sort by date to ensure chronological order
        if 'Date' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values('Date')
        else:
            cleaned_data = cleaned_data.sort_index()
        
        logger.info(f"Validation complete for {symbol}: {len(cleaned_data)} clean rows")
        
        return ValidationResult(
            is_valid=True,
            issues=issues,
            warnings=warnings,
            stats=stats,
            cleaned_data=cleaned_data
        )
    
    def _validate_structure(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For OHLC, forward fill then backward fill
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # For volume, fill with median
        if 'Volume' in data.columns:
            median_volume = data['Volume'].median()
            data['Volume'] = data['Volume'].fillna(median_volume)
        
        # Drop any remaining rows with NaN values
        data = data.dropna()
        
        return data
    
    def _validate_prices(self, data: pd.DataFrame) -> List[str]:
        """Validate price data for anomalies"""
        issues = []
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(f"{col} has {negative_count} non-positive values")
        
        # Check OHLC relationships
        if all(col in data.columns for col in price_columns):
            # High should be >= Low
            invalid_high_low = (data['High'] < data['Low']).sum()
            if invalid_high_low > 0:
                issues.append(f"{invalid_high_low} rows where High < Low")
            
            # Close should be between Low and High
            invalid_close = ((data['Close'] < data['Low']) | (data['Close'] > data['High'])).sum()
            if invalid_close > 0:
                issues.append(f"{invalid_close} rows where Close outside High-Low range")
        
        # Check for extreme price movements (>50% in one day)
        if 'Close' in data.columns:
            returns = data['Close'].pct_change().abs()
            extreme_moves = (returns > 0.5).sum()
            if extreme_moves > 0:
                issues.append(f"{extreme_moves} extreme price movements (>50% daily change)")
        
        return issues
    
    def _fix_price_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix common price data anomalies"""
        data = data.copy()
        
        # Remove rows with non-positive prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Fix OHLC relationships
        if all(col in data.columns for col in price_columns):
            # Ensure High >= Low
            swap_mask = data['High'] < data['Low']
            data.loc[swap_mask, ['High', 'Low']] = data.loc[swap_mask, ['Low', 'High']].values
            
            # Ensure Close is within High-Low range
            data['Close'] = data['Close'].clip(lower=data['Low'], upper=data['High'])
        
        return data
    
    def _validate_volume(self, data: pd.DataFrame) -> List[str]:
        """Validate volume data"""
        issues = []
        
        if 'Volume' in data.columns:
            # Check for negative volume
            negative_volume = (data['Volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"{negative_volume} rows with negative volume")
            
            # Check for zero volume (might be valid but worth noting)
            zero_volume = (data['Volume'] == 0).sum()
            if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                issues.append(f"{zero_volume} rows with zero volume ({zero_volume/len(data)*100:.1f}%)")
        
        return issues
    
    def _fix_volume_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix volume data anomalies"""
        data = data.copy()
        
        if 'Volume' in data.columns:
            # Replace negative volume with absolute value
            data['Volume'] = data['Volume'].abs()
            
            # Replace zero volume with median volume
            median_volume = data['Volume'].median()
            data.loc[data['Volume'] == 0, 'Volume'] = median_volume
        
        return data
    
    def _check_continuity(self, data: pd.DataFrame) -> List[str]:
        """Check for data continuity issues"""
        issues = []
        
        # Check for large gaps in the data
        if hasattr(data.index, 'date') or 'Date' in data.columns:
            date_col = data.index if hasattr(data.index, 'date') else data['Date']
            date_diffs = pd.Series(date_col).diff().dt.days.dropna()
            
            # Flag gaps longer than 10 days (weekends and holidays are normal)
            large_gaps = (date_diffs > 10).sum()
            if large_gaps > 0:
                issues.append(f"{large_gaps} data gaps longer than 10 days")
                max_gap = date_diffs.max()
                issues.append(f"Maximum gap: {max_gap} days")
        
        return issues
    
    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality statistics"""
        stats = {
            'total_rows': len(data),
            'missing_values': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        }
        
        if 'Close' in data.columns:
            stats.update({
                'price_mean': data['Close'].mean(),
                'price_std': data['Close'].std(),
                'price_min': data['Close'].min(),
                'price_max': data['Close'].max(),
                'daily_return_std': data['Close'].pct_change().std()
            })
        
        if 'Volume' in data.columns:
            stats.update({
                'volume_mean': data['Volume'].mean(),
                'volume_std': data['Volume'].std(),
                'zero_volume_pct': (data['Volume'] == 0).mean() * 100
            })
        
        return stats

class DataPipeline:
    """End-to-end data pipeline for ML model training"""
    
    def __init__(self, validator: StockDataValidator):
        self.validator = validator
        
    def prepare_training_data(self, raw_data: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete pipeline from raw data to ML-ready data
        
        Args:
            raw_data: Raw stock data
            symbol: Stock symbol
            
        Returns:
            Tuple of (clean_data, pipeline_report)
        """
        pipeline_report = {
            'symbol': symbol,
            'input_rows': len(raw_data),
            'validation_passed': False,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Step 1: Validate and clean
        validation_result = self.validator.validate_and_clean(raw_data, symbol)
        
        pipeline_report.update({
            'validation_passed': validation_result.is_valid,
            'issues': validation_result.issues,
            'warnings': validation_result.warnings,
            'stats': validation_result.stats
        })
        
        if not validation_result.is_valid:
            logger.error(f"Data validation failed for {symbol}: {validation_result.issues}")
            raise ValueError(f"Data validation failed: {'; '.join(validation_result.issues)}")
        
        clean_data = validation_result.cleaned_data
        pipeline_report['clean_rows'] = len(clean_data)
        
        # Step 2: Additional preprocessing for ML
        ml_ready_data = self._prepare_for_ml(clean_data)
        pipeline_report['final_rows'] = len(ml_ready_data)
        
        logger.info(f"Pipeline complete for {symbol}: {len(raw_data)} -> {len(ml_ready_data)} rows")
        
        return ml_ready_data, pipeline_report
    
    def _prepare_for_ml(self, data: pd.DataFrame) -> pd.DataFrame:
        """Additional preprocessing steps for ML model"""
        # Ensure data is sorted chronologically
        if 'Date' in data.columns:
            data = data.sort_values('Date')
        else:
            data = data.sort_index()
        
        # Remove any remaining outliers using IQR method for Close prices
        if 'Close' in data.columns:
            Q1 = data['Close'].quantile(0.25)
            Q3 = data['Close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More lenient than 1.5 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)
            data = data[outlier_mask]
        
        # Ensure we have enough data after all cleaning
        if len(data) < 252:  # At least 1 year of trading days
            raise ValueError(f"Insufficient data after cleaning: {len(data)} rows")
        
        return data
