"""
Fundamental Analysis Module
Handles earnings reports, financial events, and fundamental indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class EarningsEvent:
    """Container for earnings event data"""
    date: datetime
    eps_estimate: Optional[float]
    eps_actual: Optional[float] 
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]
    surprise_percent: Optional[float]
    days_until: int  # Days until/since earnings (negative = past, positive = future)

@dataclass
class FundamentalMetrics:
    """Container for fundamental analysis results"""
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    roe: Optional[float]  # Return on Equity
    roa: Optional[float]  # Return on Assets
    profit_margin: Optional[float]
    operating_margin: Optional[float]
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    analyst_rating: Optional[str]
    price_target: Optional[float]

class FundamentalAnalyzer:
    """Analyzer for fundamental data and earnings events"""
    
    def __init__(self):
        self.cache = {}
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        if not self.fmp_api_key:
            logger.warning("FMP_API_KEY not found in environment variables - earnings data will be limited")
        
    def get_earnings_calendar(self, symbol: str, periods: int = 8) -> List[EarningsEvent]:
        """
        Get real earnings calendar data using multiple APIs
        
        Args:
            symbol: Stock ticker symbol
            periods: Number of quarters to look back/forward
            
        Returns:
            List of EarningsEvent objects
        """
        earnings_calendar = []
        
        try:
            # First try Alpha Vantage (most comprehensive historical data)
            if self.alpha_vantage_api_key:
                earnings_calendar.extend(self._get_alpha_vantage_earnings(symbol, periods))
            
            # Supplement with Finnhub for additional context
            if self.finnhub_api_key and len(earnings_calendar) < periods:
                finnhub_earnings = self._get_finnhub_earnings(symbol, periods)
                # Merge without duplicates based on date
                existing_dates = {e.date.strftime('%Y-%m-%d') for e in earnings_calendar}
                for fe in finnhub_earnings:
                    if fe.date.strftime('%Y-%m-%d') not in existing_dates:
                        earnings_calendar.append(fe)
            
            # If we still don't have enough data, supplement with yfinance
            if len(earnings_calendar) < 2:
                earnings_calendar.extend(self._get_yfinance_earnings_fallback(symbol))
            
            # Sort by date (most recent first)
            earnings_calendar.sort(key=lambda x: x.date, reverse=True)
            
            logger.info(f"Retrieved {len(earnings_calendar)} earnings events for {symbol}")
            return earnings_calendar[:periods]  # Limit to requested periods
            
        except Exception as e:
            logger.error(f"Error getting earnings calendar for {symbol}: {e}")
            return self._get_yfinance_earnings_fallback(symbol)
    
    def _get_finnhub_earnings(self, symbol: str, periods: int) -> List[EarningsEvent]:
        """Get earnings data from Finnhub API"""
        try:
            earnings_events = []
            current_date = datetime.now()
            
            # Get earnings calendar for the past year and future
            from_date = (current_date - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = (current_date + timedelta(days=180)).strftime('%Y-%m-%d')
            
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {
                'from': from_date,
                'to': to_date,
                'symbol': symbol.upper(),
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            earnings_list = data.get('earningsCalendar', [])
            
            for earning in earnings_list[:periods]:  # Limit to requested periods
                try:
                    earnings_date = datetime.strptime(earning['date'], '%Y-%m-%d')
                    
                    # Calculate surprise percentage
                    eps_estimate = earning.get('epsEstimate', 0)
                    eps_actual = earning.get('epsActual', 0)
                    surprise_pct = None
                    
                    if eps_estimate and eps_actual and eps_estimate != 0:
                        surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    
                    event = EarningsEvent(
                        date=earnings_date,
                        eps_estimate=eps_estimate,
                        eps_actual=eps_actual,
                        revenue_estimate=earning.get('revenueEstimate'),
                        revenue_actual=earning.get('revenueActual'),
                        surprise_percent=surprise_pct,
                        days_until=(earnings_date.date() - current_date.date()).days
                    )
                    earnings_events.append(event)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed Finnhub earnings data: {e}")
                    continue
            
            return earnings_events
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub earnings for {symbol}: {e}")
            return []
    
    def _get_alpha_vantage_earnings(self, symbol: str, periods: int) -> List[EarningsEvent]:
        """Get earnings data from Alpha Vantage API"""
        try:
            earnings_events = []
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'EARNINGS',
                'symbol': symbol.upper(),
                'apikey': self.alpha_vantage_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data or 'Note' in data:
                logger.warning(f"Alpha Vantage API issue for {symbol}: {data}")
                return []
            
            quarterly_earnings = data.get('quarterlyEarnings', [])
            
            for earning in quarterly_earnings[:periods]:  # Limit to requested periods
                try:
                    # Parse reported date
                    reported_date_str = earning.get('reportedDate', '')
                    if not reported_date_str:
                        continue
                        
                    reported_date = datetime.strptime(reported_date_str, '%Y-%m-%d')
                    
                    # Parse EPS values
                    eps_actual = None
                    eps_estimate = None
                    surprise_pct = None
                    
                    try:
                        if earning.get('reportedEPS') and earning.get('reportedEPS') != 'None':
                            eps_actual = float(earning['reportedEPS'])
                    except (ValueError, TypeError):
                        pass
                        
                    try:
                        if earning.get('estimatedEPS') and earning.get('estimatedEPS') != 'None':
                            eps_estimate = float(earning['estimatedEPS'])
                    except (ValueError, TypeError):
                        pass
                    
                    try:
                        if earning.get('surprisePercentage') and earning.get('surprisePercentage') != 'None':
                            surprise_pct = float(earning['surprisePercentage'])
                    except (ValueError, TypeError):
                        pass
                    
                    event = EarningsEvent(
                        date=reported_date,
                        eps_estimate=eps_estimate,
                        eps_actual=eps_actual,
                        revenue_estimate=None,  # Alpha Vantage doesn't provide revenue estimates
                        revenue_actual=None,
                        surprise_percent=surprise_pct,
                        days_until=(reported_date.date() - datetime.now().date()).days
                    )
                    earnings_events.append(event)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed Alpha Vantage earnings data: {e}")
                    continue
            
            return earnings_events
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage earnings for {symbol}: {e}")
            return []
    
    def _get_fmp_earnings_history(self, symbol: str, periods: int) -> List[EarningsEvent]:
        """Get historical earnings data from Financial Modeling Prep"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}"
            params = {
                'limit': periods * 4,  # Get more than needed
                'apikey': self.fmp_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            earnings_events = []
            current_date = datetime.now()
            
            for earning in data[:periods]:  # Limit to requested periods
                try:
                    earnings_date = datetime.strptime(earning['date'], '%Y-%m-%d')
                    
                    # Calculate surprise percentage
                    eps_estimate = earning.get('epsEstimated', 0)
                    eps_actual = earning.get('eps', 0)
                    surprise_pct = None
                    
                    if eps_estimate and eps_actual and eps_estimate != 0:
                        surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    
                    event = EarningsEvent(
                        date=earnings_date,
                        eps_estimate=eps_estimate,
                        eps_actual=eps_actual,
                        revenue_estimate=earning.get('revenueEstimated'),
                        revenue_actual=earning.get('revenue'),
                        surprise_percent=surprise_pct,
                        days_until=(earnings_date.date() - current_date.date()).days
                    )
                    earnings_events.append(event)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed earnings data: {e}")
                    continue
            
            return earnings_events
            
        except Exception as e:
            logger.error(f"Error fetching FMP earnings history for {symbol}: {e}")
            return []
    
    def _get_fmp_earnings_calendar(self, symbol: str) -> List[EarningsEvent]:
        """Get upcoming earnings from Financial Modeling Prep earnings calendar"""
        try:
            # Get earnings calendar for next few months
            url = "https://financialmodelingprep.com/api/v3/earning_calendar"
            
            current_date = datetime.now()
            end_date = current_date + timedelta(days=120)  # Next 4 months
            
            params = {
                'from': current_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'apikey': self.fmp_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            earnings_events = []
            
            # Filter for our symbol
            for earning in data:
                if earning.get('symbol') == symbol.upper():
                    try:
                        earnings_date = datetime.strptime(earning['date'], '%Y-%m-%d')
                        
                        event = EarningsEvent(
                            date=earnings_date,
                            eps_estimate=earning.get('epsEstimated'),
                            eps_actual=None,  # Future earnings
                            revenue_estimate=earning.get('revenueEstimated'),
                            revenue_actual=None,
                            surprise_percent=None,
                            days_until=(earnings_date.date() - current_date.date()).days
                        )
                        earnings_events.append(event)
                        
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping malformed calendar data: {e}")
                        continue
            
            return earnings_events
            
        except Exception as e:
            logger.error(f"Error fetching FMP earnings calendar for {symbol}: {e}")
            return []
    
    def _get_yfinance_earnings_fallback(self, symbol: str) -> List[EarningsEvent]:
        """Fallback to yfinance for basic earnings data"""
        try:
            ticker = yf.Ticker(symbol)
            earnings_events = []
            
            # Try to get basic earnings data from info
            try:
                info = ticker.info
                next_earnings_date = info.get('earningsDate')
                if next_earnings_date:
                    # Handle next earnings date - could be datetime or list
                    if isinstance(next_earnings_date, list) and len(next_earnings_date) > 0:
                        earnings_date = next_earnings_date[0]
                    else:
                        earnings_date = next_earnings_date
                    
                    # Convert to datetime if needed
                    if hasattr(earnings_date, 'date'):
                        days_until = (earnings_date.date() - datetime.now().date()).days
                        event = EarningsEvent(
                            date=earnings_date,
                            eps_estimate=info.get('earningsQuarterlyGrowth'),
                            eps_actual=None,
                            revenue_estimate=None,
                            revenue_actual=None,
                            surprise_percent=None,
                            days_until=days_until
                        )
                        earnings_events.append(event)
                        
            except Exception as e:
                logger.debug(f"yfinance earnings data not available for {symbol}: {e}")
            
            # Try to get historical earnings data
            try:
                earnings_df = ticker.earnings
                if earnings_df is not None and hasattr(earnings_df, 'empty') and not earnings_df.empty:
                    current_date = datetime.now()
                    for i, (year_idx, row) in enumerate(earnings_df.iterrows()):
                        if i >= 4:  # Limit to last 4 years
                            break
                        
                        # Convert year to int safely
                        try:
                            year = int(year_idx) if hasattr(year_idx, '__int__') else int(str(year_idx))
                        except (ValueError, TypeError):
                            continue
                        
                        # Estimate quarterly earnings dates (end of quarters)
                        q4_date = datetime(year, 12, 31)
                        
                        earnings_value = row.get('Earnings', 0) if hasattr(row, 'get') else 0
                        revenue_value = row.get('Revenue', 0) if hasattr(row, 'get') else 0
                        
                        event = EarningsEvent(
                            date=q4_date,
                            eps_estimate=None,
                            eps_actual=earnings_value,
                            revenue_estimate=None,
                            revenue_actual=revenue_value,
                            surprise_percent=None,
                            days_until=(q4_date.date() - current_date.date()).days
                        )
                        earnings_events.append(event)
                        
            except Exception as e:
                logger.debug(f"yfinance historical earnings not available for {symbol}: {e}")
            
            return earnings_events
            
        except Exception as e:
            logger.error(f"Error in yfinance earnings fallback for {symbol}: {e}")
            return []
    
    def get_fundamental_metrics(self, symbol: str) -> FundamentalMetrics:
        """
        Get comprehensive fundamental metrics for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            FundamentalMetrics object
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get analyst recommendations
            analyst_rating = None
            price_target = None
            try:
                recommendations = ticker.recommendations
                if hasattr(recommendations, 'empty') and hasattr(recommendations, 'iloc'):
                    if not recommendations.empty:
                        latest_rec = recommendations.iloc[-1]
                        analyst_rating = latest_rec.get('To Grade', 'Unknown')
                        
                # Get price targets
                upgrades_downgrades = ticker.upgrades_downgrades
                if hasattr(upgrades_downgrades, 'empty') and hasattr(upgrades_downgrades, 'iloc'):
                    if not upgrades_downgrades.empty:
                        latest_target = upgrades_downgrades.iloc[-1]
                        price_target = latest_target.get('ToGrade', None)
                    
            except Exception as e:
                logger.debug(f"Analyst data not available for {symbol}: {e}")
            
            # Calculate growth rates with better error handling
            revenue_growth = None
            earnings_growth = None
            try:
                financials = ticker.financials
                if hasattr(financials, 'empty') and hasattr(financials, 'iloc'):
                    if not financials.empty and 'Total Revenue' in financials.index:
                        revenue_series = financials.loc['Total Revenue']
                        if len(revenue_series) >= 2:
                            recent_revenue = revenue_series.iloc[0]
                            prev_revenue = revenue_series.iloc[1]
                            
                            # Convert to scalar values if they're Series
                            if hasattr(recent_revenue, 'iloc'):
                                recent_revenue = recent_revenue.iloc[0] if len(recent_revenue) > 0 else recent_revenue
                            if hasattr(prev_revenue, 'iloc'):
                                prev_revenue = prev_revenue.iloc[0] if len(prev_revenue) > 0 else prev_revenue
                            
                            if (pd.notna(recent_revenue) and pd.notna(prev_revenue) and 
                                not isinstance(recent_revenue, pd.Series) and not isinstance(prev_revenue, pd.Series) and
                                prev_revenue != 0):
                                revenue_growth = float(((recent_revenue - prev_revenue) / abs(prev_revenue)) * 100)
                
                # Earnings growth
                earnings = ticker.earnings
                if hasattr(earnings, 'empty') and hasattr(earnings, 'iloc'):
                    if not earnings.empty and 'Earnings' in earnings.columns and len(earnings) >= 2:
                        recent_earnings = earnings['Earnings'].iloc[-1]
                        prev_earnings = earnings['Earnings'].iloc[-2]
                        if (pd.notna(recent_earnings) and pd.notna(prev_earnings) and 
                            prev_earnings != 0):
                            earnings_growth = float(((recent_earnings - prev_earnings) / abs(prev_earnings)) * 100)
                        
            except Exception as e:
                logger.debug(f"Financial data calculation error for {symbol}: {e}")
            
            metrics = FundamentalMetrics(
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                roe=info.get('returnOnEquity'),
                roa=info.get('returnOnAssets'),
                profit_margin=info.get('profitMargins'),
                operating_margin=info.get('operatingMargins'),
                revenue_growth=revenue_growth,
                earnings_growth=earnings_growth,
                analyst_rating=analyst_rating,
                price_target=price_target
            )
            
            logger.info(f"Retrieved fundamental metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting fundamental metrics for {symbol}: {e}")
            return FundamentalMetrics(
                pe_ratio=None, forward_pe=None, peg_ratio=None, price_to_book=None,
                price_to_sales=None, debt_to_equity=None, current_ratio=None,
                roe=None, roa=None, profit_margin=None, operating_margin=None,
                revenue_growth=None, earnings_growth=None, analyst_rating=None,
                price_target=None
            )
    
    def calculate_earnings_proximity_features(self, prices_df: pd.DataFrame, 
                                            earnings_events: List[EarningsEvent]) -> pd.DataFrame:
        """
        Calculate features based on proximity to earnings events
        
        Args:
            prices_df: DataFrame with stock prices indexed by date
            earnings_events: List of earnings events
            
        Returns:
            DataFrame with earnings proximity features
        """
        if not earnings_events:
            # Return empty features if no earnings data
            features_df = pd.DataFrame(index=prices_df.index)
            features_df['days_to_earnings'] = np.nan
            features_df['earnings_surprise_impact'] = 0.0
            features_df['pre_earnings_period'] = 0.0
            features_df['post_earnings_period'] = 0.0
            return features_df
        
        features_df = pd.DataFrame(index=prices_df.index)
        
        # Initialize feature columns
        features_df['days_to_earnings'] = np.nan
        features_df['earnings_surprise_impact'] = 0.0
        features_df['pre_earnings_period'] = 0.0  # 1 if within 5 days before earnings
        features_df['post_earnings_period'] = 0.0  # 1 if within 5 days after earnings
        
        for price_date in prices_df.index:
            price_datetime = price_date.to_pydatetime() if hasattr(price_date, 'to_pydatetime') else price_date
            
            # Find the closest earnings event
            min_days_diff = float('inf')
            closest_event = None
            
            for event in earnings_events:
                days_diff = abs((event.date.date() - price_datetime.date()).days)
                if days_diff < min_days_diff:
                    min_days_diff = days_diff
                    closest_event = event
            
            if closest_event:
                days_to_earnings = (closest_event.date.date() - price_datetime.date()).days
                features_df.loc[price_date, 'days_to_earnings'] = days_to_earnings
                
                # Pre-earnings period (5 days before)
                if -5 <= days_to_earnings <= 0:
                    features_df.loc[price_date, 'pre_earnings_period'] = 1.0
                
                # Post-earnings period (5 days after)
                if 0 <= days_to_earnings <= 5:
                    features_df.loc[price_date, 'post_earnings_period'] = 1.0
                
                # Earnings surprise impact (for past earnings)
                if (closest_event.surprise_percent is not None and 
                    0 <= days_to_earnings <= 10):  # Impact lasts ~10 days
                    # Decay the impact over time
                    decay_factor = max(0, 1 - (days_to_earnings / 10))
                    impact = closest_event.surprise_percent * decay_factor / 100  # Normalize
                    features_df.loc[price_date, 'earnings_surprise_impact'] = impact
        
        # Fill NaN values
        features_df = features_df.ffill().fillna(0)
        
        logger.info(f"Generated earnings proximity features for {len(features_df)} trading days")
        return features_df
    
    def calculate_fundamental_score(self, metrics: FundamentalMetrics) -> float:
        """
        Calculate a composite fundamental score from 0-100
        
        Args:
            metrics: FundamentalMetrics object
            
        Returns:
            Fundamental score (0-100, higher is better)
        """
        score = 0.0
        components = 0
        
        # PE Ratio scoring (lower is better, but not too low)
        if metrics.pe_ratio is not None and metrics.pe_ratio > 0:
            if 10 <= metrics.pe_ratio <= 25:
                score += 20  # Ideal range
            elif 5 <= metrics.pe_ratio < 10 or 25 < metrics.pe_ratio <= 35:
                score += 15  # Good range
            elif metrics.pe_ratio < 5 or metrics.pe_ratio > 35:
                score += 5   # Concerning
            components += 1
        
        # Growth scoring
        if metrics.revenue_growth is not None:
            if metrics.revenue_growth > 15:
                score += 20
            elif metrics.revenue_growth > 5:
                score += 15
            elif metrics.revenue_growth > 0:
                score += 10
            else:
                score += 0
            components += 1
        
        if metrics.earnings_growth is not None:
            if metrics.earnings_growth > 20:
                score += 20
            elif metrics.earnings_growth > 10:
                score += 15
            elif metrics.earnings_growth > 0:
                score += 10
            else:
                score += 0
            components += 1
        
        # Profitability
        if metrics.roe is not None and metrics.roe > 0:
            if metrics.roe > 0.20:  # 20%+
                score += 15
            elif metrics.roe > 0.15:
                score += 12
            elif metrics.roe > 0.10:
                score += 8
            else:
                score += 3
            components += 1
        
        if metrics.profit_margin is not None and metrics.profit_margin > 0:
            if metrics.profit_margin > 0.20:
                score += 15
            elif metrics.profit_margin > 0.10:
                score += 12
            elif metrics.profit_margin > 0.05:
                score += 8
            else:
                score += 3
            components += 1
        
        # Financial health
        if metrics.debt_to_equity is not None:
            if metrics.debt_to_equity < 0.3:
                score += 10  # Low debt
            elif metrics.debt_to_equity < 0.6:
                score += 7
            elif metrics.debt_to_equity < 1.0:
                score += 3
            else:
                score += 0  # High debt
            components += 1
        
        # Normalize score to 0-100
        if components > 0:
            final_score = (score / (components * 20)) * 100
            return min(100, max(0, final_score))
        
        return 50.0  # Neutral score if no data
