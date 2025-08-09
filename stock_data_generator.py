"""
Stock Data Feature Engineering Module

This module generates feature-engineered stock data that can be used for training AI models.
It includes a variety of technical indicators, price patterns, volatility measures, and other
relevant features commonly used in stock price prediction models.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import talib
from sklearn.preprocessing import MinMaxScaler
import logging

# Helper to normalize yfinance download DataFrame (handles MultiIndex etc.)
def _normalize_ohlcv(df, symbol):
    """Return a clean single-level OHLCV DataFrame for one symbol.
    Handles cases where yfinance returns MultiIndex columns or duplicated columns.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        last_level = df.columns.get_level_values(-1)
        if symbol in last_level:
            try:
                df = df.xs(symbol, axis=1, level=-1)
            except Exception:
                df.columns = ['_'.join([str(x) for x in col if x]) for col in df.columns]
        else:
            df.columns = ['_'.join([str(x) for x in col if x]) for col in df.columns]
    # Drop duplicate columns keeping the first
    try:
        cols_index = pd.Index(df.columns)
        if cols_index.duplicated().any():
            df = df.loc[:, ~cols_index.duplicated()]
    except Exception:
        pass
    rename_map = {'Adj Close': 'Adj_Close', 'adjclose': 'Adj_Close'}
    df = df.rename(columns=rename_map)
    required = ['Open','High','Low','Close','Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns after normalization: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_feature_engineered_stock_data(symbols=None, start_date=None, end_date=None, min_rows=10000):
    """
    Generate feature-engineered stock data for training AI models.
    
    Parameters:
    -----------
    symbols : list, optional
        List of stock symbols to fetch data for. Default is a predefined list of popular stocks.
    start_date : str, optional
        Start date for data in format 'YYYY-MM-DD'. Default is 5 years ago.
    end_date : str, optional
        End date for data in format 'YYYY-MM-DD'. Default is current date.
    min_rows : int, optional
        Minimum number of rows to generate. Default is 10,000.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the feature-engineered stock data.
    """
    # Default values
    if symbols is None:
        # Include a diverse set of stocks from different sectors
        symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META',  # Tech
            'JPM', 'BAC', 'GS', 'WFC', 'C',           # Finance
            'JNJ', 'PFE', 'MRK', 'ABBV', 'UNH',       # Healthcare
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',        # Energy
            'PG', 'KO', 'PEP', 'WMT', 'COST',         # Consumer Staples
            'HD', 'NKE', 'SBUX', 'MCD', 'AMZN'        # Consumer Discretionary
        ]
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    if start_date is None:
        # Calculate how many years back we need to go to ensure we have enough data
        # Assuming ~252 trading days per year and ~30 stocks, we need about 2 years
        # of data to get 10,000+ rows
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching stock data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    all_data = []
    
    # Fetch data for each symbol
    for symbol in symbols:
        try:
            # Fetch data from Yahoo Finance
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False, group_by='column', threads=True)
            # Normalize possible multiindex / structure
            stock_data = _normalize_ohlcv(stock_data, symbol)
            if len(stock_data) < 100:  # Skip if not enough data
                logger.warning(f"Not enough data for {symbol}, skipping")
                continue
            # Add symbol column early
            stock_data['Symbol'] = symbol
            # Basic features
            stock_data['Return'] = stock_data['Close'].pct_change()
            stock_data['Log_Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
            stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
            # Price-based features (safe operations)
            stock_data['Price_Range'] = stock_data['High'] - stock_data['Low']
            with np.errstate(divide='ignore', invalid='ignore'):
                open_col = stock_data['Open']
                # If duplicate column selection returned DataFrame, take first column
                if isinstance(open_col, pd.DataFrame):
                    open_col = open_col.iloc[:, 0]
                stock_data['Price_Range_Pct'] = stock_data['Price_Range'] / open_col
            # Replace inf with NaN
            stock_data['Price_Range_Pct'] = stock_data['Price_Range_Pct'].replace([np.inf, -np.inf], np.nan)
            
            # Create target variables for prediction (next day's return)
            stock_data['Target_Next_Day_Return'] = stock_data['Return'].shift(-1)
            stock_data['Target_Next_Day_Direction'] = np.where(stock_data['Target_Next_Day_Return'] > 0, 1, 0)
            
            # Add technical indicators using TA-Lib
            
            # Moving Averages
            for window in [5, 10, 20, 50, 100, 200]:
                stock_data[f'MA_{window}'] = talib.SMA(stock_data['Close'], timeperiod=window)
                stock_data[f'EMA_{window}'] = talib.EMA(stock_data['Close'], timeperiod=window)
                
                # Add features that compare price to moving averages
                stock_data[f'Close_to_MA_{window}'] = stock_data['Close'] / stock_data[f'MA_{window}'] - 1
                
                # Add slope of moving averages (momentum)
                stock_data[f'MA_{window}_Slope'] = stock_data[f'MA_{window}'].pct_change(periods=5)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(stock_data['Close'])
            stock_data['MACD'] = macd
            stock_data['MACD_Signal'] = macd_signal
            stock_data['MACD_Hist'] = macd_hist
            
            # RSI (Relative Strength Index)
            for window in [7, 14, 21]:
                stock_data[f'RSI_{window}'] = talib.RSI(stock_data['Close'], timeperiod=window)
            
            # Bollinger Bands
            for window in [5, 20]:
                upper, middle, lower = talib.BBANDS(stock_data['Close'], timeperiod=window)
                stock_data[f'BB_Upper_{window}'] = upper
                stock_data[f'BB_Middle_{window}'] = middle
                stock_data[f'BB_Lower_{window}'] = lower
                stock_data[f'BB_Width_{window}'] = (upper - lower) / middle
                stock_data[f'BB_Position_{window}'] = (stock_data['Close'] - lower) / (upper - lower)
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'])
            stock_data['Stochastic_K'] = slowk
            stock_data['Stochastic_D'] = slowd
            
            # Commodity Channel Index
            stock_data['CCI'] = talib.CCI(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # Average Directional Index
            stock_data['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # On-Balance Volume (OBV)
            stock_data['OBV'] = talib.OBV(stock_data['Close'], stock_data['Volume'])
            stock_data['OBV_Change'] = stock_data['OBV'].pct_change()
            
            # Volatility Indicators
            stock_data['ATR'] = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # Compute historical volatility
            for window in [5, 10, 20, 30]:
                stock_data[f'Volatility_{window}'] = stock_data['Log_Return'].rolling(window=window).std() * np.sqrt(252)
            
            # Add date-based features
            stock_data['Day_of_Week'] = stock_data.index.dayofweek
            stock_data['Month'] = stock_data.index.month
            stock_data['Year'] = stock_data.index.year
            
            # Create dummy variables for day of week and month
            for i in range(7):
                stock_data[f'Day_{i}'] = (stock_data['Day_of_Week'] == i).astype(int)
            for i in range(1, 13):
                stock_data[f'Month_{i}'] = (stock_data['Month'] == i).astype(int)
            
            # Add market regime features
            # Bull/Bear market indicator (200-day moving average)
            stock_data['Bull_Market'] = (stock_data['Close'] > stock_data['MA_200']).astype(int)
            
            # Trend strength
            stock_data['ADX_Trend'] = np.where(stock_data['ADX'] > 25, 1, 0)
            
            # Market momentum
            stock_data['Market_Momentum'] = np.where(stock_data['Return'].rolling(window=10).sum() > 0, 1, 0)
            
            # Add lagged features (past returns)
            for lag in [1, 2, 3, 5, 10, 21, 63]:
                stock_data[f'Return_Lag_{lag}'] = stock_data['Return'].shift(lag)
                stock_data[f'Log_Return_Lag_{lag}'] = stock_data['Log_Return'].shift(lag)
                stock_data[f'Volume_Change_Lag_{lag}'] = stock_data['Volume_Change'].shift(lag)
            
            # Add rolling window statistics for returns
            for window in [5, 10, 21]:
                stock_data[f'Return_Mean_{window}'] = stock_data['Return'].rolling(window=window).mean()
                stock_data[f'Return_Std_{window}'] = stock_data['Return'].rolling(window=window).std()
                stock_data[f'Return_Min_{window}'] = stock_data['Return'].rolling(window=window).min()
                stock_data[f'Return_Max_{window}'] = stock_data['Return'].rolling(window=window).max()
                
                # Rolling window correlation between price and volume
                stock_data[f'Price_Volume_Corr_{window}'] = stock_data['Return'].rolling(window=window).corr(stock_data['Volume_Change'])
            
            # Gap features
            stock_data['Gap_Up'] = (stock_data['Open'] > stock_data['Close'].shift(1)).astype(int)
            stock_data['Gap_Down'] = (stock_data['Open'] < stock_data['Close'].shift(1)).astype(int)
            stock_data['Gap_Size'] = (stock_data['Open'] / stock_data['Close'].shift(1) - 1)
            
            # Add sector information (simplified version)
            sectors = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'AMZN': 'Consumer_Discretionary', 
                'GOOGL': 'Technology', 'META': 'Technology',
                'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'WFC': 'Finance', 'C': 'Finance',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'UNH': 'Healthcare',
                'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
                'PG': 'Consumer_Staples', 'KO': 'Consumer_Staples', 'PEP': 'Consumer_Staples', 
                'WMT': 'Consumer_Staples', 'COST': 'Consumer_Staples',
                'HD': 'Consumer_Discretionary', 'NKE': 'Consumer_Discretionary', 
                'SBUX': 'Consumer_Discretionary', 'MCD': 'Consumer_Discretionary'
            }
            stock_data['Sector'] = sectors.get(symbol, 'Other')
            
            # Create dummy variables for sectors
            for sector in sectors.values():
                stock_data[f'Sector_{sector}'] = (stock_data['Sector'] == sector).astype(int)
            
            # Add candlestick pattern recognition
            for pattern_func in [
                talib.CDLDOJI, talib.CDLHAMMER, talib.CDLENGULFING, 
                talib.CDLMORNINGSTAR, talib.CDLEVENINGSTAR,
                talib.CDLHARAMI, talib.CDLSHOOTINGSTAR
            ]:
                pattern_name = pattern_func.__name__.replace('talib.', '')
                stock_data[pattern_name] = pattern_func(
                    stock_data['Open'], 
                    stock_data['High'], 
                    stock_data['Low'], 
                    stock_data['Close']
                )
            
            # Drop rows with NaN values (mostly from the beginning due to rolling windows)
            stock_data = stock_data.dropna()
            
            # Append to our list of dataframes
            all_data.append(stock_data)
            
            logger.info(f"Successfully processed {symbol}, added {len(stock_data)} rows")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    # Combine all stock data
    if not all_data:
        raise ValueError("No valid stock data was processed")
        
    combined_data = pd.concat(all_data)
    
    # Reset index to turn date into a column
    combined_data = combined_data.reset_index()
    combined_data.rename(columns={'index': 'Date'}, inplace=True)
    
    # Ensure we have enough data
    if len(combined_data) < min_rows:
        logger.warning(f"Only generated {len(combined_data)} rows, less than the requested {min_rows}")
    else:
        logger.info(f"Successfully generated {len(combined_data)} rows of feature-engineered stock data")
    
    return combined_data

def save_data_to_csv(data, file_path='stock_data_features.csv'):
    """
    Save the generated data to a CSV file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to save
    file_path : str, optional
        Path where to save the CSV file
    """
    data.to_csv(file_path, index=False)
    logger.info(f"Data saved to {file_path}")
    
def normalize_features(data, exclude_columns=None):
    """
    Normalize numerical features to the range [0, 1]
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to normalize
    exclude_columns : list, optional
        Columns to exclude from normalization
        
    Returns:
    --------
    pandas.DataFrame
        Normalized data
    """
    if exclude_columns is None:
        exclude_columns = ['Date', 'Symbol', 'Sector', 'Target_Next_Day_Direction']
    
    # Create a copy of the data
    normalized_data = data.copy()
    
    # Get columns to normalize
    cols_to_normalize = [col for col in data.columns if col not in exclude_columns]
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Normalize the selected columns
    normalized_data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
    
    return normalized_data

def split_train_test(data, test_size=0.2, time_based=True):
    """
    Split data into training and test sets.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to split
    test_size : float, optional
        Proportion of the data to include in the test split
    time_based : bool, optional
        If True, split based on time (latest data as test)
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Define features and target (exclude non-numeric columns)
    all_feature_candidates = [col for col in data.columns if not col.startswith('Target_')]
    # Keep only numeric columns
    numeric_cols = data[all_feature_candidates].select_dtypes(include=[np.number]).columns.tolist()
    features = numeric_cols
    target = 'Target_Next_Day_Direction'  # Binary classification target
    
    if time_based:
        # Sort by date
        data = data.sort_values('Date')
        
        # Determine the split point
        split_idx = int(len(data) * (1 - test_size))
        
        # Split the data
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
    else:
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data[target], test_size=test_size, random_state=42
        )
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to demonstrate the usage."""
    try:
        # Generate the feature-engineered stock data
        stock_data = get_feature_engineered_stock_data()
        
        # Save the data to CSV
        save_data_to_csv(stock_data)
        
        # Normalize the data
        normalized_data = normalize_features(stock_data)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = split_train_test(normalized_data)
        
        print(f"Generated {len(stock_data)} rows of feature-engineered stock data")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Print some example features
        print("\nExample features:")
        for col in sorted(stock_data.columns)[:20]:  # First 20 columns alphabetically
            print(f"- {col}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
