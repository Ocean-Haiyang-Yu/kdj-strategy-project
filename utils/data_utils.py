import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def fetch_data(ticker, period='10y', benchmark='SPY', include_benchmark=True):
    """
    Fetch historical data for a ticker and optionally a benchmark.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to fetch data for
    period : str, default '10y'
        The period to fetch data for (e.g., '1y', '5y', '10y', 'max')
    benchmark : str, default 'SPY'
        The benchmark ticker symbol
    include_benchmark : bool, default True
        Whether to include benchmark data
        
    Returns:
    --------
    DataFrame
        Historical data with benchmark data if requested
    """
    # Calculate start date based on period
    end_date = datetime.now()
    if period.endswith('y'):
        years = int(period[:-1])
        start_date = end_date - timedelta(days=365 * years)
    elif period.endswith('m'):
        months = int(period[:-1])
        start_date = end_date - timedelta(days=30 * months)
    elif period.endswith('d'):
        days = int(period[:-1])
        start_date = end_date - timedelta(days=days)
    else:
        start_date = None  # yfinance will use default for 'max'
    
    # Fetch data for the ticker
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Fix the column structure
    # Display the original columns
    print("Original columns:", data.columns.tolist())

    # Create a new DataFrame with standardized column names
    fixed_data = pd.DataFrame(index=data.index)

    # Extract data based on column name patterns
    for col in data.columns:
        col_str = str(col).lower()
        if 'high' in col_str:
            fixed_data['high'] = data[col]
        elif 'low' in col_str:
            fixed_data['low'] = data[col]
        elif 'close' in col_str:
            fixed_data['close'] = data[col]
        elif 'open' in col_str:
            fixed_data['open'] = data[col]
        elif 'volume' in col_str:
            fixed_data['volume'] = data[col]
        elif 'adj' in col_str:
            fixed_data['adj_close'] = data[col]

    # Verify the new columns
    print("Fixed columns:", fixed_data.columns.tolist())

    # Use this fixed data for calculations
    data = fixed_data

    # Display the first few rows of the fixed data
    print(data.head())
    
    
    # Rename columns to include ticker name
    data.columns = [f"{ticker}_{col}" for col in data.columns]
    
    # Fetch benchmark data if requested
    if include_benchmark and benchmark:
        benchmark_data = yf.download(benchmark, start=start_date, end=end_date)
        # Fix the column structure
        # Display the original columns
        print("Original benchmark columns:", benchmark_data.columns.tolist())

        # Create a new DataFrame with standardized column names
        fixed_benchmark_data = pd.DataFrame(index=benchmark_data.index)

        # Extract data based on column name patterns
        for col in benchmark_data.columns:
            col_str = str(col).lower()
            if 'high' in col_str:
                fixed_benchmark_data['high'] = benchmark_data[col]
            elif 'low' in col_str:
                fixed_benchmark_data['low'] = benchmark_data[col]
            elif 'close' in col_str:
                fixed_benchmark_data['close'] = benchmark_data[col]
            elif 'open' in col_str:
                fixed_benchmark_data['open'] = benchmark_data[col]
            elif 'volume' in col_str:
                fixed_benchmark_data['volume'] = benchmark_data[col]
            elif 'adj' in col_str:
                fixed_benchmark_data['adj_close'] = benchmark_data[col]

        # Verify the new columns
        print("Fixed benchmark columns:", fixed_benchmark_data.columns.tolist())

        # Use this fixed data for calculations
        benchmark_data = fixed_benchmark_data

        # Display the first few rows of the fixed data
        print(benchmark_data.head())
        benchmark_data.columns = [f"{benchmark}_{col}" for col in benchmark_data.columns]
        
        # Merge the dataframes on date
        data = pd.merge(data, benchmark_data, left_index=True, right_index=True, how='inner')
    
    return data

def preprocess_data(df, ticker, benchmark=None, fill_method='ffill'):
    """
    Preprocess the data by handling missing values, calculating returns, etc.
    
    Parameters:
    -----------
    df : DataFrame
        The raw data
    ticker : str
        The ticker symbol
    benchmark : str, default None
        The benchmark ticker symbol
    fill_method : str, default 'ffill'
        Method to fill missing values ('ffill', 'bfill', or 'interpolate')
        
    Returns:
    --------
    DataFrame
        Preprocessed data with additional columns for returns, etc.
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    if fill_method == 'ffill':
        data = data.fillna(method='ffill')
    elif fill_method == 'bfill':
        data = data.fillna(method='bfill')
    elif fill_method == 'interpolate':
        data = data.interpolate()
    
    # Calculate daily returns for the ticker
    data[f'{ticker}_daily_return'] = data[f'{ticker}_close'].pct_change()
    
    # Calculate log returns (useful for some analyses)
    data[f'{ticker}_log_return'] = np.log(data[f'{ticker}_close'] / data[f'{ticker}_close'].shift(1))
    
    # Calculate returns for benchmark if provided and exists in data
    if benchmark and f'{benchmark}_close' in data.columns:
        data[f'{benchmark}_daily_return'] = data[f'{benchmark}_close'].pct_change()
        data[f'{benchmark}_log_return'] = np.log(data[f'{benchmark}_close'] / data[f'{benchmark}_close'].shift(1))
    
    # Drop rows with NaN (usually just the first row)
    data = data.dropna()
    
    return data

def resample_data(df, freq='W'):
    """
    Resample data to different frequencies.
    
    Parameters:
    -----------
    df : DataFrame
        The data to resample
    freq : str, default 'W'
        The frequency to resample to ('W' for weekly, 'M' for monthly)
        
    Returns:
    --------
    DataFrame
        Resampled data
    """
    # Define how to resample OHLC data
    resampled = df.resample(freq).agg({
        col: 'last' if 'close' in col or 'volume' in col 
             else ('first' if 'open' in col 
                  else ('max' if 'high' in col 
                       else ('min' if 'low' in col else 'mean'))) 
        for col in df.columns
    })
    
    return resampled

def get_risk_free_rate(start_date=None, end_date=None, ticker='^IRX'):
    """
    Fetch daily risk-free rates from Yahoo Finance (using 13-week Treasury Bill as proxy).
    
    Parameters:
    -----------
    start_date : datetime or str, default None
        The start date for fetching data
    end_date : datetime or str, default None
        The end date for fetching data
    ticker : str, default '^IRX'
        The ticker symbol for the risk-free rate (default is 13-week Treasury Bill)
        
    Returns:
    --------
    pandas.Series
        Daily risk-free rates (annualized, in decimal form)
    """
    try:
        # Fetch Treasury Bill rate
        rf_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Convert from percentage to decimal
        daily_rf = rf_data[rf_data.columns[0]] / 100
        
        # Handle missing values (forward fill, then backward fill)
        daily_rf = daily_rf.fillna(method='ffill').fillna(method='bfill')
        
        return daily_rf
    except Exception as e:
        print(f"Warning: Could not fetch risk-free rate: {e}")
        print("Using default value of 2%")
        
        # If fetching fails, return a Series with default value
        if start_date and end_date:
            # Create date range from start to end
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
            return pd.Series(0.02, index=date_range)  # 2% as default
        else:
            # Just return a single value
            return pd.Series(0.02, index=[pd.Timestamp.now()])