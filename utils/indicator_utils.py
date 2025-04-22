import pandas as pd
import numpy as np

def calculate_kdj(df, ticker, k_period=9, d_period=3, j_period=3):
    """
    Calculate KDJ indicator for a ticker.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing price data
    ticker : str
        The ticker symbol
    k_period : int, default 9
        The period for calculating K
    d_period : int, default 3
        The period for calculating D (moving average of K)
    j_period : int, default 3
        The period for calculating J (weighted K and D)
        
    Returns:
    --------
    DataFrame
        The input dataframe with KDJ columns added
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Extract high, low, close prices
    high_col = f'{ticker}_high'
    low_col = f'{ticker}_low'
    close_col = f'{ticker}_close'
    
    # Calculate RSV (Raw Stochastic Value)
    low_min = data[low_col].rolling(window=k_period).min()
    high_max = data[high_col].rolling(window=k_period).max()
    
    # Avoid division by zero
    denominator = high_max - low_min
    denominator = denominator.replace(0, np.nan)
    
    data['rsv'] = (data[close_col] - low_min) / denominator * 100
    
    # Calculate K (first line of KDJ)
    data[f'{ticker}_kdj_k'] = data['rsv'].rolling(window=d_period).mean()
    
    # Calculate D (second line of KDJ)
    data[f'{ticker}_kdj_d'] = data[f'{ticker}_kdj_k'].rolling(window=d_period).mean()
    
    # Calculate J (third line of KDJ)
    data[f'{ticker}_kdj_j'] = 3 * data[f'{ticker}_kdj_k'] - 2 * data[f'{ticker}_kdj_d']
    
    # Drop the temporary column
    data = data.drop(columns=['rsv'])
    
    return data

def calculate_sma(df, ticker, periods=[20, 50, 200]):
    """
    Calculate Simple Moving Averages for multiple periods.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing price data
    ticker : str
        The ticker symbol
    periods : list, default [20, 50, 200]
        The periods for calculating SMAs
        
    Returns:
    --------
    DataFrame
        The input dataframe with SMA columns added
    """
    data = df.copy()
    close_col = f'{ticker}_close'
    
    for period in periods:
        data[f'{ticker}_sma_{period}'] = data[close_col].rolling(window=period).mean()
    
    return data

def calculate_macd(df, ticker, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing price data
    ticker : str
        The ticker symbol
    fast : int, default 12
        The fast period for EMA calculation
    slow : int, default 26
        The slow period for EMA calculation
    signal : int, default 9
        The signal period for signal line calculation
        
    Returns:
    --------
    DataFrame
        The input dataframe with MACD columns added
    """
    data = df.copy()
    close_col = f'{ticker}_close'
    
    # Calculate fast and slow EMAs
    ema_fast = data[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = data[close_col].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    data[f'{ticker}_macd'] = ema_fast - ema_slow
    
    # Calculate signal line
    data[f'{ticker}_macd_signal'] = data[f'{ticker}_macd'].ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    data[f'{ticker}_macd_hist'] = data[f'{ticker}_macd'] - data[f'{ticker}_macd_signal']
    
    return data

def calculate_rsi(df, ticker, period=14):
    """
    Calculate RSI (Relative Strength Index) indicator.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing price data
    ticker : str
        The ticker symbol
    period : int, default 14
        The period for calculating RSI
        
    Returns:
    --------
    DataFrame
        The input dataframe with RSI column added
    """
    data = df.copy()
    close_col = f'{ticker}_close'
    
    # Calculate price changes
    delta = data[close_col].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    data[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
    
    return data

def generate_signals(df, ticker, strategy='kdj', **kwargs):
    """
    Generate buy/sell signals based on a specified strategy.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing price and indicator data
    ticker : str
        The ticker symbol
    strategy : str, default 'kdj'
        The strategy to use for generating signals
    **kwargs : dict
        Additional parameters for the strategy
        
    Returns:
    --------
    DataFrame
        The input dataframe with signal columns added
    """
    data = df.copy()
    
    if strategy == 'kdj':
        # Extract parameters with defaults
        j_buy_threshold = kwargs.get('j_buy_threshold', 20)
        j_sell_threshold = kwargs.get('j_sell_threshold', 80)
        
        # Generate signals based on KDJ J line crossing thresholds
        data[f'{ticker}_signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Buy signal: J crosses below buy threshold
        data.loc[data[f'{ticker}_kdj_j'] < j_buy_threshold, f'{ticker}_signal'] = 1
        
        # Sell signal: J crosses above sell threshold
        data.loc[data[f'{ticker}_kdj_j'] > j_sell_threshold, f'{ticker}_signal'] = -1
    
    # Add more strategy options here as needed
    
    return data