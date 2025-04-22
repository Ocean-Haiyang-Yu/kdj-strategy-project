import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import get_risk_free_rate

def calculate_returns(df, ticker, benchmark='SPY', risk_free_rate=0.02):
    """
    Calculate various performance metrics based on strategy returns.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing strategy returns
    ticker : str
        The ticker symbol
    benchmark : str, default 'SPY'
        The benchmark ticker symbol
    risk_free_rate : float or pandas.Series, default 0.02
        The risk-free rate (annualized)
        If float: used as a constant rate
        If Series: used as daily rates (must have dates as index matching df)
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Extract relevant columns
    strategy_returns = df[f'{ticker}_strategy_return']
    
    # Calculate or extract benchmark returns
    if f'{benchmark}_daily_return' in df.columns:
        benchmark_returns = df[f'{benchmark}_daily_return']
    elif f'{benchmark}_close' in df.columns:
        #print(f'Calculating {benchmark} returns...')
        # Calculate benchmark returns if they haven't been calculated yet
        benchmark_returns = df[f'{benchmark}_close'].pct_change().dropna()
    else:
        benchmark_returns = None
    
    # Handle risk-free rate based on type
    if isinstance(risk_free_rate, pd.Series):
        # For Series, align with our data and convert annual rates to daily
        daily_rf = pd.Series(index=df.index)
        
        # For each date in our dataframe, find the matching or nearest previous date in risk_free_rate
        for date in df.index:
            try:
                # Get the most recent rate up to this date
                rate_idx = risk_free_rate.index.get_indexer([date], method='pad')[0]
                if rate_idx >= 0:  # Valid index found
                    # Convert annual rate to daily rate
                    daily_rf[date] = risk_free_rate.iloc[rate_idx] / 252
                else:  # No valid index, use default
                    daily_rf[date] = 0.02 / 252
            except:
                # Fallback to default if any error occurs
                daily_rf[date] = 0.02 / 252
    else:
        # For scalar, use the same daily rate for all days
        daily_rf = pd.Series(risk_free_rate / 252, index=df.index)
    
    # Calculate excess returns (return - risk-free rate)
    excess_returns = strategy_returns - daily_rf
    
    # Calculate annualized metrics
    annual_return = (1 + strategy_returns.mean()) ** 252 - 1
    annual_volatility = strategy_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio using excess returns
    sharpe_ratio = excess_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Calculate beta and alpha if benchmark data is available
    beta = alpha = None
    if benchmark_returns is not None:
        # Ensure benchmark_returns and strategy_returns have the same length and are aligned by date
        # This is necessary because sometimes the arrays might have slightly different lengths
        aligned_benchmark, aligned_strategy = benchmark_returns.align(strategy_returns, join='inner')
        
        # Only proceed if we have enough data points
        if len(aligned_benchmark) > 1:
            # Calculate beta using linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                aligned_benchmark.values, aligned_strategy.values
            )
            beta = slope
            
            # Calculate alpha (Jensen's Alpha)
            benchmark_annual_return = (1 + aligned_benchmark.mean()) ** 252 - 1
            
            # Use average annual risk-free rate for alpha calculation
            if isinstance(risk_free_rate, pd.Series):
                avg_annual_rf = risk_free_rate.mean()
            else:
                avg_annual_rf = risk_free_rate
                
            alpha = annual_return - (avg_annual_rf + beta * (benchmark_annual_return - avg_annual_rf))
    
    # Create a dictionary of metrics
    metrics = {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'beta': beta,
        'alpha': alpha,
        'total_return': (1 + strategy_returns).cumprod().iloc[-1] - 1,
        'win_rate': len(strategy_returns[strategy_returns > 0]) / len(strategy_returns)
    }
    
    return metrics

def plot_cumulative_returns(df, ticker, benchmark=None, figsize=(12, 6)):
    """
    Plot cumulative returns of the strategy and benchmark.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing strategy returns
    ticker : str
        The ticker symbol
    benchmark : str, default None
        The benchmark ticker symbol
    figsize : tuple, default (12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + df[f'{ticker}_strategy_return']).cumprod()
    
    # Plot strategy returns
    ax.plot(strategy_cumulative, label=f'{ticker} Strategy', linewidth=2)
    
    # Plot benchmark returns if available
    if benchmark and f'{benchmark}_daily_return' in df.columns:
        benchmark_cumulative = (1 + df[f'{benchmark}_daily_return']).cumprod()
        ax.plot(benchmark_cumulative, label=f'{benchmark} Buy & Hold', linewidth=2, alpha=0.7)
    
    ax.set_title('Cumulative Returns', fontsize=14)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_rolling_metrics(df, ticker, window=252, figsize=(15, 12)):
    """
    Plot rolling metrics (volatility, Sharpe ratio).
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing strategy returns
    ticker : str
        The ticker symbol
    window : int, default 252
        Rolling window size
    figsize : tuple, default (15, 12)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Extract returns
    returns = df[f'{ticker}_strategy_return']
    
    # Calculate rolling volatility (annualized)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Calculate rolling Sharpe ratio (annualized)
    rolling_mean = returns.rolling(window=window).mean() * 252
    rolling_sharpe = rolling_mean / rolling_vol
    
    # Plot rolling volatility
    axes[0].plot(rolling_vol, linewidth=2)
    axes[0].set_title(f'Rolling {window}-day Volatility (Annualized)', fontsize=14)
    axes[0].set_ylabel('Volatility', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot rolling Sharpe ratio
    axes[1].plot(rolling_sharpe, linewidth=2)
    axes[1].set_title(f'Rolling {window}-day Sharpe Ratio', fontsize=14)
    axes[1].set_ylabel('Sharpe Ratio', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig

def plot_drawdowns(df, ticker, figsize=(12, 6)):
    """
    Plot drawdowns over time.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe containing strategy returns
    ticker : str
        The ticker symbol
    figsize : tuple, default (12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate drawdowns
    cumulative_returns = (1 + df[f'{ticker}_strategy_return']).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / peak) - 1
    
    # Plot drawdowns
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown, color='red', linewidth=1)
    
    ax.set_title('Strategy Drawdowns', fontsize=14)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines at common drawdown levels
    ax.axhline(y=-0.1, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=-0.3, color='gray', linestyle='--', alpha=0.7)
    
    return fig

def generate_performance_report(metrics, figsize=(8, 6)):
    """
    Generate a performance report table as a figure.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of performance metrics
    figsize : tuple, default (8, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Hide axes
    ax.axis('off')
    
    # Create table data
    metrics_to_display = {
        'Annual Return': f"{metrics['annual_return']:.2%}",
        'Annual Volatility': f"{metrics['annual_volatility']:.2%}",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
        'Total Return': f"{metrics['total_return']:.2%}",
        'Win Rate': f"{metrics['win_rate']:.2%}",
        'ALPHA': f"{metrics['alpha']:.2%}" if metrics['alpha'] is not None else None,
    }
    
    # Add beta and alpha if available
    if metrics['beta'] is not None:
        metrics_to_display['Beta'] = f"{metrics['beta']:.2f}"
    if metrics['alpha'] is not None:
        metrics_to_display['Alpha'] = f"{metrics['alpha']:.2%}"
    
    # Create table
    table_data = list(metrics_to_display.items())
    
    # Create table
    table = ax.table(
        cellText=[[k, v] for k, v in table_data],
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.6, 0.4]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Set title
    ax.set_title('Performance Metrics', fontsize=16, pad=20)
    
    fig.tight_layout()
    
    return fig