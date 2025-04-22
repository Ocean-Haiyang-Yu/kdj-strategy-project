"""
Utilities for visualizing trading performance and strategies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.performance_utils import (
    calculate_cumulative_returns,
    calculate_rolling_volatility,
    calculate_rolling_sharpe
)

def set_plotting_style():
    """Set the plotting style for visualizations."""
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.grid'] = True

def plot_cumulative_returns(returns, benchmark_returns=None, title='Cumulative Returns'):
    """
    Plot cumulative returns of strategy and benchmark.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of strategy returns
    benchmark_returns : pandas.Series, optional
        Series of benchmark returns
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.pyplot.Figure
        Figure object for further customization
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative returns
    cum_returns = calculate_cumulative_returns(returns)
    
    # Plot strategy returns
    cum_returns.mul(100).plot(ax=ax, label='Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        cum_benchmark = calculate_cumulative_returns(benchmark_returns)
        cum_benchmark.mul(100).plot(ax=ax, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Cumulative Returns (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
    plt.tight_layout()
    
    return fig

def plot_rolling_volatility(returns, benchmark_returns=None, window=252, title='Rolling Volatility'):
    """
    Plot rolling volatility of strategy and benchmark.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of strategy returns
    benchmark_returns : pandas.Series, optional
        Series of benchmark returns
    window : int, optional
        Rolling window size, default is 252 (1 year of trading days)
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.pyplot.Figure
        Figure object for further customization
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate rolling volatility
    rolling_vol = calculate_rolling_volatility(returns, window)
    
    # Plot strategy volatility
    rolling_vol.mul(100).plot(ax=ax, label='Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        benchmark_vol = calculate_rolling_volatility(benchmark_returns, window)
        benchmark_vol.mul(100).plot(ax=ax, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
    plt.tight_layout()
    
    return fig

def plot_rolling_sharpe(returns, benchmark_returns=None, window=252, title='Rolling Sharpe Ratio'):
    """
    Plot rolling Sharpe ratio of strategy and benchmark.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of strategy returns
    benchmark_returns : pandas.Series, optional
        Series of benchmark returns
    window : int, optional
        Rolling window size, default is 252 (1 year of trading days)
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.pyplot.Figure
        Figure object for further customization
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate rolling Sharpe ratio
    rolling_sharpe = calculate_rolling_sharpe(returns, window)
    
    # Plot strategy Sharpe ratio
    rolling_sharpe.plot(ax=ax, label='Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        benchmark_sharpe = calculate_rolling_sharpe(benchmark_returns, window)
        benchmark_sharpe.plot(ax=ax, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_drawdown(returns, title='Drawdown'):
    """
    Plot drawdown of strategy returns.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of strategy returns
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.pyplot.Figure
        Figure object for further customization
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative returns
    cum_returns = calculate_cumulative_returns(returns)
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / (1 + running_max)
    
    # Plot drawdown
    drawdown.mul(100).plot(ax=ax, linewidth=2, color='red')
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
    ax.fill_between(drawdown.index, 0, drawdown.mul(100), color='red', alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_monthly_returns_heatmap(returns, title='Monthly Returns (%)'):
    """
    Plot heatmap of monthly returns.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of daily returns
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.pyplot.Figure
        Figure object for further customization
    """
    set_plotting_style()
    
    # Resample to monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table with years and months
    monthly_returns.index = pd.MultiIndex.from_arrays([
        monthly_returns.index.year,
        monthly_returns.index.month
    ], names=['Year', 'Month'])
    
    monthly_pivot = monthly_returns.unstack(level=1) * 100
    
    # Convert month numbers to names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_pivot.columns = [month_names[m] for m in monthly_pivot.columns]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    
    sns.heatmap(
        monthly_pivot,
        annot=True,
        fmt='.1f',
        center=0,
        cmap=cmap,
        linewidths=1,
        cbar_kws={'label': 'Return (%)'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_performance_summary(returns, benchmark_returns=None):
    """
    Create a 2x2 grid of key performance charts.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of strategy returns
    benchmark_returns : pandas.Series, optional
        Series of benchmark returns
        
    Returns:
    --------
    matplotlib.pyplot.Figure
        Figure object for further customization
    """
    set_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cumulative returns
    cum_returns = calculate_cumulative_returns(returns)
    cum_returns.mul(100).plot(ax=axes[0, 0], label='Strategy', linewidth=2)
    
    if benchmark_returns is not None:
        cum_benchmark = calculate_cumulative_returns(benchmark_returns)
        cum_benchmark.mul(100).plot(ax=axes[0, 0], label='Benchmark', linewidth=2, alpha=0.7)
    
    axes[0, 0].set_title('Cumulative Returns (%)', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
    
    # Rolling volatility
    rolling_vol = calculate_rolling_volatility(returns)
    rolling_vol.mul(100).plot(ax=axes[0, 1], label='Strategy', linewidth=2)
    
    if benchmark_returns is not None:
        benchmark_vol = calculate_rolling_volatility(benchmark_returns)
        benchmark_vol.mul(100).plot(ax=axes[0, 1], label='Benchmark', linewidth=2, alpha=0.7)
    
    axes[0, 1].set_title('Rolling Volatility (%)', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
    
    # Rolling Sharpe ratio
    rolling_sharpe = calculate_rolling_sharpe(returns)
    rolling_sharpe.plot(ax=axes[1, 0], label='Strategy', linewidth=2)
    
    if benchmark_returns is not None:
        benchmark_sharpe = calculate_rolling_sharpe(benchmark_returns)
        benchmark_sharpe.plot(ax=axes[1, 0], label='Benchmark', linewidth=2, alpha=0.7)
    
    axes[1, 0].set_title('Rolling Sharpe Ratio', fontsize=14)
    axes[1, 0].legend()
    
    # Drawdown
    cum_returns = calculate_cumulative_returns(returns)
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / (1 + running_max)
    drawdown.mul(100).plot(ax=axes[1, 1], linewidth=2, color='red')
    
    axes[1, 1].set_title('Drawdown (%)', fontsize=14)
    axes[1, 1].yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
    axes[1, 1].fill_between(drawdown.index, 0, drawdown.mul(100), color='red', alpha=0.3)
    
    plt.tight_layout()
    
    return fig