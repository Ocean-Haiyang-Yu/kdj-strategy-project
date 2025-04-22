# KDJ Multi-Timeframe Trading Strategy

## Overview
This project implements a comprehensive algorithmic trading system based on the KDJ indicator with multi-timeframe analysis capabilities. Developed as part of the MGMTMFE 412 course at UCLA Anderson School of Management, the framework enables back-testing, optimization, and performance evaluation of trading strategies.

## Project Structure
```
kdj-strategy-project/
├── backtest/             # Backtesting engine and performance evaluation
│   └── engine.py         # Event-driven backtesting implementation
├── data/                 # Data handling utilities
├── notebooks/            # Jupyter notebooks for strategy exploration
├── optimize/             # Parameter optimization tools
│   └── grid_search.py    # Grid search implementation for parameter tuning
├── results/              # Storage for backtest results
├── strategies/           # Strategy implementations
│   ├── base_strategy.py  # Abstract base class for trading strategies
│   └── kdj_strategy.py   # KDJ indicator strategy implementation
└── utils/                # Utility functions
    ├── data_utils.py     # Data loading and preprocessing
    ├── indicator_utils.py # Technical indicator calculations
    ├── performance_utils.py # Performance metrics calculation
    └── visualization_utils.py # Visualization functions
```

## Features
- Multi-timeframe analysis (daily, weekly, monthly)
- KDJ stochastic oscillator implementation with customizable parameters
- Event-driven backtesting engine with transaction costs and cash interest
- Comprehensive performance metrics:
  - Annualized return and volatility
  - Sharpe ratio
  - Maximum drawdown
  - Beta calculation (via regression)
  - Performance alpha
- Parameter optimization via grid search
- Visualization suite for equity curves, drawdowns, and performance metrics

## Installation & Setup
```bash
# Clone the repository
git clone https://github.com/username/kdj-strategy-project.git
cd kdj-strategy-project

# Create and activate a virtual environment (optional)
conda create -n trading python=3.9
conda activate trading

# Install dependencies
pip install -r requirements.txt
```

## Usage
The primary interface for the trading system is through Jupyter notebooks:

```python
# Example usage in a notebook
from utils.data_utils import load_data
from strategies.kdj_strategy import KDJStrategy
from backtest.engine import Backtest

# Load data
data = load_data('AAPL', start_date='2020-01-01', end_date='2023-01-01')

# Create strategy
strategy = KDJStrategy(
    ticker='AAPL',
    params={
        'k_period': 14,
        'j_buy_threshold': 0.0,
        'j_sell_threshold': 100.0,
        'daily_enabled': True,
        'weekly_enabled': True,
        'monthly_enabled': True
    }
)

# Run backtest
backtest = Backtest(
    strategy=strategy,
    data=data,
    initial_capital=100000,
    commission=0.001
)
results = backtest.run()

# Display performance metrics
print(results.stats)
```

## Requirements
- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- yfinance
- scikit-learn
- statsmodels
- jupyter

## Project Requirements
This project fulfills the requirements for the UCLA Anderson MGMTMFE 412 Final Group Project, which include:
- Implementation of a trading strategy with economic intuition
- Comprehensive performance metrics and visualization
- Presentation of backtesting results

## License
[MIT License](LICENSE)

## Author
Haiyang Yu

## Acknowledgements
- UCLA Anderson School of Management
- Financial data provided by YahooFinance
