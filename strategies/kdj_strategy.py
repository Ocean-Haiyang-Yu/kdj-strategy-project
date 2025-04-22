from .base_strategy import BaseStrategy
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.indicator_utils import calculate_kdj

class KDJStrategy(BaseStrategy):
    """
    KDJ indicator based trading strategy.
    
    This strategy generates buy signals when the J value is less than a buy threshold
    and sell signals when the J value is greater than a sell threshold.
    """
    
    def __init__(self, ticker, params=None):
        """
        Initialize the KDJ strategy.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol
        params : dict, default None
            Strategy-specific parameters
        """
        # Default parameters
        default_params = {
            'k_period': 9,
            'd_period': 3,
            'j_period': 3,
            'j_buy_threshold': 20,
            'j_sell_threshold': 80,
            'daily_enabled': True,
            'weekly_enabled': False,
            'monthly_enabled': False,
            'daily_weight': 0.5,
            'weekly_weight': 1.0,
            'monthly_weight': 0.3,
            'position_size': 1.0  # Full position by default
        }
        
        # Update default parameters with provided parameters
        if params:
            default_params.update(params)
        
        super().__init__(ticker, default_params)
        self.name = "KDJ Strategy"
    
    def prepare_data(self, data):
        """
        Prepare data for the KDJ strategy (calculate KDJ indicator).
        
        Parameters:
        -----------
        data : DataFrame
            The dataframe containing price data
            
        Returns:
        --------
        DataFrame
            The input dataframe with KDJ indicator added
        """
        # Calculate KDJ for daily data
        if self.params['daily_enabled']:
            data = calculate_kdj(
                data, 
                self.ticker, 
                k_period=self.params['k_period'],
                d_period=self.params['d_period'],
                j_period=self.params['j_period']
            )
        
        # Calculate KDJ for weekly data if enabled
        if self.params['weekly_enabled']:
            # Resample to weekly
            import pandas as pd
            weekly_data = data.resample('W').agg({
                f'{self.ticker}_high': 'max',
                f'{self.ticker}_low': 'min',
                f'{self.ticker}_close': 'last'
            })
            
            # Calculate KDJ on weekly data
            weekly_data = calculate_kdj(
                weekly_data, 
                self.ticker, 
                k_period=self.params['k_period'],
                d_period=self.params['d_period'],
                j_period=self.params['j_period']
            )
            
            # Merge weekly KDJ back to daily data
            weekly_j = weekly_data[f'{self.ticker}_kdj_j'].resample('D').ffill()
            data[f'{self.ticker}_kdj_j_weekly'] = weekly_j
        
        # Calculate KDJ for monthly data if enabled
        if self.params['monthly_enabled']:
            # Resample to monthly
            monthly_data = data.resample('M').agg({
                f'{self.ticker}_high': 'max',
                f'{self.ticker}_low': 'min',
                f'{self.ticker}_close': 'last'
            })
            
            # Calculate KDJ on monthly data
            monthly_data = calculate_kdj(
                monthly_data, 
                self.ticker, 
                k_period=self.params['k_period'],
                d_period=self.params['d_period'],
                j_period=self.params['j_period']
            )
            
            # Merge monthly KDJ back to daily data
            monthly_j = monthly_data[f'{self.ticker}_kdj_j'].resample('D').ffill()
            data[f'{self.ticker}_kdj_j_monthly'] = monthly_j
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the KDJ strategy.
        
        Parameters:
        -----------
        data : DataFrame
            The dataframe containing price and KDJ indicator data
            
        Returns:
        --------
        DataFrame
            The input dataframe with signal column added
        """
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Initialize signal column (0 = no signal, 1 = buy, -1 = sell)
        df[f'{self.ticker}_signal'] = 0
        
        # Create a weighted J value based on enabled timeframes
        weighted_j = 0
        weight_sum = 0
        
        # Add daily J value if enabled
        if self.params['daily_enabled']:
            weighted_j += df[f'{self.ticker}_kdj_j'] * self.params['daily_weight']
            weight_sum += self.params['daily_weight']
        
        # Add weekly J value if enabled
        if self.params['weekly_enabled'] and f'{self.ticker}_kdj_j_weekly' in df.columns:
            weighted_j += df[f'{self.ticker}_kdj_j_weekly'] * self.params['weekly_weight']
            weight_sum += self.params['weekly_weight']
        
        # Add monthly J value if enabled
        if self.params['monthly_enabled'] and f'{self.ticker}_kdj_j_monthly' in df.columns:
            weighted_j += df[f'{self.ticker}_kdj_j_monthly'] * self.params['monthly_weight']
            weight_sum += self.params['monthly_weight']
        
        # Normalize the weighted J value
        if weight_sum > 0:
            weighted_j /= weight_sum
            df[f'{self.ticker}_kdj_j_weighted'] = weighted_j
        else:
            # If no timeframes are enabled, use daily J value as fallback
            df[f'{self.ticker}_kdj_j_weighted'] = df[f'{self.ticker}_kdj_j']
        
        # Generate signals based on weighted J value crossing thresholds
        # Buy signal: J crosses below buy threshold
        df.loc[df[f'{self.ticker}_kdj_j_weighted'] < self.params['j_buy_threshold'], f'{self.ticker}_signal'] = 1
        
        # Sell signal: J crosses above sell threshold
        df.loc[df[f'{self.ticker}_kdj_j_weighted'] > self.params['j_sell_threshold'], f'{self.ticker}_signal'] = -1
        
        # Determine position size based on signal strength
        df[f'{self.ticker}_position_size'] = df[f'{self.ticker}_signal'] * self.params['position_size']
        
        return df