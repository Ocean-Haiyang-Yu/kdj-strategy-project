import pandas as pd
import numpy as np
import itertools
import concurrent.futures
import time
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine

class GridSearch:
    """
    Grid search for optimizing strategy parameters.
    
    This class performs grid search to find the best parameters for a strategy.
    """
    
    def __init__(self, data, ticker, strategy_class, param_grid, metric='sharpe_ratio', maximize=True, n_jobs=1):
        """
        Initialize the grid search.
        
        Parameters:
        -----------
        data : DataFrame
            The dataframe containing price data
        ticker : str
            The ticker symbol
        strategy_class : class
            The strategy class to optimize
        param_grid : dict
            Dictionary of parameter grids
        metric : str, default 'sharpe_ratio'
            The metric to optimize ('sharpe_ratio', 'annual_return', 'max_drawdown', etc.)
        maximize : bool, default True
            Whether to maximize or minimize the metric
        n_jobs : int, default 1
            Number of parallel jobs to run
        """
        self.data = data
        self.ticker = ticker
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.metric = metric
        self.maximize = maximize
        self.n_jobs = n_jobs
        
        self.results = []
        self.best_params = None
        self.best_score = None
        self.best_backtest = None
    
    def _evaluate_params(self, params):
        """
        Evaluate a single set of parameters.
        
        Parameters:
        -----------
        params : dict
            The parameters to evaluate
            
        Returns:
        --------
        dict
            Evaluation results
        """
        # Create strategy instance with parameters
        strategy = self.strategy_class(self.ticker, params)
        
        # Create backtester
        engine = BacktestEngine()
        
        # Run backtest
        backtest_results = engine.run(self.data, self.ticker, strategy)
        
        # Extract performance metrics
        from utils.performance_utils import calculate_returns
        metrics = calculate_returns(backtest_results, self.ticker)
        
        # Get the score based on the specified metric
        score = metrics.get(self.metric, 0)
        
        # Return evaluation results
        return {
            'params': params,
            'score': score,
            'metrics': metrics,
            'backtest_results': backtest_results
        }
    
    def fit(self, show_progress=True):
        """
        Perform grid search to find the best parameters.
        
        Parameters:
        -----------
        show_progress : bool, default True
            Whether to show progress during the search
            
        Returns:
        --------
        self
            The grid search instance for method chaining
        """
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Print number of combinations
        if show_progress:
            print(f"Grid search with {len(param_combinations)} parameter combinations")
        
        # Initialize results list and tracking variables
        self.results = []
        start_time = time.time()
        
        # Evaluate parameters (parallel or sequential)
        if self.n_jobs > 1:
            # Parallel evaluation
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                
                # Submit all parameter combinations for evaluation
                for params_tuple in param_combinations:
                    params = dict(zip(param_names, params_tuple))
                    futures.append(executor.submit(self._evaluate_params, params))
                
                # Collect results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    self.results.append(future.result())
                    
                    # Show progress if requested
                    if show_progress and (i+1) % max(1, len(param_combinations) // 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"Progress: {i+1}/{len(param_combinations)} combinations evaluated ({elapsed:.2f}s)")
        else:
            # Sequential evaluation
            for i, params_tuple in enumerate(param_combinations):
                params = dict(zip(param_names, params_tuple))
                result = self._evaluate_params(params)
                self.results.append(result)
                
                # Show progress if requested
                if show_progress and (i+1) % max(1, len(param_combinations) // 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"Progress: {i+1}/{len(param_combinations)} combinations evaluated ({elapsed:.2f}s)")
        
        # Find the best parameters
        compare_func = max if self.maximize else min
        best_result = compare_func(self.results, key=lambda x: x['score'])
        
        self.best_params = best_result['params']
        self.best_score = best_result['score']
        self.best_backtest = best_result['backtest_results']
        
        # Print result summary
        if show_progress:
            total_time = time.time() - start_time
            print(f"\nGrid search completed in {total_time:.2f}s")
            print(f"Best {self.metric}: {self.best_score:.4f}")
            print(f"Best parameters: {self.best_params}")
        
        return self
    
    def get_results_df(self):
        """
        Get the grid search results as a dataframe.
        
        Returns:
        --------
        DataFrame
            Grid search results
        """
        # Extract parameter values and scores
        results_list = []
        
        for result in self.results:
            row = result['params'].copy()
            row['score'] = result['score']
            
            # Add other metrics
            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value
            
            results_list.append(row)
        
        # Create dataframe
        results_df = pd.DataFrame(results_list)
        
        # Sort by score
        results_df = results_df.sort_values('score', ascending=not self.maximize)
        
        return results_df
    
    def plot_param_impact(self, param_name, metric=None, figsize=(10, 6)):
        """
        Plot the impact of a parameter on the specified metric.
        
        Parameters:
        -----------
        param_name : str
            The parameter name to plot
        metric : str, default None
            The metric to plot (uses the optimization metric if None)
        figsize : tuple, default (10, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set default metric if not specified
        if metric is None:
            metric = self.metric
        
        # Get results dataframe
        results_df = self.get_results_df()
        
        # Check if parameter exists
        if param_name not in results_df.columns:
            raise ValueError(f"Parameter '{param_name}' not found in grid search results")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot parameter impact
        sns.boxplot(x=param_name, y=metric, data=results_df, ax=ax)
        
        ax.set_title(f'Impact of {param_name} on {metric}', fontsize=14)
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        return fig
    
    def plot_param_heatmap(self, param1, param2, metric=None, figsize=(10, 8)):
        """
        Plot a heatmap of two parameters' impact on the specified metric.
        
        Parameters:
        -----------
        param1 : str
            The first parameter name
        param2 : str
            The second parameter name
        metric : str, default None
            The metric to plot (uses the optimization metric if None)
        figsize : tuple, default (10, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set default metric if not specified
        if metric is None:
            metric = self.metric
        
        # Get results dataframe
        results_df = self.get_results_df()
        
        # Check if parameters exist
        if param1 not in results_df.columns or param2 not in results_df.columns:
            raise ValueError(f"Parameters '{param1}' or '{param2}' not found in grid search results")
        
        # Create pivot table for heatmap
        pivot_table = results_df.pivot_table(
            index=param1, 
            columns=param2, 
            values=metric,
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(pivot_table, annot=True, cmap='viridis', ax=ax)
        
        ax.set_title(f'Impact of {param1} and {param2} on {metric}', fontsize=14)
        
        fig.tight_layout()
        
        return fig
