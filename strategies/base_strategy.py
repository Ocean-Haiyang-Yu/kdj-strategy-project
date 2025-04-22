class BaseStrategy:
    """
    Base class for all trading strategies.
    
    This class defines the interface that all strategy classes should implement.
    """
    
    def __init__(self, ticker, params=None):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol
        params : dict, default None
            Strategy-specific parameters
        """
        self.ticker = ticker
        self.params = params or {}
        self.name = "Base Strategy"
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy.
        
        Parameters:
        -----------
        data : DataFrame
            The dataframe containing price and indicator data
            
        Returns:
        --------
        DataFrame
            The input dataframe with signal column added
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def prepare_data(self, data):
        """
        Prepare data for the strategy (calculate indicators, etc.).
        
        Parameters:
        -----------
        data : DataFrame
            The dataframe containing price data
            
        Returns:
        --------
        DataFrame
            The input dataframe with necessary indicators added
        """
        raise NotImplementedError("Subclasses must implement prepare_data")
    
    def get_parameters(self):
        """
        Get the strategy parameters.
        
        Returns:
        --------
        dict
            The strategy parameters
        """
        return self.params
    
    def set_parameters(self, params):
        """
        Set the strategy parameters.
        
        Parameters:
        -----------
        params : dict
            The strategy parameters
            
        Returns:
        --------
        self
            The strategy instance for method chaining
        """
        self.params.update(params)
        return self
    
    def __str__(self):
        """
        String representation of the strategy.
        
        Returns:
        --------
        str
            String representation
        """
        return f"{self.name} for {self.ticker} with parameters: {self.params}"
