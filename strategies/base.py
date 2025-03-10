from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration parameters."""
        self.config = config
        self.position = 0  # Current position: 0 (none), 1 (long), -1 (short)
        self.indicators: Dict[str, Any] = {}  # Store calculated indicators
        self.last_update_time: Optional[datetime] = None
        self._initialized = False
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        if not self._initialized:
            self.calculate_indicators(data)
            self._initialized = True
    
    @abstractmethod
    def generate_signal(self, bar: pd.Series) -> int:
        """
        Generate trading signal based on current market data.
        
        Args:
            bar (pd.Series): Current market data bar containing OHLCV data
            
        Returns:
            int: Trading signal where:
                 1 = long signal
                 0 = no signal
                -1 = short signal
        """
        pass
    
    def calculate_indicators(self, data: pd.DataFrame) -> None:
        """
        Calculate technical indicators used by the strategy.
        Override this method in child classes to implement specific indicators.
        
        Args:
            data (pd.DataFrame): Historical market data
        """
        pass
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        Override this method in child classes to implement specific validation.
        
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        return True
    
    def update_position(self, signal: int) -> None:
        """
        Update the current position based on the signal.
        
        Args:
            signal (int): Trading signal (1, 0, -1)
        """
        self.position = signal
        self.last_update_time = datetime.now()
    
    def get_required_columns(self) -> list[str]:
        """
        Get the list of required data columns for this strategy.
        Override this method in child classes if they need specific columns.
        
        Returns:
            list[str]: List of required column names
        """
        return ['timestamp', 'open', 'high', 'low', 'close', 'volume'] 