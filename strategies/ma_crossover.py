import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from .base import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    Generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy with configuration parameters.
        
        Required config parameters:
        - fast_ma_period: Period for the fast moving average
        - slow_ma_period: Period for the slow moving average
        """
        super().__init__(config)
        self.fast_ma_period = int(config.get('fast_ma_period', 10))
        self.slow_ma_period = int(config.get('slow_ma_period', 20))
        self.fast_ma: Optional[pd.Series] = None
        self.slow_ma: Optional[pd.Series] = None
        
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not all(key in self.config for key in ['fast_ma_period', 'slow_ma_period']):
            return False
        if self.fast_ma_period >= self.slow_ma_period:
            return False
        if self.fast_ma_period < 2 or self.slow_ma_period < 2:
            return False
        return True
        
    def calculate_indicators(self, data: pd.DataFrame) -> None:
        """Calculate moving averages."""
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
            
        if len(data) < self.slow_ma_period:
            raise ValueError(f"Not enough data for calculation. Need at least {self.slow_ma_period} bars")
            
        self.fast_ma = data['close'].rolling(window=self.fast_ma_period).mean()
        self.slow_ma = data['close'].rolling(window=self.slow_ma_period).mean()
        
        # Store in indicators dict for potential external access
        self.indicators['fast_ma'] = self.fast_ma
        self.indicators['slow_ma'] = self.slow_ma
        
    def generate_signal(self, bar: pd.Series) -> int:
        """
        Generate trading signals based on MA crossover.
        
        Returns:
            int: 1 for buy signal, -1 for sell signal, 0 for no signal
        """
        if not self._initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first")
            
        if self.fast_ma is None or self.slow_ma is None:
            raise RuntimeError("Moving averages not calculated")
            
        current_idx = bar.name
        
        # Skip if we don't have enough data
        if current_idx < self.slow_ma_period:
            return 0
            
        try:
            # Get current and previous values
            curr_fast = self.fast_ma[current_idx]
            curr_slow = self.slow_ma[current_idx]
            prev_fast = self.fast_ma[current_idx - 1]
            prev_slow = self.slow_ma[current_idx - 1]
            
            # Check for crossover
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                return 1  # Buy signal
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                return -1  # Sell signal
                
            return 0  # No signal
            
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Error accessing data at index {current_idx}: {str(e)}")
            
    def get_required_columns(self) -> list[str]:
        """Get required data columns for this strategy."""
        return ['close']  # We only need the close price for MA calculation 