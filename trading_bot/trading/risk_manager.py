import logging
from typing import Dict, Optional

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        
        # Core risk parameters
        self.max_exposure = self.config.get("max_exposure", 0.1)  # Maximum exposure as a percentage of total capital
        self.stop_loss_level = self.config.get("stop_loss_level", 0.05)  # Default stop-loss level (5%)
        self.slippage_tolerance = self.config.get("slippage_tolerance", 0.01)  # Tolerance for slippage (1%)
        
        # Additional risk parameters
        self.trailing_stop = self.config.get("trailing_stop", 0.02)  # Trailing stop distance (2%)
        self.max_drawdown = self.config.get("max_drawdown", 0.15)  # Maximum drawdown allowed (15%)
        self.take_profit = self.config.get("take_profit", 0.1)  # Take profit level (10%)
        self.min_risk_reward = self.config.get("min_risk_reward", 2.0)  # Minimum risk-reward ratio
        self.position_scaling = self.config.get("position_scaling", 0.5)  # Scale factor for subsequent positions
        
        # State tracking
        self.highest_price = {}  # Track highest price per asset for trailing stops
        self.initial_capital = None
        self.current_drawdown = 0.0

    def check_risk(self, current_price: float, order_type: str, position_size: float, 
                  current_position: Dict, total_capital: float, asset_id: str = None) -> bool:
        """
        Perform comprehensive pre-trade risk checks.
        
        Args:
            current_price: The price at which the trade is being made
            order_type: 'buy' or 'sell'
            position_size: The size of the trade (amount to buy/sell)
            current_position: Current token holdings including entry price
            total_capital: Total capital the bot has access to
            asset_id: Unique identifier for the asset being traded
        
        Returns:
            bool: Whether the trade is approved or rejected
        """
        try:
            self.logger.info("Running comprehensive pre-trade risk checks...")
            
            # Initialize tracking if needed
            if self.initial_capital is None:
                self.initial_capital = total_capital
            
            # Update drawdown
            self.current_drawdown = max(self.current_drawdown, 
                                      (self.initial_capital - total_capital) / self.initial_capital)
            
            # Basic position and exposure checks
            if not self._check_position_size(position_size, total_capital):
                return False
                
            if not self._check_max_exposure(current_price, position_size, total_capital):
                return False
            
            # Advanced risk checks
            if not self._check_drawdown():
                self.logger.warning("Trade rejected: Maximum drawdown limit reached.")
                return False
            
            if order_type == 'buy':
                if not self._check_risk_reward_ratio(current_price, current_position):
                    self.logger.warning("Trade rejected: Risk-reward ratio below minimum threshold.")
                    return False
                    
                # Scale position size if we already have a position
                if current_position and current_position.get("size", 0) > 0:
                    position_size *= self.position_scaling
                    
            elif order_type == 'sell':
                # Check both traditional and trailing stop-loss
                if not self._check_stop_loss(current_price, current_position):
                    return False
                    
                if asset_id and not self._check_trailing_stop(current_price, asset_id):
                    self.logger.warning("Trade rejected: Trailing stop-loss triggered.")
                    return False
                    
                if self._check_take_profit(current_price, current_position):
                    self.logger.info("Take profit target reached.")
                    return True
            
            # Update highest price for trailing stop
            if asset_id:
                self.highest_price[asset_id] = max(
                    self.highest_price.get(asset_id, current_price),
                    current_price
                )
            
            if not self._check_slippage(current_price):
                return False
            
            self.logger.info("All risk checks passed. Trade approved.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during risk check: {e}")
            return False

    def _check_position_size(self, position_size: float, total_capital: float) -> bool:
        """Check if the position size is within allowed limits."""
        max_position_size = self.max_exposure * total_capital
        if position_size > max_position_size:
            self.logger.info(f"Position size {position_size} exceeds max allowed {max_position_size}.")
            return False
        return True

    def _check_max_exposure(self, current_price: float, position_size: float, total_capital: float) -> bool:
        """Check if the trade would exceed maximum allowable exposure."""
        exposure = (current_price * position_size) / total_capital
        if exposure > self.max_exposure:
            self.logger.info(f"Exposure {exposure:.2%} exceeds max allowable {self.max_exposure:.2%}.")
            return False
        return True

    def _check_stop_loss(self, current_price: float, current_position: Dict) -> bool:
        """Check if the current price has hit the stop-loss threshold."""
        if not current_position or "entry_price" not in current_position:
            return True
            
        stop_loss_price = (1 - self.stop_loss_level) * current_position["entry_price"]
        if current_price <= stop_loss_price:
            self.logger.info(f"Current price {current_price} has hit stop-loss {stop_loss_price}.")
            return False
        return True

    def _check_trailing_stop(self, current_price: float, asset_id: str) -> bool:
        """Check if the current price has hit the trailing stop-loss."""
        if asset_id not in self.highest_price:
            return True
            
        trailing_stop_price = self.highest_price[asset_id] * (1 - self.trailing_stop)
        if current_price <= trailing_stop_price:
            self.logger.info(f"Current price {current_price} has hit trailing stop {trailing_stop_price}.")
            return False
        return True

    def _check_take_profit(self, current_price: float, current_position: Dict) -> bool:
        """Check if the take-profit level has been reached."""
        if not current_position or "entry_price" not in current_position:
            return False
            
        take_profit_price = (1 + self.take_profit) * current_position["entry_price"]
        return current_price >= take_profit_price

    def _check_risk_reward_ratio(self, current_price: float, current_position: Dict) -> bool:
        """Check if the trade meets minimum risk-reward ratio requirements."""
        if not current_position:
            return True
            
        stop_loss_price = (1 - self.stop_loss_level) * current_price
        take_profit_price = (1 + self.take_profit) * current_price
        
        risk = current_price - stop_loss_price
        reward = take_profit_price - current_price
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        return risk_reward_ratio >= self.min_risk_reward

    def _check_drawdown(self) -> bool:
        """Check if the maximum drawdown limit has been reached."""
        return self.current_drawdown <= self.max_drawdown

    def _check_slippage(self, current_price: float, expected_price: Optional[float] = None) -> bool:
        """
        Check if the slippage is within tolerance limits.
        Args:
            current_price: The actual execution price
            expected_price: The expected price (if None, uses current_price as expected)
        """
        expected_price = expected_price or current_price
        slippage = abs(current_price - expected_price) / expected_price

        if slippage > self.slippage_tolerance:
            self.logger.info(f"Slippage {slippage:.2%} exceeds tolerance {self.slippage_tolerance:.2%}.")
            return False
        return True

    def reset_tracking(self):
        """Reset all tracking variables."""
        self.highest_price = {}
        self.initial_capital = None
        self.current_drawdown = 0.0