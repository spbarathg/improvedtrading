import logging

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.max_exposure = self.config.get("max_exposure", 0.1)  # Maximum exposure as a percentage of total capital
        self.stop_loss_level = self.config.get("stop_loss_level", 0.05)  # Default stop-loss level (5%)
        self.slippage_tolerance = self.config.get("slippage_tolerance", 0.01)  # Tolerance for slippage (1%)

    def check_risk(self, current_price, order_type, position_size, current_position, total_capital):
        """
        Perform pre-trade checks to assess risk before placing an order.
        current_price: The price at which the trade is being made.
        order_type: 'buy' or 'sell'.
        position_size: The size of the trade (amount to buy/sell).
        current_position: Current token holdings.
        total_capital: Total capital the bot has access to.
        
        Returns a boolean indicating whether the trade is approved or rejected.
        """
        try:
            self.logger.info("Running pre-trade risk checks...")

            # Check position sizing
            if not self._check_position_size(position_size, total_capital):
                self.logger.warning("Trade rejected: Position size exceeds allowed limits.")
                return False

            # Check exposure limits (maximum risk exposure)
            if not self._check_max_exposure(current_price, position_size, total_capital):
                self.logger.warning("Trade rejected: Maximum exposure limit reached.")
                return False

            # Check stop-loss for sell orders
            if order_type == 'sell' and not self._check_stop_loss(current_price, current_position):
                self.logger.warning("Trade rejected: Stop-loss condition triggered.")
                return False

            # Check slippage tolerance
            if not self._check_slippage(current_price):
                self.logger.warning("Trade rejected: Slippage exceeds tolerance.")
                return False

            self.logger.info("Risk checks passed. Trade approved.")
            return True

        except Exception as e:
            self.logger.error(f"Error during risk check: {e}")
            return False

    def _check_position_size(self, position_size, total_capital):
        """Check if the position size is within allowed limits based on risk management rules."""
        max_position_size = self.max_exposure * total_capital
        if position_size > max_position_size:
            self.logger.info(f"Position size {position_size} exceeds max allowed {max_position_size}.")
            return False
        return True

    def _check_max_exposure(self, current_price, position_size, total_capital):
        """Check if the trade would exceed the bot's maximum allowable exposure."""
        exposure = (current_price * position_size) / total_capital
        if exposure > self.max_exposure:
            self.logger.info(f"Exposure {exposure:.2%} exceeds max allowable {self.max_exposure:.2%}.")
            return False
        return True

    def _check_stop_loss(self, current_price, current_position):
        """Check if the current price has hit the stop-loss threshold."""
        stop_loss_price = (1 - self.stop_loss_level) * current_position["entry_price"]
        if current_price <= stop_loss_price:
            self.logger.info(f"Current price {current_price} has hit stop-loss {stop_loss_price}.")
            return False
        return True

    def _check_slippage(self, current_price):
        """Check if the slippage is within tolerance limits."""
        # Simulated example. In practice, slippage is calculated by comparing expected and executed prices.
        expected_price = current_price  # Replace with actual expected price
        actual_price = current_price  # Replace with actual executed price
        slippage = abs(actual_price - expected_price) / expected_price

        if slippage > self.slippage_tolerance:
            self.logger.info(f"Slippage {slippage:.2%} exceeds tolerance {self.slippage_tolerance:.2%}.")
            return False
        return True