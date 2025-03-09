import asyncio
import logging
from typing import Dict, Optional
from trading_bot.trading.exchange import Exchange
from trading_bot.trading.risk_manager import RiskManager

class OrderManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.exchange = Exchange(self.config)
        self.risk_manager = RiskManager(self.config)
        self.open_orders: Dict[str, dict] = {}  # Dictionary to track open orders with their details
        self.positions: Dict[str, dict] = {}  # Dictionary to track token holdings with entry prices
        self._tracking_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self):
        """Start the order manager and its background tasks"""
        if not self._is_running:
            self._is_running = True
            self._tracking_task = asyncio.create_task(self.track_orders())
            self.logger.info("OrderManager started")

    async def stop(self):
        """Stop the order manager and clean up resources"""
        if self._is_running:
            self._is_running = False
            if self._tracking_task:
                self._tracking_task.cancel()
                try:
                    await self._tracking_task
                except asyncio.CancelledError:
                    pass
            await self.exchange.close()
            self.logger.info("OrderManager stopped")

    async def place_order(self, order_type, input_token, output_token, amount, stop_loss_price=None):
        """
        Place an order (buy/sell) and track its status.
        order_type: 'buy' or 'sell'
        input_token: Token to sell (for buy) or buy (for sell)
        output_token: Token to buy (for buy) or sell (for sell)
        amount: Amount of input_token to trade
        stop_loss_price: Optional stop loss price for the order
        Returns: order_id if successful
        """
        try:
            # Check risk before placing order
            current_price = await self.exchange.get_current_price(input_token, output_token)
            total_capital = self.config.get("total_capital", 0)
            
            if not await self.risk_manager.check_risk(
                current_price=current_price,
                order_type=order_type,
                position_size=amount,
                current_position=self.positions.get(input_token, {}),
                total_capital=total_capital
            ):
                raise ValueError("Risk check failed")

            self.logger.info(f"Placing {order_type} order: {amount} {input_token} -> {output_token}")
            
            # Execute the trade using the Exchange
            order_result = await self.exchange.execute_trade(input_token, output_token, amount)
            
            # Generate or get order_id from exchange result
            order_id = order_result.get('order_id', f"{order_type}-{input_token}-{output_token}-{amount}")
            
            # Store order details including stop loss
            self.open_orders[order_id] = {
                "status": "pending",
                "type": order_type,
                "input_token": input_token,
                "output_token": output_token,
                "amount": amount,
                "entry_price": current_price,
                "stop_loss_price": stop_loss_price
            }

            # Start monitoring the order
            asyncio.create_task(self.monitor_order(order_id))
            
            return order_id

        except Exception as e:
            self.logger.error(f"Error placing {order_type} order: {e}")
            raise

    async def get_order_status(self, order_id: str) -> str:
        """
        Get the current status of an order
        Returns: Order status string ('pending', 'confirmed', 'cancelled', 'failed', etc.)
        """
        try:
            if order_id not in self.open_orders:
                return "not_found"
            
            # Get real-time status from exchange
            status = await self.exchange.get_order_status(order_id)
            self.open_orders[order_id]["status"] = status
            return status

        except Exception as e:
            self.logger.error(f"Error getting status for order {order_id}: {e}")
            return "error"

    async def monitor_order(self, order_id: str):
        """Monitor the status of an open order and update its status."""
        try:
            max_attempts = self.config.max_status_checks
            check_interval = self.config.status_check_interval
            
            for _ in range(max_attempts):
                if order_id not in self.open_orders:
                    break
                    
                status = await self.get_order_status(order_id)
                
                if status in ['confirmed', 'cancelled', 'failed']:
                    self.logger.info(f"Order {order_id} final status: {status}")
                    break
                    
                await asyncio.sleep(check_interval)
            
            # If still pending after max attempts, mark as timeout
            if self.open_orders.get(order_id)["status"] == "pending":
                self.open_orders[order_id]["status"] = "timeout"
                self.logger.warning(f"Order {order_id} monitoring timed out")

        except Exception as e:
            self.logger.error(f"Error monitoring order {order_id}: {e}")
            self.open_orders[order_id]["status"] = "error"

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order
        Returns: True if cancellation was successful, False otherwise
        """
        try:
            if order_id not in self.open_orders:
                self.logger.warning(f"Order {order_id} not found for cancellation")
                return False

            current_status = self.open_orders[order_id]["status"]
            if current_status != "pending":
                self.logger.warning(f"Cannot cancel order {order_id}, status: {current_status}")
                return False

            # Attempt to cancel on exchange
            success = await self.exchange.cancel_order(order_id)
            
            if success:
                self.open_orders[order_id]["status"] = "cancelled"
                self.logger.info(f"Order {order_id} has been cancelled")
                return True
            
            return False

        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def update_position(self, token, amount, price, is_buy):
        """Update the bot's token holdings based on buy/sell orders."""
        try:
            if is_buy:
                # Calculate average entry price for buys
                current_position = self.positions.get(token, {"amount": 0, "entry_price": 0})
                total_value = (current_position["amount"] * current_position["entry_price"]) + (amount * price)
                new_amount = current_position["amount"] + amount
                new_entry_price = total_value / new_amount if new_amount > 0 else price
                
                self.positions[token] = {
                    "amount": new_amount,
                    "entry_price": new_entry_price
                }
                self.logger.info(f"Updated position: Bought {amount} {token}. Total: {new_amount} @ {new_entry_price}")
            else:
                # Decrease the position for sells
                if token in self.positions:
                    current_amount = self.positions[token]["amount"]
                    new_amount = max(0, current_amount - amount)
                    self.positions[token]["amount"] = new_amount
                    self.logger.info(f"Updated position: Sold {amount} {token}. Remaining: {new_amount}")
                    
                    # Remove position if fully sold
                    if new_amount == 0:
                        del self.positions[token]
        
        except Exception as e:
            self.logger.error(f"Error updating position for {token}: {e}")

    async def track_orders(self):
        """Continuously track the status of open orders and handle logic like stop-loss."""
        while self._is_running:
            try:
                pending_orders = [
                    order_id for order_id, order in self.open_orders.items()
                    if order["status"] == "pending"
                ]
                
                # Create tasks for checking each pending order
                tasks = [self.get_order_status(order_id) for order_id in pending_orders]
                if tasks:
                    await asyncio.gather(*tasks)

                # Optional: Check for stop-loss conditions
                for order_id in pending_orders:
                    if await self._should_trigger_stop_loss(order_id):
                        await self.cancel_order(order_id)

                await asyncio.sleep(self.config.order_tracking_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error tracking orders: {e}")
                await asyncio.sleep(5)  # Backoff on error

    async def _should_trigger_stop_loss(self, order_id: str) -> bool:
        """Check if stop-loss should be triggered for an order"""
        try:
            order = self.open_orders.get(order_id)
            if not order or not order.get("stop_loss_price"):
                return False

            # Get current price from exchange
            current_price = await self.exchange.get_current_price(
                order["input_token"],
                order["output_token"]
            )

            # Check if price has hit stop loss
            if order["type"] == "buy":
                return current_price <= order["stop_loss_price"]
            else:  # sell order
                return current_price >= order["stop_loss_price"]

        except Exception as e:
            self.logger.error(f"Error checking stop loss for order {order_id}: {e}")
            return False