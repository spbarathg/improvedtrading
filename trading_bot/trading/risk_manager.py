import asyncio
import numpy as np
import orjson
from datetime import datetime, timedelta
from cachetools import TTLCache
from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, List, Optional, Tuple, Deque, DefaultDict, Any
from collections import defaultdict, deque
from numba import jit, njit, prange
import mmap
from concurrent.futures import ThreadPoolExecutor
import threading
from array import array
import struct
from typing import NamedTuple, TypeVar, Generic
import os
import time
import structlog
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class RiskManagerError(Exception):
    """Base exception for risk manager errors"""
    pass

class ValidationError(RiskManagerError):
    """Error when validating input data"""
    pass

class CalculationError(RiskManagerError):
    """Error during risk calculations"""
    pass

class PositionError(RiskManagerError):
    """Error when managing positions"""
    pass

class MetricsError(RiskManagerError):
    """Error when handling metrics"""
    pass

# Type hints for better performance
T = TypeVar('T')

class Position(NamedTuple):
    size: float
    entry: float
    current_price: float
    max_price: float
    last_update: float

class AssetHistory(NamedTuple):
    returns: np.ndarray
    index: int
    cumulative: float
    max_cumulative: float

@njit(nogil=True, parallel=True, fastmath=True)
def _vectorized_risk_check(
    prices: np.ndarray,
    sizes: np.ndarray,
    capitals: np.ndarray,
    entries: np.ndarray,
    max_exposure: float,
    stop_loss: float,
    order_type: str
) -> np.ndarray:
    """JIT-compiled vectorized risk assessment with SIMD optimization"""
    try:
        n = len(prices)
        results = np.ones(n, dtype=np.bool_)
        max_sizes = capitals * max_exposure
        
        for i in prange(n):
            # Position size check
            if sizes[i] > max_sizes[i]:
                results[i] = False
                continue
                
            # Stop loss check
            if order_type == "long":
                if prices[i] < entries[i] * (1 - stop_loss):
                    results[i] = False
            else:  # short
                if prices[i] > entries[i] * (1 + stop_loss):
                    results[i] = False
                    
        return results
    except Exception as e:
        # Numba doesn't support exception handling well, so we return a conservative result
        return np.zeros(len(prices), dtype=np.bool_)  # Fail safe: reject all trades

@njit(nogil=True, fastmath=True)
def _update_returns_vectorized(
    returns_buffer: np.ndarray,
    new_returns: np.ndarray,
    current_index: int,
    window_size: int
) -> Tuple[int, float, float]:
    """JIT-compiled returns update with incremental statistics"""
    count = len(new_returns)
    sum_ret = 0.0
    sum_sq = 0.0
    
    # Calculate new sum and sum of squares
    for i in prange(count):
        sum_ret += new_returns[i]
        sum_sq += new_returns[i] ** 2
    
    # Handle circular buffer overwrite
    if current_index + count > window_size:
        overflow = current_index + count - window_size
        for i in prange(overflow):
            old_val = returns_buffer[current_index + i]
            sum_ret -= old_val
            sum_sq -= old_val ** 2
        current_index = 0
    
    # Update buffer and index
    end_index = current_index + count
    returns_buffer[current_index:end_index] = new_returns
    new_index = end_index % window_size
    
    return new_index, sum_ret, sum_sq

class ShardedPositions:
    """Lock-striped positions for concurrent access"""
    _NUM_SHARDS = 16  # Align with typical CPU cores
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.logger.debug("Initializing sharded positions", num_shards=self._NUM_SHARDS)
        self._shards = [
            (threading.Lock(), {})
            for _ in range(self._NUM_SHARDS)
        ]
    
    def _get_shard(self, asset: str) -> int:
        return hash(asset) % self._NUM_SHARDS
    
    def get(self, asset: str) -> Optional[Position]:
        shard_id = self._get_shard(asset)
        lock, positions = self._shards[shard_id]
        with lock:
            position = positions.get(asset)
            if position:
                self.logger.debug("Retrieved position",
                              asset=asset,
                              shard=shard_id,
                              position=position._asdict())
            return position
    
    def update(self, asset: str, position: Position) -> bool:
        shard_id = self._get_shard(asset)
        lock, positions = self._shards[shard_id]
        with lock:
            positions[asset] = position
            self.logger.debug("Updated position",
                          asset=asset,
                          shard=shard_id,
                          position=position._asdict())
            return True

class RiskManager:
    _CACHE_SIZE = 10_000
    _CACHE_TTL = 300
    _VOLATILITY_WINDOW = 1000
    _BATCH_SIZE = 4096  # Optimize for cache lines
    _ALIGNMENT = 64
    _METRIC_UPDATE_INTERVAL = 1.0  # Seconds
    _MAX_RETRIES = 3
    _RETRY_DELAY = 1  # seconds
    
    def __init__(self, config):
        self.config = config
        self.daily_loss = 0.0
        self.open_trades = 0
        self.trade_history = deque(maxlen=1000)
        self.last_reset = datetime.now()
        logger.info("Initializing risk manager")
        
        self._positions = ShardedPositions()
        self._returns_buffer = np.zeros(self._VOLATILITY_WINDOW, dtype=np.float64)
        self._current_index = 0
        self._cumulative_return = 0.0
        self._max_cumulative_return = 0.0
        
        # Initialize metrics
        self.metrics = {
            'total_exposure': Gauge('risk_total_exposure', 'Total position exposure'),
            'max_drawdown': Gauge('risk_max_drawdown', 'Maximum drawdown'),
            'volatility': Gauge('risk_volatility', 'Portfolio volatility'),
            'sharpe_ratio': Gauge('risk_sharpe_ratio', 'Sharpe ratio'),
            'position_count': Gauge('risk_position_count', 'Number of open positions'),
            'risk_checks': Counter('risk_checks_total', 'Total number of risk checks'),
            'failed_checks': Counter('risk_checks_failed', 'Number of failed risk checks'),
            'check_latency': Histogram('risk_check_latency', 'Risk check latency')
        }
        
        # Initialize caches
        self._risk_cache = TTLCache(
            maxsize=self._CACHE_SIZE,
            ttl=self._CACHE_TTL
        )
        
        # Initialize thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=os.cpu_count(),
            thread_name_prefix="RiskManager"
        )
        
        self.logger.info("RiskManager initialized",
                      cache_size=self._CACHE_SIZE,
                      cache_ttl=self._CACHE_TTL,
                      volatility_window=self._VOLATILITY_WINDOW,
                      batch_size=self._BATCH_SIZE)

    async def start(self):
        try:
            if self._running:
                logger.warning("RiskManager is already running")
                return

            self._running = True
            self._tasks.append(asyncio.create_task(self._update_metrics_background()))
            logger.info("RiskManager started successfully")
        except Exception as e:
            logger.exception("Failed to start RiskManager")
            raise RiskManagerError("Start failed") from e

    async def stop(self):
        try:
            self._running = False
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._executor.shutdown(wait=True)
            logger.info("RiskManager stopped successfully")
        except Exception as e:
            logger.exception("Error during RiskManager shutdown")
            raise RiskManagerError("Stop failed") from e

    async def check_risk(self, order_type: str, assets: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Check risk parameters with comprehensive error handling"""
        start_time = time.time()
        error_messages = []
        
        try:
            # Input validation
            if not assets:
                raise ValidationError("Empty assets list")
            if order_type not in ("long", "short"):
                raise ValidationError(f"Invalid order type: {order_type}")
                
            # Prepare arrays for vectorized operations
            n = len(assets)
            prices = np.zeros(n, dtype=np.float32)
            sizes = np.zeros(n, dtype=np.float32)
            capitals = np.zeros(n, dtype=np.float32)
            entries = np.zeros(n, dtype=np.float32)
            
            # Extract and validate data
            for i, asset in enumerate(assets):
                try:
                    prices[i] = float(asset.get('price', 0))
                    sizes[i] = float(asset.get('size', 0))
                    capitals[i] = float(asset.get('capital', 0))
                    entries[i] = float(asset.get('entry', 0))
                    
                    if any(v <= 0 for v in (prices[i], sizes[i], capitals[i], entries[i])):
                        error_messages.append(f"Invalid values for asset {asset.get('symbol', 'unknown')}")
                except (TypeError, ValueError) as e:
                    error_messages.append(f"Data conversion error for asset {asset.get('symbol', 'unknown')}: {str(e)}")
            
            if error_messages:
                raise ValidationError("Multiple validation errors occurred")
            
            # Perform risk check
            results = _vectorized_risk_check(
                prices,
                sizes,
                capitals,
                entries,
                self.config.MAX_EXPOSURE,
                self.config.STOP_LOSS,
                order_type
            )
            
            # Record metrics
            self.risk_check_latency.observe(time.time() - start_time)
            violations = np.sum(~results)
            if violations > 0:
                self.risk_violations.labels(type=order_type).inc(violations)
            
            return results, error_messages
            
        except ValidationError as e:
            logger.error(f"Validation error in risk check: {str(e)}", exc_info=True)
            return np.zeros(len(assets), dtype=np.bool_), [str(e)] + error_messages
        except Exception as e:
            logger.exception("Unexpected error in risk check")
            return np.zeros(len(assets), dtype=np.bool_), [f"Unexpected error: {str(e)}"] + error_messages

    async def update_positions(self, updates: List[Dict]) -> Tuple[bool, List[str]]:
        """Update positions with error handling"""
        error_messages = []
        try:
            if not updates:
                return True, []
                
            # Process updates in thread pool to avoid blocking
            processed = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._process_updates,
                updates
            )
            
            success = True
            for asset, position in processed.items():
                try:
                    if not self._positions.update(asset, position):
                        success = False
                        error_messages.append(f"Failed to update position for {asset}")
                except Exception as e:
                    success = False
                    error_messages.append(f"Error updating {asset}: {str(e)}")
            
            self.position_updates.inc(len(updates))
            return success, error_messages
            
        except Exception as e:
            logger.exception("Error in position updates")
            return False, [f"Position update failed: {str(e)}"] + error_messages

    def _process_updates(self, updates: List[Dict]) -> Dict[str, Position]:
        """Process position updates with error handling"""
        try:
            processed = {}
            current_time = time.time()
            
            for update in updates:
                try:
                    asset = update.get('symbol')
                    if not asset:
                        logger.error("Missing symbol in update")
                        continue
                        
                    # Get current position
                    current = self._positions.get(asset)
                    
                    # Calculate new position
                    new_size = float(update.get('size', 0))
                    new_price = float(update.get('price', 0))
                    
                    if current:
                        max_price = max(current.max_price, new_price)
                        entry = (current.entry * current.size + new_price * new_size) / (current.size + new_size)
                    else:
                        max_price = new_price
                        entry = new_price
                    
                    processed[asset] = Position(
                        size=new_size,
                        entry=entry,
                        current_price=new_price,
                        max_price=max_price,
                        last_update=current_time
                    )
                    
                except (TypeError, ValueError) as e:
                    logger.error(f"Data conversion error in update for {update.get('symbol', 'unknown')}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing update for {update.get('symbol', 'unknown')}: {str(e)}")
                    
            return processed
            
        except Exception as e:
            logger.exception("Error in update processing")
            return {}

    async def _update_metrics_background(self):
        """Background metrics update with error handling"""
        while self._running:
            try:
                metrics = self._collect_metrics()
                self._send_metrics(metrics)
                await asyncio.sleep(self._METRIC_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in metrics update: {str(e)}", exc_info=True)
                await asyncio.sleep(self._RETRY_DELAY)

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current metrics with error handling"""
        try:
            metrics = {}
            
            # Calculate total exposure
            total_exposure = 0.0
            total_pnl = 0.0
            
            for shard in self._positions._shards:
                for asset, pos in shard.items():
                    try:
                        exposure = pos.size * pos.current_price
                        pnl = (pos.current_price - pos.entry) * pos.size
                        
                        metrics[f"exposure_{asset}"] = exposure
                        metrics[f"pnl_{asset}"] = pnl
                        
                        total_exposure += exposure
                        total_pnl += pnl
                    except Exception as e:
                        logger.error(f"Error calculating metrics for {asset}: {str(e)}")
            
            metrics["total_exposure"] = total_exposure
            metrics["total_pnl"] = total_pnl
            metrics["volatility"] = self.volatility
            
            return metrics
            
        except Exception as e:
            logger.exception("Error collecting metrics")
            return {}

    def _send_metrics(self, metrics: Dict[str, float]):
        """Send metrics with error handling"""
        try:
            for name, value in metrics.items():
                if name.startswith("exposure_"):
                    asset = name.split("_")[1]
                    self.current_exposure.labels(asset=asset).set(value)
                elif name == "total_exposure":
                    self.current_exposure.labels(asset="total").set(value)
        except Exception as e:
            logger.error(f"Error sending metrics: {str(e)}", exc_info=True)

    @property
    def volatility(self) -> float:
        """Calculate volatility with error handling"""
        try:
            with self._buffer_lock:
                if self._current_index < 2:
                    return 0.0
                    
                # Calculate rolling standard deviation
                active_window = self._returns_buffer[:self._current_index]
                return float(np.std(active_window, ddof=1))
                
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}", exc_info=True)
            return 0.0  # Return safe default

    def validate_trade(self, signal: str, market_data: Dict) -> bool:
        """Validate if a trade meets risk management criteria"""
        try:
            # Reset daily metrics if needed
            self._reset_daily_metrics()
            
            # Check if trading is allowed
            if not self._check_trading_allowed():
                return False
                
            # Validate position size
            if not self._validate_position_size(market_data['size']):
                return False
                
            # Check daily loss limit
            if not self._check_daily_loss_limit():
                return False
                
            # Check maximum open trades
            if not self._check_max_open_trades():
                return False
                
            logger.info(f"Trade validation successful for signal: {signal}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}", exc_info=True)
            return False

    def _reset_daily_metrics(self):
        """Reset daily metrics at the start of each trading day"""
        try:
            current_time = datetime.now()
            if (current_time - self.last_reset).days >= 1:
                logger.info("Resetting daily risk metrics")
                self.daily_loss = 0.0
                self.last_reset = current_time
                logger.debug("Daily metrics reset successful")
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {str(e)}", exc_info=True)

    def _check_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        if not self.config.TRADING_ENABLED:
            logger.warning("Trading is currently disabled in configuration")
            return False
        return True

    def _validate_position_size(self, size: float) -> bool:
        """Validate if the position size is within allowed limits"""
        try:
            if size <= 0:
                logger.warning(f"Invalid position size: {size}")
                return False
                
            if size > self.config.MAX_POSITION_SIZE:
                logger.warning(f"Position size {size} exceeds maximum allowed {self.config.MAX_POSITION_SIZE}")
                return False
                
            logger.debug(f"Position size validation successful: {size}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating position size: {str(e)}", exc_info=True)
            return False

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        try:
            max_daily_loss = self.config.MAX_DAILY_LOSS_PCT
            if self.daily_loss >= max_daily_loss:
                logger.warning(f"Daily loss limit reached: {self.daily_loss:.2f}% >= {max_daily_loss}%")
                return False
                
            logger.debug(f"Daily loss check passed: Current loss {self.daily_loss:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {str(e)}", exc_info=True)
            return False

    def _check_max_open_trades(self) -> bool:
        """Check if maximum number of open trades has been reached"""
        try:
            if self.open_trades >= self.config.MAX_OPEN_TRADES:
                logger.warning(f"Maximum open trades limit reached: {self.open_trades} >= {self.config.MAX_OPEN_TRADES}")
                return False
                
            logger.debug(f"Open trades check passed: Current open trades {self.open_trades}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking max open trades: {str(e)}", exc_info=True)
            return False

    def update_trade_metrics(self, trade_result: Dict):
        """Update risk metrics based on trade result"""
        try:
            # Update daily loss/profit
            pnl_pct = trade_result.get('pnl_percentage', 0.0)
            self.daily_loss += abs(pnl_pct) if pnl_pct < 0 else 0
            
            # Update open trades count
            if trade_result['status'] == 'closed':
                self.open_trades = max(0, self.open_trades - 1)
            elif trade_result['status'] == 'open':
                self.open_trades += 1
                
            # Record trade in history
            self.trade_history.append({
                'timestamp': datetime.now(),
                'trade_id': trade_result['trade_id'],
                'pnl_percentage': pnl_pct,
                'status': trade_result['status']
            })
            
            logger.info(f"Updated trade metrics - Daily Loss: {self.daily_loss:.2f}%, Open Trades: {self.open_trades}")
            
        except Exception as e:
            logger.error(f"Error updating trade metrics: {str(e)}", exc_info=True)

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        try:
            metrics = {
                'daily_loss': self.daily_loss,
                'open_trades': self.open_trades,
                'trade_history_count': len(self.trade_history),
                'last_reset': self.last_reset.isoformat()
            }
            logger.debug(f"Retrieved risk metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}", exc_info=True)
            return {}

    def __init__(self, config):
        """Initialize risk manager with configuration."""
        self.config = config
        self.position_limits = config.POSITION_LIMITS
        self.risk_limits = config.RISK_LIMITS
        self.active_positions = {}
        logger.info("Risk manager initialized with position limits: %s", self.position_limits)

    async def check_trade(self, trade_info: Dict) -> bool:
        """
        Check if a proposed trade meets risk management criteria.
        
        Args:
            trade_info: Dictionary containing trade details
            
        Returns:
            bool: Whether the trade is allowed
        """
        try:
            # Check position limits
            symbol = trade_info['symbol']
            size = Decimal(str(trade_info['size']))
            current_position = self.active_positions.get(symbol, Decimal('0'))
            new_position = current_position + size
            
            if abs(new_position) > self.position_limits.get(symbol, Decimal('0')):
                logger.warning(
                    "Trade rejected: Position limit exceeded",
                    extra={
                        'symbol': symbol,
                        'current_position': str(current_position),
                        'trade_size': str(size),
                        'new_position': str(new_position),
                        'limit': str(self.position_limits.get(symbol))
                    }
                )
                return False
            
            # Check risk limits
            risk_score = self._calculate_risk_score(trade_info)
            if risk_score > self.risk_limits['max_risk_score']:
                logger.warning(
                    "Trade rejected: Risk score too high",
                    extra={
                        'risk_score': risk_score,
                        'max_allowed': self.risk_limits['max_risk_score'],
                        'trade_info': trade_info
                    }
                )
                return False
            
            logger.info(
                "Trade approved",
                extra={
                    'symbol': symbol,
                    'size': str(size),
                    'risk_score': risk_score
                }
            )
            return True
            
        except Exception as e:
            logger.error(
                "Error checking trade",
                exc_info=True,
                extra={
                    'error': str(e),
                    'trade_info': trade_info
                }
            )
            return False

    def _calculate_risk_score(self, trade_info: Dict) -> float:
        """
        Calculate risk score for a trade.
        
        Args:
            trade_info: Dictionary containing trade details
            
        Returns:
            float: Risk score between 0 and 1
        """
        try:
            # Extract trade parameters
            size = Decimal(str(trade_info['size']))
            price = Decimal(str(trade_info['price']))
            volatility = trade_info.get('volatility', 0)
            
            # Calculate component scores
            size_score = float(size / self.risk_limits['max_position_size'])
            value_score = float((size * price) / self.risk_limits['max_position_value'])
            volatility_score = float(volatility / self.risk_limits['max_volatility'])
            
            # Combine scores with weights
            risk_score = (
                size_score * self.risk_limits['size_weight'] +
                value_score * self.risk_limits['value_weight'] +
                volatility_score * self.risk_limits['volatility_weight']
            )
            
            logger.debug(
                "Calculated risk score",
                extra={
                    'risk_score': risk_score,
                    'size_score': size_score,
                    'value_score': value_score,
                    'volatility_score': volatility_score
                }
            )
            
            return risk_score
            
        except Exception as e:
            logger.error(
                "Error calculating risk score",
                exc_info=True,
                extra={
                    'error': str(e),
                    'trade_info': trade_info
                }
            )
            return float('inf')

    def update_position(self, trade_info: Dict):
        """
        Update tracked position after trade execution.
        
        Args:
            trade_info: Dictionary containing trade details
        """
        try:
            symbol = trade_info['symbol']
            size = Decimal(str(trade_info['size']))
            
            # Update position
            current = self.active_positions.get(symbol, Decimal('0'))
            new_position = current + size
            self.active_positions[symbol] = new_position
            
            logger.info(
                "Position updated",
                extra={
                    'symbol': symbol,
                    'previous_position': str(current),
                    'trade_size': str(size),
                    'new_position': str(new_position)
                }
            )
            
        except Exception as e:
            logger.error(
                "Error updating position",
                exc_info=True,
                extra={
                    'error': str(e),
                    'trade_info': trade_info
                }
            )

    def get_position(self, symbol: str) -> Decimal:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Decimal: Current position size
        """
        try:
            position = self.active_positions.get(symbol, Decimal('0'))
            logger.debug(
                "Retrieved position",
                extra={
                    'symbol': symbol,
                    'position': str(position)
                }
            )
            return position
            
        except Exception as e:
            logger.error(
                "Error getting position",
                exc_info=True,
                extra={
                    'error': str(e),
                    'symbol': symbol
                }
            )
            return Decimal('0')

    async def check_risk_limits(self) -> bool:
        """
        Check if any risk limits are breached.
        
        Returns:
            bool: Whether all positions are within limits
        """
        try:
            for symbol, position in self.active_positions.items():
                if abs(position) > self.position_limits.get(symbol, Decimal('0')):
                    logger.warning(
                        "Risk limit breached",
                        extra={
                            'symbol': symbol,
                            'position': str(position),
                            'limit': str(self.position_limits.get(symbol))
                        }
                    )
                    return False
            
            logger.debug("All positions within risk limits")
            return True
            
        except Exception as e:
            logger.error(
                "Error checking risk limits",
                exc_info=True,
                extra={'error': str(e)}
            )
            return False