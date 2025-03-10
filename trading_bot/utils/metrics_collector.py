import time
import logging
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime, timedelta
import asyncio
from prometheus_client import Counter, Gauge, Histogram
from trading_bot.utils.decorators import error_handler

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Advanced metrics collector with real-time monitoring and alerting capabilities.
    """
    
    def __init__(self, config):
        self.config = config
        self.window_size = getattr(config, 'METRICS_WINDOW_SIZE', 1000)
        self.alert_threshold = getattr(config, 'METRICS_ALERT_THRESHOLD', 0.95)
        
        # Performance metrics
        self.latency = Histogram(
            'api_latency_seconds',
            'API call latency in seconds',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
        self.requests = Counter(
            'api_requests_total',
            'Total API requests made',
            ['endpoint', 'method', 'status']
        )
        self.errors = Counter(
            'api_errors_total',
            'Total API errors encountered',
            ['endpoint', 'error_type']
        )
        
        # Trading metrics
        self.trade_volume = Gauge(
            'trade_volume_total',
            'Total trading volume in base currency'
        )
        self.position_size = Gauge(
            'position_size',
            'Current position size',
            ['symbol']
        )
        self.profit_loss = Gauge(
            'profit_loss',
            'Current profit/loss',
            ['symbol']
        )
        
        # Model metrics
        self.prediction_accuracy = Gauge(
            'model_prediction_accuracy',
            'Model prediction accuracy'
        )
        self.model_latency = Histogram(
            'model_prediction_latency',
            'Model prediction latency in seconds',
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0)
        )
        
        # Rolling windows for real-time analysis
        self._latencies = deque(maxlen=self.window_size)
        self._errors = deque(maxlen=self.window_size)
        self._trades = deque(maxlen=self.window_size)
        
        # Alert state
        self._alert_sent = {}
        self._alert_cooldown = timedelta(minutes=5)
        
    @error_handler("record_latency")
    async def record_latency(self, endpoint: str, latency: float):
        """Record API call latency with automatic alerting."""
        self.latency.observe(latency)
        self._latencies.append((datetime.now(), endpoint, latency))
        
        # Check for latency spikes
        await self._check_latency_alert(endpoint, latency)
        
    @error_handler("record_request")
    def record_request(self, endpoint: str, method: str, status: int):
        """Record API request with status."""
        self.requests.labels(endpoint=endpoint, method=method, status=status).inc()
        
    @error_handler("record_error")
    async def record_error(self, endpoint: str, error_type: str, error_msg: str):
        """Record API error with context."""
        self.errors.labels(endpoint=endpoint, error_type=error_type).inc()
        self._errors.append((datetime.now(), endpoint, error_type, error_msg))
        
        # Check for error patterns
        await self._check_error_alert(endpoint, error_type)
        
    @error_handler("record_trade")
    async def record_trade(self, symbol: str, size: float, price: float, side: str):
        """Record trade execution with volume tracking."""
        self.trade_volume.inc(size * price)
        self.position_size.labels(symbol=symbol).set(size)
        
        trade_info = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'size': size,
            'price': price,
            'side': side
        }
        self._trades.append(trade_info)
        
        # Update profit/loss
        await self._update_pnl(symbol)
        
    @error_handler("record_model_prediction")
    def record_model_prediction(self, accuracy: float, latency: float):
        """Record model prediction metrics."""
        self.prediction_accuracy.set(accuracy)
        self.model_latency.observe(latency)
        
    async def _check_latency_alert(self, endpoint: str, latency: float):
        """Check for concerning latency patterns."""
        if len(self._latencies) < 10:
            return
            
        recent_latencies = [l for _, e, l in self._latencies[-10:] if e == endpoint]
        avg_latency = sum(recent_latencies) / len(recent_latencies)
        
        if avg_latency > self.alert_threshold:
            await self._send_alert(
                f"High latency detected for {endpoint}",
                f"Average latency: {avg_latency:.2f}s"
            )
            
    async def _check_error_alert(self, endpoint: str, error_type: str):
        """Check for concerning error patterns."""
        if len(self._errors) < 10:
            return
            
        recent_errors = [
            (e, t) for _, e, t, _ in self._errors[-10:]
            if e == endpoint
        ]
        error_count = len(recent_errors)
        
        if error_count >= 3:  # Three or more errors in recent window
            await self._send_alert(
                f"High error rate for {endpoint}",
                f"Recent errors: {error_count}, Type: {error_type}"
            )
            
    async def _update_pnl(self, symbol: str):
        """Calculate and update profit/loss metrics."""
        if not self._trades:
            return
            
        symbol_trades = [
            t for t in self._trades
            if t['symbol'] == symbol
        ]
        
        if symbol_trades:
            # Simple P&L calculation - can be enhanced based on requirements
            pnl = sum(
                t['size'] * t['price'] * (1 if t['side'] == 'sell' else -1)
                for t in symbol_trades
            )
            self.profit_loss.labels(symbol=symbol).set(pnl)
            
    async def _send_alert(self, title: str, message: str):
        """Send alert with cooldown period."""
        alert_key = f"{title}:{message}"
        now = datetime.now()
        
        if alert_key in self._alert_sent:
            last_sent = self._alert_sent[alert_key]
            if now - last_sent < self._alert_cooldown:
                return
                
        self._alert_sent[alert_key] = now
        logger.warning(f"ALERT - {title}: {message}")
        
        # Here you could integrate with external alerting systems
        # For example, sending to Slack, email, etc.
        
    async def get_summary(self) -> Dict:
        """Get summary of current metrics."""
        return {
            'latency': {
                'avg': sum((l for _, _, l in self._latencies), 0.0) / len(self._latencies) if self._latencies else 0,
                'max': max((l for _, _, l in self._latencies), default=0),
                'count': len(self._latencies)
            },
            'errors': {
                'count': len(self._errors),
                'types': set((t for _, _, t, _ in self._errors))
            },
            'trades': {
                'count': len(self._trades),
                'volume': float(self.trade_volume._value.get())
            }
        }
        
    async def cleanup_old_data(self):
        """Cleanup old metric data periodically."""
        while True:
            try:
                now = datetime.now()
                cutoff = now - timedelta(days=1)
                
                # Clean up old data while preserving recent entries
                self._latencies = deque(
                    ((ts, e, l) for ts, e, l in self._latencies if ts > cutoff),
                    maxlen=self.window_size
                )
                
                self._errors = deque(
                    ((ts, e, t, m) for ts, e, t, m in self._errors if ts > cutoff),
                    maxlen=self.window_size
                )
                
                self._trades = deque(
                    (t for t in self._trades if t['timestamp'] > cutoff),
                    maxlen=self.window_size
                )
                
                # Clean up old alerts
                self._alert_sent = {
                    k: v for k, v in self._alert_sent.items()
                    if now - v < self._alert_cooldown
                }
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {str(e)}")
                
            await asyncio.sleep(3600)  # Run cleanup hourly 