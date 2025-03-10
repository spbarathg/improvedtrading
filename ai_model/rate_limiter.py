import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for AI model predictions with adaptive throttling"""
    
    def __init__(self, config):
        self.config = config
        
        # Base rate limits
        self.max_requests_per_second = config.MAX_PREDICTIONS_PER_SECOND
        self.max_concurrent_predictions = config.MAX_CONCURRENT_PREDICTIONS
        self.max_burst_size = config.MAX_BURST_SIZE
        
        # Tracking windows
        self.short_window = timedelta(seconds=1)
        self.medium_window = timedelta(seconds=10)
        self.long_window = timedelta(minutes=1)
        
        # Request tracking
        self.request_times = {
            'short': deque(maxlen=self.max_requests_per_second),
            'medium': deque(maxlen=self.max_requests_per_second * 10),
            'long': deque(maxlen=self.max_requests_per_second * 60)
        }
        
        # Concurrent requests tracking
        self._concurrent_semaphore = asyncio.Semaphore(self.max_concurrent_predictions)
        self._active_requests = 0
        
        # Adaptive rate limiting
        self.error_window = deque(maxlen=100)
        self.latency_window = deque(maxlen=100)
        self._backoff_until: Optional[datetime] = None
        self._current_backoff = 1.0
        
        # Resource monitoring
        self._last_resource_check = datetime.now()
        self._resource_check_interval = timedelta(seconds=5)
        
        logger.info(
            f"Rate limiter initialized with {self.max_requests_per_second} req/s, "
            f"{self.max_concurrent_predictions} concurrent, "
            f"{self.max_burst_size} burst size"
        )
    
    async def acquire(self) -> bool:
        """
        Attempt to acquire permission to make a prediction.
        Returns True if allowed, False if rate limited.
        """
        try:
            # Check backoff period
            if self._backoff_until and datetime.now() < self._backoff_until:
                logger.warning("Rate limited due to backoff period")
                return False
            
            # Check resource availability
            if not await self._check_resources():
                logger.warning("Rate limited due to resource constraints")
                return False
            
            # Check rate limits
            if not self._check_rate_limits():
                logger.warning("Rate limited due to request frequency")
                return False
            
            # Acquire concurrent semaphore
            if not await self._try_acquire_semaphore():
                logger.warning("Rate limited due to concurrent requests")
                return False
            
            # Update request tracking
            now = datetime.now()
            for window in self.request_times.values():
                window.append(now)
            
            self._active_requests += 1
            return True
            
        except Exception as e:
            logger.error(f"Error in rate limiter acquire: {e}")
            return False
    
    async def release(self, success: bool, latency_ms: float):
        """Release the rate limiter and update metrics"""
        try:
            self._active_requests -= 1
            self._concurrent_semaphore.release()
            
            # Update metrics
            self.latency_window.append(latency_ms)
            self.error_window.append(0 if success else 1)
            
            # Adjust backoff if needed
            await self._adjust_backoff()
            
        except Exception as e:
            logger.error(f"Error in rate limiter release: {e}")
    
    def _check_rate_limits(self) -> bool:
        """Check if current request rates are within limits"""
        now = datetime.now()
        
        # Check each window
        windows = {
            'short': (self.short_window, self.max_requests_per_second),
            'medium': (self.medium_window, self.max_requests_per_second * 10),
            'long': (self.long_window, self.max_requests_per_second * 60)
        }
        
        for window_name, (duration, limit) in windows.items():
            # Clean old requests
            cutoff = now - duration
            while self.request_times[window_name] and self.request_times[window_name][0] < cutoff:
                self.request_times[window_name].popleft()
            
            # Check if within limit
            if len(self.request_times[window_name]) >= limit:
                return False
        
        return True
    
    async def _check_resources(self) -> bool:
        """Check system resource availability"""
        now = datetime.now()
        if now - self._last_resource_check < self._resource_check_interval:
            return True
        
        try:
            # Add resource checks here (CPU, memory, etc.)
            # For now, just using a simple concurrent request check
            if self._active_requests >= self.max_concurrent_predictions:
                return False
            
            self._last_resource_check = now
            return True
            
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return False
    
    async def _try_acquire_semaphore(self) -> bool:
        """Try to acquire the concurrent semaphore with timeout"""
        try:
            return await asyncio.wait_for(
                self._concurrent_semaphore.acquire(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            return False
    
    async def _adjust_backoff(self):
        """Adjust backoff period based on error rate and latency"""
        if not self.error_window or not self.latency_window:
            return
        
        error_rate = sum(self.error_window) / len(self.error_window)
        avg_latency = sum(self.latency_window) / len(self.latency_window)
        
        # Increase backoff if error rate or latency is too high
        if error_rate > 0.1 or avg_latency > 1000:  # 10% errors or 1s latency
            self._current_backoff = min(self._current_backoff * 2, 60)  # Max 60s backoff
            self._backoff_until = datetime.now() + timedelta(seconds=self._current_backoff)
            logger.warning(
                f"Increasing backoff to {self._current_backoff}s due to "
                f"error rate: {error_rate:.2%}, latency: {avg_latency:.0f}ms"
            )
        else:
            # Gradually reduce backoff
            self._current_backoff = max(self._current_backoff * 0.5, 1.0)
            self._backoff_until = None 