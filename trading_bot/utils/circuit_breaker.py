import time
import logging
import asyncio
from enum import Enum
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from trading_bot.utils.decorators import error_handler
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Circuit is broken
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external API calls.
    Prevents cascading failures by stopping calls to failing services.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_timeout: int = 30,
        exception_types: tuple = (Exception,)
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.half_open_timeout = timedelta(seconds=half_open_timeout)
        self.exception_types = exception_types
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._half_open_start: Optional[datetime] = None
        
        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._consecutive_successes = 0
        self._last_error: Optional[Exception] = None
        
        # Locks
        self._state_lock = asyncio.Lock()
        
        logger.info(
            f"Circuit breaker '{name}' initialized with "
            f"failure threshold: {failure_threshold}, "
            f"recovery timeout: {recovery_timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def metrics(self) -> Dict:
        """Get current metrics"""
        return {
            'name': self.name,
            'state': self._state.value,
            'total_calls': self._total_calls,
            'successful_calls': self._successful_calls,
            'failed_calls': self._failed_calls,
            'failure_rate': (
                self._failed_calls / self._total_calls 
                if self._total_calls > 0 else 0
            ),
            'last_failure': (
                self._last_failure_time.isoformat() 
                if self._last_failure_time else None
            ),
            'last_success': (
                self._last_success_time.isoformat() 
                if self._last_success_time else None
            ),
            'last_error': str(self._last_error) if self._last_error else None
        }
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute the wrapped function with circuit breaker protection.
        Raises CircuitBreakerError if circuit is open.
        """
        self._total_calls += 1
        
        try:
            # Check circuit state
            await self._check_state()
            
            # Execute function
            start_time = datetime.now()
            result = await func(*args, **kwargs)
            
            # Update success metrics
            await self._handle_success()
            
            return result
            
        except self.exception_types as e:
            # Update failure metrics
            await self._handle_failure(e)
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' prevented call: {str(e)}"
            ) from e
            
        except Exception as e:
            # Log unexpected errors but don't count them as circuit breaker failures
            logger.error(
                f"Unexpected error in circuit breaker '{self.name}': {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise
    
    async def _check_state(self):
        """Check if circuit breaker allows calls"""
        async with self._state_lock:
            now = datetime.now()
            
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self._last_failure_time and now - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_start = now
                    logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
                    
            elif self._state == CircuitState.HALF_OPEN:
                # Check if half-open timeout has elapsed
                if now - self._half_open_start >= self.half_open_timeout:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' half-open timeout elapsed, "
                        "returning to OPEN state"
                    )
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' half-open timeout elapsed"
                    )
    
    async def _handle_success(self):
        """Handle successful call"""
        async with self._state_lock:
            self._successful_calls += 1
            self._last_success_time = datetime.now()
            self._consecutive_successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.failure_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._consecutive_successes = 0
                    logger.info(f"Circuit breaker '{self.name}' closed after recovery")
    
    async def _handle_failure(self, error: Exception):
        """Handle failed call"""
        async with self._state_lock:
            self._failed_calls += 1
            self._last_failure_time = datetime.now()
            self._last_error = error
            self._consecutive_successes = 0
            
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after "
                        f"{self._failure_count} consecutive failures"
                    )
            
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' returned to OPEN state "
                    "after failure in HALF_OPEN state"
                )

class CircuitBreakerError(Exception):
    """Raised when circuit breaker prevents a call"""
    pass 