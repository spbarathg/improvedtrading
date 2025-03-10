import functools
import logging
import asyncio
from typing import Callable, Any, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar('T')

def error_handler(operation_name: str):
    """Decorator for standardized error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def rate_limit(calls: int, period: float):
    """Decorator for rate limiting function calls."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        calls_made = []
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            now = asyncio.get_event_loop().time()
            calls_made[:] = [t for t in calls_made if now - t < period]
            
            if len(calls_made) >= calls:
                wait_time = calls_made[0] + period - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            calls_made.append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def retry(max_retries: int = 3, backoff_base: float = 2.0):
    """Decorator for retrying failed operations with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_base ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator 