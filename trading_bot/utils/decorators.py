import logging
import functools
import time
from typing import Any, Callable, TypeVar, cast
from datetime import datetime

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

def log_execution_time(func: F) -> F:
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.4f} seconds",
                extra={
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'success'
                }
            )
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.4f} seconds: {str(e)}",
                extra={
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'error',
                    'error': str(e)
                },
                exc_info=True
            )
            raise
    return cast(F, wrapper)

def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """Decorator to retry failed function calls with exponential backoff"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {wait_time:.2f} seconds",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_attempts': max_attempts,
                            'wait_time': wait_time,
                            'error': str(e)
                        }
                    )
                    time.sleep(wait_time)
            logger.error(
                f"All {max_attempts} attempts failed for {func.__name__}",
                extra={
                    'function': func.__name__,
                    'max_attempts': max_attempts,
                    'final_error': str(last_exception)
                },
                exc_info=last_exception
            )
            raise last_exception
        return cast(F, wrapper)
    return decorator

def validate_args(*types: Any) -> Callable[[F], F]:
    """Decorator to validate function argument types"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for arg, expected_type in zip(args[1:], types):
                if not isinstance(arg, expected_type):
                    error_msg = f"Argument {arg} is not of type {expected_type}"
                    logger.error(
                        f"Type validation failed for {func.__name__}: {error_msg}",
                        extra={
                            'function': func.__name__,
                            'argument': str(arg),
                            'expected_type': str(expected_type),
                            'actual_type': type(arg).__name__
                        }
                    )
                    raise TypeError(error_msg)
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def log_exceptions(func: F) -> F:
    """Decorator to log exceptions with detailed context"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Exception in {func.__name__}: {str(e)}",
                extra={
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                },
                exc_info=True
            )
            raise
    return cast(F, wrapper)

def deprecated(func: F) -> F:
    """Decorator to mark functions as deprecated"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.warning(
            f"Function {func.__name__} is deprecated and will be removed in a future version",
            extra={
                'function': func.__name__,
                'file': func.__code__.co_filename,
                'line': func.__code__.co_firstlineno
            }
        )
        return func(*args, **kwargs)
    return cast(F, wrapper)

def error_handler(operation_name: str):
    """Decorator for standardized error handling."""
    def decorator(func: Callable[..., F]) -> Callable[..., F]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> F:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> F:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def rate_limit(calls: int, period: float):
    """Decorator for rate limiting function calls."""
    def decorator(func: Callable[..., F]) -> Callable[..., F]:
        calls_made = []
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> F:
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