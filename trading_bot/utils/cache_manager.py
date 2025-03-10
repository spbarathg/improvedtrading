import time
import logging
from typing import Any, Dict, Optional, TypeVar, Generic
from cachetools import TTLCache, LRUCache
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
from trading_bot.utils.decorators import error_handler

logger = logging.getLogger(__name__)
T = TypeVar('T')

class CacheManager(Generic[T]):
    """
    Advanced cache manager with TTL, memory limits, and async support.
    Implements a two-level caching strategy with memory-sensitive eviction.
    """
    
    def __init__(self, 
                 ttl: int = 3600,
                 max_size: int = 1000,
                 memory_limit_mb: int = 100):
        self.ttl_cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.lru_cache = LRUCache(maxsize=max_size)
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._locks: Dict[str, asyncio.Lock] = {}
        
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache with async support."""
        try:
            # Try TTL cache first
            if key in self.ttl_cache:
                return self.ttl_cache[key]
            
            # Try LRU cache next
            if key in self.lru_cache:
                value = self.lru_cache[key]
                # Promote to TTL cache if within memory limits
                if self._check_memory_usage():
                    self.ttl_cache[key] = value
                return value
                
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
            
    @error_handler("cache_set")
    async def set(self, key: str, value: T) -> bool:
        """Set value in cache with memory management."""
        try:
            # Ensure we have a lock for this key
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
                
            async with self._locks[key]:
                # Check memory limits
                if self._check_memory_usage():
                    self.ttl_cache[key] = value
                else:
                    # Fallback to LRU cache if memory limit reached
                    self.lru_cache[key] = value
                    
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache value: {str(e)}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Remove value from both caches."""
        try:
            self.ttl_cache.pop(key, None)
            self.lru_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False
            
    async def clear(self) -> bool:
        """Clear all caches."""
        try:
            self.ttl_cache.clear()
            self.lru_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
            
    def _check_memory_usage(self) -> bool:
        """Check if current memory usage is within limits."""
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            return memory_usage < self.memory_limit
        except ImportError:
            logger.warning("psutil not available, skipping memory check")
            return True
            
    async def get_or_set(self, key: str, value_func: callable) -> Optional[T]:
        """Get value from cache or compute and store it."""
        value = await self.get(key)
        if value is None:
            try:
                if asyncio.iscoroutinefunction(value_func):
                    value = await value_func()
                else:
                    value = await asyncio.to_thread(value_func)
                    
                if value is not None:
                    await self.set(key, value)
                    
            except Exception as e:
                logger.error(f"Error in get_or_set: {str(e)}")
                
        return value
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=False) 