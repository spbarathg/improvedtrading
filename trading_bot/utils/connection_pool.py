import asyncio
import aiohttp
import logging
from typing import Dict, Optional
from contextlib import asynccontextmanager
from aiohttp import ClientSession, ClientTimeout
from trading_bot.utils.decorators import error_handler, retry

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Manages connection pools for various services."""
    
    def __init__(self, config):
        self.config = config
        self._pools: Dict[str, ClientSession] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._timeouts = {
            'default': ClientTimeout(total=10),
            'long': ClientTimeout(total=30),
            'short': ClientTimeout(total=5)
        }
        
    @error_handler("connection_pool_init")
    async def initialize(self):
        """Initialize connection pools."""
        await self.get_session('default')
        await self.get_session('api')
        await self.get_session('websocket', timeout=self._timeouts['long'])
        
    @error_handler("get_session")
    @retry(max_retries=3)
    async def get_session(self, pool_name: str, timeout: Optional[ClientTimeout] = None) -> ClientSession:
        """Get or create a session from the specified pool."""
        if pool_name not in self._pools:
            if pool_name not in self._locks:
                self._locks[pool_name] = asyncio.Lock()
                
            async with self._locks[pool_name]:
                if pool_name not in self._pools:  # Double-check pattern
                    self._pools[pool_name] = ClientSession(
                        timeout=timeout or self._timeouts['default'],
                        headers={'User-Agent': 'TradingBot/1.0'},
                        raise_for_status=True
                    )
                    
        return self._pools[pool_name]
        
    @asynccontextmanager
    async def session(self, pool_name: str = 'default'):
        """Context manager for getting a session."""
        session = await self.get_session(pool_name)
        try:
            yield session
        except Exception as e:
            logger.error(f"Error in session {pool_name}: {str(e)}")
            raise
            
    async def close(self):
        """Close all connection pools."""
        for pool_name, session in self._pools.items():
            try:
                await session.close()
                logger.info(f"Closed connection pool: {pool_name}")
            except Exception as e:
                logger.error(f"Error closing pool {pool_name}: {str(e)}")
                
        self._pools.clear()
        self._locks.clear()
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close() 