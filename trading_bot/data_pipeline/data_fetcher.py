import asyncio
import aiohttp
import structlog
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property, partial
from typing import Any, Dict, List, Optional, Deque, Callable, Awaitable, Tuple
from aiolimiter import AsyncLimiter
from aiohttp import ClientTimeout, TCPConnector, ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_log,
    after_log
)
import orjson
from trading_bot.data_pipeline.data_storage import DataStorage
from trading_bot.config import config
import logging

logger = logging.getLogger(__name__)

class DataFetcherError(Exception):
    """Base exception class for DataFetcher errors"""
    pass

class ValidationError(DataFetcherError):
    """Raised when data validation fails"""
    pass

class ConnectionError(DataFetcherError):
    """Raised when connection issues occur"""
    pass

class CacheError(DataFetcherError):
    """Raised when cache operations fail"""
    pass

# Default configuration values
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5
DEFAULT_API_ENDPOINTS = {
    "jupiter": "https://quote-api.jup.ag/v4",
    "solana": str(config.SOLANA_RPC_URL)
}
DEFAULT_SOLANA_RPC_WS = str(config.SOLANA_RPC_URL).replace("https", "wss")

# Metrics (replace with actual metrics implementation)
class Metrics:
    @staticmethod
    def increment(name: str, tags: Dict[str, str] = None):
        pass

    @staticmethod
    def histogram(name: str, value: float, tags: Dict[str, str] = None):
        pass

@dataclass(slots=True)
class CacheEntry:
    data: Any
    expires_at: float
    access_count: int = 0

class CircuitBreaker:
    def __init__(self, max_failures: int = 5, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure = 0.0

    def is_open(self) -> bool:
        return (self.failure_count >= self.max_failures and 
                (time.monotonic() - self.last_failure) < self.reset_timeout)

    def record_failure(self):
        self.failure_count += 1
        self.last_failure = time.monotonic()

    def record_success(self):
        self.failure_count = max(0, self.failure_count - 1)

class DataFetcher:
    _POOL_SIZE = 100
    _TTL_MARGIN = 0.1  # 10% of TTL for cache expiration
    _MAX_RECONNECT_ATTEMPTS = 3
    _RECONNECT_DELAY = 5  # seconds
    
    def __init__(self, config, data_storage: DataStorage):
        self.config = config
        self.data_storage = data_storage
        self._cache: Dict[str, CacheEntry] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(
            lambda: CircuitBreaker()
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._limiter = AsyncLimiter(100, 1)  # 100 requests per second
        self.last_fetch_time = datetime.now()
        logger.info("Initializing data fetcher")

    async def __aenter__(self):
        try:
            await self.start()
            return self
        except Exception as e:
            logger.exception("Failed to enter context")
            raise DataFetcherError("Context entry failed") from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.stop()
        except Exception as e:
            logger.exception("Error during context exit")
            raise DataFetcherError("Context exit failed") from e

    @cached_property
    def connector(self) -> TCPConnector:
        try:
            return TCPConnector(
                limit=self._POOL_SIZE,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
        except Exception as e:
            logger.exception("Failed to create TCP connector")
            raise ConnectionError("Failed to create connector") from e

    async def start(self):
        try:
            if self._running:
                logger.warning("DataFetcher is already running")
                return

            self._session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=ClientTimeout(total=30),
                json_serialize=orjson.dumps
            )
            self._running = True
            self._tasks.append(asyncio.create_task(self._monitor_tasks()))
            logger.info("DataFetcher started successfully")
        except Exception as e:
            logger.exception("Failed to start DataFetcher")
            raise DataFetcherError("Start operation failed") from e

    async def stop(self):
        try:
            self._running = False
            await self._cancel_pending()
            
            if self._session and not self._session.closed:
                await self._session.close()
            
            if self._ws and not self._ws.closed:
                await self._ws.close()

            self._session = None
            self._ws = None
            logger.info("DataFetcher stopped successfully")
        except Exception as e:
            logger.exception("Error during DataFetcher shutdown")
            raise DataFetcherError("Stop operation failed") from e

    async def _monitor_tasks(self):
        while self._running:
            try:
                while self._tasks and self._tasks[0].done():
                    task = self._tasks.pop(0)
                    if task.exception():
                        logger.error("Task failed: %s", task.exception())
                await asyncio.sleep(1)
            except Exception as e:
                logger.error("Monitor error: %s", e)

    async def _cancel_pending(self):
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def fetch(self, endpoint: str, *, 
                   path: str = "",
                   params: Optional[Dict] = None,
                   validator: Optional[Callable] = None) -> Tuple[Any, bool]:
        """
        Fetch data from an endpoint with comprehensive error handling
        Returns: Tuple[data, success_flag]
        """
        if not self._session:
            raise ConnectionError("Session not initialized")

        circuit_breaker = self._circuit_breakers[endpoint]
        if circuit_breaker.is_open():
            logger.warning(f"Circuit breaker open for endpoint: {endpoint}")
            return None, False

        try:
            async with self._limiter:
                url = f"{endpoint}/{path}".rstrip('/')
                async with self._session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if validator and not validator(data):
                        raise ValidationError(f"Data validation failed for {url}")

                    circuit_breaker.record_success()
                    return data, True

        except ClientError as e:
            circuit_breaker.record_failure()
            logger.error(f"HTTP error for {endpoint}: {str(e)}", exc_info=True)
            return None, False
        except asyncio.TimeoutError:
            circuit_breaker.record_failure()
            logger.error(f"Timeout while fetching from {endpoint}", exc_info=True)
            return None, False
        except ValidationError as e:
            circuit_breaker.record_failure()
            logger.error(f"Validation error for {endpoint}: {str(e)}", exc_info=True)
            return None, False
        except Exception as e:
            circuit_breaker.record_failure()
            logger.exception(f"Unexpected error fetching from {endpoint}")
            return None, False

    async def fetch_market_data(self) -> Optional[Dict]:
        """Fetch current market data for configured trading pairs"""
        try:
            if not self._session or self._session.closed:
                await self.start()

            market_data = {}
            for pair in self.config.TRADING_PAIRS:
                try:
                    async with self._session.get(f"/quote/{pair}") as response:
                        if response.status == 200:
                            data = await response.json()
                            market_data[pair] = self._process_market_data(data)
                            logger.debug(f"Fetched market data for {pair}: {market_data[pair]}")
                        else:
                            logger.warning(f"Failed to fetch market data for {pair}: Status {response.status}")
                except Exception as e:
                    logger.error(f"Error fetching market data for {pair}: {str(e)}", exc_info=True)

            if market_data:
                await self.data_storage.store_market_data(market_data)
                logger.info(f"Successfully fetched and stored market data for {len(market_data)} pairs")
                return market_data
            else:
                logger.warning("No market data was fetched")
                return None

        except Exception as e:
            logger.error(f"Error in fetch_market_data: {str(e)}", exc_info=True)
            return None

    def _process_market_data(self, raw_data: Dict) -> Dict:
        """Process raw market data into standardized format"""
        try:
            processed_data = {
                'timestamp': datetime.now().isoformat(),
                'price': float(raw_data['price']),
                'volume': float(raw_data.get('volume', 0)),
                'bid': float(raw_data.get('bid', 0)),
                'ask': float(raw_data.get('ask', 0)),
                'high': float(raw_data.get('high', 0)),
                'low': float(raw_data.get('low', 0))
            }
            logger.debug(f"Processed market data: {processed_data}")
            return processed_data
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}", exc_info=True)
            return {}

    async def fetch_historical_data(self, pair: str, timeframe: str = '1h', limit: int = 1000) -> Optional[List[Dict]]:
        """Fetch historical market data"""
        try:
            if not self._session or self._session.closed:
                await self.start()

            logger.info(f"Fetching historical data for {pair} ({timeframe} timeframe, {limit} candles)")
            async with self._session.get(f"/history/{pair}", params={'timeframe': timeframe, 'limit': limit}) as response:
                if response.status == 200:
                    data = await response.json()
                    processed_data = [self._process_market_data(candle) for candle in data]
                    await self.data_storage.store_historical_data(pair, processed_data)
                    logger.info(f"Successfully fetched and stored {len(processed_data)} historical candles for {pair}")
                    return processed_data
                else:
                    logger.warning(f"Failed to fetch historical data for {pair}: Status {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching historical data for {pair}: {str(e)}", exc_info=True)
            return None

    async def fetch_orderbook(self, pair: str) -> Optional[Dict]:
        """Fetch current orderbook data"""
        try:
            if not self._session or self._session.closed:
                await self.start()

            logger.info(f"Fetching orderbook for {pair}")
            async with self._session.get(f"/orderbook/{pair}") as response:
                if response.status == 200:
                    data = await response.json()
                    processed_data = {
                        'timestamp': datetime.now().isoformat(),
                        'bids': data.get('bids', []),
                        'asks': data.get('asks', [])
                    }
                    await self.data_storage.store_orderbook(pair, processed_data)
                    logger.info(f"Successfully fetched and stored orderbook for {pair}")
                    return processed_data
                else:
                    logger.warning(f"Failed to fetch orderbook for {pair}: Status {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching orderbook for {pair}: {str(e)}", exc_info=True)
            return None

    def _get_cache(self, key: str) -> Optional[CacheEntry]:
        try:
            entry = self._cache.get(key)
            if not entry:
                return None

            if time.time() > entry.expires_at:
                del self._cache[key]
                return None

            entry.access_count += 1
            return entry

        except Exception as e:
            logger.error(f"Cache retrieval error for key {key}: {str(e)}", exc_info=True)
            raise CacheError(f"Failed to get cache entry: {str(e)}") from e

    def _set_cache(self, key: str, data: Any, ttl: int):
        try:
            expires_at = time.time() + (ttl * (1 - self._TTL_MARGIN))
            self._cache[key] = CacheEntry(data=data, expires_at=expires_at)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}", exc_info=True)
            raise CacheError(f"Failed to set cache entry: {str(e)}") from e

    @staticmethod
    def validate_market_data(data: Dict) -> bool:
        try:
            return isinstance(data, dict) and all(
                isinstance(v, (int, float)) for v in data.values()
            )
        except Exception as e:
            logger.error(f"Market data validation error: {str(e)}", exc_info=True)
            return False

    async def batch_process(self, processor: Callable[[Any], Awaitable[Any]], items: List[Any]):
        """Parallel batch processing with backpressure"""
        sem = asyncio.Semaphore(100)  # Max concurrent processes
        async def _process(item):
            async with sem:
                return await processor(item)
        return await asyncio.gather(*(_process(i) for i in items), return_exceptions=True)

    async def stream_solana_transactions(self):
        reconnect_attempts = 0
        while self._running and reconnect_attempts < self._MAX_RECONNECT_ATTEMPTS:
            try:
                if not self._session:
                    raise ConnectionError("Session not initialized")

                async with self._session.ws_connect(DEFAULT_SOLANA_RPC_WS) as ws:
                    self._ws = ws
                    logger.info("Connected to Solana WebSocket")
                    reconnect_attempts = 0  # Reset counter on successful connection

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = orjson.loads(msg.data)
                                await self.data_storage.store_transaction(data)
                            except Exception as e:
                                logger.error(f"Failed to process message: {str(e)}", exc_info=True)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

            except (ClientError, asyncio.TimeoutError) as e:
                reconnect_attempts += 1
                logger.error(
                    f"WebSocket connection error (attempt {reconnect_attempts}/{self._MAX_RECONNECT_ATTEMPTS}): {str(e)}",
                    exc_info=True
                )
                if reconnect_attempts < self._MAX_RECONNECT_ATTEMPTS:
                    await asyncio.sleep(self._RECONNECT_DELAY * reconnect_attempts)
            except Exception as e:
                logger.exception("Unexpected error in Solana transaction stream")
                break

        if reconnect_attempts >= self._MAX_RECONNECT_ATTEMPTS:
            logger.error("Max reconnection attempts reached for Solana WebSocket")