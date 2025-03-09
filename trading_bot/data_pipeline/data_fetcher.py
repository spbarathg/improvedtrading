import asyncio
import aiohttp
import logging
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property, partial
from typing import Any, Dict, List, Optional, Deque, Callable, Awaitable
from aiolimiter import AsyncLimiter
from aiohttp import ClientTimeout, TCPConnector
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

logger = logging.getLogger(__name__)

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
    
    def __init__(self, data_storage: DataStorage):
        self.data_storage = data_storage
        self.cache: Dict[str, Deque[CacheEntry]] = {}
        self.breakers: Dict[str, CircuitBreaker] = defaultdict(
            partial(CircuitBreaker, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY * 10)
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.limiter = AsyncLimiter(config.RPC_RATE_LIMIT + config.JUPITER_RATE_LIMIT)
        self._shutdown = asyncio.Event()
        self._pending: Deque[asyncio.Task] = deque(maxlen=1000)
        self.api_endpoints = DEFAULT_API_ENDPOINTS
        self.solana_rpc_ws = DEFAULT_SOLANA_RPC_WS

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    @cached_property
    def connector(self):
        return TCPConnector(
            limit=self._POOL_SIZE,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=False,
            use_dns_cache=True,
            ssl=False
        )

    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=ClientTimeout(total=30, connect=10),
            json_serialize=orjson.dumps,
            headers={'User-Agent': 'TradingBot/2.0'},
            auto_decompress=True
        )
        asyncio.create_task(self._monitor_tasks())

    async def stop(self):
        self._shutdown.set()
        await self._cancel_pending()
        if self.session:
            await self.session.close()
        self.connector.close()

    async def _monitor_tasks(self):
        while not self._shutdown.is_set():
            try:
                while self._pending and self._pending[0].done():
                    task = self._pending.popleft()
                    if task.exception():
                        logger.error("Task failed: %s", task.exception())
                await asyncio.sleep(1)
            except Exception as e:
                logger.error("Monitor error: %s", e)

    async def _cancel_pending(self):
        for task in self._pending:
            task.cancel()
        await asyncio.gather(*self._pending, return_exceptions=True)

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential_jitter(max=60),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    async def fetch(self, endpoint: str, *, 
                   path: str = "",
                   params: Optional[Dict] = None,
                   validator: Optional[Callable] = None) -> Any:
        """Generic fetch method with circuit breaking and validation"""
        if self.breakers[endpoint].is_open():
            raise RuntimeError(f"Circuit open for {endpoint}")

        url = f"{self.api_endpoints[endpoint]}/{path}"
        async with self.limiter:
            try:
                async with self.session.get(url, params=params) as resp:
                    data = await resp.json(loads=orjson.loads)
                    if validator and not validator(data):
                        raise ValueError("Validation failed")
                    self.breakers[endpoint].record_success()
                    Metrics.increment("fetch.success", {"endpoint": endpoint})
                    return data
            except Exception as e:
                self.breakers[endpoint].record_failure()
                Metrics.increment("fetch.error", {"endpoint": endpoint})
                logger.error("Fetch error: %s", e, exc_info=True)
                raise

    async def fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Optimized market data fetcher with cache stampede prevention"""
        cache_key = f"market:{symbol}"
        if entry := self._get_cache(cache_key):
            Metrics.increment("cache.hit", {"type": "market"})
            return entry.data

        Metrics.increment("cache.miss", {"type": "market"})
        data = await self.fetch("jupiter", path="quote", params={
            "inputMint": symbol,
            "outputMint": "SOL",
            "amount": 1000000000
        }, validator=self.validate_market_data)

        if data:
            self._set_cache(cache_key, data, config.CACHE_TTL)
            await self.data_storage.store("market_data", data)
        return data

    def _get_cache(self, key: str) -> Optional[CacheEntry]:
        """LFU cache implementation with TTL"""
        now = time.monotonic()
        if key not in self.cache:
            return None

        entries = sorted(
            [entry for entry in self.cache[key] if entry.expires_at > now],
            key=lambda e: (-e.access_count, e.expires_at)
        )
        if not entries:
            del self.cache[key]
            return None

        entries[0].access_count += 1
        return entries[0]

    def _set_cache(self, key: str, data: Any, ttl: int):
        """Set cache with probabilistic early expiration"""
        expires = time.monotonic() + ttl * (1 - self._TTL_MARGIN + random.uniform(0, 2 * self._TTL_MARGIN))
        entry = CacheEntry(data=data, expires_at=expires)
        self.cache.setdefault(key, deque(maxlen=config.CACHE_SIZE)).append(entry)

    @staticmethod
    def validate_market_data(data: Dict) -> bool:
        """Fast validation using set theory"""
        required = {"price", "amount", "route"}
        return required.issubset(data) and all(isinstance(data[k], (int, float)) for k in required)

    async def batch_process(self, processor: Callable[[Any], Awaitable[Any]], items: List[Any]):
        """Parallel batch processing with backpressure"""
        sem = asyncio.Semaphore(100)  # Max concurrent processes
        async def _process(item):
            async with sem:
                return await processor(item)
        return await asyncio.gather(*(_process(i) for i in items), return_exceptions=True)

    async def stream_solana_transactions(self):
        """WebSocket streamer with reconnect logic"""
        while not self._shutdown.is_set():
            try:
                async with self.session.ws_connect(self.solana_rpc_ws) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = orjson.loads(msg.data)
                            await self.data_storage.store("solana_tx", data)
                            Metrics.increment("ws.message")
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                await asyncio.sleep(random.uniform(1, 5))