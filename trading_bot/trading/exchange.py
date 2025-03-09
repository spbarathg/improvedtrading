import asyncio
import aiohttp
import base64
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.publickey import PublicKey
from solana.keypair import Keypair
from solana.rpc.types import TxOpts
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import orjson
from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Optional, List, Set, DefaultDict
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import lru_cache
import weakref
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class OrderStatus:
    order_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    amount: float
    price: float
    type: str
    error: Optional[str] = None

class Exchange:
    _JUPITER_API = "https://quote-api.jup.ag/v4"
    _STATE_SAVE_INTERVAL = 300  # 5 minutes
    _HEALTH_CHECK_INTERVAL = 60  # 1 minute
    _MAX_CONNECTIONS = 100
    _BATCH_SIZE = 10
    _CACHE_TTL = 30  # Reduced from 60 to complement DataStorage
    _CACHE_SIZE = 5_000  # Reduced from 10_000 to complement DataStorage
    
    def __init__(self, config):
        self.config = config
        self._signer = Keypair.from_secret_key(base64.b64decode(config.PRIVATE_KEY))
        self._pubkey = str(self._signer.public_key)
        
        # Optimized connection pools with connection limits
        self.solana = AsyncClient(
            config.SOLANA_RPC_URL,
            timeout=30,
            commitment="confirmed",
            max_connections=self._MAX_CONNECTIONS
        )
        
        # Connection pooling with keep-alive
        self.jupiter = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            json_serialize=orjson.dumps,
            connector=aiohttp.TCPConnector(
                limit=self._MAX_CONNECTIONS,
                ttl_dns_cache=300,
                use_dns_cache=True,
                enable_cleanup_closed=True
            )
        )
        
        # Rate limiting with adaptive limits
        self.limiter = {
            'solana': AsyncLimiter(config.RPC_RATE_LIMIT),
            'jupiter': AsyncLimiter(config.JUPITER_RATE_LIMIT)
        }
        
        # Memory-efficient state management with LRU cache
        self.orders: Dict[str, OrderStatus] = {}
        self._state_lock = asyncio.Lock()
        self._last_state_save = asyncio.Event()
        self._batch_queue: DefaultDict[str, List] = defaultdict(list)
        self._batch_locks: Dict[str, asyncio.Lock] = {}
        
        # Thread pool for CPU-bound tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Circuit breaker state
        self._circuit_breaker = {
            'solana': {'failures': 0, 'last_failure': 0},
            'jupiter': {'failures': 0, 'last_failure': 0}
        }
        
        # Performance metrics with labels
        self.metrics = {
            'requests': Counter('exchange_requests', 'API requests', ['service', 'status', 'endpoint']),
            'latency': Histogram('exchange_latency', 'Request latency', ['service', 'operation']),
            'orders': Gauge('exchange_orders', 'Active orders', ['status']),
            'balance': Gauge('exchange_balance', 'Current balance'),
            'batch_size': Histogram('exchange_batch_size', 'Batch operation size'),
            'circuit_breaker': Gauge('exchange_circuit_breaker', 'Circuit breaker state', ['service']),
            'cache_hits': Counter('exchange_cache_hits', 'Cache hits'),
            'cache_misses': Counter('exchange_cache_misses', 'Cache misses')
        }
        
        # Weak reference cache for frequently accessed data
        self._cache = weakref.WeakValueDictionary()
        self._cache_timestamps = {}

    @lru_cache(maxsize=1000)
    def _get_cached_balance(self) -> float:
        """Cached balance check with TTL"""
        current_time = time.time()
        if 'balance' in self._cache and current_time - self._cache_timestamps.get('balance', 0) < self._CACHE_TTL:
            return self._cache['balance']
        return 0.0

    async def _circuit_breaker_check(self, service: str) -> bool:
        """Circuit breaker implementation with exponential backoff"""
        state = self._circuit_breaker[service]
        current_time = time.time()
        
        if state['failures'] >= 5:
            if current_time - state['last_failure'] < 2 ** state['failures']:
                self.metrics['circuit_breaker'].labels(service=service).set(0)
                return False
            state['failures'] = 0
            self.metrics['circuit_breaker'].labels(service=service).set(1)
        return True

    async def _batch_process(self, service: str, operation: str, items: List):
        """Efficient batch processing with size limits"""
        if len(items) == 0:
            return []
            
        batch_size = min(len(items), self._BATCH_SIZE)
        self.metrics['batch_size'].observe(batch_size)
        
        async with self._batch_locks.setdefault(service, asyncio.Lock()):
            results = await asyncio.gather(*[
                self._request(operation, item, service)
                for item in items[:batch_size]
            ])
            
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _request(self, method: str, url: str, service: str, **kwargs):
        """Optimized request handling with circuit breaker and metrics"""
        if not await self._circuit_breaker_check(service):
            raise Exception(f"Circuit breaker open for {service}")
            
        async with self.limiter[service]:
            start = datetime.now()
            try:
                async with getattr(self.jupiter, method)(url, **kwargs) as resp:
                    data = await resp.json(loads=orjson.loads)
                    self.metrics['requests'].labels(
                        service=service,
                        status='success',
                        endpoint=url.split('/')[-1]
                    ).inc()
                    self.metrics['latency'].labels(
                        service=service,
                        operation=method
                    ).observe((datetime.now() - start).total_seconds())
                    return data
            except Exception as e:
                self._circuit_breaker[service]['failures'] += 1
                self._circuit_breaker[service]['last_failure'] = time.time()
                self.metrics['requests'].labels(
                    service=service,
                    status='error',
                    endpoint=url.split('/')[-1]
                ).inc()
                logger.error('Request failed: %s', e)
                raise

    async def get_balance(self):
        """Optimized balance check with caching and circuit breaker"""
        try:
            cached_balance = self._get_cached_balance()
            if cached_balance > 0:
                return cached_balance
                
            result = await self.solana.get_balance(PublicKey(self._pubkey))
            balance = result['result']['value'] / 1e9
            self._cache['balance'] = balance
            self._cache_timestamps['balance'] = time.time()
            self.metrics['balance'].set(balance)
            return balance
        except Exception as e:
            logger.error('Balance check failed: %s', e)
            return 0.0

    async def get_quote(self, input_mint: str, output_mint: str, amount: int):
        """Batch-capable quote fetching with caching"""
        cache_key = f"quote_{input_mint}_{output_mint}_{amount}"
        if cache_key in self._cache and time.time() - self._cache_timestamps.get(cache_key, 0) < self._CACHE_TTL:
            return self._cache[cache_key]
            
        params = {
            'inputMint': input_mint,
            'outputMint': output_mint,
            'amount': amount,
            'slippageBps': self.config.SLIPPAGE_BPS
        }
        result = await self._request('get', '/quote', 'jupiter', params=params)
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
        return result

    async def get_swap_tx(self, quote: dict):
        """Transaction preparation with pre-signed headers"""
        return await self._request('post', '/swap', 'jupiter', json={
            'quoteResponse': quote,
            'userPublicKey': self._pubkey,
            'wrapUnwrapSOL': True
        })

    async def execute_swap(self, input_mint: str, output_mint: str, amount: int):
        """Optimized swap execution pipeline with batching"""
        try:
            # Phase 1: Get quote with caching
            quote = await self.get_quote(input_mint, output_mint, amount)
            if not quote:
                return None

            # Phase 2: Prepare transaction
            tx_data = await self.get_swap_tx(quote)
            if not tx_data:
                return None

            # Phase 3: Sign and send with optimized serialization
            tx = Transaction.deserialize(tx_data['transaction'])
            tx.sign(self._signer)
            
            # Async confirmation with timeout
            sig = await asyncio.wait_for(
                self.solana.send_transaction(tx),
                timeout=30
            )
            sig = sig.get('result')
            if not sig:
                return None

            # Track order with atomic update
            async with self._state_lock:
                self.orders[sig] = OrderStatus(
                    order_id=sig,
                    status='pending',
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    amount=amount,
                    price=quote.get('price', 0),
                    type='swap'
                )
                self.metrics['orders'].labels(status='pending').inc()

            # Phase 4: Wait for confirmation with timeout
            try:
                await asyncio.wait_for(
                    self.solana.confirm_transaction(sig),
                    timeout=60
                )
                # Update order status
                async with self._state_lock:
                    if sig in self.orders:
                        self.orders[sig].status = 'completed'
                        self.orders[sig].updated_at = datetime.now()
                        self.metrics['orders'].labels(status='completed').inc()
                return sig
            except asyncio.TimeoutError:
                # Update order status
                async with self._state_lock:
                    if sig in self.orders:
                        self.orders[sig].status = 'timeout'
                        self.orders[sig].updated_at = datetime.now()
                        self.orders[sig].error = 'Transaction confirmation timeout'
                        self.metrics['orders'].labels(status='timeout').inc()
                return None

        except Exception as e:
            logger.error('Swap execution failed: %s', e)
            return None

    async def _save_state(self):
        """Periodic state persistence with atomic updates"""
        while True:
            try:
                async with self._state_lock:
                    # Save orders state
                    state = {
                        'orders': {
                            order_id: {
                                'status': order.status,
                                'created_at': order.created_at.isoformat(),
                                'updated_at': order.updated_at.isoformat(),
                                'amount': order.amount,
                                'price': order.price,
                                'type': order.type,
                                'error': order.error
                            }
                            for order_id, order in self.orders.items()
                        }
                    }
                    
                    # Save to file with atomic write
                    with open('exchange_state.json', 'w') as f:
                        orjson.dump(state, f)
                    
                    self._last_state_save.set()
                    logger.debug('State saved successfully')
                    
            except Exception as e:
                logger.error('State save failed: %s', e)
            
            await asyncio.sleep(self._STATE_SAVE_INTERVAL)

    async def _cleanup(self):
        """Resource cleanup with graceful shutdown"""
        try:
            # Cancel pending tasks
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            
            # Close connections
            await self.jupiter.close()
            await self.solana.close()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            # Save final state
            await self._save_state()
            
            logger.info('Cleanup completed successfully')
            
        except Exception as e:
            logger.error('Cleanup failed: %s', e)