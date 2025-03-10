import asyncio
import aiohttp
import base64
import json
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
from typing import Dict, Optional, List, Set, DefaultDict, Tuple, Any
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import lru_cache
import weakref
from concurrent.futures import ThreadPoolExecutor
import time
import logging

logger = logging.getLogger(__name__)

class ExchangeError(Exception):
    """Base exception for exchange-related errors"""
    pass

class QuoteError(ExchangeError):
    """Error when fetching quotes"""
    pass

class SwapError(ExchangeError):
    """Error during swap execution"""
    pass

class BalanceError(ExchangeError):
    """Error when fetching or updating balances"""
    pass

class CircuitBreakerError(ExchangeError):
    """Error when circuit breaker is triggered"""
    pass

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
    _MAX_RETRIES = 3
    _RETRY_DELAY = 1  # seconds
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Exchange")
        
        self.config = config
        self.client = AsyncClient(config.SOLANA_RPC_URL)
        
        # Initialize rate limiters
        self.rpc_limiter = AsyncLimiter(
            config.RPC_RATE_LIMIT,
            time_period=1.0
        )
        self.jupiter_limiter = AsyncLimiter(
            config.JUPITER_RATE_LIMIT,
            time_period=1.0
        )
        
        # Initialize connection pool
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=self._MAX_CONNECTIONS,
                ttl_dns_cache=300
            )
        )
        
        # Initialize circuit breaker state
        self._circuit_breakers = {
            'rpc': {'failures': 0, 'last_failure': None},
            'jupiter': {'failures': 0, 'last_failure': None}
        }
        
        # Initialize metrics
        self.metrics = {
            'quotes_total': Counter('exchange_quotes_total', 'Total quote requests'),
            'swaps_total': Counter('exchange_swaps_total', 'Total swap executions'),
            'quote_latency': Histogram('exchange_quote_latency', 'Quote request latency'),
            'swap_latency': Histogram('exchange_swap_latency', 'Swap execution latency'),
            'balance_checks': Counter('exchange_balance_checks', 'Total balance checks'),
            'errors': Counter('exchange_errors_total', 'Total errors', ['type']),
            'circuit_breaks': Counter('exchange_circuit_breaks', 'Circuit breaker activations', ['service'])
        }
        
        # Initialize state
        self._running = False
        self._tasks = set()
        self._last_state_save = time.time()
        
        self.logger.info("Exchange initialized",
                      rpc_url=config.SOLANA_RPC_URL,
                      rpc_rate_limit=config.RPC_RATE_LIMIT,
                      jupiter_rate_limit=config.JUPITER_RATE_LIMIT,
                      max_connections=self._MAX_CONNECTIONS)
        
        try:
            self._signer = Keypair.from_secret_key(base64.b64decode(config.PRIVATE_KEY))
            self._pubkey = str(self._signer.public_key)
            self._executor = ThreadPoolExecutor(max_workers=4)
            self._cache: Dict[str, Tuple[Any, float]] = {}  # (value, expiry)
            self._last_errors: DefaultDict[str, List[float]] = defaultdict(list)
            
            # Metrics
            self.request_latency = Histogram(
                'exchange_request_latency_seconds',
                'Request latency in seconds',
                ['service', 'operation']
            )
            self.error_counter = Counter(
                'exchange_errors_total',
                'Total number of exchange errors',
                ['service', 'error_type']
            )
            self.circuit_breaker_status = Gauge(
                'exchange_circuit_breaker_status',
                'Circuit breaker status (0=closed, 1=open)',
                ['service']
            )
            
            logger.info("Exchange initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize Exchange")
            raise ExchangeError(f"Exchange initialization failed: {str(e)}") from e

    async def start(self):
        try:
            if self._running:
                logger.warning("Exchange is already running")
                return

            self._running = True
            self._tasks.extend([
                asyncio.create_task(self._save_state()),
                asyncio.create_task(self._cleanup())
            ])
            logger.info("Exchange started successfully")
        except Exception as e:
            logger.exception("Failed to start Exchange")
            raise ExchangeError("Exchange start failed") from e

    async def stop(self):
        try:
            self._running = False
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            if self.session and not self.session.closed:
                await self.session.close()
            
            await self.client.close()
            self._executor.shutdown(wait=True)
            logger.info("Exchange stopped successfully")
        except Exception as e:
            logger.exception("Error during Exchange shutdown")
            raise ExchangeError("Exchange stop failed") from e

    @lru_cache(maxsize=1000)
    def _get_cached_balance(self) -> float:
        try:
            return float(self._cache.get("balance", (0.0, 0))[0])
        except (TypeError, ValueError) as e:
            logger.error(f"Error retrieving cached balance: {str(e)}", exc_info=True)
            return 0.0

    async def _circuit_breaker_check(self, service: str) -> bool:
        try:
            now = time.time()
            self._last_errors[service] = [t for t in self._last_errors[service] if now - t < 60]
            
            if len(self._last_errors[service]) >= 5:  # 5 errors in 1 minute
                self._circuit_breakers[service]['failures'] = now + 300  # Open for 5 minutes
                self.circuit_breaker_status.labels(service=service).set(1)
                logger.warning(f"Circuit breaker opened for {service}")
                return True
            
            if self._circuit_breakers[service]['failures'] > now:
                return True
                
            self.circuit_breaker_status.labels(service=service).set(0)
            return False
        except Exception as e:
            logger.error(f"Error in circuit breaker check for {service}: {str(e)}", exc_info=True)
            return True  # Fail safe: assume circuit is open on error

    async def _request(self, method: str, url: str, service: str, **kwargs) -> Tuple[Any, bool]:
        """Make HTTP request with comprehensive error handling"""
        if not self.session:
            raise ExchangeError("Session not initialized")

        if await self._circuit_breaker_check(service):
            raise CircuitBreakerError(f"Circuit breaker is open for {service}")

        start_time = time.time()
        try:
            async with self.rpc_limiter:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    self.request_latency.labels(
                        service=service,
                        operation=method
                    ).observe(time.time() - start_time)
                    
                    return data, True

        except aiohttp.ClientError as e:
            self.error_counter.labels(service=service, error_type="http").inc()
            self._last_errors[service].append(time.time())
            logger.error(f"HTTP error for {service} at {url}: {str(e)}", exc_info=True)
            return None, False
            
        except asyncio.TimeoutError:
            self.error_counter.labels(service=service, error_type="timeout").inc()
            self._last_errors[service].append(time.time())
            logger.error(f"Timeout for {service} at {url}", exc_info=True)
            return None, False
            
        except Exception as e:
            self.error_counter.labels(service=service, error_type="unknown").inc()
            self._last_errors[service].append(time.time())
            logger.exception(f"Unexpected error for {service} at {url}")
            return None, False

    async def get_balance(self) -> float:
        """Get account balance with caching and error handling"""
        try:
            cache_key = f"balance_{self._pubkey}"
            now = time.time()
            
            # Check cache
            if cache_key in self._cache:
                value, expiry = self._cache[cache_key]
                if now < expiry:
                    return float(value)
            
            # Fetch fresh balance
            response = await self.client.get_balance(self._pubkey)
            if not response or "value" not in response["result"]:
                raise BalanceError("Invalid response from get_balance")
                
            balance = float(response["result"]["value"]) / 1e9  # Convert lamports to SOL
            self._cache[cache_key] = (balance, now + self._CACHE_TTL)
            return balance
            
        except Exception as e:
            logger.exception("Error fetching balance")
            self.error_counter.labels(service="solana", error_type="balance").inc()
            raise BalanceError(f"Failed to get balance: {str(e)}") from e

    async def get_quote(self, input_mint: str, output_mint: str, amount: int) -> Dict:
        """Get swap quote with error handling"""
        try:
            url = f"{self._JUPITER_API}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": self.config.SLIPPAGE_BPS
            }
            
            data, success = await self._request(
                "GET", url, "jupiter", params=params
            )
            
            if not success or not data:
                raise QuoteError("Failed to fetch quote")
                
            return data
            
        except CircuitBreakerError as e:
            logger.warning(str(e))
            raise QuoteError("Service temporarily unavailable") from e
        except Exception as e:
            logger.exception("Error getting quote")
            raise QuoteError(f"Quote fetch failed: {str(e)}") from e

    async def get_swap_tx(self, quote: dict) -> Transaction:
        """Get swap transaction with error handling"""
        try:
            url = f"{self._JUPITER_API}/swap"
            data, success = await self._request(
                "POST",
                url,
                "jupiter",
                json={
                    "quoteResponse": quote,
                    "userPublicKey": self._pubkey,
                    "wrapUnwrapSOL": True
                }
            )
            
            if not success or not data or "swapTransaction" not in data:
                raise SwapError("Invalid swap transaction response")
                
            return Transaction.deserialize(base64.b64decode(data["swapTransaction"]))
            
        except Exception as e:
            logger.exception("Error getting swap transaction")
            raise SwapError(f"Failed to get swap transaction: {str(e)}") from e

    async def execute_swap(self, input_mint: str, output_mint: str, amount: int) -> OrderStatus:
        """Execute swap with comprehensive error handling and monitoring"""
        start_time = time.time()
        order_id = f"swap_{int(start_time)}_{input_mint}_{output_mint}"
        
        try:
            # Get quote
            quote = await self.get_quote(input_mint, output_mint, amount)
            
            # Get transaction
            transaction = await self.get_swap_tx(quote)
            
            # Sign and send transaction
            transaction.sign(self._signer)
            opts = TxOpts(skip_preflight=True)
            
            response = await self.client.send_transaction(
                transaction,
                self._signer,
                opts=opts
            )
            
            if "result" not in response:
                raise SwapError("Invalid transaction response")
                
            signature = response["result"]
            
            # Monitor transaction
            status = await self.client.confirm_transaction(signature)
            if not status or not status.get("result", {}).get("value", False):
                raise SwapError("Transaction failed to confirm")
            
            # Record metrics
            self.request_latency.labels(
                service="swap",
                operation="complete"
            ).observe(time.time() - start_time)
            
            return OrderStatus(
                order_id=order_id,
                status="completed",
                created_at=datetime.fromtimestamp(start_time),
                updated_at=datetime.now(),
                amount=amount,
                price=float(quote["price"]),
                type="swap"
            )
            
        except (QuoteError, SwapError) as e:
            logger.error(f"Swap failed: {str(e)}", exc_info=True)
            self.error_counter.labels(service="swap", error_type="execution").inc()
            return OrderStatus(
                order_id=order_id,
                status="failed",
                created_at=datetime.fromtimestamp(start_time),
                updated_at=datetime.now(),
                amount=amount,
                price=0.0,
                type="swap",
                error=str(e)
            )
            
        except Exception as e:
            logger.exception("Unexpected error during swap execution")
            self.error_counter.labels(service="swap", error_type="unknown").inc()
            return OrderStatus(
                order_id=order_id,
                status="error",
                created_at=datetime.fromtimestamp(start_time),
                updated_at=datetime.now(),
                amount=amount,
                price=0.0,
                type="swap",
                error=f"Unexpected error: {str(e)}"
            )

    async def _save_state(self):
        """Periodic state saving with error handling"""
        while self._running:
            try:
                # Save critical state
                state = {
                    "circuit_breakers": dict(self._circuit_breakers),
                    "last_errors": {k: list(v) for k, v in self._last_errors.items()},
                    "cache": {k: list(v) for k, v in self._cache.items()}
                }
                
                await self.client.send_transaction(
                    Transaction().add(
                        memo_program.create_memo_instruction(
                            orjson.dumps(state).decode()
                        )
                    ),
                    self._signer
                )
                
                logger.debug("State saved successfully")
                await asyncio.sleep(self._STATE_SAVE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error saving state: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Retry after 1 minute on error

    async def _cleanup(self):
        """Periodic cleanup with error handling"""
        while self._running:
            try:
                now = time.time()
                
                # Clean up cache
                expired_keys = [
                    k for k, (_, expiry) in self._cache.items()
                    if now > expiry
                ]
                for k in expired_keys:
                    del self._cache[k]
                
                # Clean up circuit breakers
                expired_breakers = [
                    k for k, v in self._circuit_breakers.items()
                    if now > v
                ]
                for k in expired_breakers:
                    del self._circuit_breakers[k]
                    self.circuit_breaker_status.labels(service=k).set(0)
                
                # Clean up error history
                for service in list(self._last_errors.keys()):
                    self._last_errors[service] = [
                        t for t in self._last_errors[service]
                        if now - t < 3600  # Keep last hour
                    ]
                
                logger.debug("Cleanup completed successfully")
                await asyncio.sleep(self._HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Retry after 1 minute on error

    async def connect(self):
        """Establish connection to the exchange"""
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession(
                    base_url=str(self.config.SOLANA_RPC_URL),
                    timeout=aiohttp.ClientTimeout(total=30)
                )
                logger.info("Successfully established exchange connection")
            return True
        except Exception as e:
            logger.error(f"Failed to establish exchange connection: {str(e)}", exc_info=True)
            return False

    async def close(self):
        """Close exchange connection"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing exchange connection: {str(e)}", exc_info=True)

    async def check_connection(self) -> bool:
        """Check if the exchange connection is alive"""
        try:
            if not self.session or self.session.closed:
                logger.warning("Exchange connection not established or closed")
                return await self.connect()
            
            # Ping exchange
            async with self.session.get("/health") as response:
                if response.status == 200:
                    logger.debug("Exchange connection check successful")
                    return True
                else:
                    logger.warning(f"Exchange health check failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Exchange connection check failed: {str(e)}", exc_info=True)
            return False

    async def reconnect(self):
        """Attempt to reconnect to the exchange"""
        logger.info("Attempting to reconnect to exchange")
        self.connection_retries += 1
        
        if self.connection_retries > self.MAX_RETRIES:
            logger.error(f"Max reconnection attempts ({self.MAX_RETRIES}) exceeded")
            raise ConnectionError("Failed to reconnect to exchange after maximum retries")
            
        try:
            await self.close()
            success = await self.connect()
            if success:
                self.connection_retries = 0
                logger.info("Successfully reconnected to exchange")
                return True
            else:
                logger.warning(f"Reconnection attempt {self.connection_retries} failed")
                return False
        except Exception as e:
            logger.error(f"Error during reconnection attempt: {str(e)}", exc_info=True)
            return False

    async def measure_latency(self) -> float:
        """Measure exchange latency"""
        try:
            start_time = datetime.now()
            await self.check_connection()
            latency = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"Exchange latency: {latency:.2f}ms")
            return latency
        except Exception as e:
            logger.error(f"Error measuring exchange latency: {str(e)}", exc_info=True)
            return -1

    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            if not await self.check_connection():
                logger.error("Cannot get market price: Exchange connection failed")
                return None
                
            async with self.session.get(f"/v1/market/{symbol}/price") as response:
                if response.status == 200:
                    data = await response.json()
                    price = float(data['price'])
                    logger.debug(f"Retrieved market price for {symbol}: {price}")
                    return price
                else:
                    logger.warning(f"Failed to get market price for {symbol}: Status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {str(e)}", exc_info=True)
            return None

    async def place_order(self, order: Dict) -> Optional[Dict]:
        """Place an order on the exchange"""
        try:
            if not await self.check_connection():
                logger.error("Cannot place order: Exchange connection failed")
                return None
                
            logger.info(f"Placing order: {order}")
            async with self.session.post("/v1/order", json=order) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Order placed successfully: {result['order_id']}")
                    return result
                else:
                    logger.warning(f"Failed to place order: Status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}", exc_info=True)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            if not await self.check_connection():
                logger.error("Cannot cancel order: Exchange connection failed")
                return False
                
            logger.info(f"Cancelling order: {order_id}")
            async with self.session.delete(f"/v1/order/{order_id}") as response:
                success = response.status == 200
                if success:
                    logger.info(f"Order {order_id} cancelled successfully")
                else:
                    logger.warning(f"Failed to cancel order {order_id}: Status {response.status}")
                return success
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}", exc_info=True)
            return False

    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get the status of an order"""
        try:
            if not await self.check_connection():
                logger.error("Cannot get order status: Exchange connection failed")
                return None
                
            async with self.session.get(f"/v1/order/{order_id}") as response:
                if response.status == 200:
                    status = await response.json()
                    logger.debug(f"Retrieved status for order {order_id}: {status['status']}")
                    return status
                else:
                    logger.warning(f"Failed to get status for order {order_id}: Status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting status for order {order_id}: {str(e)}", exc_info=True)
            return None