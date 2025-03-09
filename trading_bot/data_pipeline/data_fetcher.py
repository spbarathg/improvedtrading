import asyncio
import aiohttp
from aiolimiter import AsyncLimiter
from trading_bot.data_pipeline.data_storage import DataStorage
import logging
import time
import random

class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.data_storage = DataStorage(self.config)

        # Setup rate limiters (requests per second for different APIs)
        self.pump_fun_limiter = AsyncLimiter(max_rate=config.pump_fun_rate_limit, time_period=1)
        self.dexscreener_limiter = AsyncLimiter(max_rate=config.dexscreener_rate_limit, time_period=1)
        self.twitter_limiter = AsyncLimiter(max_rate=config.twitter_rate_limit, time_period=15 * 60)  # Per 15-minute window
        self.logger = logging.getLogger()

        # Create aiohttp session to reuse
        self.session = aiohttp.ClientSession()

    async def exponential_backoff(self, attempt):
        """Exponential backoff strategy."""
        await asyncio.sleep(min(2 ** attempt + random.uniform(0, 1), 60))  # Limit to 60 seconds

    async def fetch_pump_fun(self):
        """Fetch new coin launches from pump.fun."""
        url = "https://api.pump.fun/new-coins"
        attempt = 0
        while attempt < 5:  # Limit retry attempts
            async with self.pump_fun_limiter:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self.data_storage.store_raw_data('pump_fun', data)
                            return  # Exit on success
                        else:
                            self.logger.warning(f"Failed to fetch from pump.fun, status code: {response.status}")
                except Exception as e:
                    self.logger.error(f"Error fetching pump.fun data: {e}")
                attempt += 1
                await self.exponential_backoff(attempt)

    async def fetch_dexscreener(self):
        """Fetch price and market data from Dexscreener."""
        url = "https://api.dexscreener.com/latest-market-data"
        attempt = 0
        while attempt < 5:  # Limit retry attempts
            async with self.dexscreener_limiter:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self.data_storage.store_raw_data('dexscreener', data)
                            return  # Exit on success
                        else:
                            self.logger.warning(f"Failed to fetch from Dexscreener, status code: {response.status}")
                except Exception as e:
                    self.logger.error(f"Error fetching Dexscreener data: {e}")
                attempt += 1
                await self.exponential_backoff(attempt)

    async def fetch_twitter(self):
        """Fetch relevant tweets and profile updates from Twitter API."""
        url = f"https://api.twitter.com/2/tweets/search/recent?query={self.config.twitter_query}"
        headers = {"Authorization": f"Bearer {self.config.twitter_bearer_token}"}
        attempt = 0
        while attempt < 5:  # Limit retry attempts
            async with self.twitter_limiter:
                try:
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self.data_storage.store_raw_data('twitter', data)
                            return  # Exit on success
                        else:
                            self.logger.warning(f"Failed to fetch from Twitter, status code: {response.status}")
                except Exception as e:
                    self.logger.error(f"Error fetching Twitter data: {e}")
                attempt += 1
                await self.exponential_backoff(attempt)

    async def fetch_solana_transactions(self):
        """Fetch transaction confirmations via Solana RPC (websockets)."""
        url = self.config.solana_rpc_url
        try:
            async with self.session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        await self.data_storage.store_raw_data('solana', data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        self.logger.error(f"Solana websocket connection error: {msg.data}")
                        break
        except Exception as e:
            self.logger.error(f"Error fetching Solana transactions: {e}")
            await asyncio.sleep(5)  # Optional backoff

    async def start_fetching(self):
        """Main loop to continuously fetch data."""
        while True:
            try:
                await asyncio.gather(
                    self.fetch_pump_fun(),
                    self.fetch_dexscreener(),
                    self.fetch_twitter(),
                    self.fetch_solana_transactions()
                )
            except Exception as e:
                self.logger.error(f"Error in data fetching loop: {e}")
            finally:
                # Sleep to avoid overwhelming APIs, adjust time as needed
                await asyncio.sleep(self.config.fetch_interval)

    async def close_session(self):
        """Close the aiohttp session."""
        await self.session.close()