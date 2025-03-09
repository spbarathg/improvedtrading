import asyncio
import logging
import signal
import sys
from trading_bot.config import Config
from trading_bot.data_pipeline.data_fetcher import DataFetcher
from trading_bot.data_pipeline.feature_extractor import FeatureExtractor
from trading_bot.data_pipeline.data_storage import DataStorage
from trading_bot.trading.exchange import Exchange
from trading_bot.trading.order_manager import OrderManager
from trading_bot.trading.risk_manager import RiskManager
from ai_model.online_model import OnlineModel
from ai_model.periodic_model import PeriodicModel
from ai_model.model_selector import ModelSelector
from ai_model.trainer import Trainer

# Initialize logging
logging.basicConfig(
    filename="trading_bot/logs/trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        # Load configuration
        self.config = Config()

        # Initialize data pipeline components
        self.data_fetcher = DataFetcher(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.data_storage = DataStorage(self.config)

        # Initialize trading components
        self.exchange = Exchange(self.config)
        self.order_manager = OrderManager(self.exchange, self.config)
        self.risk_manager = RiskManager(self.config)

        # Initialize AI model components
        self.online_model = OnlineModel(self.config)
        self.periodic_model = PeriodicModel(self.config)
        self.model_selector = ModelSelector(self.online_model, self.periodic_model, self.config)
        self.trainer = Trainer(self.data_storage, self.feature_extractor, self.config)

        # Flag for graceful shutdown
        self.shutdown_flag = False

    async def start_fetching_data(self):
        """Start the data fetching loop."""
        while not self.shutdown_flag:
            try:
                await self.data_fetcher.start_fetching()
            except Exception as e:
                logger.error(f"Error in data fetching loop: {e}")
            await asyncio.sleep(self.config.DATA_FETCH_INTERVAL)

    async def start_feature_extraction(self):
        """Start the feature extraction loop."""
        while not self.shutdown_flag:
            try:
                await self.feature_extractor.start_extraction()
            except Exception as e:
                logger.error(f"Error in feature extraction loop: {e}")
            await asyncio.sleep(self.config.FEATURE_EXTRACTION_INTERVAL)

    async def start_model_training(self):
        """Start the AI model training loop."""
        while not self.shutdown_flag:
            try:
                await self.trainer.train_models()
            except Exception as e:
                logger.error(f"Error in model training loop: {e}")
            await asyncio.sleep(self.config.MODEL_TRAINING_INTERVAL)

    async def trading_loop(self):
        """Main trading loop."""
        while not self.shutdown_flag:
            try:
                # Get the latest features from DataStorage
                features = self.data_storage.get_latest_features()

                # Get predictions from the selected model
                predictions = await self.model_selector.predict(features)

                # Perform risk management checks
                if self.risk_manager.pre_trade_checks(predictions, features):
                    # Place orders if risk checks pass
                    await self.order_manager.place_orders(predictions, features)

                # Log the results
                logger.info(f"Predictions: {predictions}, Features: {features}")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

            # Control loop frequency
            await asyncio.sleep(self.config.TRADING_LOOP_INTERVAL)

    async def shutdown(self):
        """Gracefully shutdown the bot."""
        logger.info("Shutting down trading bot...")
        self.shutdown_flag = True

        # Close connections and clean up
        await self.exchange.close()
        await self.data_storage.close()

        logger.info("Trading bot shutdown complete.")

    def handle_shutdown_signal(self):
        """Handle shutdown signals (e.g., Ctrl+C)."""
        logger.info("Shutdown signal received.")
        asyncio.create_task(self.shutdown())

async def main():
    # Initialize the trading bot
    bot = TradingBot()

    # Register shutdown signal handler
    signal.signal(signal.SIGINT, lambda *args: bot.handle_shutdown_signal())

    # Start persistent tasks
    tasks = [
        asyncio.create_task(bot.start_fetching_data()),
        asyncio.create_task(bot.start_feature_extraction()),
        asyncio.create_task(bot.start_model_training()),
        asyncio.create_task(bot.trading_loop()),
    ]

    # Wait for all tasks to complete (or shutdown signal)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())