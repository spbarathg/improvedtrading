import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    def __init__(self):
        # API Keys
        self.PUMP_FUN_API_KEY = os.getenv("PUMP_FUN_API_KEY")
        self.DEXSCREENER_API_KEY = os.getenv("DEXSCREENER_API_KEY")
        self.TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
        self.JUPITER_API_KEY = os.getenv("JUPITER_API_KEY")

        # Trading Parameters
        self.TRADING_AMOUNT = float(os.getenv("TRADING_AMOUNT", 100.0))  # Default: 100 USD
        self.MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 1000.0))  # Default: 1000 USD
        self.STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 5.0))  # Default: 5%
        self.TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 10.0))  # Default: 10%

        # Risk Management
        self.MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", 2.0))  # Default: 2%
        self.MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 5))  # Default: 5 trades

        # Model Hyperparameters
        self.ONLINE_MODEL_LEARNING_RATE = float(os.getenv("ONLINE_MODEL_LEARNING_RATE", 0.01))
        self.PERIODIC_MODEL_RETRAIN_INTERVAL = int(os.getenv("PERIODIC_MODEL_RETRAIN_INTERVAL", 3600))  # Default: 1 hour
        self.MODEL_SELECTION_STRATEGY = os.getenv("MODEL_SELECTION_STRATEGY", "weighted_average")  # Default: weighted_average

        # Update Intervals (in seconds)
        self.DATA_FETCH_INTERVAL = int(os.getenv("DATA_FETCH_INTERVAL", 60))  # Default: 1 minute
        self.FEATURE_EXTRACTION_INTERVAL = int(os.getenv("FEATURE_EXTRACTION_INTERVAL", 60))  # Default: 1 minute
        self.MODEL_TRAINING_INTERVAL = int(os.getenv("MODEL_TRAINING_INTERVAL", 3600))  # Default: 1 hour
        self.TRADING_LOOP_INTERVAL = int(os.getenv("TRADING_LOOP_INTERVAL", 10))  # Default: 10 seconds

        # Data Source URLs
        self.PUMP_FUN_API_URL = os.getenv("PUMP_FUN_API_URL", "https://api.pump.fun")
        self.DEXSCREENER_API_URL = os.getenv("DEXSCREENER_API_URL", "https://api.dexscreener.com")
        self.TWITTER_API_URL = os.getenv("TWITTER_API_URL", "https://api.twitter.com")
        self.JUPITER_API_URL = os.getenv("JUPITER_API_URL", "https://api.jup.ag")

        # File Paths
        self.DATA_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH", "trading_bot/data_pipeline/data_storage.db")
        self.LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "trading_bot/logs/trading_bot.log")
        self.MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "ai_model/saved_models/")

        # Other Configurations
        self.DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"