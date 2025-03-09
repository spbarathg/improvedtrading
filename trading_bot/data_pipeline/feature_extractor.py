import asyncio
import numpy as np
from trading_bot.data_pipeline.data_storage import DataStorage
import logging

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.data_storage = DataStorage(self.config)
        self.logger = logging.getLogger()

    def calculate_features(self, raw_data):
        """Calculate features from raw data. Uses NumPy for efficiency."""
        try:
            # Example feature calculation (adjust according to your needs)
            prices = np.array([item['price'] for item in raw_data if 'price' in item])
            volumes = np.array([item['volume'] for item in raw_data if 'volume' in item])

            # Handle cases where data might be missing
            if len(prices) < 2 or len(volumes) < 5:
                self.logger.warning("Not enough data points for feature calculation.")
                return None

            # Calculate percentage change in price as a feature
            pct_price_change = np.diff(prices) / prices[:-1] * 100

            # Calculate rolling average of volumes (ensure lengths match)
            rolling_avg_volume = np.convolve(volumes, np.ones(5)/5, mode='valid')

            # Ensure that the feature sizes match for stacking
            min_length = min(len(pct_price_change), len(rolling_avg_volume))
            pct_price_change = pct_price_change[-min_length:]
            rolling_avg_volume = rolling_avg_volume[-min_length:]

            # Create feature matrix (ensure the feature sizes match)
            features = np.column_stack((pct_price_change, rolling_avg_volume))
            return features

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return None

    async def extract_features(self):
        """Continuously retrieve raw data and extract features."""
        while True:
            try:
                # Fetch raw data from DataStorage
                raw_data = await self.data_storage.get_raw_data()

                if raw_data:
                    # Calculate features from the raw data
                    features = self.calculate_features(raw_data)

                    if features is not None:
                        # Store extracted features back into DataStorage
                        await self.data_storage.store_features(features)

                    # Control the loop frequency (adjust the interval if necessary)
                    await asyncio.sleep(self.config.feature_extraction_interval)
                else:
                    self.logger.warning("No raw data available, skipping feature extraction.")

            except Exception as e:
                self.logger.error(f"Error in feature extraction loop: {e}")
                await asyncio.sleep(5)  # Optional backoff on error

    async def start_extraction(self):
        """Start the feature extraction loop."""
        await self.extract_features()