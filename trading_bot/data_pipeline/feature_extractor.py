import asyncio
import numpy as np
from trading_bot.data_pipeline.data_storage import DataStorage
import logging
from typing import Dict, List, Optional
import re

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.data_storage = DataStorage(self.config)
        self.logger = logging.getLogger()

    async def analyze_token_name(self, token_name: str) -> Dict[str, float]:
        """Analyze token name for potential features."""
        features = {
            'name_length': len(token_name),
            'has_numbers': float(bool(re.search(r'\d', token_name))),
            'is_all_caps': float(token_name.isupper()),
            'special_chars_count': float(len(re.findall(r'[^a-zA-Z0-9]', token_name)))
        }
        return features

    async def calculate_liquidity_features(self, raw_data: List[Dict]) -> Optional[Dict[str, float]]:
        """Calculate liquidity-related features."""
        try:
            liquidity_data = [item['liquidity'] for item in raw_data if 'liquidity' in item]
            if not liquidity_data:
                return None
            
            liquidity_array = np.array(liquidity_data)
            return {
                'initial_liquidity': float(liquidity_array[0]),
                'current_liquidity': float(liquidity_array[-1]),
                'liquidity_change': float((liquidity_array[-1] - liquidity_array[0]) / liquidity_array[0] * 100),
                'liquidity_volatility': float(np.std(liquidity_array))
            }
        except Exception as e:
            self.logger.error(f"Error calculating liquidity features: {e}")
            return None

    async def calculate_price_features(self, raw_data: List[Dict]) -> Optional[Dict[str, float]]:
        """Calculate price-related features."""
        try:
            prices = np.array([item['price'] for item in raw_data if 'price' in item])
            if len(prices) < 2:
                return None

            returns = np.diff(prices) / prices[:-1]
            return {
                'price_change_pct': float((prices[-1] - prices[0]) / prices[0] * 100),
                'price_volatility': float(np.std(returns) * 100),
                'max_drawdown': float(self._calculate_max_drawdown(prices)),
                'price_momentum': float(self._calculate_momentum(prices))
            }
        except Exception as e:
            self.logger.error(f"Error calculating price features: {e}")
            return None

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate the maximum drawdown from peak."""
        peak = prices[0]
        max_drawdown = 0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def _calculate_momentum(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate price momentum using ROC (Rate of Change)."""
        if len(prices) < period:
            return 0
        return (prices[-1] - prices[-period]) / prices[-period] * 100

    async def calculate_volume_features(self, raw_data: List[Dict]) -> Optional[Dict[str, float]]:
        """Calculate volume-related features."""
        try:
            volumes = np.array([item['volume'] for item in raw_data if 'volume' in item])
            if len(volumes) < 5:
                return None

            return {
                'volume_mean': float(np.mean(volumes)),
                'volume_std': float(np.std(volumes)),
                'volume_trend': float(self._calculate_trend(volumes)),
                'volume_acceleration': float(np.mean(np.diff(volumes)))
            }
        except Exception as e:
            self.logger.error(f"Error calculating volume features: {e}")
            return None

    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate linear trend coefficient."""
        try:
            x = np.arange(len(data))
            z = np.polyfit(x, data, 1)
            return z[0]
        except:
            return 0.0

    async def calculate_features(self, raw_data: List[Dict]) -> Optional[Dict[str, float]]:
        """Calculate comprehensive features from raw data."""
        try:
            if not raw_data:
                return None

            # Gather all features asynchronously
            features = {}
            
            # Token name features
            if 'token_name' in raw_data[0]:
                token_features = await self.analyze_token_name(raw_data[0]['token_name'])
                features.update(token_features)

            # Calculate different feature categories concurrently
            feature_tasks = [
                self.calculate_liquidity_features(raw_data),
                self.calculate_price_features(raw_data),
                self.calculate_volume_features(raw_data)
            ]
            
            feature_results = await asyncio.gather(*feature_tasks)
            
            # Combine all features
            for result in feature_results:
                if result:
                    features.update(result)

            return features if features else None

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
                    features = await self.calculate_features(raw_data)

                    if features is not None:
                        # Store extracted features back into DataStorage
                        await self.data_storage.store_features(features)
                    else:
                        self.logger.warning("Feature calculation returned None")

                    # Control the loop frequency
                    await asyncio.sleep(self.config.feature_extraction_interval)
                else:
                    self.logger.warning("No raw data available, skipping feature extraction.")
                    await asyncio.sleep(5)  # Wait before retrying

            except Exception as e:
                self.logger.error(f"Error in feature extraction loop: {e}")
                await asyncio.sleep(5)  # Backoff on error

    async def start_extraction(self):
        """Start the feature extraction loop."""
        self.logger.info("Starting feature extraction process...")
        await self.extract_features()