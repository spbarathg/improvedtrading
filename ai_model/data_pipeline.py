import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import asyncio
from collections import deque
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from ai_model.configs import config as ai_config
from ai_model.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Handles data flow between trading bot and AI model components.
    Implements efficient data processing, caching, and validation.
    """

    def __init__(self, config=ai_config):
        self.config = config
        self.preprocessor = DataPreprocessor(
            scaling_method=config.SCALING_METHOD,
            normalization=config.NORMALIZATION,
            cache_size=config.FEATURE_CACHE_SIZE
        )
        
        # Implement adaptive buffer sizing
        self.min_buffer_size = 100
        self.max_buffer_size = config.FEATURE_CACHE_SIZE
        self.current_buffer_size = self.min_buffer_size
        self.resize_threshold = 0.8  # Resize when buffer is 80% full
        
        # Use dynamic arrays for better memory efficiency
        self.market_data_buffer = self._create_market_data_buffer(self.current_buffer_size)
        self.feature_buffer = np.zeros((self.current_buffer_size, len(config.FEATURE_CONFIG.feature_names)), dtype=np.float32)
        self.buffer_index = 0
        
        # Implement adaptive LRU cache
        self.indicator_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = self._calculate_optimal_cache_size()
        
        # Pre-allocate arrays with adaptive sizing
        self.initial_history_size = 50
        self.price_history = np.zeros(self.initial_history_size, dtype=np.float32)
        self.volume_history = np.zeros(self.initial_history_size, dtype=np.float32)
        self.history_index = 0
        
        # Performance tracking with circular buffer
        self.processing_times = deque(maxlen=100)
        self.processing_index = 0
        
        # Concurrent processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self._calculate_optimal_workers())
        
        # Feature calculation flags with adaptive computation
        self.feature_flags = {
            'basic': True,
            'technical': True,
            'market_depth': True
        }
        self.feature_importance = {}
        self.last_feature_update = datetime.now()

    def _create_market_data_buffer(self, size):
        """Create market data buffer with optimal dtype"""
        return np.zeros(size, dtype=[
            ('timestamp', 'i8'),
            ('price', 'f4'),  # Using float32 instead of float64
            ('volume', 'f4'),
            ('bid', 'f4'),
            ('ask', 'f4')
        ])

    def _calculate_optimal_cache_size(self):
        """Calculate optimal cache size based on available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Use 1% of available memory for cache
            return min(
                self.cache_size,
                int((memory.available * 0.01) / (8 * len(self.config.FEATURE_CONFIG.feature_names)))
            )
        except ImportError:
            return 1000

    def _calculate_optimal_workers(self):
        """Calculate optimal number of workers based on CPU cores"""
        import multiprocessing
        return min(self.config.MAX_WORKERS, multiprocessing.cpu_count())

    def _resize_buffers_if_needed(self):
        """Resize buffers if threshold is reached"""
        if self.buffer_index >= self.current_buffer_size * self.resize_threshold:
            new_size = min(self.current_buffer_size * 2, self.max_buffer_size)
            if new_size > self.current_buffer_size:
                # Create new larger buffers
                new_market_data = self._create_market_data_buffer(new_size)
                new_feature_buffer = np.zeros((new_size, self.feature_buffer.shape[1]), dtype=np.float32)
                
                # Copy existing data
                new_market_data[:self.current_buffer_size] = self.market_data_buffer
                new_feature_buffer[:self.current_buffer_size] = self.feature_buffer
                
                # Update references
                self.market_data_buffer = new_market_data
                self.feature_buffer = new_feature_buffer
                self.current_buffer_size = new_size
                logger.info(f"Resized buffers to {new_size}")

    async def process_market_data(self, market_data: Dict) -> Optional[Dict]:
        """Optimized market data processing with adaptive features."""
        try:
            start_time = datetime.now()
            
            # Quick validation of required fields
            if not self._quick_validate(market_data):
                return None
            
            # Check if buffers need resizing
            self._resize_buffers_if_needed()
            
            # Update price history efficiently
            self._update_price_history(market_data['price'], market_data['volume'])
            
            # Adaptive feature calculation based on importance
            current_time = datetime.now()
            if (current_time - self.last_feature_update).total_seconds() >= 3600:  # Update every hour
                self._update_feature_importance()
                self.last_feature_update = current_time
            
            # Process features concurrently with importance-based filtering
            feature_tasks = []
            if self.feature_flags['basic']:
                feature_tasks.append(self._extract_basic_features(market_data))
            if self.feature_flags['technical'] and self.feature_importance.get('technical', 0) > 0.3:
                feature_tasks.append(self._calculate_technical_indicators(market_data))
            if self.feature_flags['market_depth'] and self.feature_importance.get('market_depth', 0) > 0.3:
                feature_tasks.append(self._process_market_depth(market_data))
            
            # Gather features concurrently
            features = await asyncio.gather(*feature_tasks)
            all_features = np.concatenate(features).reshape(1, -1)
            
            # Update buffers efficiently
            self._update_buffers(market_data, all_features)
            
            # Track processing time with deque
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_times.append(processing_time)
            
            return {
                'features': all_features,
                'timestamp': datetime.now(),
                'market_data': market_data,
                'processing_time': processing_time,
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None

    def _quick_validate(self, data: Dict) -> bool:
        """Fast validation of required fields."""
        return all(field in data for field in self.config.FEATURE_CONFIG.required_fields)

    def _update_price_history(self, price: float, volume: float):
        """Efficient price history update using circular buffer."""
        self.price_history[self.history_index] = price
        self.volume_history[self.history_index] = volume
        self.history_index = (self.history_index + 1) % len(self.price_history)

    def _update_buffers(self, market_data: Dict, features: np.ndarray):
        """Efficient buffer updates using numpy operations."""
        # Update market data buffer
        self.market_data_buffer[self.buffer_index] = (
            market_data['timestamp'],
            market_data['price'],
            market_data['volume'],
            market_data['bid'],
            market_data['ask']
        )
        
        # Update feature buffer
        self.feature_buffer[self.buffer_index] = features.flatten()
        self.buffer_index = (self.buffer_index + 1) % self.current_buffer_size

    @lru_cache(maxsize=1000)
    def _calculate_technical_indicators(self, data: Dict) -> np.ndarray:
        """Optimized technical indicator calculation with caching."""
        try:
            # Use pre-allocated price history
            prices = self.price_history
            if np.all(prices == 0):
                return np.zeros(4, dtype=np.float32)
            
            # Vectorized calculations
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Calculate indicators using vectorized operations
            rsi = self._vectorized_rsi(returns)
            macd = self._vectorized_macd(prices)
            bb_position = self._vectorized_bollinger(prices)
            
            return np.array([rsi, macd, bb_position, volatility], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return np.zeros(4, dtype=np.float32)

    def _vectorized_rsi(self, returns: np.ndarray, periods: int = 14) -> float:
        """Vectorized RSI calculation."""
        if len(returns) < periods:
            return 50.0
            
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        avg_gain = np.mean(gains[-periods:])
        avg_loss = np.mean(losses[-periods:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _vectorized_macd(self, prices: np.ndarray) -> float:
        """Vectorized MACD calculation."""
        if len(prices) < 26:
            return 0.0
            
        # Use numpy's exponential weighted functions
        weights12 = np.exp(np.linspace(-1., 0., 12))
        weights26 = np.exp(np.linspace(-1., 0., 26))
        
        weights12 = weights12 / weights12.sum()
        weights26 = weights26 / weights26.sum()
        
        ema12 = np.sum(prices[-12:] * weights12)
        ema26 = np.sum(prices[-26:] * weights26)
        
        return ema12 - ema26

    def _vectorized_bollinger(self, prices: np.ndarray, window: int = 20) -> float:
        """Vectorized Bollinger Bands calculation."""
        if len(prices) < window:
            return 0.5
            
        rolling_mean = np.mean(prices[-window:])
        rolling_std = np.std(prices[-window:])
        
        upper = rolling_mean + (2 * rolling_std)
        lower = rolling_mean - (2 * rolling_std)
        
        position = (prices[-1] - lower) / (upper - lower)
        return np.clip(position, 0, 1)

    def get_performance_metrics(self) -> Dict:
        """Efficient performance metrics calculation."""
        return {
            'avg_processing_time': np.mean(self.processing_times[self.processing_times != 0]),
            'buffer_usage': {
                'market_data': self.buffer_index,
                'features': self.buffer_index
            },
            'cache_stats': {
                'indicator_cache_size': len(self.indicator_cache),
                'price_history_size': len(self.price_history)
            }
        }

    async def cleanup(self):
        """Efficient resource cleanup."""
        try:
            # Clear numpy arrays
            self.market_data_buffer.fill(0)
            self.feature_buffer.fill(0)
            self.price_history.fill(0)
            self.volume_history.fill(0)
            self.processing_times.clear()
            
            # Clear caches
            self.indicator_cache.clear()
            
            # Reset indices
            self.buffer_index = 0
            self.history_index = 0
            self.processing_index = 0
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=False)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise 

    def _get_memory_usage(self):
        """Get current memory usage of buffers"""
        return {
            'market_data_buffer': self.market_data_buffer.nbytes / 1024 / 1024,  # MB
            'feature_buffer': self.feature_buffer.nbytes / 1024 / 1024,  # MB
            'cache_size': len(self.indicator_cache),
            'cache_efficiency': self.cache_hits / (self.cache_hits + self.cache_misses + 1)
        }

    def _update_feature_importance(self):
        """Update feature importance based on model feedback"""
        # Implementation would depend on model feedback mechanism
        pass 