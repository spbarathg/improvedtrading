import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from cachetools import TTLCache
from prometheus_client import Counter, Histogram, Gauge
from trading_bot.utils.helpers import Helpers

logger = logging.getLogger(__name__)

class FeatureExtractor:
    _VECTOR_SIZE = 1000  # Pre-allocated vector size
    _TREND_WINDOW = 10
    _BATCH_SIZE = 1000
    _CACHE_TTL = 300  # 5 minutes cache TTL
    
    def __init__(self, config, data_storage):
        self.config = config
        self.data_storage = data_storage
        
        # Optimized cache with larger size and longer TTL
        self.cache = TTLCache(maxsize=50_000, ttl=self._CACHE_TTL)
        
        # Pre-allocated numpy arrays with optimal memory alignment
        self._price_buffer = np.zeros(self._VECTOR_SIZE, dtype=np.float32, order='C')
        self._volume_buffer = np.zeros(self._VECTOR_SIZE, dtype=np.float32, order='C')
        self._liquidity_buffer = np.zeros(self._VECTOR_SIZE, dtype=np.float32, order='C')
        self._buffer_index = 0
        
        # Pre-allocate arrays for calculations
        self._price_changes = np.zeros(self._VECTOR_SIZE, dtype=np.float32, order='C')
        self._volume_changes = np.zeros(self._VECTOR_SIZE, dtype=np.float32, order='C')
        self._liquidity_changes = np.zeros(self._VECTOR_SIZE, dtype=np.float32, order='C')
        
        # Validation ranges with optimized bounds
        self._validation = {
            'price': (0.0, 1e6),  # Reasonable upper bound
            'volume': (0.0, 1e9),
            'liquidity': (0.0, 1e9),
            'sentiment': (-1.0, 1.0),
            'price_change': (-100.0, 100.0),
            'price_trend': (-1e3, 1e3),
            'price_volatility': (0.0, 1e3)
        }
        
        # Metrics with optimized labels
        self.metrics = {
            'processed': Counter('features_processed', 'Total processed items'),
            'errors': Counter('feature_errors', 'Processing errors'),
            'latency': Histogram('feature_latency', 'Processing latency in seconds', buckets=(0.1, 0.5, 1.0, 2.0, 5.0)),
            'cache': Gauge('feature_cache', 'Current cache size'),
            'throughput': Gauge('feature_throughput', 'Items/sec', ['window'])
        }
        
        # Throughput tracking with optimized window size
        self._throughput_window = np.zeros(100, dtype=np.int32)  # Pre-allocated array
        self._window_index = 0
        self._last_throughput = datetime.now()

    def _vectorized_validation(self, values: np.ndarray, feature: str) -> np.ndarray:
        """Vectorized validation using numpy with SIMD optimization"""
        min_val, max_val = self._validation[feature]
        return np.logical_and(values >= min_val, values <= max_val)

    async def _batch_process(self, data_batch: List[Dict]) -> List[Dict]:
        """Process batch of data using optimized vectorized operations"""
        if not data_batch:
            return []

        # Pre-allocate structured array with optimal memory layout
        dtype = np.dtype([
            ('price', 'f4'), ('volume', 'f4'), ('liquidity', 'f4'),
            ('prev_price', 'f4'), ('prev_volume', 'f4'), ('prev_liquidity', 'f4')
        ])
        
        # Create array with list comprehension for better performance
        arr = np.array([
            (d.get('price', 0), d.get('volume', 0), d.get('liquidity', 0),
             d.get('previous_price', 0), d.get('previous_volume', 0),
             d.get('previous_liquidity', 0))
            for d in data_batch
        ], dtype=dtype)

        # Vectorized calculations with SIMD optimization
        valid = np.ones(len(arr), dtype=bool)
        
        # Price features with optimized division
        price_mask = self._vectorized_validation(arr['price'], 'price')
        valid &= price_mask
        np.divide(
            arr['price'] - arr['prev_price'],
            arr['prev_price'],
            out=self._price_changes[:len(arr)],
            where=arr['prev_price'] != 0
        )
        self._price_changes[:len(arr)] *= 100

        # Volume features
        volume_mask = self._vectorized_validation(arr['volume'], 'volume')
        valid &= volume_mask
        np.divide(
            arr['volume'] - arr['prev_volume'],
            arr['prev_volume'],
            out=self._volume_changes[:len(arr)],
            where=arr['prev_volume'] != 0
        )
        self._volume_changes[:len(arr)] *= 100

        # Liquidity features
        liquidity_mask = self._vectorized_validation(arr['liquidity'], 'liquidity')
        valid &= liquidity_mask
        np.divide(
            arr['liquidity'] - arr['prev_liquidity'],
            arr['prev_liquidity'],
            out=self._liquidity_changes[:len(arr)],
            where=arr['prev_liquidity'] != 0
        )
        self._liquidity_changes[:len(arr)] *= 100

        # Build results using list comprehension for better performance
        return [
            {
                'price': arr['price'][i],
                'price_change': self._price_changes[i],
                'volume': arr['volume'][i],
                'volume_change': self._volume_changes[i],
                'liquidity': arr['liquidity'][i],
                'liquidity_change': self._liquidity_changes[i]
            }
            for i in range(len(arr))
            if valid[i]
        ]

    async def _calculate_trends(self, buffer: np.ndarray) -> np.ndarray:
        """Calculate rolling trends using optimized matrix operations"""
        if len(buffer) < self._TREND_WINDOW:
            return np.zeros(len(buffer), dtype=np.float32)
            
        # Create optimized sliding window view
        shape = (len(buffer) - self._TREND_WINDOW + 1, self._TREND_WINDOW)
        strides = (buffer.strides[0], buffer.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            buffer, shape=shape, strides=strides
        )
        
        # Pre-compute means for better performance
        x = np.arange(self._TREND_WINDOW, dtype=np.float32)
        x_mean = x.mean()
        y_mean = windows.mean(axis=1)
        
        # Vectorized trend calculation with optimized memory access
        numerator = np.sum((windows - y_mean[:, None]) * (x - x_mean), axis=1)
        denominator = np.sum((x - x_mean) ** 2)
        
        return numerator / denominator

    async def process_stream(self):
        """Main processing loop with optimized throughput and resource usage"""
        while True:
            start_time = datetime.now()
            
            try:
                # Batch data collection with timeout
                raw_data = await asyncio.wait_for(
                    self.data_storage.get_latest_batch(self._BATCH_SIZE),
                    timeout=0.1
                )
                
                if not raw_data:
                    await asyncio.sleep(0.01)  # Reduced sleep time
                    continue
                
                # Process batch with optimized memory usage
                batch_start = datetime.now()
                features = await self._batch_process(raw_data)
                
                # Update metrics efficiently
                if features:
                    self.metrics['processed'].inc(len(features))
                    self.metrics['latency'].observe(
                        (datetime.now() - batch_start).total_seconds()
                    )
                    await self.data_storage.batch_store_features(features)
                
                # Optimized throughput calculation
                self._throughput_window[self._window_index] = len(features)
                self._window_index = (self._window_index + 1) % 100
                
                if (datetime.now() - self._last_throughput).seconds >= 5:
                    tps = np.sum(self._throughput_window) / 5
                    self.metrics['throughput'].labels(window='5s').set(tps)
                    self._last_throughput = datetime.now()
                
                # Adaptive sleep with optimized timing
                processing_time = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0.001, min(0.1, (len(features) / self._BATCH_SIZE) * 0.01))
                await asyncio.sleep(sleep_time)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics['errors'].inc()
                logger.error("Processing error: %s", e, exc_info=True)
                await asyncio.sleep(0.1)  # Reduced error sleep time