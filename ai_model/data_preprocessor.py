from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import numpy as np
import asyncio
from functools import lru_cache
import logging
from typing import Tuple, Optional, Union
import concurrent.futures

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Enhanced data preprocessor with memory-efficient processing, caching, and better error handling.
    """

    def __init__(self, scaling_method='standard', normalization=None, cache_size=1000):
        """
        Initializes the DataPreprocessor with specified scaling and normalization methods.
        
        :param scaling_method: str, The scaling method to use ('standard', 'minmax', None)
        :param normalization: str, The normalization method to use ('l1', 'l2', None)
        :param cache_size: int, Size of the LRU cache for feature processing
        """
        self.scaling_method = scaling_method
        self.normalization = normalization
        self._validate_parameters()
        
        # Initialize scalers with better memory management
        self.standard_scaler = StandardScaler(copy=False) if scaling_method == 'standard' else None
        self.minmax_scaler = MinMaxScaler(copy=False) if scaling_method == 'minmax' else None
        
        # Initialize normalizers with better memory management
        self.l1_normalizer = Normalizer(norm='l1', copy=False) if normalization == 'l1' else None
        self.l2_normalizer = Normalizer(norm='l2', copy=False) if normalization == 'l2' else None
        
        self.is_fitted = False
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    def _validate_parameters(self):
        """Validates initialization parameters."""
        valid_scaling = {'standard', 'minmax', None}
        valid_norm = {'l1', 'l2', None}
        
        if self.scaling_method not in valid_scaling:
            raise ValueError(f"Invalid scaling_method. Must be one of {valid_scaling}")
        if self.normalization not in valid_norm:
            raise ValueError(f"Invalid normalization. Must be one of {valid_norm}")

    @lru_cache(maxsize=1000)
    def _cache_key(self, features_tuple: Tuple) -> str:
        """Generate cache key for feature arrays."""
        return hash(features_tuple)

    async def preprocess(self, features: np.ndarray, labels: np.ndarray, 
                        online_model: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asynchronously preprocesses the features and labels with improved error handling.
        
        :param features: Array of raw features
        :param labels: Array of labels
        :param online_model: Boolean for online model processing
        :return: Tuple of preprocessed features and labels
        """
        try:
            if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
                raise TypeError("Features and labels must be numpy arrays")

            if features.size == 0 or labels.size == 0:
                raise ValueError("Empty features or labels provided")

            if features.shape[0] != labels.shape[0]:
                raise ValueError("Features and labels must have the same number of samples")

            # Process features asynchronously with better error handling
            X_scaled = await self._scale_features(features, online_model)
            X_normalized = await self._normalize_features(X_scaled)
            
            # Transform labels with validation
            y_transformed = await asyncio.to_thread(self._transform_labels, labels)
            
            return X_normalized, y_transformed

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    async def _scale_features(self, features: np.ndarray, online_model: bool) -> np.ndarray:
        """
        Memory-efficient feature scaling with improved error handling.
        """
        if self.scaling_method is None:
            return features

        try:
            scaler = self.standard_scaler if self.scaling_method == 'standard' else self.minmax_scaler
            
            if online_model:
                return await asyncio.to_thread(
                    lambda: scaler.partial_fit(features).transform(features)
                )
            
            if not self.is_fitted:
                result = await asyncio.to_thread(
                    lambda: scaler.fit_transform(features)
                )
                self.is_fitted = True
                return result
            
            return await asyncio.to_thread(
                lambda: scaler.transform(features)
            )

        except Exception as e:
            logger.error(f"Error in feature scaling: {str(e)}")
            raise

    async def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Memory-efficient feature normalization with improved error handling.
        """
        if self.normalization is None:
            return features

        try:
            normalizer = self.l1_normalizer if self.normalization == 'l1' else self.l2_normalizer
            return await asyncio.to_thread(normalizer.fit_transform, features)

        except Exception as e:
            logger.error(f"Error in feature normalization: {str(e)}")
            raise

    def _transform_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Optimized label transformation with validation.
        """
        label_map = {"Win": 1, "Hold": 0, "Loss": -1}
        
        try:
            if isinstance(labels[0], (str, bytes)):
                transformed = np.array([label_map.get(label, None) for label in labels])
                if None in transformed:
                    invalid_labels = set(label for label in labels if label not in label_map)
                    raise ValueError(f"Invalid labels found: {invalid_labels}")
                return transformed
            return labels  # Assume already transformed if not string labels

        except Exception as e:
            logger.error(f"Error in label transformation: {str(e)}")
            raise

    async def reset(self):
        """
        Efficiently reset the preprocessor state.
        """
        try:
            # Only recreate necessary objects based on configuration
            if self.scaling_method == 'standard':
                self.standard_scaler = StandardScaler(copy=False)
            elif self.scaling_method == 'minmax':
                self.minmax_scaler = MinMaxScaler(copy=False)
            
            if self.normalization == 'l1':
                self.l1_normalizer = Normalizer(norm='l1', copy=False)
            elif self.normalization == 'l2':
                self.l2_normalizer = Normalizer(norm='l2', copy=False)
            
            self.is_fitted = False
            
        except Exception as e:
            logger.error(f"Error resetting preprocessor: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources."""
        self._thread_pool.shutdown(wait=False)