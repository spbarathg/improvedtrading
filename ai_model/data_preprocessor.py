from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import numpy as np
import asyncio

class DataPreprocessor:
    """
    Handles data preprocessing for AI models asynchronously. This includes feature scaling,
    label transformations, and various normalization techniques.
    """

    def __init__(self, scaling_method='standard', normalization=None):
        """
        Initializes the DataPreprocessor with specified scaling and normalization methods.
        
        :param scaling_method: str, The scaling method to use ('standard', 'minmax', None)
        :param normalization: str, The normalization method to use ('l1', 'l2', None)
        """
        # Initialize scalers and normalizers
        self.scaling_method = scaling_method
        self.normalization = normalization
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Normalizers for L1 and L2
        self.l1_normalizer = Normalizer(norm='l1')
        self.l2_normalizer = Normalizer(norm='l2')
        
        # Store fitted state
        self.is_fitted = False

    async def preprocess(self, features, labels, online_model=True):
        """
        Asynchronously preprocesses the features and labels for the AI models.
        
        :param features: Array of raw features from the data pipeline
        :param labels: Array of labels (e.g., price movement classification)
        :param online_model: Boolean indicating whether we are processing data for the online model
        :return: Tuple (X_processed, y_processed) of preprocessed features and labels
        """
        # Process features asynchronously
        X_scaled = await self._scale_features(features, online_model)
        X_normalized = await self._normalize_features(X_scaled)
        
        # Transform labels (this is typically fast, so no need for async)
        y_transformed = self._transform_labels(labels)
        
        return X_normalized, y_transformed

    async def _scale_features(self, features, online_model):
        """
        Asynchronously scales the features using the specified scaling method.
        
        :param features: Array of features to scale
        :param online_model: Boolean indicating whether to use partial_fit
        :return: Scaled features
        """
        if self.scaling_method is None:
            return features
            
        # Use asyncio.to_thread for CPU-bound operations
        if self.scaling_method == 'standard':
            if online_model:
                return await asyncio.to_thread(
                    self.standard_scaler.partial_fit_transform, features
                )
            return await asyncio.to_thread(
                self.standard_scaler.fit_transform if not self.is_fitted 
                else self.standard_scaler.transform, features
            )
        elif self.scaling_method == 'minmax':
            if online_model:
                return await asyncio.to_thread(
                    self.minmax_scaler.partial_fit_transform, features
                )
            return await asyncio.to_thread(
                self.minmax_scaler.fit_transform if not self.is_fitted 
                else self.minmax_scaler.transform, features
            )
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    async def _normalize_features(self, features):
        """
        Asynchronously normalizes the features using the specified normalization method.
        
        :param features: Array of features to normalize
        :return: Normalized features
        """
        if self.normalization is None:
            return features
            
        if self.normalization == 'l1':
            return await asyncio.to_thread(self.l1_normalizer.fit_transform, features)
        elif self.normalization == 'l2':
            return await asyncio.to_thread(self.l2_normalizer.fit_transform, features)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

    def _transform_labels(self, labels):
        """
        Transforms raw labels into a multi-class format.
        Example:
        - "Win" -> 1 (for buy/sell opportunities)
        - "Hold" -> 0 (no action needed)
        - "Loss" -> -1 (indicates a potential sell)
        
        :param labels: Array of raw labels
        :return: Array of transformed labels
        """
        transformed_labels = np.array([self._label_to_class(label) for label in labels])
        return transformed_labels

    def _label_to_class(self, label):
        """
        Maps a label to its corresponding class.
        :param label: String label
        :return: Integer class label
        """
        if label == "Win":
            return 1
        elif label == "Hold":
            return 0
        elif label == "Loss":
            return -1
        else:
            raise ValueError(f"Unknown label: {label}")

    async def reset(self):
        """
        Asynchronously resets all scalers and normalizers.
        """
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.l1_normalizer = Normalizer(norm='l1')
        self.l2_normalizer = Normalizer(norm='l2')
        self.is_fitted = False