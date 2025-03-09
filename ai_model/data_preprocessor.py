from sklearn.preprocessing import StandardScaler
import numpy as np

class DataPreprocessor:
    """
    Handles data preprocessing for AI models. This includes feature scaling, label transformations, and any other
    data preparation steps that are specific to the models and not part of the main data pipeline.
    """

    def __init__(self):
        """
        Initializes the DataPreprocessor.
        """
        # StandardScaler for online model's incremental scaling
        self.scaler = StandardScaler()

    def preprocess(self, features, labels, online_model=True):
        """
        Preprocesses the features and labels for the AI models.
        - Scales the features (StandardScaler) for online models using partial_fit.
        - Handles multi-class labels (Win/Hold/Loss).
        
        :param features: Array of raw features from the data pipeline.
        :param labels: Array of labels (e.g., price movement classification).
        :param online_model: Boolean indicating whether we are processing data for the online model (requires partial fit).
        :return: Tuple (X_processed, y_processed) of preprocessed features and labels.
        """
        # Scale the features
        if online_model:
            X_scaled = self.scaler.partial_fit_transform(features)
        else:
            X_scaled = self.scaler.fit_transform(features)  # Use fit for the periodic model

        # Transform labels for multi-class classification (e.g., Win/Hold/Loss)
        y_transformed = self._transform_labels(labels)

        return X_scaled, y_transformed

    def _transform_labels(self, labels):
        """
        Transforms raw labels into a multi-class format.
        Example:
        - "Win" -> 1 (for buy/sell opportunities)
        - "Hold" -> 0 (no action needed)
        - "Loss" -> -1 (indicates a potential sell)
        
        :param labels: Array of raw labels.
        :return: Array of transformed labels.
        """
        transformed_labels = np.array([self._label_to_class(label) for label in labels])
        return transformed_labels

    def _label_to_class(self, label):
        """
        Maps a label to its corresponding class.
        :param label: String label.
        :return: Integer class label.
        """
        if label == "Win":
            return 1
        elif label == "Hold":
            return 0
        elif label == "Loss":
            return -1
        else:
            raise ValueError(f"Unknown label: {label}")