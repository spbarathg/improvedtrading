import os
import pickle
import logging
import numpy as np
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from ai_model.base_model import BaseModel

logger = logging.getLogger(__name__)

class OnlineModel(BaseModel):
    """
    Implements an online learning model using SGDClassifier or PassiveAggressiveClassifier.
    Supports incremental learning with mini-batches, feature scaling, and learning rate scheduling.
    """

    def __init__(self, config):
        self.config = config
        self.model_type = self.config.MODEL_TYPE  # e.g., 'SGD' or 'PA' (Passive Aggressive)
        self.batch_size = getattr(self.config, 'BATCH_SIZE', 32)
        self.initial_learning_rate = getattr(self.config, 'LEARNING_RATE', 0.01)
        self.learning_rate_schedule = getattr(self.config, 'LEARNING_RATE_SCHEDULE', 'optimal')
        self.scaler = StandardScaler()
        
        # Initialize learning rate parameters
        self.n_iterations = 0
        self.current_learning_rate = self.initial_learning_rate

        # Initialize the model based on the config
        if self.model_type == "SGD":
            self.model = SGDClassifier(
                loss='log',
                learning_rate=self.learning_rate_schedule,
                eta0=self.initial_learning_rate,
                warm_start=True  # Enable warm start for continuous training
            )
        elif self.model_type == "PA":
            self.model = PassiveAggressiveClassifier(
                C=self.initial_learning_rate,
                warm_start=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.is_trained = False
        self.classes_ = None  # Store unique classes for partial_fit

    def _update_learning_rate(self):
        """Updates the learning rate based on the chosen schedule."""
        if self.learning_rate_schedule == 'inverse_scaling':
            self.current_learning_rate = (
                self.initial_learning_rate / (1 + self.initial_learning_rate * self.n_iterations)
            )
            if isinstance(self.model, SGDClassifier):
                self.model.eta0 = self.current_learning_rate
        self.n_iterations += 1

    def _create_mini_batches(self, X, y, batch_size):
        """Creates mini-batches from input data."""
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    def train(self, data):
        """
        Trains the model on the provided data using mini-batches.
        :param data: A tuple (X_train, y_train) where X_train are features and y_train are labels.
        """
        X_train, y_train = data
        self.classes_ = np.unique(y_train)
        
        # Initial fit of the scaler
        self.scaler.fit(X_train)
        
        # Train in mini-batches
        for X_batch, y_batch in self._create_mini_batches(X_train, y_train, self.batch_size):
            X_batch_scaled = self.scaler.transform(X_batch)
            if not self.is_trained:
                self.model.fit(X_batch_scaled, y_batch)
                self.is_trained = True
            else:
                self.model.partial_fit(X_batch_scaled, y_batch, classes=self.classes_)
            self._update_learning_rate()
        
        logger.info(f"Model trained on initial data with {len(X_train)} samples")

    async def predict(self, features):
        """
        Makes predictions based on the input features. If the model isn't trained yet, returns None.
        :param features: The input features for making predictions.
        :return: Predictions or None if the model isn't trained.
        """
        if not self.is_trained:
            logger.warning("Model hasn't been trained yet. Skipping prediction.")
            return None
        
        features_scaled = self.scaler.transform([features])
        predictions = self.model.predict(features_scaled)
        return predictions[0]  # Return single prediction

    def partial_fit(self, data):
        """
        Updates the model incrementally using mini-batches of new data.
        :param data: A tuple (X_train, y_train) where X_train are features and y_train are labels.
        """
        X_train, y_train = data

        if not self.is_trained:
            logger.warning("Model hasn't been trained yet. Performing initial training.")
            self.train(data)
            return

        # Process data in mini-batches
        for X_batch, y_batch in self._create_mini_batches(X_train, y_train, self.batch_size):
            # Update scaler and transform batch
            X_batch_scaled = self.scaler.partial_fit(X_batch).transform(X_batch)
            
            # Update model
            self.model.partial_fit(X_batch_scaled, y_batch, classes=self.classes_)
            self._update_learning_rate()
        
        logger.info(f"Model updated with {len(X_train)} new samples")

    def save(self, filepath):
        """
        Saves the model and scaler to the specified file.
        :param filepath: Path where the model will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        logger.info(f"Model saved to {filepath}.")

    def load(self, filepath):
        """
        Loads the model and scaler from the specified file.
        :param filepath: Path from where the model will be loaded.
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} not found. Cannot load model.")
            return

        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}.")