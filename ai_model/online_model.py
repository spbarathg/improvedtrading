import os
import pickle
import logging
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from ai_model.base_model import BaseModel

logger = logging.getLogger(__name__)

class OnlineModel(BaseModel):
    """
    Implements an online learning model using SGDClassifier or PassiveAggressiveClassifier.
    Supports incremental learning with mini-batches and feature scaling.
    """

    def __init__(self, config):
        self.config = config
        self.model_type = self.config.MODEL_TYPE  # e.g., 'SGD' or 'PA' (Passive Aggressive)
        self.learning_rate = self.config.LEARNING_RATE
        self.scaler = StandardScaler()

        # Initialize the model based on the config
        if self.model_type == "SGD":
            self.model = SGDClassifier(loss='log', learning_rate='optimal')
        elif self.model_type == "PA":
            self.model = PassiveAggressiveClassifier()

        self.is_trained = False  # Keeps track of whether the model has been trained

    def train(self, data):
        """
        Trains the model on the provided data (initial batch training).
        :param data: A tuple (X_train, y_train) where X_train are features and y_train are labels.
        """
        X_train, y_train = data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model on the scaled data
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        logger.info("Model trained on initial data batch.")

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
        Updates the model incrementally using mini-batches of new data (online learning).
        :param data: A tuple (X_train, y_train) where X_train are features and y_train are labels.
        """
        X_train, y_train = data

        if not self.is_trained:
            logger.warning("Model hasn't been trained yet. Performing initial training.")
            self.train(data)
            return

        # Perform feature scaling with incremental fitting
        X_train_scaled = self.scaler.partial_fit(X_train).transform(X_train)
        self.model.partial_fit(X_train_scaled, y_train)
        logger.info("Model parameters updated using partial_fit.")

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