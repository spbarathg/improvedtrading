import asyncio
import logging
from sklearn.model_selection import train_test_split
from ai_model.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class Trainer:
    """
    Manages the training of both online and periodic models.
    """

    def __init__(self, data_storage, feature_extractor, config):
        """
        Initializes the Trainer with dependencies.
        :param data_storage: DataStorage component for fetching training data.
        :param feature_extractor: FeatureExtractor for processing raw data into features.
        :param config: Configuration settings (e.g., training intervals, batch sizes, model save frequency).
        """
        self.data_storage = data_storage
        self.feature_extractor = feature_extractor
        self.config = config

        # Data Preprocessor
        self.data_preprocessor = DataPreprocessor()

        # Initialize validation set
        self.X_val, self.y_val = None, None
        self.validation_size = self.config.VALIDATION_SET_SIZE

    async def train_online_model(self, online_model):
        """
        Continuously fetches data and trains the online model using partial_fit.
        :param online_model: The online learning model to train.
        """
        while True:
            try:
                # Fetch the latest batch of data
                raw_data = self.data_storage.get_latest_data(batch_size=self.config.ONLINE_BATCH_SIZE)
                features, labels = self.feature_extractor.extract_features(raw_data)

                # Preprocess data (scaling, cleaning, etc.)
                X_train, y_train = self.data_preprocessor.preprocess(features, labels)

                # Perform online training (incremental)
                online_model.partial_fit(X_train, y_train)

                # Periodically save the online model
                if self.should_save_model():
                    online_model.save()

                logger.info("Online model updated with new mini-batch of data.")

            except Exception as e:
                logger.error(f"Error during online model training: {e}")

            # Control training frequency
            await asyncio.sleep(self.config.ONLINE_TRAINING_INTERVAL)

    async def train_periodic_model(self, periodic_model):
        """
        Trains the periodic model at regular intervals using historical data.
        :param periodic_model: The periodically retrained model.
        """
        while True:
            try:
                # Fetch a rolling window of historical data
                raw_data = self.data_storage.get_historical_data(window=self.config.ROLLING_WINDOW_SIZE)
                features, labels = self.feature_extractor.extract_features(raw_data)

                # Preprocess data
                X_train, y_train = self.data_preprocessor.preprocess(features, labels)

                # Perform periodic model training
                periodic_model.train(X_train, y_train)

                # Save the periodic model
                periodic_model.save()

                logger.info("Periodic model retrained with new historical data.")

            except Exception as e:
                logger.error(f"Error during periodic model training: {e}")

            # Control training frequency (train periodically)
            await asyncio.sleep(self.config.PERIODIC_TRAINING_INTERVAL)

    def should_save_model(self):
        """
        Determines if the model should be saved (based on config).
        :return: Boolean indicating whether the model should be saved.
        """
        # Logic to determine if models should be saved (e.g., every N iterations)
        return True  # Can customize based on iteration count, time intervals, etc.

    async def train_models(self):
        """
        Orchestrates the training of both online and periodic models.
        """
        online_model = self.config.online_model
        periodic_model = self.config.periodic_model

        # Kick off asynchronous training tasks for both models
        await asyncio.gather(
            self.train_online_model(online_model),
            self.train_periodic_model(periodic_model)
        )

    def setup_validation_set(self):
        """
        Initializes the validation set by splitting the historical data.
        The validation set is used for model selection.
        """
        # Fetch historical data for creating validation set
        raw_data = self.data_storage.get_historical_data(window=self.config.VALIDATION_WINDOW_SIZE)
        features, labels = self.feature_extractor.extract_features(raw_data)

        # Preprocess data
        X_data, y_data = self.data_preprocessor.preprocess(features, labels)

        # Split the data into training and validation sets
        self.X_val, _, self.y_val, _ = train_test_split(
            X_data, y_data, test_size=self.validation_size, random_state=self.config.RANDOM_SEED
        )

        logger.info("Validation set initialized for model selection.")

    def get_validation_data(self):
        """
        Provides the validation set for model evaluation.
        :return: Tuple (X_val, y_val) containing validation features and labels.
        """
        return self.X_val, self.y_val