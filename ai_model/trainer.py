import asyncio
import logging
import numpy as np
from datetime import datetime
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

        # Training state
        self.training_task = None
        self.last_save_time = datetime.now()
        self.training_iteration = 0

    async def fetch_and_process_batch(self, batch_size):
        """
        Asynchronously fetches and processes a batch of data.
        """
        raw_data = await self.data_storage.get_latest_data_async(batch_size=batch_size)
        features, labels = await self.feature_extractor.extract_features_async(raw_data)
        return await self.data_preprocessor.preprocess_async(features, labels)

    async def train_online_model(self, online_model):
        """
        Continuously fetches data and trains the online model using partial_fit.
        :param online_model: The online learning model to train.
        """
        while True:
            try:
                # Fetch and process mini-batch asynchronously
                X_batch, y_batch = await self.fetch_and_process_batch(self.config.ONLINE_BATCH_SIZE)
                
                # Split mini-batch into smaller chunks for incremental learning
                n_samples = len(X_batch)
                chunk_size = min(self.config.MINI_BATCH_SIZE, n_samples)
                
                for i in range(0, n_samples, chunk_size):
                    X_mini = X_batch[i:i + chunk_size]
                    y_mini = y_batch[i:i + chunk_size]
                    
                    # Perform online training (incremental)
                    await online_model.partial_fit_async(X_mini, y_mini)
                    self.training_iteration += 1

                    # Evaluate on validation set periodically
                    if self.training_iteration % self.config.VALIDATION_INTERVAL == 0:
                        score = await online_model.score_async(self.X_val, self.y_val)
                        logger.info(f"Online model validation score: {score}")

                # Save model if needed
                await self.save_model_if_needed(online_model, "online")
                logger.info(f"Online model updated with {n_samples} samples")

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
                # Fetch rolling window of historical data asynchronously
                window_size = self.config.ROLLING_WINDOW_SIZE
                raw_data = await self.data_storage.get_historical_data_async(window=window_size)
                
                # Process data in chunks to avoid memory issues
                chunk_size = self.config.PROCESSING_CHUNK_SIZE
                X_chunks, y_chunks = [], []
                
                for i in range(0, len(raw_data), chunk_size):
                    chunk = raw_data[i:i + chunk_size]
                    features, labels = await self.feature_extractor.extract_features_async(chunk)
                    X_proc, y_proc = await self.data_preprocessor.preprocess_async(features, labels)
                    X_chunks.append(X_proc)
                    y_chunks.append(y_proc)

                # Combine chunks
                X_train = np.vstack(X_chunks)
                y_train = np.concatenate(y_chunks)

                # Perform full model retraining
                await periodic_model.train_async(X_train, y_train)

                # Evaluate on validation set
                score = await periodic_model.score_async(self.X_val, self.y_val)
                logger.info(f"Periodic model validation score: {score}")

                # Save the periodic model
                await self.save_model_if_needed(periodic_model, "periodic")
                logger.info(f"Periodic model retrained with {len(X_train)} samples")

            except Exception as e:
                logger.error(f"Error during periodic model training: {e}")

            # Control training frequency
            await asyncio.sleep(self.config.PERIODIC_TRAINING_INTERVAL)

    async def save_model_if_needed(self, model, model_type):
        """
        Saves the model if the configured time interval has passed.
        """
        current_time = datetime.now()
        time_since_last_save = (current_time - self.last_save_time).total_seconds()
        
        if time_since_last_save >= self.config.MODEL_SAVE_INTERVAL:
            save_path = f"{self.config.MODEL_SAVE_DIR}/{model_type}_model_{current_time.strftime('%Y%m%d_%H%M%S')}.pkl"
            await model.save_async(save_path)
            self.last_save_time = current_time
            logger.info(f"Saved {model_type} model to {save_path}")

    def start_training(self):
        """
        Starts the training process in the background.
        Returns the background task for monitoring.
        """
        if self.training_task is not None:
            logger.warning("Training is already running")
            return self.training_task

        async def initialize_and_train():
            # Initialize models
            online_model = await self.config.online_model.load_async(self.config.ONLINE_MODEL_PATH)
            periodic_model = await self.config.periodic_model.load_async(self.config.PERIODIC_MODEL_PATH)
            
            # Initialize validation set
            await self.setup_validation_set()
            
            # Start training
            await self.train_models(online_model, periodic_model)

        self.training_task = asyncio.create_task(initialize_and_train())
        logger.info("Training started in background")
        return self.training_task

    def stop_training(self):
        """
        Stops the training process gracefully.
        """
        if self.training_task is not None:
            self.training_task.cancel()
            self.training_task = None
            logger.info("Training stopped")

    async def setup_validation_set(self):
        """
        Initializes the validation set by splitting the historical data.
        """
        raw_data = await self.data_storage.get_historical_data_async(window=self.config.VALIDATION_WINDOW_SIZE)
        features, labels = await self.feature_extractor.extract_features_async(raw_data)
        X_data, y_data = await self.data_preprocessor.preprocess_async(features, labels)

        self.X_val, _, self.y_val, _ = train_test_split(
            X_data, y_data, test_size=self.validation_size, random_state=self.config.RANDOM_SEED
        )
        logger.info("Validation set initialized for model selection")

    async def train_models(self, online_model, periodic_model):
        """
        Orchestrates the training of both online and periodic models.
        """
        await asyncio.gather(
            self.train_online_model(online_model),
            self.train_periodic_model(periodic_model)
        )

    def get_validation_data(self):
        """
        Provides the validation set for model evaluation.
        """
        return self.X_val, self.y_val