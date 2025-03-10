import os
import pickle
import logging
import numpy as np
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from ai_model.base_model import BaseModel
from typing import Tuple, Optional, Union, Dict
import asyncio
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

class OnlineModel(BaseModel):
    """
    Enhanced online learning model with improved efficiency and reliability.
    Supports dynamic learning rate scheduling, automatic model checkpointing,
    and robust error handling.
    """

    def __init__(self, config):
        self.config = config
        self.model_type = self.config.MODEL_TYPE
        self.batch_size = getattr(self.config, 'BATCH_SIZE', 32)
        self.initial_learning_rate = getattr(self.config, 'LEARNING_RATE', 0.01)
        self.learning_rate_schedule = getattr(self.config, 'LEARNING_RATE_SCHEDULE', 'optimal')
        self.checkpoint_interval = getattr(self.config, 'CHECKPOINT_INTERVAL', 1000)
        self.model_dir = getattr(self.config, 'MODEL_DIR', 'model_checkpoints')
        
        # Enhanced learning rate parameters
        self.n_iterations = 0
        self.current_learning_rate = self.initial_learning_rate
        self.min_learning_rate = getattr(self.config, 'MIN_LEARNING_RATE', 1e-6)
        self.learning_rate_decay = getattr(self.config, 'LEARNING_RATE_DECAY', 0.1)
        
        # Performance tracking
        self.performance_history = {
            'train_loss': [],
            'validation_accuracy': [],
            'learning_rates': []
        }
        
        # Initialize model with optimized parameters
        self._initialize_model()
        
        self.is_trained = False
        self.classes_ = None
        self.samples_seen = 0
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def _initialize_model(self):
        """Initialize the model with optimized parameters based on type."""
        try:
            if self.model_type == "SGD":
                self.model = SGDClassifier(
                    loss='log',
                    learning_rate=self.learning_rate_schedule,
                    eta0=self.initial_learning_rate,
                    warm_start=True,
                    n_jobs=-1,  # Use all available cores
                    max_iter=1,  # For online learning
                    tol=1e-3,
                    class_weight='balanced',
                    random_state=42
                )
            elif self.model_type == "PA":
                self.model = PassiveAggressiveClassifier(
                    C=self.initial_learning_rate,
                    warm_start=True,
                    n_jobs=-1,
                    max_iter=1,
                    tol=1e-3,
                    class_weight='balanced',
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _adaptive_learning_rate(self) -> float:
        """
        Implements adaptive learning rate scheduling with early stopping detection.
        """
        if len(self.performance_history['train_loss']) > 1:
            current_loss = self.performance_history['train_loss'][-1]
            prev_loss = self.performance_history['train_loss'][-2]
            
            # If loss is increasing, reduce learning rate
            if current_loss > prev_loss:
                self.current_learning_rate *= self.learning_rate_decay
                self.current_learning_rate = max(self.current_learning_rate, self.min_learning_rate)
                
        return self.current_learning_rate

    def _update_learning_rate(self):
        """Updates learning rate based on schedule and performance."""
        if self.learning_rate_schedule == 'adaptive':
            self.current_learning_rate = self._adaptive_learning_rate()
        elif self.learning_rate_schedule == 'inverse_scaling':
            self.current_learning_rate = (
                self.initial_learning_rate / (1 + self.learning_rate_decay * self.n_iterations)
            )
        
        if isinstance(self.model, SGDClassifier):
            self.model.eta0 = self.current_learning_rate
            
        self.performance_history['learning_rates'].append(self.current_learning_rate)
        self.n_iterations += 1

    async def _create_mini_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        """
        Asynchronously creates mini-batches with improved memory efficiency.
        """
        try:
            total_samples = len(X)
            indices = np.arange(total_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Use memory views instead of copies where possible
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                yield X_batch, y_batch
                
        except Exception as e:
            logger.error(f"Error creating mini-batches: {str(e)}")
            raise

    async def train(self, data: Tuple[np.ndarray, np.ndarray]):
        """
        Enhanced training with automatic checkpointing and performance tracking.
        """
        try:
            X_train, y_train = data
            self.classes_ = np.unique(y_train)
            
            async for X_batch, y_batch in self._create_mini_batches(X_train, y_train, self.batch_size):
                if not self.is_trained:
                    self.model.fit(X_batch, y_batch)
                    self.is_trained = True
                else:
                    self.model.partial_fit(X_batch, y_batch, classes=self.classes_)
                
                # Update learning rate and track performance
                loss = self.model.loss_
                self.performance_history['train_loss'].append(loss)
                self._update_learning_rate()
                
                # Update samples seen and check for checkpointing
                self.samples_seen += len(X_batch)
                if self.samples_seen % self.checkpoint_interval == 0:
                    await self._save_checkpoint()
            
            logger.info(f"Model trained on {len(X_train)} samples with final learning rate {self.current_learning_rate:.6f}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    async def predict(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Makes predictions with improved error handling and validation.
        """
        try:
            if not self.is_trained:
                logger.warning("Model hasn't been trained yet. Skipping prediction.")
                return None
                
            if not isinstance(features, np.ndarray):
                features = np.array(features).reshape(1, -1)
                
            # Validate feature dimensions
            if features.ndim == 1:
                features = features.reshape(1, -1)
            elif features.ndim != 2:
                raise ValueError("Features must be 2-dimensional")
                
            predictions = self.model.predict(features)
            return predictions[0] if len(predictions) == 1 else predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    async def partial_fit(self, data: Tuple[np.ndarray, np.ndarray]):
        """
        Enhanced incremental training with automatic performance monitoring.
        """
        try:
            if not self.is_trained:
                logger.warning("Model hasn't been trained yet. Performing initial training.")
                await self.train(data)
                return

            X_train, y_train = data
            
            async for X_batch, y_batch in self._create_mini_batches(X_train, y_train, self.batch_size):
                self.model.partial_fit(X_batch, y_batch, classes=self.classes_)
                
                # Track performance and update learning rate
                loss = self.model.loss_
                self.performance_history['train_loss'].append(loss)
                self._update_learning_rate()
                
                # Update samples seen and check for checkpointing
                self.samples_seen += len(X_batch)
                if self.samples_seen % self.checkpoint_interval == 0:
                    await self._save_checkpoint()
            
            logger.info(f"Model updated with {len(X_train)} new samples")
            
        except Exception as e:
            logger.error(f"Error during partial_fit: {str(e)}")
            raise

    async def _save_checkpoint(self):
        """
        Asynchronously saves model checkpoint with metadata.
        """
        try:
            checkpoint_path = os.path.join(
                self.model_dir,
                f"model_checkpoint_{self.samples_seen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            
            checkpoint_data = {
                'model': self.model,
                'performance_history': self.performance_history,
                'samples_seen': self.samples_seen,
                'current_learning_rate': self.current_learning_rate,
                'classes': self.classes_,
                'timestamp': datetime.now().isoformat()
            }
            
            await asyncio.to_thread(joblib.dump, checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            # Continue execution even if checkpoint fails
            pass

    async def save(self, filepath: str):
        """
        Enhanced model saving with compression and metadata.
        """
        try:
            save_data = {
                'model': self.model,
                'performance_history': self.performance_history,
                'samples_seen': self.samples_seen,
                'current_learning_rate': self.current_learning_rate,
                'classes': self.classes_,
                'timestamp': datetime.now().isoformat()
            }
            
            await asyncio.to_thread(joblib.dump, save_data, filepath, compress=3)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    async def load(self, filepath: str):
        """
        Enhanced model loading with validation and error handling.
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file {filepath} not found")

            loaded_data = await asyncio.to_thread(joblib.load, filepath)
            
            # Validate loaded data
            required_keys = {'model', 'performance_history', 'samples_seen', 'current_learning_rate', 'classes'}
            if not all(key in loaded_data for key in required_keys):
                raise ValueError("Loaded model file is missing required data")

            # Update model state
            self.model = loaded_data['model']
            self.performance_history = loaded_data['performance_history']
            self.samples_seen = loaded_data['samples_seen']
            self.current_learning_rate = loaded_data['current_learning_rate']
            self.classes_ = loaded_data['classes']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath} (trained on {self.samples_seen} samples)")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_stats(self) -> Dict:
        """
        Returns current model statistics and performance metrics.
        """
        return {
            'samples_seen': self.samples_seen,
            'current_learning_rate': self.current_learning_rate,
            'performance_history': self.performance_history,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }