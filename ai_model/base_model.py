import logging
from typing import Dict, Optional, Any
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, config):
        """Initialize base model with configuration."""
        self.config = config
        self.model = None
        self.initialized = False
        logger.info("Base model initialized")

    def load_model(self, model_path: str) -> bool:
        """
        Load model from disk.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            self.model = joblib.load(model_path)
            self.initialized = True
            logger.info(
                "Model loaded successfully",
                extra={'model_path': model_path}
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to load model",
                exc_info=True,
                extra={
                    'error': str(e),
                    'model_path': model_path
                }
            )
            return False

    def save_model(self, model_path: str) -> bool:
        """
        Save model to disk.
        
        Args:
            model_path: Path to save model
            
        Returns:
            bool: Whether saving was successful
        """
        try:
            if not self.initialized:
                logger.error("Cannot save uninitialized model")
                return False
                
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, model_path)
            logger.info(
                "Model saved successfully",
                extra={'model_path': model_path}
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to save model",
                exc_info=True,
                extra={
                    'error': str(e),
                    'model_path': model_path
                }
            )
            return False

    def predict(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Make predictions using the model.
        
        Args:
            features: Input features
            
        Returns:
            np.ndarray: Predictions or None if failed
        """
        try:
            if not self.initialized:
                logger.error("Cannot predict with uninitialized model")
                return None
                
            predictions = self.model.predict(features)
            logger.debug(
                "Generated predictions",
                extra={
                    'input_shape': features.shape,
                    'output_shape': predictions.shape
                }
            )
            return predictions
        except Exception as e:
            logger.error(
                "Error making predictions",
                exc_info=True,
                extra={
                    'error': str(e),
                    'input_shape': features.shape
                }
            )
            return None

    def train(self, features: np.ndarray, labels: np.ndarray) -> bool:
        """
        Train the model.
        
        Args:
            features: Training features
            labels: Training labels
            
        Returns:
            bool: Whether training was successful
        """
        try:
            if not self.initialized:
                logger.error("Cannot train uninitialized model")
                return False
                
            self.model.fit(features, labels)
            logger.info(
                "Model training completed",
                extra={
                    'input_shape': features.shape,
                    'labels_shape': labels.shape
                }
            )
            return True
        except Exception as e:
            logger.error(
                "Error training model",
                exc_info=True,
                extra={
                    'error': str(e),
                    'input_shape': features.shape,
                    'labels_shape': labels.shape
                }
            )
            return False

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            features: Test features
            labels: Test labels
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        try:
            if not self.initialized:
                logger.error("Cannot evaluate uninitialized model")
                return {}
                
            predictions = self.predict(features)
            if predictions is None:
                return {}
                
            metrics = {
                'accuracy': float(np.mean(predictions == labels)),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                "Model evaluation completed",
                extra={
                    'metrics': metrics,
                    'input_shape': features.shape
                }
            )
            return metrics
        except Exception as e:
            logger.error(
                "Error evaluating model",
                exc_info=True,
                extra={
                    'error': str(e),
                    'input_shape': features.shape
                }
            )
            return {}