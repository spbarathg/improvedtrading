import logging
import numpy as np
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import asyncio
from typing import Dict, Tuple, List, Any, Optional, Union
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Enhanced model selector with advanced model selection strategies,
    cross-validation, and comprehensive performance tracking.
    """

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.performance_metrics = {}
        self.best_model = None
        self.best_model_score = float('-inf')
        self.cv_splits = getattr(config, 'CV_SPLITS', 5)
        self.validation_window = getattr(config, 'VALIDATION_WINDOW', 1000)
        self.metric_weights = {
            'accuracy': 0.3,
            'precision': 0.3,
            'recall': 0.2,
            'f1': 0.2
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.scaler = StandardScaler()

    async def evaluate_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Comprehensively evaluates a model using multiple metrics and time series cross-validation.
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            model = self.models[model_name]
            metrics = {}

            # Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            
            # Calculate metrics for each fold
            fold_metrics = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Train and predict
                await model.train((X_train_scaled, y_train))
                y_pred = await model.predict(X_val_scaled)
                
                # Calculate metrics
                fold_metric = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1': f1_score(y_val, y_pred, average='weighted')
                }
                fold_metrics.append(fold_metric)

            # Average metrics across folds
            metrics = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            }
            
            # Calculate weighted score
            weighted_score = sum(
                metrics[metric] * weight
                for metric, weight in self.metric_weights.items()
            )
            metrics['weighted_score'] = weighted_score

            # Update performance tracking
            self.performance_metrics[model_name] = metrics
            
            # Update best model if necessary
            if weighted_score > self.best_model_score:
                self.best_model = model
                self.best_model_score = weighted_score
                logger.info(f"New best model: {model_name} with score {weighted_score:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise

    async def select_best_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, float]:
        """
        Selects the best model based on comprehensive evaluation metrics.
        """
        try:
            evaluation_tasks = [
                self.evaluate_model(model_name, X, y)
                for model_name in self.models.keys()
            ]
            
            # Evaluate all models concurrently
            results = await asyncio.gather(*evaluation_tasks)
            
            # Find best model based on weighted score
            best_score = float('-inf')
            best_model_name = None
            
            for model_name, metrics in zip(self.models.keys(), results):
                score = metrics['weighted_score']
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            
            logger.info(f"Best model selected: {best_model_name} with score {best_score:.4f}")
            return best_model_name, best_score

        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            raise

    async def add_model(self, name: str, model) -> None:
        """
        Adds a new model to the selection pool with validation.
        """
        try:
            if name in self.models:
                logger.warning(f"Model {name} already exists. Updating...")
            
            self.models[name] = model
            self.performance_metrics[name] = {}
            logger.info(f"Added model: {name}")

        except Exception as e:
            logger.error(f"Error adding model {name}: {str(e)}")
            raise

    async def remove_model(self, name: str) -> None:
        """
        Removes a model from the selection pool.
        """
        try:
            if name not in self.models:
                raise ValueError(f"Model {name} not found")
            
            del self.models[name]
            del self.performance_metrics[name]
            
            if self.best_model == name:
                self.best_model = None
                self.best_model_score = float('-inf')
            
            logger.info(f"Removed model: {name}")

        except Exception as e:
            logger.error(f"Error removing model {name}: {str(e)}")
            raise

    def get_performance_summary(self) -> pd.DataFrame:
        """
        Returns a detailed performance summary of all models.
        """
        try:
            summary_data = []
            for model_name, metrics in self.performance_metrics.items():
                metrics_copy = metrics.copy()
                metrics_copy['model_name'] = model_name
                summary_data.append(metrics_copy)
            
            if not summary_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(summary_data)
            if not df.empty:
                df = df.set_index('model_name')
                df = df.round(4)
            return df

        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            raise

    async def save_state(self, filepath: str) -> None:
        """
        Saves the complete state of the model selector.
        """
        try:
            state = {
                'models': self.models,
                'performance_metrics': self.performance_metrics,
                'best_model_score': self.best_model_score,
                'metric_weights': self.metric_weights,
                'timestamp': datetime.now().isoformat()
            }
            
            await asyncio.to_thread(joblib.dump, state, filepath, compress=3)
            logger.info(f"Model selector state saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model selector state: {str(e)}")
            raise

    async def load_state(self, filepath: str) -> None:
        """
        Loads the complete state of the model selector with validation.
        """
        try:
            if not filepath.endswith('.joblib'):
                filepath += '.joblib'
            
            state = await asyncio.to_thread(joblib.load, filepath)
            
            # Validate state
            required_keys = {'models', 'performance_metrics', 'best_model_score', 'metric_weights'}
            if not all(key in state for key in required_keys):
                raise ValueError("Invalid state file: missing required data")
            
            self.models = state['models']
            self.performance_metrics = state['performance_metrics']
            self.best_model_score = state['best_model_score']
            self.metric_weights = state['metric_weights']
            
            # Find best model based on score
            for model_name, metrics in self.performance_metrics.items():
                if metrics.get('weighted_score', float('-inf')) == self.best_model_score:
                    self.best_model = self.models[model_name]
                    break
            
            logger.info(f"Model selector state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model selector state: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources."""
        self.thread_pool.shutdown(wait=False)