import logging
import numpy as np
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import asyncio
from typing import Dict, Tuple, List, Any

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Handles the dynamic selection between online and periodic models based on their performance.
    Uses a validation set to evaluate and compare model performance using multiple metrics.
    """

    def __init__(self, 
                 performance_window: int = 1000,
                 selection_threshold: float = 0.05,
                 metrics_weights: Dict[str, float] = None):
        """
        Initialize the ModelSelector with configuration parameters.
        
        Args:
            performance_window: Number of recent predictions to consider for performance evaluation
            selection_threshold: Minimum performance difference to switch models
            metrics_weights: Dictionary of metric names and their weights in the final score
        """
        self.performance_window = performance_window
        self.selection_threshold = selection_threshold
        
        # Default weights for different metrics if none provided
        self.metrics_weights = metrics_weights or {
            'accuracy': 0.3,
            'precision': 0.3,
            'recall': 0.2,
            'f1': 0.2
        }
        
        # Performance history for both models
        self.online_performance_history = []
        self.periodic_performance_history = []
        
        # Currently selected model type
        self.current_model = 'periodic'  # Start with periodic as default
        
    async def evaluate_models(self, 
                            validation_features: np.ndarray,
                            validation_labels: np.ndarray,
                            online_predictions: np.ndarray,
                            periodic_predictions: np.ndarray) -> str:
        """
        Evaluate both models on validation data and select the better performing one.
        
        Args:
            validation_features: Features from validation set
            validation_labels: True labels from validation set
            online_predictions: Predictions from online model
            periodic_predictions: Predictions from periodic model
            
        Returns:
            str: Selected model type ('online' or 'periodic')
        """
        # Calculate performance metrics for both models
        online_metrics = await self._calculate_metrics(validation_labels, online_predictions)
        periodic_metrics = await self._calculate_metrics(validation_labels, periodic_predictions)
        
        # Calculate weighted scores
        online_score = self._calculate_weighted_score(online_metrics)
        periodic_score = self._calculate_weighted_score(periodic_metrics)
        
        # Update performance history
        self.online_performance_history.append(online_score)
        self.periodic_performance_history.append(periodic_score)
        
        # Maintain window size
        if len(self.online_performance_history) > self.performance_window:
            self.online_performance_history.pop(0)
            self.periodic_performance_history.pop(0)
        
        # Calculate average performance over the window
        avg_online = np.mean(self.online_performance_history)
        avg_periodic = np.mean(self.periodic_performance_history)
        
        # Determine if we should switch models
        if abs(avg_online - avg_periodic) > self.selection_threshold:
            self.current_model = 'online' if avg_online > avg_periodic else 'periodic'
            
        return self.current_model
    
    async def _calculate_metrics(self, 
                               true_labels: np.ndarray, 
                               predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics asynchronously.
        
        Args:
            true_labels: Array of true labels
            predictions: Array of model predictions
            
        Returns:
            Dictionary containing different performance metrics
        """
        metrics = {}
        
        # Use asyncio.to_thread for CPU-bound operations
        metrics['accuracy'] = await asyncio.to_thread(
            accuracy_score, true_labels, predictions
        )
        metrics['precision'] = await asyncio.to_thread(
            precision_score, true_labels, predictions, average='weighted'
        )
        metrics['recall'] = await asyncio.to_thread(
            recall_score, true_labels, predictions, average='weighted'
        )
        metrics['f1'] = await asyncio.to_thread(
            f1_score, true_labels, predictions, average='weighted'
        )
        
        return metrics
    
    def _calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate a weighted score from multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and their values
            
        Returns:
            float: Weighted performance score
        """
        return sum(
            metrics[metric] * weight 
            for metric, weight in self.metrics_weights.items()
        )
    
    def get_current_model(self) -> str:
        """
        Get the currently selected model type.
        
        Returns:
            str: Current model type ('online' or 'periodic')
        """
        return self.current_model
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent performance for both models.
        
        Returns:
            Dictionary containing performance statistics for both models
        """
        return {
            'online': {
                'current_score': self.online_performance_history[-1] if self.online_performance_history else None,
                'average_score': np.mean(self.online_performance_history) if self.online_performance_history else None,
                'trend': np.gradient(self.online_performance_history).tolist() if len(self.online_performance_history) > 1 else None
            },
            'periodic': {
                'current_score': self.periodic_performance_history[-1] if self.periodic_performance_history else None,
                'average_score': np.mean(self.periodic_performance_history) if self.periodic_performance_history else None,
                'trend': np.gradient(self.periodic_performance_history).tolist() if len(self.periodic_performance_history) > 1 else None
            },
            'current_model': self.current_model
        }