import logging
from typing import Dict, List, Optional, Deque
from collections import deque
from datetime import datetime, timedelta
import numpy as np
from prometheus_client import Gauge, Histogram, Counter
import json
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelMetricsTracker:
    """Advanced metrics tracking for AI model performance"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_dir = Path("logs/model_metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance windows
        self.short_window = 100
        self.medium_window = 1000
        self.long_window = 10000
        
        # Deques for rolling metrics
        self._init_metric_deques()
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # Periodic metrics logging
        self.last_log_time = datetime.now()
        self.log_interval = timedelta(minutes=5)
    
    def _init_metric_deques(self):
        """Initialize deques for rolling window metrics"""
        self.prediction_accuracy = {
            'short': deque(maxlen=self.short_window),
            'medium': deque(maxlen=self.medium_window),
            'long': deque(maxlen=self.long_window)
        }
        
        self.prediction_latency = {
            'short': deque(maxlen=self.short_window),
            'medium': deque(maxlen=self.medium_window)
        }
        
        self.confidence_scores = deque(maxlen=self.medium_window)
        self.market_conditions = deque(maxlen=self.medium_window)
        self.prediction_distribution = deque(maxlen=self.long_window)
        self.profit_loss = deque(maxlen=self.long_window)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors"""
        self.prom_metrics = {
            'accuracy': {
                'short': Gauge('model_accuracy_short', 'Short-term model accuracy'),
                'medium': Gauge('model_accuracy_medium', 'Medium-term model accuracy'),
                'long': Gauge('model_accuracy_long', 'Long-term model accuracy')
            },
            'latency': Histogram(
                'model_prediction_latency',
                'Model prediction latency in milliseconds',
                buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)
            ),
            'confidence': Gauge('model_confidence', 'Average prediction confidence'),
            'predictions': Counter('model_predictions_total', 'Total number of predictions'),
            'errors': Counter('model_errors_total', 'Total number of prediction errors'),
            'profit_loss': Gauge('model_profit_loss', 'Cumulative profit/loss')
        }
    
    async def log_prediction(self, prediction: Dict, latency_ms: float):
        """Log a new prediction with its metadata"""
        try:
            # Update prediction metrics
            self.prediction_distribution.append(prediction['signal'])
            self.confidence_scores.append(prediction['confidence'])
            self.prediction_latency['short'].append(latency_ms)
            self.prediction_latency['medium'].append(latency_ms)
            
            # Update Prometheus metrics
            self.prom_metrics['latency'].observe(latency_ms)
            self.prom_metrics['confidence'].set(np.mean(self.confidence_scores))
            self.prom_metrics['predictions'].inc()
            
            # Periodic logging
            await self._periodic_log()
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            self.prom_metrics['errors'].inc()
    
    async def log_outcome(self, prediction: Dict, actual_outcome: int, profit_loss: float):
        """Log the actual outcome and update accuracy metrics"""
        try:
            # Calculate accuracy (1 if correct, 0 if incorrect)
            accuracy = 1 if prediction['signal'] == actual_outcome else 0
            
            # Update accuracy deques
            self.prediction_accuracy['short'].append(accuracy)
            self.prediction_accuracy['medium'].append(accuracy)
            self.prediction_accuracy['long'].append(accuracy)
            
            # Update profit/loss tracking
            self.profit_loss.append(profit_loss)
            
            # Update Prometheus metrics
            for window in ['short', 'medium', 'long']:
                acc = np.mean(self.prediction_accuracy[window])
                self.prom_metrics['accuracy'][window].set(acc)
            
            self.prom_metrics['profit_loss'].set(sum(self.profit_loss))
            
            # Periodic logging
            await self._periodic_log()
            
        except Exception as e:
            logger.error(f"Error logging outcome: {e}")
            self.prom_metrics['errors'].inc()
    
    async def log_market_context(self, market_data: Dict):
        """Log market conditions during prediction"""
        try:
            context = {
                'timestamp': datetime.now().isoformat(),
                'volatility': market_data.get('volatility'),
                'volume': market_data.get('volume'),
                'trend': market_data.get('trend')
            }
            self.market_conditions.append(context)
            
            # Periodic logging
            await self._periodic_log()
            
        except Exception as e:
            logger.error(f"Error logging market context: {e}")
    
    async def _periodic_log(self):
        """Periodically save detailed metrics to disk"""
        now = datetime.now()
        if now - self.last_log_time >= self.log_interval:
            try:
                metrics = {
                    'timestamp': now.isoformat(),
                    'accuracy': {
                        'short': float(np.mean(self.prediction_accuracy['short'])),
                        'medium': float(np.mean(self.prediction_accuracy['medium'])),
                        'long': float(np.mean(self.prediction_accuracy['long']))
                    },
                    'latency': {
                        'mean': float(np.mean(self.prediction_latency['short'])),
                        'p95': float(np.percentile(list(self.prediction_latency['medium']), 95))
                    },
                    'confidence': {
                        'mean': float(np.mean(self.confidence_scores)),
                        'std': float(np.std(self.confidence_scores))
                    },
                    'profit_loss': float(sum(self.profit_loss)),
                    'prediction_distribution': self._get_prediction_distribution()
                }
                
                # Save to file
                filename = self.metrics_dir / f"metrics_{now.strftime('%Y%m%d_%H%M%S')}.json"
                async with aiofiles.open(filename, 'w') as f:
                    await f.write(json.dumps(metrics, indent=2))
                
                self.last_log_time = now
                logger.info(f"Saved detailed metrics to {filename}")
                
            except Exception as e:
                logger.error(f"Error in periodic logging: {e}")
    
    def _get_prediction_distribution(self) -> Dict[str, float]:
        """Calculate distribution of predictions"""
        if not self.prediction_distribution:
            return {}
        
        total = len(self.prediction_distribution)
        return {
            str(signal): count/total 
            for signal, count in zip(*np.unique(list(self.prediction_distribution), return_counts=True))
        }
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics summary"""
        return {
            'accuracy': {
                'short_term': float(np.mean(self.prediction_accuracy['short'])),
                'medium_term': float(np.mean(self.prediction_accuracy['medium'])),
                'long_term': float(np.mean(self.prediction_accuracy['long']))
            },
            'latency': {
                'current': float(np.mean(list(self.prediction_latency['short'])[-10:])),
                'mean': float(np.mean(self.prediction_latency['medium']))
            },
            'confidence': {
                'current': float(np.mean(list(self.confidence_scores)[-10:])),
                'mean': float(np.mean(self.confidence_scores))
            },
            'profit_loss': float(sum(self.profit_loss)),
            'total_predictions': len(self.prediction_distribution)
        } 