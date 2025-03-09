import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class AIUtils:
    """
    A collection of utility functions used in AI model training and evaluation.
    These include functions for calculating performance metrics, handling predictions, and more.
    """

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculates standard performance metrics for classification models:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        
        :param y_true: Array of true labels.
        :param y_pred: Array of predicted labels.
        :return: Dictionary containing calculated metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1_score': f1_score(y_true, y_pred, average='macro')
        }
        return metrics

    @staticmethod
    def weighted_average_predictions(predictions, weights):
        """
        Computes a weighted average of predictions (useful for model ensembling).
        
        :param predictions: List of prediction arrays (from different models).
        :param weights: List of weights to apply to each prediction array.
        :return: Final weighted prediction array.
        """
        if len(predictions) != len(weights):
            raise ValueError("The number of predictions and weights must match.")
        
        # Normalize the weights so they sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Compute the weighted average of predictions
        weighted_preds = np.average(predictions, axis=0, weights=weights)
        
        return np.round(weighted_preds).astype(int)

    @staticmethod
    def calculate_moving_average(data, window_size):
        """
        Calculates the moving average over a rolling window of data.
        
        :param data: Array or list of numerical values.
        :param window_size: Size of the moving window.
        :return: Array of moving averages.
        """
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    @staticmethod
    def split_train_test(data, labels, test_size=0.2, random_seed=None):
        """
        Splits the dataset into training and testing sets.
        
        :param data: Array or DataFrame of features.
        :param labels: Array or list of labels.
        :param test_size: Proportion of the data to be used as test data (default is 0.2).
        :param random_seed: Optional random seed for reproducibility.
        :return: Tuple of (X_train, X_test, y_train, y_test)
        """
        if random_seed:
            np.random.seed(random_seed)
        
        # Shuffle the data
        indices = np.random.permutation(len(data))
        test_size = int(len(data) * test_size)
        
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]

    @staticmethod
    def calculate_cumulative_returns(prices):
        """
        Calculates cumulative returns from a series of prices.
        
        :param prices: Array or list of asset prices.
        :return: Array of cumulative returns.
        """
        returns = np.diff(prices) / prices[:-1]
        cumulative_returns = np.cumsum(returns)
        return cumulative_returns