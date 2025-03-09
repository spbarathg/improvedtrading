import logging
import numpy as np

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Selects between the online and periodic models based on their performance.
    Implements dynamic model switching or weighted averaging.
    """

    def __init__(self, online_model, periodic_model, config):
        """
        Initializes the ModelSelector.
        :param online_model: Instance of the online learning model.
        :param periodic_model: Instance of the periodically retrained model.
        :param config: Configuration settings (e.g., evaluation intervals, performance threshold).
        """
        self.online_model = online_model
        self.periodic_model = periodic_model
        self.config = config
        self.current_model = None

        # Performance tracking (accuracy, or another relevant metric)
        self.online_performance = None
        self.periodic_performance = None

    def evaluate_model(self, model, validation_data):
        """
        Evaluates a model's performance on the validation set.
        :param model: The model to evaluate.
        :param validation_data: A tuple (X_val, y_val) of validation features and labels.
        :return: The performance metric (accuracy or another relevant metric).
        """
        X_val, y_val = validation_data

        # Ensure the model is trained before evaluation
        if not model.is_trained:
            logger.warning(f"{model.__class__.__name__} is not trained. Skipping evaluation.")
            return 0

        # Get predictions and calculate accuracy or other performance metrics
        predictions = model.model.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        return accuracy

    def get_current_model(self):
        """
        Returns the currently selected model (either online or periodic) based on performance.
        If dynamic model switching is enabled, it compares performance and selects the best one.
        """
        if self.config.DYNAMIC_MODEL_SWITCHING:
            if self.online_performance is None or self.periodic_performance is None:
                logger.info("Model performance has not been evaluated yet. Selecting online model by default.")
                self.current_model = self.online_model
            else:
                # Compare performance and select the model with the better score
                if self.online_performance >= self.periodic_performance:
                    logger.info("Online model selected based on better performance.")
                    self.current_model = self.online_model
                else:
                    logger.info("Periodic model selected based on better performance.")
                    self.current_model = self.periodic_model
        else:
            logger.info("Dynamic model switching disabled. Using online model by default.")
            self.current_model = self.online_model

        return self.current_model

    def update_model_performance(self, validation_data):
        """
        Evaluates both models on the validation set at regular intervals and updates their performance.
        :param validation_data: A tuple (X_val, y_val) of validation features and labels.
        """
        self.online_performance = self.evaluate_model(self.online_model, validation_data)
        self.periodic_performance = self.evaluate_model(self.periodic_model, validation_data)

        logger.info(f"Online model performance: {self.online_performance}")
        logger.info(f"Periodic model performance: {self.periodic_performance}")

    async def predict(self, features):
        """
        Uses the currently selected model to make predictions.
        :param features: The input features for prediction.
        :return: The prediction from the selected model.
        """
        current_model = self.get_current_model()
        prediction = await current_model.predict(features)
        return prediction