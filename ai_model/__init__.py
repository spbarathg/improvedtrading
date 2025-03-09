class ModelPredictor:
    def __init__(self, model_selector):
        """
        Initializes the ModelPredictor class with a model selector that determines which model to use.

        :param model_selector: Instance of ModelSelector that chooses the right model (online or periodic).
        """
        self.model_selector = model_selector

    async def predict(self, features):
        """
        Makes predictions based on the provided features using the selected model.

        :param features: The input features from the feature extractor for making a prediction.
        :return: Predicted values or signals for trading decisions.
        """
        # Use the model selector to get predictions from the appropriate model
        predictions = await self.model_selector.predict(features)
        return predictions

    def retrain(self, historical_data):
        """
        Triggers retraining of the models using historical data stored in data storage.

        :param historical_data: Data to use for retraining the models.
        :return: None
        """
        # Retrain the models using the historical data provided
        self.model_selector.retrain(historical_data)