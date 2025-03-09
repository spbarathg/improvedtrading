from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for AI models. Defines the common interface that all concrete models must implement.
    """

    @abstractmethod
    def train(self, data):
        """
        Trains the model with the provided data.

        :param data: Training data to be used for model training.
        :return: None
        """
        pass

    @abstractmethod
    async def predict(self, features):
        """
        Makes a prediction based on the provided features.

        :param features: The input features for making a prediction.
        :return: Predicted values or signals for trading decisions.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Saves the trained model to the specified filepath.

        :param filepath: Path to save the model.
        :return: None
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Loads a pre-trained model from the specified filepath.

        :param filepath: Path to load the model from.
        :return: None
        """
        pass

    @abstractmethod
    def partial_fit(self, data):
        """
        Performs incremental training on the model using new data.

        :param data: New data for partial training.
        :return: None
        """
        pass