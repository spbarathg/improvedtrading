class ModelConfig:
    """
    Configuration class containing constants and hyperparameters for AI model training, evaluation, 
    and prediction processes.
    """

    # General Model Settings
    RANDOM_SEED = 42  # Seed for reproducibility
    VALIDATION_SPLIT = 0.2  # Proportion of data to be used as validation set
    BATCH_SIZE = 64  # Batch size for online training
    EPOCHS = 100  # Number of epochs for training periodic models
    EARLY_STOPPING_PATIENCE = 10  # Early stopping patience for periodic model training

    # Online Model Configurations
    ONLINE_MODEL_CLASS = "SGDClassifier"  # Class name for the online model
    ONLINE_LEARNING_RATE = 0.01  # Learning rate for online model
    ONLINE_SCALER_TYPE = "StandardScaler"  # Scaler type for online feature scaling
    ONLINE_MODEL_SAVE_PATH = "models/online_model.pkl"  # Path to save the online model

    # Periodic Model Configurations
    PERIODIC_MODEL_CLASS = "RandomForestClassifier"  # Class name for the periodic model
    N_ESTIMATORS = 100  # Number of trees in the Random Forest
    MAX_DEPTH = None  # Maximum depth of trees in Random Forest
    PERIODIC_MODEL_SAVE_PATH = "models/periodic_model.pkl"  # Path to save the periodic model

    # Model Selector Configurations
    MODEL_EVALUATION_INTERVAL = 3600  # Interval (in seconds) for model performance evaluation
    MODEL_SELECTION_CRITERIA = "f1_score"  # Criteria used for dynamic model switching

    # Data Preprocessing
    FEATURE_SCALING = True  # Whether to apply feature scaling
    SCALER_TYPE = "StandardScaler"  # Type of scaler used for feature scaling

    # Training Intervals
    ONLINE_TRAINING_INTERVAL = 60  # Interval (in seconds) between each online model update
    PERIODIC_TRAINING_INTERVAL = 86400  # Interval (in seconds) for periodic model retraining

    # Data Settings
    TRAINING_WINDOW_DAYS = 7  # Number of days to use for periodic training (rolling window)
    MINI_BATCH_SIZE = 32  # Mini-batch size for online training

    # Logging
    LOGGING_LEVEL = "INFO"  # Logging level for model training

    # Miscellaneous
    MAX_RETRIES = 3  # Maximum number of retries for any failed operation

    # Model Performance Thresholds
    MIN_ACCURACY_THRESHOLD = 0.6  # Minimum accuracy required for model deployment
    MIN_F1_THRESHOLD = 0.65  # Minimum F1 score required for model deployment
    DRIFT_DETECTION_THRESHOLD = 0.1  # Maximum allowed drift before model retraining

    # Cross-validation Settings
    N_SPLITS = 5  # Number of folds for cross-validation
    CV_SCORING = ['accuracy', 'f1', 'precision', 'recall']  # Metrics for cross-validation

    # Feature Engineering
    FEATURE_SELECTION_METHOD = "recursive"  # Method for feature selection
    MAX_FEATURES = 20  # Maximum number of features to select
    FEATURE_IMPORTANCE_THRESHOLD = 0.01  # Minimum importance threshold for feature selection

    # Model Versioning
    MODEL_VERSION_FORMAT = "v{major}.{minor}.{patch}"  # Format for model versioning
    AUTO_VERSION_INCREMENT = True  # Whether to automatically increment version on training
    MODEL_REGISTRY_PATH = "models/registry"  # Path to store model versions

    # Monitoring Configurations
    ENABLE_MONITORING = True  # Whether to enable model monitoring
    MONITORING_METRICS = [  # List of metrics to monitor
        "accuracy",
        "latency",
        "prediction_drift",
        "feature_drift"
    ]
    MONITORING_INTERVAL = 300  # Interval (in seconds) for monitoring metrics collection
    ALERT_THRESHOLD = 0.2  # Threshold for monitoring alerts