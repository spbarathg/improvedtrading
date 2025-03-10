from pydantic import BaseModel, Field, ConfigDict, validator
from typing import Literal, Dict, Optional, List
from pathlib import Path

class MarketDataSchema(BaseModel):
    """Schema for market data validation."""
    price: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    timestamp: int = Field(...)
    bid: float = Field(..., gt=0)
    ask: float = Field(..., gt=0)
    symbol: str = Field(...)
    depth_asks: List[List[float]] = Field(default_factory=list)
    depth_bids: List[List[float]] = Field(default_factory=list)
    
    @validator('ask')
    def ask_greater_than_bid(cls, v, values):
        if 'bid' in values and v <= values['bid']:
            raise ValueError('Ask price must be greater than bid price')
        return v

class ModelFeatureConfig(BaseModel):
    """Configuration for model features."""
    feature_names: List[str] = Field(
        default=[
            'price_change',
            'volume_change',
            'bid_ask_spread',
            'order_imbalance',
            'volatility',
            'volume_price_correlation',
            'rsi',
            'macd',
            'total_bid_volume',
            'total_ask_volume'
        ]
    )
    required_fields: List[str] = Field(
        default=['price', 'volume', 'timestamp', 'bid', 'ask']
    )
    technical_indicators: List[str] = Field(
        default=['rsi', 'macd', 'bollinger_bands']
    )
    feature_scaling: Dict[str, Dict] = Field(
        default={
            'price_change': {'min': -100, 'max': 100},
            'volume_change': {'min': -100, 'max': 100},
            'bid_ask_spread': {'min': 0, 'max': 0.1},
            'volatility': {'min': 0, 'max': 1}
        }
    )

class AIModelConfig(BaseModel):
    model_config = ConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        validate_default=True
    )

    # Model selection
    MODEL_TYPE: Literal['SGD', 'PA'] = 'SGD'
    BATCH_SIZE: int = Field(32, ge=1, le=1024)
    LEARNING_RATE: float = Field(0.01, ge=0.0001, le=1.0)
    LEARNING_RATE_SCHEDULE: Literal['optimal', 'adaptive', 'inverse_scaling'] = 'adaptive'
    MIN_LEARNING_RATE: float = Field(1e-6, ge=1e-10, le=0.1)
    LEARNING_RATE_DECAY: float = Field(0.1, ge=0.01, le=1.0)

    # Model checkpointing
    CHECKPOINT_INTERVAL: int = Field(1000, ge=100, le=10000)
    MODEL_DIR: Path = Field(Path("models/checkpoints"))
    
    # Cross-validation
    CV_SPLITS: int = Field(5, ge=2, le=10)
    VALIDATION_WINDOW: int = Field(1000, ge=100, le=10000)
    
    # Feature preprocessing
    SCALING_METHOD: Literal['standard', 'minmax', None] = 'standard'
    NORMALIZATION: Literal['l1', 'l2', None] = 'l2'
    FEATURE_CACHE_SIZE: int = Field(1000, ge=100, le=10000)
    
    # Performance metrics weights
    METRIC_WEIGHTS: Dict[str, float] = Field(
        default={
            'accuracy': 0.3,
            'precision': 0.3,
            'recall': 0.2,
            'f1': 0.2
        }
    )

    # Resource management
    MAX_WORKERS: int = Field(4, ge=1, le=16)
    MEMORY_LIMIT_MB: Optional[int] = Field(None, ge=128, le=32768)
    
    # Trading specific configurations
    MIN_CONFIDENCE_THRESHOLD: float = Field(0.6, ge=0.5, le=1.0)
    MAX_POSITION_HOLD_TIME: int = Field(3600, ge=60, le=86400)  # in seconds
    MARKET_HOURS: Dict[str, Dict[str, int]] = Field(
        default={
            'default': {'start': 8, 'end': 22},
            'weekend': {'start': 10, 'end': 20}
        }
    )
    
    # Feature configuration
    FEATURE_CONFIG: ModelFeatureConfig = Field(default_factory=ModelFeatureConfig)
    
    # Market data validation
    MARKET_DATA_SCHEMA: MarketDataSchema = Field(default_factory=MarketDataSchema)

    def __init__(self, **data):
        super().__init__(**data)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        """Returns the path for saving/loading the model."""
        return self.MODEL_DIR / f"{self.MODEL_TYPE.lower()}_model.joblib"

    @property
    def checkpoint_path(self) -> Path:
        """Returns the base path for model checkpoints."""
        return self.MODEL_DIR / "checkpoints"

    @validator('METRIC_WEIGHTS')
    def validate_metric_weights(cls, v):
        if not abs(sum(v.values()) - 1.0) < 1e-6:
            raise ValueError("Metric weights must sum to 1.0")
        return v

    def validate_market_data(self, data: Dict) -> bool:
        """Validate market data against schema."""
        try:
            MarketDataSchema(**data)
            return True
        except Exception as e:
            logger.error(f"Market data validation failed: {e}")
            return False

    def get_feature_scaling_params(self, feature_name: str) -> Dict:
        """Get scaling parameters for a feature."""
        return self.FEATURE_CONFIG.feature_scaling.get(
            feature_name,
            {'min': -1, 'max': 1}  # Default scaling
        )

    class Config:
        validate_assignment = True
        json_encoders = {
            Path: str
        }

# Global configuration instance
config = AIModelConfig()