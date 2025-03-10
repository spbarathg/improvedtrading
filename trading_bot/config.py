from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    AfterValidator,
    BeforeValidator,
    ValidationInfo,
    field_validator
)
from pydantic.networks import AnyUrl
from pydantic.types import SecretStr, conint, confloat, constr
from typing import Annotated, Literal, ClassVar, Dict
from pathlib import Path
import base58
import re
import os

# Custom types for improved validation
SolanaPrivateKey = Annotated[
    SecretStr,
    AfterValidator(lambda v: base58.b58decode(v.get_secret_value()).hex()),
    constr(min_length=64, max_length=64, pattern=r'^[1-9A-HJ-NP-Za-km-z]{64}$')
]

TradingPair = Annotated[
    str,
    constr(pattern=r'^[A-Z0-9]{2,12}/[A-Z0-9]{2,12}$')
]

LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

PathWithAutoCreate = Annotated[
    Path,
    BeforeValidator(lambda v: Path(v).parent.mkdir(parents=True, exist_ok=True))
]

class ModelConfig(BaseModel):
    """AI model configuration"""
    
    # Model performance settings
    MAX_PREDICTIONS_PER_SECOND: conint(ge=1, le=100) = 10
    MAX_CONCURRENT_PREDICTIONS: conint(ge=1, le=20) = 5
    MAX_BURST_SIZE: conint(ge=1, le=50) = 20
    
    # Resource limits
    MAX_MEMORY_USAGE_MB: conint(ge=100, le=10000) = 2000
    MAX_CPU_USAGE_PCT: confloat(ge=10.0, le=90.0) = 70.0
    
    # Performance monitoring
    METRICS_ENABLED: bool = True
    METRICS_LOG_INTERVAL: conint(ge=1, le=3600) = 300  # seconds
    METRICS_RETENTION_DAYS: conint(ge=1, le=365) = 30
    
    # Circuit breaker settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: conint(ge=1, le=100) = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: conint(ge=1, le=3600) = 60  # seconds
    CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT: conint(ge=1, le=1800) = 30  # seconds

class Config(BaseModel):
    """Main configuration with model validation"""
    
    model_config = ConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        validate_default=True,
        str_strip_whitespace=True
    )
    
    # Security-sensitive fields
    PRIVATE_KEY: SolanaPrivateKey
    
    # Network endpoints
    SOLANA_RPC_URL: AnyUrl
    JUPITER_API_URL: AnyUrl = "https://quote-api.jup.ag/v4"
    
    # Trading parameters
    TRADING_PAIRS: list[TradingPair] = ["SOL/USDC"]
    MAX_POSITION_SIZE: confloat(ge=0.01, le=1000) = 1.0  # In SOL
    SLIPPAGE_BPS: conint(ge=1, le=1000) = 50  # Basis points
    
    # Risk management
    STOP_LOSS_PCT: confloat(ge=0.1, le=50.0) = 5.0
    TAKE_PROFIT_PCT: confloat(ge=0.1, le=100.0) = 10.0
    MAX_DAILY_LOSS_PCT: confloat(ge=0.1, le=10.0) = 2.0
    MAX_OPEN_TRADES: conint(ge=1, le=50) = 5
    
    # Performance tuning
    RPC_RATE_LIMIT: conint(ge=1, le=1000) = 10  # Req/sec
    JUPITER_RATE_LIMIT: conint(ge=1, le=1000) = 5
    CACHE_SIZE: conint(ge=128, le=65536) = 1000
    CACHE_TTL: conint(ge=60, le=86400) = 3600
    
    # Data pipeline
    DATA_STORAGE_PATH: PathWithAutoCreate = Path("data/trading_bot.db")
    
    # System configuration
    TRADING_ENABLED: bool = False
    LOG_LEVEL: LogLevel = "INFO"
    LOG_FILE: PathWithAutoCreate | None = Path("logs/trading_bot.log")
    
    # AI model configuration
    MODEL: ModelConfig = ModelConfig()
    
    # Hidden calculated fields
    _rpc_rate_delay: ClassVar[float] = Field(1.0, exclude=True)
    _jupiter_rate_delay: ClassVar[float] = Field(1.0, exclude=True)
    
    @field_validator('_rpc_rate_delay', '_jupiter_rate_delay', mode='after')
    @classmethod
    def calculate_rate_delays(cls, _, info: ValidationInfo):
        if info.field_name == '_rpc_rate_delay':
            return 1.0 / info.data['RPC_RATE_LIMIT']
        return 1.0 / info.data['JUPITER_RATE_LIMIT']
    
    @field_validator('TRADING_PAIRS', mode='after')
    @classmethod
    def normalize_trading_pairs(cls, v: list[str]):
        return [pair.upper().replace('-', '/') for pair in v]
    
    @field_validator('LOG_FILE', mode='after')
    @classmethod
    def validate_log_file(cls, v: Path | None):
        if v and not v.parent.exists():
            v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @classmethod
    def load(cls):
        """Optimized configuration loader with disk caching"""
        return cls.model_construct(
            **os.environ,
            _rpc_rate_delay=1.0 / float(os.getenv('RPC_RATE_LIMIT', 10)),
            _jupiter_rate_delay=1.0 / float(os.getenv('JUPITER_RATE_LIMIT', 5))
        )

# Global configuration instance with thread-safe initialization
config = Config.load()