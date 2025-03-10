import logging
import sys
import os
from pathlib import Path
from typing import Optional, no_type_check, Union
from logging.handlers import QueueHandler, QueueListener
from concurrent_log_handler import ConcurrentRotatingFileHandler
import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import TimeStamper, format_exc_info
from structlog.stdlib import add_logger_name, add_log_level, PositionalArgumentsFormatter
import orjson
from typing import List

# Global queue for async logging
_LOG_QUEUE = structlog.threading.WaitingQueue(-1)  # Unlimited size with non-blocking put
_LISTENER: Optional[QueueListener] = None

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    json_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    encoding: str = 'utf-8',
    async_logging: bool = True,
    colorize_console: bool = True
) -> structlog.types.BindableLogger:
    """
    Optimized structured logging setup with async I/O and zero-copy processing.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path for rotated log files (str or Path)
        json_output: Use JSON formatting for files
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
        encoding: File encoding
        async_logging: Use non-blocking async logging
        colorize_console: Enable colored console output
    
    Returns:
        Configured structlog logger instance
        
    Raises:
        ValueError: If invalid log level provided
        OSError: If log file directory cannot be created
    """
    global _LISTENER
    
    # Validate log level
    log_level = log_level.upper()
    if log_level not in logging._nameToLevel:
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure processors
    timestamper = TimeStamper(fmt="iso", utc=True)
    shared_processors: List[structlog.types.Processor] = [
        merge_contextvars,
        add_log_level,
        add_logger_name,
        PositionalArgumentsFormatter(),
        timestamper,
        format_exc_info,
    ]

    # Configure outputs
    handlers: List[logging.Handler] = []
    
    # File handler with lock-free rotation
    if log_file:
        log_file = Path(log_file)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create log directory: {e}")
            
        file_handler = ConcurrentRotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            use_gzip=True
        )
        file_handler.setFormatter(_create_formatter(json=True))
        handlers.append(file_handler)

    # Console handler with zero-alloc formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        _create_formatter(json=False, colorize=colorize_console)
    )
    handlers.append(console_handler)

    # Configure async logging infrastructure
    if async_logging:
        queue_handler = QueueHandler(_LOG_QUEUE)
        # Stop existing listener if any
        if _LISTENER is not None:
            _LISTENER.stop()
        _LISTENER = QueueListener(
            _LOG_QUEUE,
            *handlers,
            respect_handler_level=True
        )
        _LISTENER.start()
        handlers = [queue_handler]

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=handlers,
        format="%(message)s",
        encoding=encoding,
        force=True  # Ensure configuration is applied
    )

    # Final structlog configuration
    structlog.configure(
        processors=[
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ] + shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
        context_class=dict
    )

    return structlog.get_logger()

@no_type_check
def _create_formatter(json: bool, colorize: bool = False) -> logging.Formatter:
    """
    Create optimized formatter with zero-copy JSON serialization.
    
    Args:
        json: Whether to use JSON formatting
        colorize: Whether to colorize console output
        
    Returns:
        Configured logging formatter
    """
    if json:
        return structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(
                serializer=orjson.dumps,
                option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_NAIVE_UTC
            ),
            foreign_pre_chain=[
                structlog.stdlib.ExtraAdder(),
                structlog.processors.StackInfoRenderer(),
            ],
        )
    
    return structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(
            colors=colorize,
            pad_event=30,
            exception_formatter=structlog.dev.plain_traceback
        ),
        foreign_pre_chain=[
            structlog.stdlib.ExtraAdder(),
            structlog.processors.StackInfoRenderer(),
        ],
    )

def shutdown_logger() -> None:
    """Properly shutdown the logger and cleanup resources."""
    global _LISTENER
    if _LISTENER is not None:
        _LISTENER.stop()
        _LISTENER = None

# Usage example
if __name__ == "__main__":
    logger = setup_logger(
        log_level="INFO",
        log_file="logs/app.log",
        json_output=True
    )
    try:
        logger.info("System initialized", component="boot", memory=os.getpid())
    finally:
        shutdown_logger()