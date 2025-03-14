import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """
    Set up logging configuration for the entire application.
    
    Args:
        log_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to the log file. If None, logs will only go to console
        log_format: The format string for log messages
        date_format: The format string for timestamps
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # If log file is specified, create file handler
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler (10MB per file, keep 5 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    # Log initial setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")

def shutdown_logger():
    """Properly shutdown logging system"""
    logging.shutdown()

# Usage example
if __name__ == "__main__":
    setup_logger(
        log_level="INFO",
        log_file="logs/app.log"
    )
    try:
        logger = logging.getLogger(__name__)
        logger.info("System initialized", component="boot", memory=os.getpid())
    finally:
        shutdown_logger()