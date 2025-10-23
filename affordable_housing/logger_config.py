"""
Centralized logging configuration for the affordable housing project.
Use this module to set up consistent logging across all scripts.
"""

from datetime import datetime
from pathlib import Path
import sys

from loguru import logger


class LoggerSetup:
    """Centralized logger configuration."""

    def __init__(self, logs_dir: Path = None):
        """
        Initialize logger setup.

        Args:
            logs_dir: Directory to store log files. Defaults to 'logs/' in project root.
        """
        self.logs_dir = logs_dir or Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

    def setup_logger(
        self,
        log_name: str = "app",
        log_level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "30 days",
        console_level: str = "INFO",
    ):
        """
        Configure loguru logger with file and console handlers.

        Args:
            log_name: Name prefix for log file (e.g., 'training', 'preprocessing')
            log_level: Minimum level for file logging (DEBUG, INFO, WARNING, ERROR)
            rotation: When to rotate log file (e.g., "10 MB", "1 day")
            retention: How long to keep old logs (e.g., "30 days", "1 week")
            console_level: Minimum level for console output

        Returns:
            Path to the log file
        """
        # Remove default handler to avoid duplicate logs
        logger.remove()

        # Add console handler with custom format and level
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=console_level,
            colorize=True,
        )

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"{log_name}_{timestamp}.log"

        # Add file handler with detailed format
        logger.add(
            log_file,
            rotation=rotation,
            retention=retention,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,  # Show full traceback on errors
            diagnose=True,  # Show variable values in traceback
        )

        logger.info(f"Logger initialized. Logging to: {log_file}")
        return log_file


def get_logger(
    log_name: str = "app",
    log_level: str = "INFO",
    console_level: str = "INFO",
):
    """
    Convenience function to get a configured logger.

    Args:
        log_name: Name prefix for log file
        log_level: Minimum level for file logging
        console_level: Minimum level for console output

    Returns:
        Tuple of (logger, log_file_path)

    Example:
        from affordable_housing.logger_config import get_logger
        logger, log_file = get_logger("training", log_level="DEBUG")
        logger.info("Starting training...")
    """
    setup = LoggerSetup()
    log_file = setup.setup_logger(
        log_name=log_name,
        log_level=log_level,
        console_level=console_level,
    )
    return logger, log_file


# Pre-configured loggers for common use cases
def setup_training_logger():
    """Get logger configured for training scripts."""
    return get_logger("training", log_level="DEBUG", console_level="INFO")


def setup_preprocessing_logger():
    """Get logger configured for data preprocessing scripts."""
    return get_logger("preprocessing", log_level="INFO", console_level="INFO")


def setup_inference_logger():
    """Get logger configured for inference/prediction scripts."""
    return get_logger("inference", log_level="INFO", console_level="WARNING")


def setup_debug_logger():
    """Get logger configured for debugging with verbose output."""
    return get_logger("debug", log_level="DEBUG", console_level="DEBUG")


if __name__ == "__main__":
    # Test the logger configuration
    test_logger, log_file = get_logger("test")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.success("Logger test completed!")
    print(f"\nLog file created at: {log_file}")
