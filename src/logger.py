import logging

# Create a logger instance
logger = logging.getLogger(__name__)


def configure_logging(log_file=None, log_level=logging.DEBUG):
    # Configure the logging module with file name and line number
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
        level=log_level,
    )

    # Add a file handler if a log file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
