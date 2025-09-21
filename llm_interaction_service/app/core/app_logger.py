import logging
import logging.config
import os

# Define the logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s] [%(module)s:%(lineno)d] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": os.path.join("logs", "app.log"),
            "formatter": "detailed",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "app": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "WARNING",
    },
}

# Apply the logging configuration
def setup_logging():
    os.makedirs("logs", exist_ok=True)  # Ensure the logs directory exists
    logging.config.dictConfig(LOGGING_CONFIG)