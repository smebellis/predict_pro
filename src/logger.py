import logging
from logging.handlers import RotatingFileHandler
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import argparse

import pandas as pd

# from src.utils.helper import load_config


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    config_file = Path(config_path).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            print(f"Configuration loaded successfully from '{config_file}'.")
            return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def setup_logging(config: Dict) -> None:
    """
    Set up logging based on the provided configuration dictionary.

    Args:
        config (dict): Configuration dictionary loaded from YAML.
    """

    logging_config = config.get("logging", {})
    log_dir = logging_config.get("log_dir", "logs")
    log_file = logging_config.get("log_file", "app.log")
    log_level_str = logging_config.get("log_level", "INFO").upper()

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Define the full path to the log file
    log_path = os.path.join(log_dir, log_file)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all messages

    # Check if handlers are already set up to prevent duplicates
    if not root_logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler with Rotation
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Stream Handler for stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # Add handlers to the root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)

        root_logger.info(f"Logging initialized. Logs are being saved to {log_path}")


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)


_config = load_config()
setup_logging(_config)

# def setup_logging(config: Dict, name: str = "APPLICATION") -> logging.Logger:
#     """
#     Sets up logging for the application.

#     Parameters
#     ----------
#     config : Dict
#         Configuration dictionary containing logging settings.
#     name : str, optional
#         Name of the logger (default is 'APPLICATION').

#     Returns
#     -------
#     logger : logging.Logger
#         Configured logger instance.
#     """

#     logging_config = config.get("logging", {})
#     log_dir = logging_config.get("log_dir", "logs")
#     log_file = logging_config.get("log_file", "app.log")
#     log_level_str = logging_config.get("log_level", "INFO").upper()

#     # Ensure the log directory exists
#     os.makedirs(log_dir, exist_ok=True)

#     # Define the full path to the log file
#     log_path = os.path.join(log_dir, log_file)

#     # Create or get the logger
#     logger = logging.getLogger(name.upper())
#     logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all messages

#     # Append current date and time to the log file name
#     current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_filename = (
#         f"{os.path.splitext(log_file)[0]}_{current_time}{os.path.splitext(log_file)[1]}"
#     )
#     log_path = os.path.join(log_dir, log_filename)

#     logger = logging.getLogger(name.upper())
#     logger.setLevel(logging.DEBUG)
#     logger.info(f"Saving logs to {log_dir}")
#     if not logger.handlers:
#         formatter = logging.Formatter(
#             "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#         )

#         # File Handler with Rotation
#         file_handler = logging.handlers.RotatingFileHandler(
#             log_path,
#             maxBytes=5 * 1024 * 1024,  # 5 MB
#             backupCount=5,
#         )
#         file_handler.setLevel(logging.DEBUG)
#         file_handler.setFormatter(formatter)

#         # Stream Handler for stdout
#         stream_handler = logging.StreamHandler()
#         stream_handler.setLevel(logging.INFO)
#         stream_handler.setFormatter(formatter)

#         logger.addHandler(file_handler)
#         logger.addHandler(stream_handler)

#     logger.debug(f"Logger initialized and handlers added for {name}.")
#     return logger


# def get_logger(name: str = __name__) -> logging.Logger:
#     """
#     Get a logger with the specified name.

#     Args:
#         name (str): Name of the logger.

#     Returns:
#         logging.Logger: Configured logger instance.
#     """
#     config = load_config()  # Assumes config.yaml is in the current working directory
#     return setup_logging(config, name)
