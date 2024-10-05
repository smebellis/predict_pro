import logging
import os
from datetime import datetime
from typing import Union
from pathlib import Path
import pandas as pd
import tqdm


def setup_logging(
    log_dir: str = "logs", log_file: str = "data_preprocessor.log"
) -> logging.Logger:
    """
    Sets up logging for the application.

    Parameters:
    ----------
    log_dir : str
        Directory to store log files.
    log_file : str
        Base name for the log file.

    Returns:
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Append current date and time to the log file name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = (
        f"{os.path.splitext(log_file)[0]}_{current_time}{os.path.splitext(log_file)[1]}"
    )
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("YourProjectLogger")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File Handler with Rotation
        file_handler = logging.handlers.RotatingFileHandler(
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

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.debug("Logger initialized and handlers added.")
    return logger


def file_load(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a file into a pandas DataFrame.

    Parameters:
        file_path (Union[str, Path]): The path to the file to be loaded.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
        pd.errors.ParserError: If there's an error parsing the file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logging.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(
            f"File does not exist, check the path and try again with the correct path: {file_path}"
        )

    try:
        file_extension = file_path.suffix.lower()
        if file_extension == ".csv":
            logging.info(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            logging.info(f"Reading Excel file: {file_path}")
            df = pd.read_excel(file_path)
        elif file_extension == ".json":
            logging.info(f"Reading JSON file: {file_path}")
            df = pd.read_json(file_path)
        else:
            logging.error(f"Unsupported file format: {file_extension}")
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported formats are CSV, Excel, and JSON."
            )
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing the file: {file_path} - {e}")
        raise
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while reading the file: {file_path} - {e}"
        )
        raise

    return df
