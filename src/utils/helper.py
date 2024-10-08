import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import argparse

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

    logger = logging.getLogger("Predict Pro Logger")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
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


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict Pro Software to detect Patterns"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=False,
        help="Path to the data input CSV file. This is the original unformatted data",
    )
    parser.add_argument(
        "--output", "-o", required=False, help="Path to save the processed CSV file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    return parser.parse_args()


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


def file_load_large_csv(
    file_path: Union[str, Path],
    chunksize: int = 100000,
    **read_csv_kwargs: Optional[Dict],
) -> pd.DataFrame:
    """
    Loads a large CSV file into a pandas DataFrame by reading it in chunks.

    Parameters:
        file_path (Union[str, Path]): The path to the CSV file to be loaded.
        chunksize (int, optional): Number of rows per chunk. Default is 100,000.
        **read_csv_kwargs: Additional keyword arguments to pass to pandas.read_csv.

    Returns:
        pd.DataFrame: The concatenated DataFrame containing all data from the CSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a CSV or if no data is read from the file.
        pd.errors.ParserError: If there's an error parsing the CSV file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        logging.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(
            f"File does not exist, check the path and try again with the correct path: {file_path}"
        )

    # Check if file has a .csv extension
    if file_path.suffix.lower() != ".csv":
        logging.error(f"Unsupported file format: {file_path.suffix}")
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. Only CSV files are supported."
        )

    logging.info(
        f"Starting to read CSV file in chunks: {file_path} with chunksize={chunksize}"
    )

    chunks = []
    try:
        for i, chunk in enumerate(
            pd.read_csv(file_path, chunksize=chunksize, **read_csv_kwargs)
        ):
            logging.debug(f"Processing chunk {i+1}")
            chunks.append(chunk)
        if not chunks:
            logging.warning(f"No data read from the file: {file_path}")
            raise ValueError(f"No data read from the file: {file_path}")
        df = pd.concat(chunks, ignore_index=True)
        logging.info(
            f"Successfully loaded CSV file: {file_path} with {len(df)} records."
        )
    except pd.errors.EmptyDataError:
        logging.error(f"No data: The file is empty: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing the file: {file_path} - {e}")
        raise
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while reading the file: {file_path} - {e}"
        )
        raise

    return df


def save_dataframe_if_not_exists(
    df: pd.DataFrame,
    file_path: str,
    file_format: str = "csv",
    **kwargs,
) -> bool:
    """
    Save a DataFrame to a file only if the file does not already exist.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    file_path : str
        The path to the file where the DataFrame should be saved.
    file_format : str, optional
        The format to save the DataFrame in (e.g., 'csv', 'excel'). Default is 'csv'.
    kwargs :
        Additional keyword arguments to pass to the pandas saving method.

    Returns:
    -------
    bool
        True if the file was saved, False if it already exists.
    """
    path = Path(file_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_format.lower() == "csv":
            df.to_csv(path, index=False, **kwargs)
        elif file_format.lower() in ["xls", "xlsx"]:
            df.to_excel(path, index=False, **kwargs)
        elif file_format.lower() == "json":
            df.to_json(path, **kwargs)
        else:
            logging.error(f"Unsupported file format: {file_format}")
            raise ValueError(f"Unsupported file format: {file_format}")

        return True
    else:

        return False


def save_dataframe_overwrite(
    df: pd.DataFrame,
    file_path: str,
    file_format: str = "csv",
    **kwargs,
) -> None:
    """
    Save a DataFrame to a file, overwriting it if it already exists.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    file_path : str
        The path to the file where the DataFrame should be saved.
    kwargs :
        Additional keyword arguments to pass to the pandas saving method.

    Returns:
    -------
    None
    """
    path = Path(file_path)
    try:
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on the specified format
        if file_format.lower() == "csv":
            df.to_csv(path, index=False, **kwargs)
        elif file_format.lower() in ["xls", "xlsx"]:
            df.to_excel(path, index=False, **kwargs)
        elif file_format.lower() == "json":
            df.to_json(path, **kwargs)
        elif file_format.lower() == "parquet":
            df.to_parquet(path, index=False, **kwargs)
        elif file_format.lower() == "feather":
            df.to_feather(path, **kwargs)
        elif file_format.lower() == "hdf":
            df.to_hdf(path, key="df", mode="w", **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logging.info(f"File saved to {file_path}. (Overwritten if it existed)")
    except PermissionError:
        logging.error(f"Permission denied: Cannot write to {file_path}.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while saving the file: {e}")
        raise
