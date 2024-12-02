import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from tqdm import tqdm
import seaborn as sns
import os
import subprocess
import zipfile

from src.logger import get_logger

logger = get_logger(__name__)

def download_dataset():
    # Check if the data/taxi-trajectory.zip file exists
        zip_file = "data/taxi-trajectory.zip"
        data_folder = "data"

        if not os.path.exists(zip_file):
            logger.info(f"{zip_file} not found. Downloading dataset...")
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            
            # Download dataset using Kaggle API
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "crailtap/taxi-trajectory",
                    "-p",
                    data_folder,
                ],
                check=True,
            )

            # Unzip the downloaded file
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(data_folder)
            logger.info(f"Dataset downloaded and extracted to {data_folder}.")
        else:
            logger.info(f"{zip_file} already exists.")

def load_config(config_path: str = "config.yaml") -> dict:
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
    root_dir = config_file.parent

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            print(f"Configuration loaded successfully from '{config_file}'.")

        # Resolve relative paths based on the config file's directory
        paths = config.get("paths", {})
        for key, path in paths.items():
            p = Path(path)
            if not p.is_absolute():
                paths[key] = str((root_dir / p).resolve())
        config["paths"] = paths

        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Run the data preprocessing pipeline.")

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        default="data/train.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument("--output", "-o", default="processed_data/taxi_data_processed.csv", help="Path to save the processed CSV file.")
    parser.add_argument(
        "--districts_path",
        default="data/porto_districts.json",
        help="Path to the districts JSON file.",
    )
    parser.add_argument("--save", default="processed_data/clustered_data.csv", help="Path to save the file")
    parser.add_argument("--input_processed", default="processed_data/taxi_data_processed.csv", help="Path to save the file")
    # Optional arguments
    parser.add_argument(
        "--missing_data_column",
        default="MISSING_DATA",
        help="Column indicating missing data.",
    )
    parser.add_argument(
        "--missing_flag",
        type=bool,
        default=True,
        help="Flag value indicating missing data.",
    )
    parser.add_argument(
        "--timestamp_column",
        default="TIMESTAMP",
        help="Column containing UNIX timestamps.",
    )
    parser.add_argument(
        "--polyline_column", default="POLYLINE", help="Column containing polyline data."
    )
    parser.add_argument(
        "--polyline_list_column",
        default="POLYLINE_LIST",
        help="Column to store polyline list.",
    )
    parser.add_argument(
        "--travel_time_column",
        default="TRAVEL_TIME",
        help="Column containing travel time.",
    )
    parser.add_argument(
        "--drop_na", type=bool, default=True, help="Flag to drop NaN values."
    )

    # District Assignment parameters
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of records to sample for district assignment.",
    )
    parser.add_argument(
        "--use_sample",
        type=bool,
        default=False,
        help="Whether to use a sample for district assignment.",
    )

    # Future Use
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    parser.add_argument(
        "--process_pipeline",
        action="store_true",
        help="Decides if the pipeline should be ran",
    )

    

    return parser.parse_args()


def categorize_time(df: pd.DataFrame) -> str:
    """
    Take an interval and maps it to a time category
    """
    # Define bins for morning, afternoon, and night
    bins = [0, 12, 18, 24]  # Morning: 0-12, Afternoon: 12-18, Night: 18-24
    labels = ["Morning", "Afternoon", "Night"]

    # Use pd.cut to categorize the 'TIME' column into the defined bins
    df["Time_of_Day"] = pd.cut(df["TIME"], bins=bins, labels=labels, right=False)


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


def read_csv_with_progress(file_path, chunksize=100000):
    """
    Reads a CSV file with a progress bar.

    Parameters:
    - file_path: Path to the CSV file.
    - chunksize: Number of rows per chunk.

    Returns:
    - DataFrame containing the concatenated chunks.
    """
    # Get the total number of lines in the file for the progress bar
    # This can be time-consuming for very large files
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        total = sum(1 for _ in f)

    # Initialize the progress bar
    progress_bar = tqdm(total=total, desc="Reading CSV", unit="lines")

    # Initialize an empty list to hold chunks
    chunks = []

    # Read the CSV file in chunks
    for chunk in pd.read_csv(
        file_path, chunksize=chunksize, iterator=True, encoding="utf-8"
    ):
        chunks.append(chunk)
        progress_bar.update(len(chunk))

    # Close the progress bar
    progress_bar.close()

    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)

    return df


def plot_route_images(route_images: torch.Tensor, num_images: int = 5):
    """
    Plots a specified number of route images to verify correctness.

    Args:
        route_images (torch.Tensor): Tensor of images representing routes.
        num_images (int): Number of images to display.
    """
    logger.info(f"Plotting {num_images} route images for verification.")
    # Determine the number of images to plot (max is the length of route_images)
    num_images = min(num_images, route_images.size(0))

    # Set up the plot
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(route_images[i].numpy(), cmap="gray")
        ax.set_title(f"Route {i+1}")
        ax.axis("off")

    plt.savefig(os.path.join("plots", "route_images_example.png"))
    plt.show()
    logger.info("Displayed route images.")


# Function to plot metrics and save them to a folder
def plot_metrics(metrics, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    for clf_name, clf_metrics in metrics.items():
        for metric, value in clf_metrics.items():
            if metric == "Confusion Matrix":
                plt.figure(figsize=(8, 6))
                sns.heatmap(value, annot=True, fmt="d", cmap="Blues")
                plt.title(f"{metric} for {clf_name}")
                plt.ylabel("Actual")
                plt.xlabel("Predicted")
                plt.savefig(os.path.join(output_dir, f"{clf_name}_{metric}.png"))
                plt.close()
            else:
                plt.figure()
                plt.bar([clf_name], [value])
                plt.title(f"{metric} for {clf_name}")
                plt.ylabel(metric)
                plt.savefig(os.path.join(output_dir, f"{clf_name}_{metric}.png"))
                plt.close()


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/smebellis/ece5831_final_project/processed_data/clustered_taxi_data.csv",
        nrows=10000,
    )
