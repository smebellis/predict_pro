import argparse
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from DataPreprocessing import DataPreprocessing
from utils.helper import setup_logging, file_load


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
        "--data_path",
        type=str,
        required=True,
        default="/home/smebellis/ece5831_final_project/processed_data/update_taxi_trajectory.csv",
        help="Path to the data file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    # Example: Choose which module to run
    # parser.add_argument(
    #     "--module",
    #     type=str,
    #     choices=["preprocess", "analyze"],
    #     default="preprocess",
    #     help="Module to execute."
    # )
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
    except Exception as e:
        logging.error(f"Error parsing arguments: {e}")
        return

    # Set up logging
    logger = setup_logging()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled.")

    logger.info("Starting the Data Preprocessing Pipeline.")

    ##################
    # LOAD THE DATA  #
    ##################
    logging.info("Loading data...")
    df = file_load(args.data_path)
    logging.info("Data loaded successfully.")

    # Initialize and run the DataPreprocessor
    preprocessor = DataPreprocessing()
    df = preprocessor.preprocess(df)

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
