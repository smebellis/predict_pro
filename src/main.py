import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from DataPreprocessing import DataPreprocessing
from pipeline import run_preprocessing_pipeline
from utils.helper import setup_logging, file_load, parse_arguments


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

    # Initialize and run the DataPreprocessor
    preprocessor = DataPreprocessing()

    # Execute the preprocessing pipeline
    try:
        run_preprocessing_pipeline(
            input_file=args.input, output_file=args.output, preprocessor=preprocessor
        )
        logger.info("Pipeline execution completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return


if __name__ == "__main__":
    main()
