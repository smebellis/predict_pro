import argparse
import logging
import os
from datetime import datetime

from DataPreprocessing import DataPreprocessing


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


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    # Add arguments as needed
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
    # Parse command-line arguments
    args = parse_arguments()

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
    preprocessor.run_pipeline()

    # Initialize and run other modules as needed
    # if args.module == "preprocess":
    #     preprocessor.run_pipeline()
    # elif args.module == "analyze":
    #     analyzer = OtherClass()
    #     analyzer.run_analysis()

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
