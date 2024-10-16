import json
import logging
import sys

from src.clustering import cluster_hdbscan
from src.DataPreprocessing import DataPreprocessing
from src.logger import get_logger, load_config
from src.pipeline import run_preprocessing_pipeline
from src.utils.helper import (
    file_load,
    parse_arguments,
    read_csv_with_progress,
    save_dataframe_if_not_exists,
)

# Set up logging
logger = get_logger(__name__)


def main():
    # Load config file
    config = load_config()

    # Access application details
    app_name = config.get("app_name", "MyApp")
    version = config.get("version", "0.0.1")
    logger.info(f"Starting {app_name} version {version}")

    try:
        args = parse_arguments()
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        return

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled.")

    logger.info("Starting the Data Preprocessing Pipeline.")

    # Path tio JSON file
    PORTO_DISTRICTS = args.districts_path

    # Load Districts Data
    logger.info("Loading district boundary data...")
    try:
        with open(PORTO_DISTRICTS, "r") as FILE:
            DISTRICTS = json.load(FILE)
    except FileNotFoundError:
        logger.error(f"Districts file not found at {PORTO_DISTRICTS}.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Error decoding JSON from districts file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading districts: {e}")
        sys.exit(1)

    if not isinstance(DISTRICTS, dict) or not DISTRICTS:
        logger.error(
            "DISTRICTS should be a non-empty dictionary. Please provide valid district data."
        )
        sys.exit(1)

    # Initialize and run the DataPreprocessor
    preprocessor = DataPreprocessing(districts=DISTRICTS)

    # TODO: Add an argument to load a smaller sample when loading the original dataset.
    if (
        args.process_pipeline
    ):  # add the flag --process_pipeline if you want to run the whole pipeline
        try:
            run_preprocessing_pipeline(
                input_file=args.input,
                output_file=args.output,
                district=DISTRICTS,
                preprocessor=preprocessor,
                # missing_data_column=args.missing_data_column,
                # missing_flag=args.missing_flag,
                # timestamp_column=args.timestamp_column,
                # polyline_column=args.polyline_column,
                # polyline_list_column=args.polyline_list_column,
                # travel_time_column=args.travel_time_column,
                # drop_na=args.drop_na,
                # sample_size=args.sample_size,
                # use_sample=args.use_sample,
            )
            logger.info("Pipeline execution completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            sys.exit(1)
    else:
        try:
            df = read_csv_with_progress(args.output)
            clustered_df = cluster_hdbscan(df)
            logger.info("CSV file read successfully.")
            save_dataframe_if_not_exists(clustered_df, args.save)
            logger.indo(f"Clustered DataFrame Save to {args.save}")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
