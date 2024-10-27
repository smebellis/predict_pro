import json
import logging
import sys

from src.cluster_districts import (
    cluster_trip_district,
    cluster_trip_time,
    HDBSCAN_Clustering,
    determine_traffic_status,
)
from src.DataPreprocessing import DataPreprocessing
from src.logger import get_logger, load_config
from src.pipeline import run_preprocessing_pipeline
from src.utils.helper import (
    file_load,
    parse_arguments,
    read_csv_with_progress,
    save_dataframe_if_not_exists,
)

from src.districts import load_districts

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

    # Load Districts Data
    logger.info("Loading district boundary data...")
    try:
        porto_districts = load_districts(args.districts_path)
    except porto_districts.DistrictLoadError:
        sys.exit(1)

    # Initialize and run the DataPreprocessor
    preprocessor = DataPreprocessing(districts=porto_districts)

    # TODO: Add an argument to load a smaller sample when loading the original dataset.
    # TODO:  Need a flag that will just load the dataset from csv.  No need to process it everytime
    # TODO:  The process pipeline needs to be made into its own module
    # TODO:  Remove logic to run pipelines from main.py, they will be in separate scripts.

    if (
        args.process_pipeline
    ):  # add the flag --process_pipeline if you want to run the whole pipeline
        try:
            run_preprocessing_pipeline(
                input_file=args.input,
                output_file=args.output,
                district=porto_districts,
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

            logger.info("CSV file read successfully.")
            clustered_df = cluster_trip_district(df, porto_districts)
            clustered_df = cluster_trip_time(df)
            clustered_df = HDBSCAN_Clustering(df)
            clustered_df = determine_traffic_status(df)
            save_dataframe_if_not_exists(clustered_df, args.save)
            logger.info(f"Clustered DataFrame Save to {args.save}")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
