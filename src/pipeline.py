from Preprocessing import Preprocessing

from districts import load_districts
from utils.helper import (
    save_dataframe_if_not_exists,
    save_dataframe_overwrite,
    parse_arguments,
    read_csv_with_progress,
)
import pandas as pd
from typing import Dict

from logger import get_logger

logger = get_logger(__name__)


def preprocessing_pipeline(
    input_file: str,
    output_file: str,
    preprocessor: Preprocessing,
    district: Dict,
    missing_data_column: str = "MISSING_DATA",
    missing_flag: bool = True,
    timestamp_column: str = "TIMESTAMP",
    polyline_column: str = "POLYLINE",
    polyline_list_column: str = "POLYLINE_LIST",
    travel_time_column: str = "TRAVEL_TIME",
    drop_na: bool = True,
) -> pd.DataFrame:

    # Load your data
    try:
        logger.info(f"Loading data from {input_file}")
        df = read_csv_with_progress(input_file)
        # df = pd.read_csv(input_file)  # add nrows=number to process a smaller sample
        logger.info(f"Loaded data from {input_file}.")
    except FileNotFoundError:
        logger.error(f"Input file {input_file} not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Input file {input_file} is empty.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

    try:
        # Step 1: Remove rows with missing GPS data
        logger.info("Removing rows with missing GPS data.")
        df = preprocessor.remove_missing_gps(df, missing_data_column, missing_flag)

        # Step 2: Drop Columns
        logger.info("Dropping unnecessary columns from DataFrame.")
        df = preprocessor.drop_columns(df)

        # Step 3: Convert UNIX timestamps to datetime
        logger.info("Converting UNIX timestamps to datetime objects.")
        df = preprocessor.convert_timestamp(df, timestamp_column)

        logger.info("Converting Coordinates from string to list object")
        df = preprocessor.convert_polyline_to_list(df, polyline_column)

        # Step 6: Extract start location from Polyline column
        logger.info("Extracting Starting and Ending locations from Polyline column.")
        df = preprocessor.extract_coordinates(df, polyline_column)

        # Step 9: Remove NaN objects from DataFrame
        if drop_na:
            logger.info("Removing rows with NaN data.")
            df = preprocessor.drop_nan(df)
        else:
            logger.info("Skipping removal of NaN data as per configuration.")

        logger.info("Separating time into individual components")
        df = preprocessor.separate_timestamp(df, timestamp_column)

        logger.info("Assigning districts to taxi data.")
        # Choose vectorized or row-wise for method.
        df = preprocessor.assign_districts(df, method="vectorized")

        # Step 17: Save the processed DataFrame
        was_saved = save_dataframe_if_not_exists(df, output_file, file_format="csv")
        if was_saved:
            logger.info(f"File saved to {output_file}.")
        else:
            logger.info(f"File {output_file} already exists. Skipping save.")

        logger.info("Preprocessing pipeline completed successfully.")
        return df

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        raise


if __name__ == "__main__":

    args = parse_arguments()
    district = load_districts(args.districts_path)
    preprocessor = Preprocessing(districts=district)

    preprocessing_pipeline(
        args.input, args.output, district=district, preprocessor=preprocessor
    )
