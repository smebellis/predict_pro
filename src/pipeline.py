import os
import subprocess
import zipfile
from typing import Dict

import pandas as pd

from src.districts import load_districts
from src.helper import (
    parse_arguments,
    read_csv_with_progress,
    save_dataframe_if_not_exists,
    save_dataframe_overwrite,
)
from src.logger import get_logger
from src.Preprocessing import Preprocessing

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
    travel_time_column: str = "TRAVEL_TIME",
    drop_na: bool = True,
) -> pd.DataFrame:
    
    # Load your data
    try:
        
        logger.info(f"Loading data from {input_file}")
        df = read_csv_with_progress(input_file)

        # For testing purposes, uncomment line to use a small sample
        # df = df.sample(n=5000, random_state=42)

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

        # Step 4: Parse and correct PolyLine Coordinates
        logger.info("Parsing and correcting POLYLINE Coordinates")
        df = preprocessor.parse_and_correct_polyline_coordinates(df, polyline_column)

        # Step 5: Extract Starting and Ending locations from POLYLINE
        logger.info("Extracting Starting and Ending locations from Polyline column.")
        df = preprocessor.extract_coordinates(df, polyline_column)

        # Step 6: Remove NaN objects from DataFrame
        if drop_na:
            logger.info("Removing rows with NaN data.")
            df = preprocessor.drop_nan(df)
        else:
            logger.info("Skipping removal of NaN data as per configuration.")

        # Step 7: Parse individual time components from TIMESTAMP
        logger.info("Separating time into individual components")
        df = preprocessor.separate_timestamp(df, timestamp_column)

        # Step 8: Assign the districts to taxi data
        logger.info("Assigning districts to taxi data.")
        # Choose vectorized or row-wise for method, row-wise is more accurate, but lengthy.
        df = preprocessor.assign_districts(df, method="vectorized")

        # Step 9: Calculate the travel time of each trip
        logger.info("Calculate the travel time")
        df = preprocessor.calculate_travel_time(df)

        # Step 10: Calculate the travel time of each trip
        logger.info("Calculate Trip Distance")
        df = preprocessor.calculate_trip_distance(df)

        # Step 11: Calculate the travel time of each trip
        logger.info("Calculate the Average speed")
        df = preprocessor.calculate_avg_speed(df, distance_column="TRIP_DISTANCE")

        # Step 11: Save the processed DataFrame
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
