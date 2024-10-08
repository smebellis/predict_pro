from DataPreprocessing import DataPreprocessing

from utils.helper import (
    setup_logging,
    save_dataframe_if_not_exists,
    save_dataframe_overwrite,
    parse_arguments,
)
import pandas as pd
from typing import Dict


def run_preprocessing_pipeline(
    input_file: str,
    output_file: str,
    preprocessor: DataPreprocessing,
    district: Dict,
    missing_data_column: str = "MISSING_DATA",
    missing_flag: bool = True,
    timestamp_column: str = "TIMESTAMP",
    polyline_column: str = "POLYLINE",
    polyline_list_column: str = "POLYLINE_LIST",
    travel_time_column: str = "TRAVEL_TIME",
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Apply a series of preprocessing steps to the DataFrame:

    1. Remove rows with missing GPS data.
    2. Drop unnecessary columns.
    3. Convert UNIX timestamps to datetime objects.
    4. Calculate travel time based on polyline data.
    5. Convert Polyline column to list.
    6. Extract start location from Polyline column.
    7. Extract end location from Polyline column.
    8. Calculate End Time of Trip.
    9. Remove NaN objects from DataFrame.
    10. Extract Lat/Long Coordinates.
    11. Add weekday information.
    12. Add month information.
    13. Add year information.
    14. Save output to CSV.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    output_file : str
        Path to save the processed CSV file.
    preprocessor : DataPreprocessing
        An instance of the DataPreprocessing class.
    missing_data_column : str, optional
        The column indicating missing data (default is "MISSING_DATA").
    missing_flag : bool, optional
        The flag value that indicates missing data (default is True).
    timestamp_column : str, optional
        The column containing UNIX timestamps (default is "TIMESTAMP").
    polyline_column : str, optional
        The column containing polyline data (default is "POLYLINE").
    polyline_list_column : str, optional
        The column to store the polyline list (default is "POLYLINE_LIST").
    travel_time_column : str, optional
        The column containing travel time (default is "TRAVEL_TIME").
    drop_na : bool, optional
        Flag to indicate whether to drop NaN values (default is True).
    inplace : bool, optional
        Flag to indicate whether to modify the DataFrame in place (default is False).

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    logger = setup_logging()

    # Load your data
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)  # add nrows=number to process a smaller sample
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

        # Step 4: Calculate travel time
        logger.info("Calculating travel time.")
        df = preprocessor.calculate_travel_time_fifteen_seconds(df, polyline_column)

        # Step 5: Convert Polyline column from string to list
        logger.info("Converting Polyline column from string to list.")
        df = preprocessor.safe_convert_string_to_list(df, polyline_column)

        # Step 6: Extract start location from Polyline column
        logger.info("Extracting start locations from Polyline column.")
        df = preprocessor.extract_start_location(df, polyline_list_column)

        # Step 7: Extract end location from Polyline column
        logger.info("Extracting end locations from Polyline column.")
        df = preprocessor.extract_end_location(df, polyline_list_column)

        # Step 8: Calculate End Time of Trip
        logger.info("Calculating end time from start time and travel time.")
        df = preprocessor.calculate_end_time(df, timestamp_column, travel_time_column)

        # Step 9: Remove NaN objects from DataFrame
        if drop_na:
            logger.info("Removing rows with NaN data.")
            df = preprocessor.drop_nan(df)
        else:
            logger.info("Skipping removal of NaN data as per configuration.")

        # Step 10: Extract Lat/Long Coordinates
        coordinate_extractions = [
            ("START", ["START_LONG", "START_LAT"]),
            ("END", ["END_LONG", "END_LAT"]),
        ]

        for column, columns_to_add in coordinate_extractions:
            logger.info(f"Extracting Lat/Long Coordinates from column '{column}'.")
            df = preprocessor.extract_coordinates(df, column, columns_to_add)

        # Step 11: Add weekday information
        logger.info("Adding weekday information.")
        df = preprocessor.add_weekday(df, timestamp_column)

        # Step 12: Add month information
        logger.info("Adding month information.")
        df = preprocessor.add_month(df, timestamp_column)

        # Step 13: Load Districts from JSON
        logger.info("Loading districts from JSON file")
        district_df = preprocessor.load_districts(districts=district)

        # Step 14: Assign Districts
        logger.info("Assigning districts to taxi data.")
        # Choose between sampling and entire dataset
        df = preprocessor.assign_districts_to_taxi(
            df, sample_size=1000, use_sample=False
        )
        # Or use the vectorized method
        # df = preprocessor.assign_districts_to_taxi_vectorized(df, sample_size=1000, use_sample=True)

        # Step 15: Add year information
        logger.info("Adding year information.")
        df = preprocessor.add_year(df, timestamp_column)

        # Step 16: Save the processed DataFrame
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
    preprocessor = DataPreprocessing()
    run_preprocessing_pipeline(args.input, args.output, preprocessor=preprocessor)
