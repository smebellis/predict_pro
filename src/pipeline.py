from DataPreprocessing import DataPreprocessing
from utils.helper import (
    setup_logging,
    save_dataframe_if_not_exists,
    save_dataframe_overwrite,
)
import pandas as pd

"""
This is a note to myself for late.  I want to move all the steps from the preprocess method into here. 
Keep the same functionality.  Have logging work as it did before.  Make sure it saves the items into the correct
location.  And that the comments are good to go.  This is a like to have, versus a need to.  The pipeline
semi works now.  It just needs the "visualizaiton" incorporated into it.  
"""


def run_preprocessing_pipeline(
    input_file: str,
    output_file: str,
    preprocessor: DataPreprocessing,
    missing_data_column: str = "MISSING_DATA",
    missing_flag: bool = True,
    timestamp_column: str = "TIMESTAMP",
    polyline_column: str = "POLYLINE",
    polyline_list_column: str = "POLYLINE_LIST",
    travel_time_column: str = "TRAVEL_TIME",
    drop_na: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply a series of preprocessing steps to the DataFrame:
    0. Drop duplicates
    1. Remove rows with missing GPS data.
    2. Convert UNIX timestamps to datetime objects.
    3. Calculate travel time based on polyline data.
    4. Convert Polyline column to list
    5. Extract start location from Polyline column
    6. Extract end location from Polyline column
    7. Calculate End Time of Trip
    8. Add weekday output
    9. Add Month output
    10. Add year output
    11. Save output to csv

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to preprocess.
    drop_na: bool = True
        The flag value that indicates drop the NaN columns
    inplace: bool = False
        The flag that indicates drop in place without modifying the dataframe
    missing_data_column : str, optional
        The column indicating missing data (default is "MISSING_DATA").
    missing_flag : bool, optional
        The flag value that indicates missing data (default is True).
    timestamp_column : str, optional
        The column containing UNIX timestamps (default is "TIMESTAMP").
    polyline_column : str, optional
        The column containing polyline data (default is "POLYLINE").
    start_time_column: str = "TIMESTAMP"
        The column containing the datetime
    travel_time_column: str = "TRAVEL_TIME"
        The column containg the travel time

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    logger = setup_logging()

    # Load your data
    df = pd.read_csv(input_file)

    try:
        logger.info("Starting preprocessing pipeline.")

        # Step 1: Remove rows with missing GPS data
        logger.info("Removing rows with missing GPS data.")
        df = preprocessor.remove_missing_gps(df, missing_data_column, missing_flag)

        # Step 2: Drop Columns
        logger.info("Removing Columns from DataFrame")
        df = preprocessor.drop_columns(df)

        # Step 2: Convert UNIX timestamps to datetime
        logger.info("Converting UNIX timestamps to datetime objects.")
        df = preprocessor.convert_timestamp(df, timestamp_column)

        # Step 3: Calculate travel time
        logger.info("Calculating travel time.")
        df = preprocessor.calculate_travel_time_fifteen_seconds(df, polyline_column)

        # Step 4: Convert Polyline column from string to list
        logger.info("Converting Polyline column from string to list.")
        df = preprocessor.safe_convert_string_to_list(df, polyline_column)

        # Step 5: Extract start location from Polyline column
        logger.info("Extracting start locations from Polyline column.")
        df = preprocessor.extract_start_location(df, polyline_list_column)

        # Step 6: Extract end location from Polyline column
        logger.info("Extracting end locations from Polyline column.")
        df = preprocessor.extract_end_location(df, polyline_list_column)

        # Step 7: Extract end location from Polyline column
        logger.info("Extracting end time from Starting Time and Travel Time.")
        df = preprocessor.calculate_end_time(df, timestamp_column, travel_time_column)

        # Step 8: Remove NaN objects from DataFrame
        logger.info("Removing rows with NaN data")
        df = preprocessor.drop_nan(df)

        # Stemp 9: Extract Lat Long Coordinates
        coordinate_extractions = [
            ("START", ["START_LONG", "START_LAT"]),
            ("END", ["END_LONG", "END_LAT"]),
        ]

        for column, columns_to_add in coordinate_extractions:
            logger.info(f"Extracting Lat/Long Coordinates from column '{column}'")
            df = preprocessor.extract_coordinates(df, column, columns_to_add)

        # Add temporal features
        # Step 10: Add weekday output
        logger.info("Adding weekday information.")
        df = preprocessor.add_weekday(df, timestamp_column)

        # Step 11: Add Month output
        logger.info("Adding Month information.")
        df = preprocessor.add_month(df, timestamp_column)

        # Step 12: Add Month output
        logger.info("Adding Year information.")
        df = preprocessor.add_year(df, timestamp_column)

        # Save the processed DataFrame
        was_saved = save_dataframe_if_not_exists(df, output_file, file_format="csv")
        if was_saved:
            preprocessor.logger.info(f"File saved to {output_file}.")
        else:
            preprocessor.logger.info(
                f"File {output_file} already exists. Skipping save."
            )

            preprocessor.logger.info("Preprocessing pipeline completed successfully.")
            return df

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        raise


# Optionally, define a main function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the data preprocessing pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output", required=True, help="Path to save the processed CSV file."
    )

    args = parser.parse_args()

    run_preprocessing_pipeline(args.input, args.output)
