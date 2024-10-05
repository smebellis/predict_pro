import numpy as np
import pandas as pd
import ast
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt


class DataPreprocessing:
    def __init__(self, log_dir: str = "logs", log_file: str = "data_preprocessor.log"):
        """
        Initialize the DataPreprocessor with a dedicated logger that logs to both a file and stdout.
        Logs are stored in a separate directory with date and time appended to the log file name.

        Parameters:
        ----------
        log_dir : str, optional
            The directory where log files will be stored. Default is 'logs'.
        log_file : str, optional
            The base filename for the log file. Default is 'data_preprocessor.log'.
        """

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Append current date and time to the log file name
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{os.path.splitext(log_file)[0]}_{current_time}{os.path.splitext(log_file)[1]}"
        log_path = os.path.join(log_dir, log_filename)

        self.logger = logging.getLogger("DataPreprocessorLogger")
        self.logger.setLevel(logging.DEBUG)  # Capture all levels of logs

        # Prevent adding multiple handlers if the logger already has them
        if not self.logger.handlers:
            # Formatter for log messages
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # File Handler with Rotation
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=5,  # Keep up to 5 backup files
            )
            file_handler.setLevel(logging.DEBUG)  # Log all levels to file
            file_handler.setFormatter(formatter)

            # Stream Handler for stdout
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)  # Log INFO and above to console
            stream_handler.setFormatter(formatter)

            # Add Handlers to the Logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

        self.logger.debug("Logger initialized and handlers added.")

    def remove_missing_gps(
        self,
        df: pd.DataFrame,
        missing_data_column: str = "MISSING_DATA",
        missing_flag: bool = True,
    ) -> pd.DataFrame:
        """
        Remove rows from the DataFrame where the specified missing data column has the missing flag.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame.
        missing_data_column : str, optional
            The name of the column that indicates missing data (default is "MISSING_DATA").
        missing_flag : bool, optional
            The flag value that indicates missing data (default is True).

        Returns:
        -------
        pd.DataFrame
            The DataFrame with rows containing missing data removed.

        Raises:
        ------
        ValueError
            If the specified missing data column does not exist in the DataFrame.
        """
        if missing_data_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{missing_data_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot extract POLYLINE from an empty DataFrame.")

        filtered_df = df.copy()

        # Using the tilde (~) operator for boolean negation (more idiomatic)
        filtered_df = df[~df[missing_data_column].astype(bool)].reset_index(drop=True)

        return filtered_df

    def convert_timestamp(
        self, df: pd.DataFrame, timestamp_column: str = "TIMESTAMP"
    ) -> pd.DataFrame:
        """
        Converts a UNIX timestamp into a windows timestamp in the year-month-day hour minute-second format

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame.
        data_column : str, optional
            The name of the column that indicates the timestamp data (default is "TIMESTAMP").

        Returns:
        -------
        pd.DataFrame
            The DataFrame with the converted timestamp in a new column named 'CONVERTED_TIMESTAMP'.

        Raises:
        ------
        ValueError
            If the specified timestamp column does not exist in the DataFrame.
        """
        if timestamp_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{timestamp_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot extract POLYLINE from an empty DataFrame.")

        # Copy df to prevent edtiting the originial
        converted_df = df.copy()

        # Make a check that the timestamp is in the correct format
        try:
            converted_df[timestamp_column] = pd.to_datetime(
                converted_df[timestamp_column], unit="s", errors="raise"
            )
        except Exception as e:
            raise ValueError(f"Error converting '{timestamp_column}' to datetime: {e}")

        return converted_df

    def calculate_travel_time_fifteen_seconds(
        self, df: pd.DataFrame, polyline_column: str = "POLYLINE"
    ) -> pd.DataFrame:
        """
        Calculate the travel time based on the polyline data where each point represents 15 seconds.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with a 'POLYLINE' column.
        polyline_column : str, optional
            The name of the column containing the polyline data.

        Returns:
        -------
        pd.DataFrame
            DataFrame with an added 'TRAVEL_TIME' column containing the travel time in seconds.
        """
        if polyline_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{polyline_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot extract travel time from an empty DataFrame.")

        # Travel time is 15 seconds multiplied by the number of points in the polyline minus one
        df["TRAVEL_TIME"] = df[polyline_column].apply(
            lambda polyline: (len(polyline) - 1) * 15
        )

        return df

    def extract_start_location(
        self, df: pd.DataFrame, polyline_column: str = "POLYLINE"
    ) -> pd.DataFrame:
        """
        Extracts the first coordinate pair from the specified polyline column and adds it as a new 'start' column.

        Args:
            df (pd.DataFrame): The input DataFrame containing the polyline data.
            polyline_column (str): The name of the column containing polylines.

        Returns:
            pd.DataFrame: A new DataFrame with an added 'start' column containing the first coordinate pair.

        Raises:
            ValueError: If the specified polyline column does not exist in the DataFrame.
        """
        if polyline_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{polyline_column}' column."
            )

        if df.empty:
            raise IndexError(
                "The input DataFrame is empty. Returning the DataFrame as is."
            )
        start_df = df.copy()

        start_df["START"] = [
            poly[0] if isinstance(poly, list) and len(poly) > 0 else None
            for poly in df["POLYLINE"]
        ]

        return start_df

    def extract_end_location(
        self, df: pd.DataFrame, polyline_column: str = "POLYLINE"
    ) -> pd.DataFrame:
        """
        Extracts the last coordinate pair from the specified polyline column and adds it as a new 'end' column.

        Args:
            df (pd.DataFrame): The input DataFrame containing the polyline data.
            polyline_column (str): The name of the column containing polylines.

        Returns:
            pd.DataFrame: A new DataFrame with an added 'END' column containing the first coordinate pair.

        Raises:
            ValueError: If the specified polyline column does not exist in the DataFrame.
        """
        if polyline_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{polyline_column}' column."
            )

        if df.empty:
            raise IndexError(
                "The input DataFrame is empty. Returning the DataFrame as is."
            )

        end_df = df.copy()

        end_df["END"] = [
            poly[-1] if isinstance(poly, list) and len(poly) > 0 else None
            for poly in df["POLYLINE"]
        ]

        return end_df

    def safe_convert_string_to_list(
        self, df: pd.DataFrame, polyline_column: str = "POLYLINE"
    ) -> Optional[list]:
        """
        Converts string representations of lists in the specified column to actual lists.

        Args:
            df (pd.DataFrame): The input DataFrame.
            polyline_column (str): The column containing string representations of lists.

        Returns:
            Optional[pd.DataFrame]: The DataFrame with the specified column converted to lists.

        Raises:
            ValueError: If the specified column does not exist.
            IndexError: If the DataFrame is empty.
            ValueError: If a cell in the specified column cannot be parsed.
        """
        if polyline_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{polyline_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot extract travel time from an empty DataFrame.")
        # Make a copy to avoid modifying the original DataFrame
        df_converted = df.copy()
        df_converted["POLYLINE"] = df_converted[polyline_column].apply(ast.literal_eval)
        return df_converted

    def add_weekday(
        self, df: pd.DataFrame, timestamp_column: str = "TIMESTAMP"
    ) -> pd.DataFrame:
        """
        Calculate the weekday from the timestamp column and add it as a new column.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with a datetime timestamp column.
        timestamp_column : str, optional
            The name of the column that contains datetime timestamp data (default is "TIMESTAMP").

        Returns:
        -------
        pd.DataFrame
            The DataFrame with a new column named 'WEEKDAY'.

        Raises:
        ------
        ValueError
            If the specified timestamp column does not exist in the DataFrame.
        """
        if timestamp_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{timestamp_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot add 'WEEKDAY' to an empty DataFrame.")

        # Make a copy to avoid altering original dataframe
        weekday_df = df.copy()

        weekday_df["WEEKDAY"] = weekday_df[timestamp_column].dt.day_name()

        return weekday_df

    def add_month(
        self, df: pd.DataFrame, timestamp_column: str = "TIMESTAMP"
    ) -> pd.DataFrame:
        """
        Calculate the month from the timestamp column and add it as a new column.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with a datetime timestamp column.
        timestamp_column : str, optional
            The name of the column that contains datetime timestamp data (default is "TIMESTAMP").

        Returns:
        -------
        pd.DataFrame
            The DataFrame with a new column named 'MONTH'.

        Raises:
        ------
        ValueError
            If the specified timestamp column does not exist in the DataFrame.
        """
        if timestamp_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{timestamp_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot add 'MONTH' to an empty DataFrame.")

        # Make a copy to avoid altering original dataframe
        month_df = df.copy()

        month_df["MONTH"] = month_df[timestamp_column].dt.month_name()

        return month_df

    def add_year(
        self, df: pd.DataFrame, timestamp_column: str = "TIMESTAMP"
    ) -> pd.DataFrame:
        """
        Calculate the year the timestamp column and adds it into the dataframe

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame.
        timestamp_column : str, optional
            The name of the column that indicates the timestamp data (default is "TIMESTAMP").

        Returns:
        -------
        pd.DataFrame
            The DataFrame with the converted timestamp in a new column named 'CONVERTED_TIMESTAMP'.

        Raises:
        ------
        ValueError
            If the specified timestamp column does not exist in the DataFrame.
        """
        if timestamp_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{timestamp_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot extract POLYLINE from an empty DataFrame.")

        # Make a copy to avoid altering original dataframe
        year_df = df.copy()

        year_df["YEAR"] = year_df[timestamp_column].dt.year

        return year_df

    def calculate_end_time(
        self,
        df: pd.DataFrame,
        start_time_column: str = "TIMESTAMP",
        travel_time_column: str = "TRAVEL_TIME",
    ) -> pd.DataFrame:
        """
        Calculate end time by adding travel time to the start time.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame.
        start_time_column : str, optional
            The name of the column that contains start time data (default is "TIMESTAMP").
        travel_time_column : str, optional
            The name of the column that contains travel time data (default is "TRAVEL_TIME").

        Returns:
        -------
        pd.DataFrame
            The DataFrame with a new column 'END_TIME'.

        Raises:
        ------
        ValueError
            If the specified start time or travel time columns do not exist or contain invalid data.
        """
        for col in [start_time_column, travel_time_column]:
            if col not in df.columns:
                raise ValueError(f"The DataFrame does not contain the '{col}' column.")

        if df.empty:
            raise IndexError("Cannot calculate end time for an empty DataFrame.")

        if not pd.api.types.is_datetime64_any_dtype(df[start_time_column]):
            raise TypeError(f"The '{start_time_column}' column must be datetime type.")

        if not pd.api.types.is_numeric_dtype(df[travel_time_column]):
            raise TypeError(f"The '{travel_time_column}' column must be numeric type.")

        end_df = df.copy()
        end_df["END_TIME"] = end_df[start_time_column] + pd.to_timedelta(
            end_df[travel_time_column], unit="s"
        )
        return end_df

    def save_dataframe_if_not_exists(
        self, df: pd.DataFrame, file_path: str, file_format: str = "csv", **kwargs
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
                self.logger.error(f"Unsupported file format: {file_format}")
                raise ValueError(f"Unsupported file format: {file_format}")

            return True
        else:

            return False

    def save_dataframe_overwrite(
        self, df: pd.DataFrame, file_path: str, file_format: str = "csv", **kwargs
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

    def drop_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by handling missing data.

        Args:
            df (pd.DataFrame): The DataFrame to clean.


        Returns:
            pd.DataFrame: The cleaned DataFrame.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
        """
        # Type Validation
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        self.logger.info("Initiating drop_na operation.")

        initial_shape = df.shape
        df_cleaned = df.dropna().reset_index(drop=True)
        final_shape = df_cleaned.shape
        dropped_nan = initial_shape[0] - final_shape[0]
        self.logger.info(f"Dropped {dropped_nan} rows containing NaN values.")

        return df_cleaned

    def drop_columns(
        self, df: pd.DataFrame, columns_to_drop: list = None
    ) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame from which columns will be dropped.
        columns_to_drop : list, optional
            A list of column names to drop. If not provided, defaults to ["ORIGIN_CALL", "ORIGIN_STAND"].

        Returns:
        -------
        pd.DataFrame
            A new DataFrame with the specified columns removed.

        Raises:
        ------
        ValueError
            If the input DataFrame is empty.
        """

        if columns_to_drop is None:
            columns_to_drop = ["ORIGIN_CALL", "ORIGIN_STAND"]
        if df.empty:
            raise ValueError("The input DataFrame is empty. Cannot drop columns.")
        if columns_to_drop is not None and not isinstance(columns_to_drop, list):
            self.logger.error("columns_to_drop is not a list.")
            raise TypeError("columns_to_drop must be a list of column names.")

        # Drop columns if they exist; ignore otherwise
        dropped_df = df.drop(columns=columns_to_drop, errors="ignore")

        return dropped_df

    def extract_coordinates(
        self, df: pd.DataFrame, column: str, columns_to_add: list = None
    ) -> pd.DataFrame:
        """
        Extracts latitude and longitude coordinates from a specified column and adds them as new columns.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing the coordinate data.
        column : str
            The name of the column containing coordinate pairs (e.g., [LONG, LAT]).
        columns_to_add : list, optional
            A list of new column names to add for the extracted coordinates.
            If not provided, defaults to ["LONG", "LAT"].

        Returns:
        -------
        pd.DataFrame
            A new DataFrame with the extracted coordinate columns added.

        Raises:
        ------
        TypeError:
            If `df` is not a pandas DataFrame.
            If `column` is not a string.
            If `columns_to_add` is provided but is not a list of strings.
        ValueError:
            If `column` does not exist in `df`.
            If the data in `column` is not in the expected format.
            If the length of `columns_to_add` does not match the number of elements in each coordinate pair.
        """
        # Type Validation
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        if not isinstance(column, str):
            self.logger.error("Parameter 'column' must be a string.")
            raise TypeError(
                "Parameter 'column' must be a string representing the column name."
            )

        if columns_to_add is not None:
            if not isinstance(columns_to_add, list):
                self.logger.error(
                    "Parameter 'columns_to_add' must be a list of strings."
                )
                raise TypeError("Parameter 'columns_to_add' must be a list of strings.")
            if not all(isinstance(col, str) for col in columns_to_add):
                self.logger.error("All elements in 'columns_to_add' must be strings.")
                raise TypeError("All elements in 'columns_to_add' must be strings.")
        else:
            columns_to_add = ["LONG", "LAT"]
            self.logger.info(
                "No 'columns_to_add' provided. Using default ['LONG', 'LAT']."
            )

        # Check if the specified column exists
        if column not in df.columns:
            self.logger.error(
                f"The specified column '{column}' does not exist in the DataFrame."
            )
            raise ValueError(
                f"The specified column '{column}' does not exist in the DataFrame."
            )

        self.logger.info(
            f"Extracting coordinates from column '{column}' and adding columns {columns_to_add}."
        )

        # Create a copy to maintain immutability
        coordinate_df = df.copy()

        # Convert the specified column to a list
        try:
            coordinates = coordinate_df[column].tolist()
            # self.logger.debug(f"Coordinates extracted: {coordinates}")
        except Exception as e:
            self.logger.error(f"Error converting column '{column}' to list: {e}")
            raise ValueError(f"Error converting column '{column}' to list: {e}")

        # Validate the format of the coordinates
        if not all(
            isinstance(coord, (list, tuple)) and len(coord) == len(columns_to_add)
            for coord in coordinates
        ):
            self.logger.error(
                f"All entries in column '{column}' must be lists or tuples of length {len(columns_to_add)}."
            )
            raise ValueError(
                f"All entries in column '{column}' must be lists or tuples of length {len(columns_to_add)}."
            )

        # Create a DataFrame from the list of coordinates
        try:
            lat_long_df = pd.DataFrame(coordinates, index=coordinate_df.index)
            # self.logger.debug(
            #     f"Created DataFrame from coordinates: {lat_long_df.head()}"
            # )
        except Exception as e:
            self.logger.error(f"Error creating DataFrame from coordinates: {e}")
            raise ValueError(f"Error creating DataFrame from coordinates: {e}")

        # Assign new column names
        lat_long_df.columns = columns_to_add
        self.logger.info(f"Renamed coordinate columns to {columns_to_add}.")

        # Join the new coordinate columns to the original DataFrame
        try:
            coordinate_df = coordinate_df.join(lat_long_df)
            self.logger.info(
                f"Successfully joined new coordinate columns to the DataFrame."
            )
        except Exception as e:
            self.logger.error(f"Error joining coordinate columns: {e}")
            raise ValueError(f"Error joining coordinate columns: {e}")

        return coordinate_df

    def preprocess(
        self,
        df: pd.DataFrame,
        missing_data_column: str = "MISSING_DATA",
        missing_flag: bool = True,
        timestamp_column: str = "TIMESTAMP",
        polyline_column: str = "POLYLINE",
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
        try:
            self.logger.info("Starting preprocessing pipeline.")

            # Step 1: Remove rows with missing GPS data
            self.logger.info("Removing rows with missing GPS data.")
            df = self.remove_missing_gps(df, missing_data_column, missing_flag)

            # Step XX: Drop Columns
            self.logger.info("Removing Columns from DataFrame")
            df = self.drop_columns(df)

            # Step 2: Convert UNIX timestamps to datetime
            self.logger.info("Converting UNIX timestamps to datetime objects.")
            df = self.convert_timestamp(df, timestamp_column)

            # Step 3: Calculate travel time
            self.logger.info("Calculating travel time.")
            df = self.calculate_travel_time_fifteen_seconds(df, polyline_column)

            # Step 4: Convert Polyline column from string to list
            self.logger.info("Converting Polyline column from string to list.")
            df = self.safe_convert_string_to_list(df, polyline_column)

            # Step 5: Extract start location from Polyline column
            self.logger.info("Extracting start locations from Polyline column.")
            df = self.extract_start_location(df, polyline_column)

            # Step 6: Extract end location from Polyline column
            self.logger.info("Extracting end locations from Polyline column.")
            df = self.extract_end_location(df, polyline_column)

            # Step 6: Extract end location from Polyline column
            self.logger.info("Extracting end time from Starting Time and Travel Time.")
            df = self.calculate_end_time(df, timestamp_column, travel_time_column)

            # Step XX: Remove NaN objects from DataFrame
            self.logger.info("Removing rows with NaN data")
            df = self.drop_nan(df)

            # Stemp XX: Extract Lat Long Coordinates
            coordinate_extractions = [
                ("START", ["START_LONG", "START_LAT"]),
                ("END", ["END_LONG", "END_LAT"]),
                # Add more tuples as needed, e.g.,
                # ("MIDDLE", ["MIDDLE_LONG", "MIDDLE_LAT"]),
            ]

            for column, columns_to_add in coordinate_extractions:
                self.logger.info(
                    f"Extracting Lat/Long Coordinates from column '{column}'"
                )
                df = self.extract_coordinates(df, column, columns_to_add)

            # Step 8: Add weekday output
            self.logger.info("Adding weekday information.")
            df = self.add_weekday(df, timestamp_column)

            # Step 9: Add Month output
            self.logger.info("Adding Month information.")
            df = self.add_month(df, timestamp_column)

            # Step 10: Add Month output
            self.logger.info("Adding Year information.")
            df = self.add_year(df, timestamp_column)

            # Step 11: Save the DataFrame to a CSV file only if it doesn't exist
            file_path = "processed_data/update_taxi_trajectory.csv"
            was_saved = self.save_dataframe_if_not_exists(
                df, file_path, file_format="csv"
            )

            if was_saved:
                self.logger.info(f"File saved to {file_path}.")
            else:
                self.logger.info(f"File {file_path} already exists. Skipping save.")

            self.logger.info("Preprocessing pipeline completed successfully.")
            return df

        except Exception as e:
            self.logger.error(f"An error occurred during preprocessing: {e}")
            raise


if __name__ == "__main__":
    data = "/home/smebellis/ece5831_final_project/data/train.csv"
    df = pd.read_csv(data, nrows=50000)
    dp = DataPreprocessing()
    df = dp.preprocess(df)

    breakpoint()
