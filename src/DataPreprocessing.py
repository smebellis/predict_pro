import ast
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from dotenv import load_dotenv
from tqdm import tqdm

# Register tqdm with pandas
tqdm.pandas()

"""There are methods that can be removed beause I moved them to the 
pipeline file.  Also, I think this logger setup in the init can be 
removed.  It is redundant.  I already have a function that sets up a 
logger.  Ideally the logger will be able to list from where is came from.  
This is something to add to the Class.  """

from src.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessing:
    def __init__(
        self,
        districts: Dict = None,
        log_dir: str = "logs",
        log_file: str = "data_preprocessor.log",
    ):
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

        self.districts_df = self.load_districts(districts) if districts else None

    @staticmethod
    def haversine(
        lon1: float,
        lat1: float,
        lon2: Union[float, np.ndarray],
        lat2: Union[float, np.ndarray],
        unit: str = "km",
    ) -> Union[float, np.ndarray]:
        """
        Calculate the great-circle distance between one point and multiple points on the Earth.

        Parameters:
        - lon1 (float): Longitude of the first point in decimal degrees.
        - lat1 (float): Latitude of the first point in decimal degrees.
        - lon2 (float or np.ndarray): Longitude(s) of the second point(s) in decimal degrees.
        - lat2 (float or np.ndarray): Latitude(s) of the second point(s) in decimal degrees.
        - unit (str, optional): Unit of distance ('km', 'miles', 'nmi'). Defaults to 'km'.

        Returns:
        - float or np.ndarray: Distance(s) between the first point and second point(s) in the specified unit.
        """
        # Validate unit
        units = {"km": 6371.0, "miles": 3956.0, "nmi": 3440.0}
        if unit not in units:
            raise ValueError("Unit must be one of 'km', 'miles', or 'nmi'.")

        # Convert decimal degrees to radians
        lon1_rad, lat1_rad = np.radians(lon1), np.radians(lat1)
        lon2_rad, lat2_rad = np.radians(lon2), np.radians(lat2)

        # Compute differences
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        # Haversine formula
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # Calculate distance
        distance = c * units[unit]

        return distance

    def load_districts(self, districts: Dict) -> pd.DataFrame:
        """
        Convert a districts dictionary to a pandas DataFrame with calculated center coordinates.

        Parameters:
        - districts (dict): Dictionary containing district boundary information.

        Returns:
        - pd.DataFrame: Processed DataFrame with district boundaries and center coordinates.
        """
        logger.debug("Converting districts dictionary to DataFrame.")
        districts_df = pd.DataFrame.from_dict(districts, orient="index").reset_index()
        districts_df = districts_df.rename(columns={"index": "DISTRICT_NAME"})
        districts_df["center_lat"] = (
            districts_df["lower_lat"] + districts_df["upper_lat"]
        ) / 2
        districts_df["center_long"] = (
            districts_df["left_long"] + districts_df["right_long"]
        ) / 2
        logger.debug("Calculated center coordinates for districts.")
        return districts_df

    def assign_district(self, row: pd.Series) -> str:
        """
        Assigns a district to a given point.

        Parameters:
        - row (pd.Series): A row containing 'Long' and 'Lat' for the point.

        Returns:
        - str: The name of the closest district containing the point or "no district" if none found.
        """
        if self.districts_df is None:
            logger.error("Districts data not loaded. Cannot assign district.")
            return "no district"

        lon, lat = row["Long"], row["Lat"]

        # Correcting the bounding box logic if necessary
        within_long = (self.districts_df["left_long"] <= lon) & (
            lon <= self.districts_df["right_long"]
        )
        within_lat = (self.districts_df["lower_lat"] <= lat) & (
            lat <= self.districts_df["upper_lat"]
        )
        filtered_districts = self.districts_df[within_long & within_lat]

        if filtered_districts.empty:
            return "no district"

        # Calculate distances to district centers
        distances = self.haversine(
            lon,
            lat,
            filtered_districts["center_long"].values,
            filtered_districts["center_lat"].values,
        )

        # Find the district with the minimum distance
        min_distance_idx = np.argmin(distances)
        closest_district = filtered_districts.iloc[min_distance_idx]["DISTRICT_NAME"]

        logger.debug(f"Assigned district '{closest_district}' to point ({lon}, {lat}).")
        return closest_district

    def assign_districts_to_taxi(
        self,
        taxi_df: pd.DataFrame,
        sample_size: int = 1000,
        use_sample: bool = True,
    ) -> pd.DataFrame:
        """
        Assign district names to taxi data points using the assign_district function.

        Parameters:
        - taxi_df (pd.DataFrame): DataFrame containing taxi trajectory data.
        - sample_size (int, optional): Number of samples to process for testing. Defaults to 1000.
        - use_sample (bool, optional): Whether to process a sample or the entire dataset. Defaults to True.

        Returns:
        - pd.DataFrame: Taxi DataFrame with assigned district names.
        """
        # Rename columns for consistency
        taxi_df = taxi_df.rename(columns={"START_LAT": "Lat", "START_LONG": "Long"})

        # Initialize the 'DISTRICT_NAME' column with 'no district'
        taxi_df["DISTRICT_NAME"] = "no district"

        if use_sample:
            logger.info(f"Sampling {sample_size} records for testing...")
            processed_df = taxi_df.sample(sample_size, random_state=42).copy()

            # Assign districts using the assign_district function
            logger.debug("Assigning districts to sample data.")
            tqdm.pandas(desc="Assigning districts to sample data")
            processed_df["DISTRICT_NAME"] = processed_df.progress_apply(
                self.assign_district, axis=1
            )

            logger.info("District assignment to sample data completed.")
            return processed_df
        else:
            logger.info("Assigning districts to the entire dataset...")
            tqdm.pandas(desc="Assigning districts to entire data")
            taxi_df["DISTRICT_NAME"] = taxi_df.progress_apply(
                self.assign_district, axis=1
            )

            logger.info("District assignment to entire dataset completed.")
            return taxi_df

    def assign_district_vectorized(self, taxi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign district names to taxi data points using vectorized operations for better performance.

        Parameters:
        - taxi_df (pd.DataFrame): DataFrame containing taxi trajectory data.

        Returns:
        - pd.DataFrame: Taxi DataFrame with assigned district names.
        """
        if self.districts_df is None:
            logger.error("Districts data not loaded. Cannot assign districts.")
            taxi_df["DISTRICT_NAME"] = "no district"
            return taxi_df

        # Initialize 'DISTRICT_NAME' with 'no district'
        taxi_df["DISTRICT_NAME"] = "no district"

        logger.info("Starting vectorized district assignment.")
        for _, district in tqdm(
            self.districts_df.iterrows(),
            total=self.districts_df.shape[0],
            desc="Assigning districts",
        ):
            condition = (
                (taxi_df["Long"] >= district["left_long"])
                & (taxi_df["Long"] <= district["right_long"])
                & (taxi_df["Lat"] >= district["lower_lat"])
                & (taxi_df["Lat"] <= district["upper_lat"])
                & (taxi_df["DISTRICT_NAME"] == "no district")  # Prevent overwriting
            )
            taxi_df.loc[condition, "DISTRICT_NAME"] = district["DISTRICT_NAME"]

        logger.info("Vectorized district assignment completed.")
        return taxi_df

    def assign_districts_to_taxi_vectorized(
        self,
        taxi_df: pd.DataFrame,
        sample_size: int = 1000,
        use_sample: bool = True,
    ) -> pd.DataFrame:
        """
        Assign district names to taxi data points using the vectorized assign_district_vectorized function.

        Parameters:
        - taxi_df (pd.DataFrame): DataFrame containing taxi trajectory data.
        - sample_size (int, optional): Number of samples to process for testing. Defaults to 1000.
        - use_sample (bool, optional): Whether to process a sample or the entire dataset. Defaults to True.

        Returns:
        - pd.DataFrame: Taxi DataFrame with assigned district names.
        """
        # Rename columns for consistency
        taxi_df = taxi_df.rename(columns={"START_LAT": "Lat", "START_LONG": "Long"})

        if use_sample:
            logger.info(f"Sampling {sample_size} records for vectorized assignment.")
            processed_df = taxi_df.sample(sample_size, random_state=42).copy()
            processed_df = self.assign_district_vectorized(processed_df)
            logger.info("Vectorized district assignment to sample data completed.")
            return processed_df
        else:
            logger.info(
                "Assigning districts to the entire dataset using vectorized method."
            )
            taxi_df = self.assign_district_vectorized(taxi_df)
            logger.info("Vectorized district assignment to entire dataset completed.")
            return taxi_df

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
            for poly in df["POLYLINE_LIST"]
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
            for poly in df["POLYLINE_LIST"]
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
        tqdm.pandas(desc="Converting POLYLINE to a list")
        df_converted["POLYLINE_LIST"] = df_converted[polyline_column].progress_apply(
            ast.literal_eval
        )
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
        weekday_df["WEEKDAY_NUM"] = weekday_df[timestamp_column].dt.weekday

        return weekday_df

    def add_time(
        self, df: pd.DataFrame, timestamp_column: str = "TIMESTAMP"
    ) -> pd.DataFrame:
        """
        Calculate the time of day from the timestamp column and add it as a new column.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with a datetime timestamp column.
        timestamp_column : str, optional
            The name of the column that contains datetime timestamp data (default is "TIMESTAMP").

        Returns:
        -------
        pd.DataFrame
            The DataFrame with a new column named 'TIME'.

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
            raise IndexError("Cannot add 'TIME' to an empty DataFrame.")

        # Make a copy to avoid altering original dataframe
        time_df = df.copy()

        time_df["TIME"] = time_df[timestamp_column].dt.hour

        return time_df

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
        self,
        df: pd.DataFrame,
        file_path: str,
        file_format: str = "csv",
        **kwargs,
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
                logger.error(f"Unsupported file format: {file_format}")
                raise ValueError(f"Unsupported file format: {file_format}")

            return True
        else:

            return False

    def save_dataframe_overwrite(
        self,
        df: pd.DataFrame,
        file_path: str,
        file_format: str = "csv",
        **kwargs,
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
            logger.error("Input is not a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        logger.info("Initiating drop_na operation.")

        initial_shape = df.shape
        df_cleaned = df.dropna().reset_index(drop=True)
        final_shape = df_cleaned.shape
        dropped_nan = initial_shape[0] - final_shape[0]
        logger.info(f"Dropped {dropped_nan} rows containing NaN values.")

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
            logger.error("columns_to_drop is not a list.")
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
            logger.error("Input is not a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        if not isinstance(column, str):
            logger.error("Parameter 'column' must be a string.")
            raise TypeError(
                "Parameter 'column' must be a string representing the column name."
            )

        if columns_to_add is not None:
            if not isinstance(columns_to_add, list):
                logger.error("Parameter 'columns_to_add' must be a list of strings.")
                raise TypeError("Parameter 'columns_to_add' must be a list of strings.")
            if not all(isinstance(col, str) for col in columns_to_add):
                logger.error("All elements in 'columns_to_add' must be strings.")
                raise TypeError("All elements in 'columns_to_add' must be strings.")
        else:
            columns_to_add = ["LONG", "LAT"]
            logger.info("No 'columns_to_add' provided. Using default ['LONG', 'LAT'].")

        # Check if the specified column exists
        if column not in df.columns:
            logger.error(
                f"The specified column '{column}' does not exist in the DataFrame."
            )
            raise ValueError(
                f"The specified column '{column}' does not exist in the DataFrame."
            )

        logger.info(
            f"Extracting coordinates from column '{column}' and adding columns {columns_to_add}."
        )

        # Create a copy to maintain immutability
        coordinate_df = df.copy()

        # Convert the specified column to a list
        try:
            coordinates = coordinate_df[column].tolist()
            # logger.debug(f"Coordinates extracted: {coordinates}")
        except Exception as e:
            logger.error(f"Error converting column '{column}' to list: {e}")
            raise ValueError(f"Error converting column '{column}' to list: {e}")

        # Validate the format of the coordinates
        if not all(
            isinstance(coord, (list, tuple)) and len(coord) == len(columns_to_add)
            for coord in coordinates
        ):
            logger.error(
                f"All entries in column '{column}' must be lists or tuples of length {len(columns_to_add)}."
            )
            raise ValueError(
                f"All entries in column '{column}' must be lists or tuples of length {len(columns_to_add)}."
            )

        # Create a DataFrame from the list of coordinates
        try:
            lat_long_df = pd.DataFrame(coordinates, index=coordinate_df.index)
            # logger.debug(
            #     f"Created DataFrame from coordinates: {lat_long_df.head()}"
            # )
        except Exception as e:
            logger.error(f"Error creating DataFrame from coordinates: {e}")
            raise ValueError(f"Error creating DataFrame from coordinates: {e}")

        # Assign new column names
        lat_long_df.columns = columns_to_add
        logger.info(f"Renamed coordinate columns to {columns_to_add}.")

        # Join the new coordinate columns to the original DataFrame
        try:
            coordinate_df = coordinate_df.join(lat_long_df)
            logger.info(f"Successfully joined new coordinate columns to the DataFrame.")
        except Exception as e:
            logger.error(f"Error joining coordinate columns: {e}")
            raise ValueError(f"Error joining coordinate columns: {e}")

        return coordinate_df

    def preprocess(
        self,
        df: pd.DataFrame,
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
        try:
            logger.info("Starting preprocessing pipeline.")

            # Step 1: Remove rows with missing GPS data
            logger.info("Removing rows with missing GPS data.")
            df = self.remove_missing_gps(df, missing_data_column, missing_flag)

            # Step XX: Drop Columns
            logger.info("Removing Columns from DataFrame")
            df = self.drop_columns(df)

            # Step 2: Convert UNIX timestamps to datetime
            logger.info("Converting UNIX timestamps to datetime objects.")
            df = self.convert_timestamp(df, timestamp_column)

            # Step 3: Calculate travel time
            logger.info("Calculating travel time.")
            df = self.calculate_travel_time_fifteen_seconds(df, polyline_column)

            # Step 4: Convert Polyline column from string to list
            logger.info("Converting Polyline column from string to list.")
            df = self.safe_convert_string_to_list(df, polyline_column)

            # Step 5: Extract start location from Polyline column
            logger.info("Extracting start locations from Polyline column.")
            df = self.extract_start_location(df, polyline_list_column)

            # Step 6: Extract end location from Polyline column
            logger.info("Extracting end locations from Polyline column.")
            df = self.extract_end_location(df, polyline_list_column)

            # Step 6: Extract end location from Polyline column
            logger.info("Extracting end time from Starting Time and Travel Time.")
            df = self.calculate_end_time(df, timestamp_column, travel_time_column)

            # Step XX: Remove NaN objects from DataFrame
            logger.info("Removing rows with NaN data")
            df = self.drop_nan(df)

            # Stemp XX: Extract Lat Long Coordinates
            coordinate_extractions = [
                ("START", ["START_LONG", "START_LAT"]),
                ("END", ["END_LONG", "END_LAT"]),
            ]

            for column, columns_to_add in coordinate_extractions:
                logger.info(f"Extracting Lat/Long Coordinates from column '{column}'")
                df = self.extract_coordinates(df, column, columns_to_add)

            # Step 8: Add weekday output
            logger.info("Adding weekday information.")
            df = self.add_weekday(df, timestamp_column)

            # Step 9: Add Month output
            logger.info("Adding Month information.")
            df = self.add_month(df, timestamp_column)

            # Step 10: Add Month output
            logger.info("Adding Year information.")
            df = self.add_year(df, timestamp_column)

            # Step 11: Save the DataFrame to a CSV file only if it doesn't exist
            file_path = "processed_data/update_taxi_trajectory1.csv"
            was_saved = self.save_dataframe_if_not_exists(
                df, file_path, file_format="csv"
            )

            if was_saved:
                logger.info(f"File saved to {file_path}.")
            else:
                logger.info(f"File {file_path} already exists. Skipping save.")

            logger.info("Preprocessing pipeline completed successfully.")
            return df

        except Exception as e:
            logger.error(f"An error occurred during preprocessing: {e}")
            raise


if __name__ == "__main__":
    data = "/home/smebellis/ece5831_final_project/data/train.csv"
    df = pd.read_csv(data, nrows=50000)
    dp = DataPreprocessing()
    df = dp.preprocess(df)

    breakpoint()
