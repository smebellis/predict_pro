import ast
import logging
import os
from datetime import datetime
import json
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

from logger import get_logger

logger = get_logger(__name__)


class Preprocessing:
    def __init__(
        self,
        districts: Dict = None,
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

    def assign_districts(
        self,
        taxi_df: pd.DataFrame,
        method: str = "vectorized",
        sample_size: int = 1000,
        use_sample: bool = False,
    ) -> pd.DataFrame:
        """
        Assign district names to taxi data points using the specified method.

        Parameters:
        ----------
        taxi_df : pd.DataFrame
            DataFrame containing taxi trajectory data.
        method : str, optional
            Method to use ('vectorized' or 'row-wise'). Defaults to 'vectorized'.
        sample_size : int, optional
            Number of samples to process for testing. Defaults to 1000.
        use_sample : bool, optional
            Whether to process a sample or the entire dataset. Defaults to False.

        Returns:
        -------
        pd.DataFrame
            Taxi DataFrame with assigned district names.
        """
        # Initialize 'DISTRICT_NAME' column if it doesn't exist
        if "DISTRICT_NAME" not in taxi_df.columns:
            taxi_df["DISTRICT_NAME"] = "no district"

        if use_sample:
            logger.info(f"Sampling {sample_size} records for testing...")
            taxi_df = taxi_df.sample(sample_size, random_state=42).copy()

        if method == "vectorized":
            return self.assign_district_vectorized(taxi_df)
        elif method == "row-wise":
            tqdm.pandas(desc="Assigning districts")
            taxi_df["DISTRICT_NAME"] = taxi_df.progress_apply(
                self.assign_district, axis=1
            )
        else:
            raise ValueError(
                f"Unknown method '{method}' specified. Use 'vectorized' or 'row-wise'."
            )

        logger.info("District assignment completed.")
        return taxi_df

    def assign_district_vectorized(self, taxi_df: pd.DataFrame) -> pd.DataFrame:
        if self.districts_df is None:
            logger.error("Districts data not loaded. Cannot assign districts.")
            taxi_df["DISTRICT_NAME"] = "no district"
            return taxi_df

        logger.info("Starting vectorized district assignment.")
        for _, district in tqdm(
            self.districts_df.iterrows(),
            total=self.districts_df.shape[0],
            desc="Assigning districts",
        ):
            condition = (
                (taxi_df["START_LONG"] >= district["left_long"])
                & (taxi_df["START_LONG"] <= district["right_long"])
                & (taxi_df["START_LAT"] >= district["lower_lat"])
                & (taxi_df["START_LAT"] <= district["upper_lat"])
                & (taxi_df["DISTRICT_NAME"] == "no district")
            )
            # Log the number of matches for debugging
            num_matches = condition.sum()
            logger.debug(
                f"Number of matches for district '{district['DISTRICT_NAME']}': {num_matches}"
            )

            taxi_df.loc[condition, "DISTRICT_NAME"] = district["DISTRICT_NAME"]

        logger.info("Vectorized district assignment completed.")
        return taxi_df

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

        lon, lat = row["START_LONG"], row["START_LAT"]

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

    def calculate_travel_time(
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

    def convert_coordinates(self, string):
        """
        Loads list of coordinates from given string and swap out longitudes & latitudes.
        We do the swapping because the standard is to have latitude values first, but
        the original datasets provided in the competition have it backwards.
        """
        return [(lat, long) for (long, lat) in json.loads(string)]

    def convert_polyline_to_list(
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
        df_converted[polyline_column] = df_converted[polyline_column].progress_apply(
            self.convert_coordinates
        )
        return df_converted

    def extract_coordinates(
        self, df: pd.DataFrame, polyline_column: str = "POLYLINE"
    ) -> pd.DataFrame:
        """
        Extract some features from the original columns in the given dataset.
        """
        if polyline_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{polyline_column}' column."
            )

        if df.empty:
            raise IndexError(
                "The input DataFrame is empty. Returning the DataFrame as is."
            )
        extract_df = df.copy()

        # Extract Starting latitudes and longitudes
        extract_df["START_LAT"] = extract_df["POLYLINE"].apply(
            lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else None
        )
        extract_df["START_LONG"] = extract_df["POLYLINE"].apply(
            lambda x: x[0][1] if len(x) > 0 and len(x[0]) > 0 else None
        )
        extract_df["END_LAT"] = extract_df["POLYLINE"].apply(
            lambda x: x[-1][0] if len(x) > 0 and len(x[0]) > 0 else None
        )
        extract_df["END_LONG"] = extract_df["POLYLINE"].apply(
            lambda x: x[-1][1] if len(x) > 0 and len(x[0]) > 0 else None
        )

        return extract_df

    def separate_timestamp(
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
        timestamp_df = df.copy()

        timestamp_df["WEEKDAY"] = timestamp_df[timestamp_column].dt.day_name()
        timestamp_df["TIME"] = timestamp_df[timestamp_column].dt.hour
        timestamp_df["MONTH"] = timestamp_df[timestamp_column].dt.month_name()
        timestamp_df["YEAR"] = timestamp_df[timestamp_column].dt.year

        return timestamp_df

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


if __name__ == "__main__":
    data = "/home/smebellis/ece5831_final_project/data/train.csv"
    df = pd.read_csv(data, nrows=50000)
    dp = Preprocessing()

    breakpoint()
