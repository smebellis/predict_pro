import numpy as np
import pandas as pd
import ast
import logging
from typing import Optional, Tuple


class DataPreprocessing:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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

        df[timestamp_column] = pd.to_datetime(
            df[timestamp_column], unit="s"
        ).dt.tz_localize(None)

        return df

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

        df["START"] = [
            poly[0] if isinstance(poly, list) and len(poly) > 0 else None
            for poly in df["POLYLINE"]
        ]

        return df

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

        df["END"] = [
            poly[-1] if isinstance(poly, list) and len(poly) > 0 else None
            for poly in df["POLYLINE"]
        ]

        return df

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

    def preprocess(
        self,
        df: pd.DataFrame,
        missing_data_column: str = "MISSING_DATA",
        missing_flag: bool = True,
        timestamp_column: str = "TIMESTAMP",
        polyline_column: str = "POLYLINE",
    ) -> pd.DataFrame:
        """
        Apply a series of preprocessing steps to the DataFrame:
        1. Remove rows with missing GPS data.
        2. Convert UNIX timestamps to datetime objects.
        3. Calculate travel time based on polyline data.
        4. Convert Polyline column to list
        5. Extract start location from Polyline column
        6. Extract end location from Polyline column
        7. Add weekday output

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to preprocess.
        missing_data_column : str, optional
            The column indicating missing data (default is "MISSING_DATA").
        missing_flag : bool, optional
            The flag value that indicates missing data (default is True).
        timestamp_column : str, optional
            The column containing UNIX timestamps (default is "TIMESTAMP").
        polyline_column : str, optional
            The column containing polyline data (default is "POLYLINE").

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

            # Step 7: Add weekday output
            self.logger.info("Adding weekday information.")
            df = self.add_weekday(df, timestamp_column)

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
