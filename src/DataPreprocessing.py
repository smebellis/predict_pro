import numpy as np
import pandas as pd
from typing import Optional


class DataPreprocessing:
    def __init__(self) -> None:
        pass

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

    def extract_polyline(
        self, df: pd.DataFrame, data_column: str = "POLYLINE"
    ) -> pd.DataFrame:
        """
        Extracts the start and end POLYLINE locations from the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with a 'POLYLINE' column.

        Returns:
        -------
        tuple:
            (start_location, end_location)
        """
        if data_column not in df.columns:
            raise ValueError(
                f"The DataFrame does not contain the '{data_column}' column."
            )
        if df.empty:
            raise IndexError("Cannot extract POLYLINE from an empty DataFrame.")
        start_location = df.iloc[0]["POLYLINE"]
        end_location = df.iloc[-1]["POLYLINE"]

        return start_location, end_location

    def preprocess(self):
        self.remove_missing_gps()
