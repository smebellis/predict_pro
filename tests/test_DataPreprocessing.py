import numpy as np
import unittest
import pandas as pd
import ast
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from src.DataPreprocessing import DataPreprocessing


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):

        self.data_preprocessor = DataPreprocessing()

    def test_remove_missing_gps(self):
        data = {"MISSING_DATA": [False, False, True, False]}
        df = pd.DataFrame(data)
        cleaned_df = self.data_preprocessor.remove_missing_gps(df)
        expected_df = pd.DataFrame({"MISSING_DATA": [False, False, False]})
        pd.testing.assert_frame_equal(cleaned_df, expected_df)

    def test_remove_nan(self):
        """Test dropping NaN with default parameters (drop_na=True, inplace=False)."""
        data = {"NAN_DATA": ["1", "2", None, "3"]}
        df = pd.DataFrame(data)

        actual_df = self.data_preprocessor.drop_nan(df)
        expected_df = pd.DataFrame({"NAN_DATA": ["1", "2", "3"]})

        pd.testing.assert_frame_equal(actual_df, expected_df)
        # Ensure original DataFrame is unchanged
        self.assertTrue(df.isnull().any().any())

    def test_convert_timestamp(self):
        data = {
            "TIMESTAMP": [1672024955, 357223268, 1286429319, 1217817752, 1057457493]
        }
        df = pd.DataFrame(data)
        actual_df = self.data_preprocessor.convert_timestamp(df)

        expected_data = {
            "TIMESTAMP": [
                pd.Timestamp("2022-12-26 03:22:35", unit="s").tz_localize(None),
                pd.Timestamp("1981-04-27 12:41:08", unit="s").tz_localize(None),
                pd.Timestamp("2010-10-07 05:28:39", unit="s").tz_localize(None),
                pd.Timestamp("2008-08-04 02:42:32", unit="s").tz_localize(None),
                pd.Timestamp("2003-07-06 02:11:33", unit="s").tz_localize(None),
            ]
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_extract_start_location(self):
        # Input DataFrame with 2 polylines
        data = {
            "POLYLINE": [
                [[-8.618643, 41.141412], [-8.618499, 41.141376]],
                [[-8.618300, 41.141200], [-8.618100, 41.141100]],
            ]
        }
        df = pd.DataFrame(data)

        # Call the method to test
        actual_output = self.data_preprocessor.extract_start_location(df)

        # Define the expected output DataFrame
        expected_data = {
            "POLYLINE": [
                [[-8.618643, 41.141412], [-8.618499, 41.141376]],
                [[-8.618300, 41.141200], [-8.618100, 41.141100]],
            ],
            "START": [
                [-8.618643, 41.141412],
                [-8.618300, 41.141200],
            ],
        }
        expected_output = pd.DataFrame(expected_data)

        # Reset index to ensure alignment
        actual_output = actual_output.reset_index(drop=True)
        expected_output = expected_output.reset_index(drop=True)

        # Assert that the actual output matches the expected output
        pd.testing.assert_frame_equal(actual_output, expected_output)

    def test_extract_coordinates(self):
        """Test extracting coordinates with default columns_to_add."""
        data = {
            "ID": [1, 2],
            "LOCATION": [[-8.618643, 41.141412], [-8.618499, 41.141376]],
            "OTHER_DATA": ["A", "B"],
        }
        df = pd.DataFrame(data)

        expected_data = {
            "ID": [1, 2],
            "LOCATION": [[-8.618643, 41.141412], [-8.618499, 41.141376]],
            "OTHER_DATA": ["A", "B"],
            "LONG": [-8.618643, -8.618499],
            "LAT": [41.141412, 41.141376],
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = self.data_preprocessor.extract_coordinates(df=df, column="LOCATION")

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_extract_end_location(self):
        # Input DataFrame with 2 polylines
        data = {
            "POLYLINE": [
                [[-8.618643, 41.141412], [-8.618499, 41.141376]],
                [[-8.618300, 41.141200], [-8.618100, 41.141100]],
            ]
        }
        df = pd.DataFrame(data)

        # Call the method to test
        actual_output = self.data_preprocessor.extract_end_location(df)

        # Define the expected output DataFrame
        expected_data = {
            "POLYLINE": [
                [[-8.618643, 41.141412], [-8.618499, 41.141376]],
                [[-8.618300, 41.141200], [-8.618100, 41.141100]],
            ],
            "END": [
                [-8.618499, 41.141376],
                [-8.618100, 41.141100],
            ],
        }
        expected_output = pd.DataFrame(expected_data)

        # Reset index to ensure alignment
        actual_output = actual_output.reset_index(drop=True)
        expected_output = expected_output.reset_index(drop=True)

        # Assert that the actual output matches the expected output
        pd.testing.assert_frame_equal(actual_output, expected_output)

    def test_add_weekday(self):
        # Prepare initial data
        data = {
            "TIMESTAMP": [1672024955, 357223268, 1286429319, 1217817752, 1057457493]
        }
        df = pd.DataFrame(data)

        # Convert timestamps first
        converted_df = self.data_preprocessor.convert_timestamp(df)

        # Add WEEKDAY
        actual_df = self.data_preprocessor.add_weekday(converted_df)

        # Define expected data
        expected_data = {
            "TIMESTAMP": [
                pd.Timestamp("2022-12-26 03:22:35"),
                pd.Timestamp("1981-04-27 12:41:08"),
                pd.Timestamp("2010-10-07 05:28:39"),
                pd.Timestamp("2008-08-04 02:42:32"),
                pd.Timestamp("2003-07-06 02:11:33"),
            ],
            "WEEKDAY": ["Monday", "Monday", "Thursday", "Monday", "Sunday"],
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert both columns
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_add_month(self):
        # Prepare initial data
        data = {
            "TIMESTAMP": [1672024955, 357223268, 1286429319, 1217817752, 1057457493]
        }
        df = pd.DataFrame(data)

        # Convert timestamps first
        converted_df = self.data_preprocessor.convert_timestamp(df)

        # Add WEEKDAY
        actual_df = self.data_preprocessor.add_month(converted_df)

        # Define expected data
        expected_data = {
            "TIMESTAMP": [
                pd.Timestamp("2022-12-26 03:22:35"),
                pd.Timestamp("1981-04-27 12:41:08"),
                pd.Timestamp("2010-10-07 05:28:39"),
                pd.Timestamp("2008-08-04 02:42:32"),
                pd.Timestamp("2003-07-06 02:11:33"),
            ],
            "MONTH": ["December", "April", "October", "August", "July"],
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert both columns
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_add_year(self):

        # Prepare initial data
        data = {
            "TIMESTAMP": [1672024955, 357223268, 1286429319, 1217817752, 1057457493]
        }
        df = pd.DataFrame(data)

        # Convert timestamps first
        converted_df = self.data_preprocessor.convert_timestamp(df)

        # Add WEEKDAY
        actual_df = self.data_preprocessor.add_year(converted_df)

        # Define expected data
        expected_data = {
            "TIMESTAMP": [
                pd.Timestamp("2022-12-26 03:22:35"),
                pd.Timestamp("1981-04-27 12:41:08"),
                pd.Timestamp("2010-10-07 05:28:39"),
                pd.Timestamp("2008-08-04 02:42:32"),
                pd.Timestamp("2003-07-06 02:11:33"),
            ],
            "YEAR": [2022, 1981, 2010, 2008, 2003],
        }
        expected_df = pd.DataFrame(expected_data)
        # Convert YEAR column to int32
        expected_df["YEAR"] = expected_df["YEAR"].astype("int32")
        # Assert both columns
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_safe_convert_string_to_list(self):
        # Input DataFrame with POLYLINE as string representations of lists
        data = {
            "POLYLINE": [
                "[[-8.618643, 41.141412], [-8.618499, 41.141376]]",
                "[[-8.618300, 41.141200], [-8.618100, 41.141100]]",
            ]
        }
        df = pd.DataFrame(data)

        # Call the method to test
        actual_output = self.data_preprocessor.safe_convert_string_to_list(df)

        # Define the expected output as a DataFrame with actual lists
        expected_data = {
            "POLYLINE": [
                [[-8.618643, 41.141412], [-8.618499, 41.141376]],
                [[-8.618300, 41.141200], [-8.618100, 41.141100]],
            ]
        }
        expected_output = pd.DataFrame(expected_data)

        # Use assert_frame_equal to compare the actual and expected DataFrames
        pd.testing.assert_frame_equal(actual_output, expected_output)

    def test_calculate_travel_time_fifteen_seconds(self):
        data = {
            "POLYLINE": [
                [
                    (41.1471, -8.6139),
                    (41.1472, -8.6140),
                    (41.1473, -8.6141),
                ],  # 2 intervals, 2 * 15 = 30 seconds
                [
                    (41.1481, -8.6131),
                    (41.1482, -8.6132),
                ],  # 1 interval, 1 * 15 = 15 seconds
                [(41.1491, -8.6121)],  # 0 interval, 0 * 15 = 0 seconds
            ]
        }
        df = pd.DataFrame(data)
        actual_time = self.data_preprocessor.calculate_travel_time_fifteen_seconds(df)
        # Assert that the travel time has been correctly calculated
        expected_times = [30, 15, 0]
        self.assertListEqual(list(df["TRAVEL_TIME"]), expected_times)

    def test_calculate_end_time(self):
        data = {
            "TIMESTAMP": [
                pd.Timestamp("2022-12-26 03:22:35"),
                pd.Timestamp("1981-04-27 12:41:08"),
            ],
            "TRAVEL_TIME": [30, 15],
        }
        df = pd.DataFrame(data)
        expected_end_time = [
            pd.Timestamp("2022-12-26 03:23:05"),
            pd.Timestamp("1981-04-27 12:41:23"),
        ]
        result_df = self.data_preprocessor.calculate_end_time(df)
        self.assertListEqual(result_df["END_TIME"].tolist(), expected_end_time)

    def test_save_dataframe_if_not_exists(self):
        self.sample_df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.csv")
            was_saved = self.data_preprocessor.save_dataframe_if_not_exists(
                self.sample_df, file_path, file_format="csv"
            )
            self.assertTrue(was_saved)
            self.assertTrue(Path(file_path).exists())

    def test_save_csv_overwrite_file_exists(self):
        self.sample_df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.csv")

            # First save
            self.data_preprocessor.save_dataframe_overwrite(
                self.sample_df, file_path, file_format="csv"
            )
            self.assertTrue(Path(file_path).exists(), "CSV file was not created.")

            # Modify the DataFrame
            modified_df = pd.DataFrame({"A": [4, 5, 6], "B": ["u", "v", "w"]})

            # Overwrite the existing file
            self.data_preprocessor.save_dataframe_overwrite(
                modified_df, file_path, file_format="csv"
            )

            # Read back the file to ensure it's overwritten
            saved_df = pd.read_csv(file_path)
            pd.testing.assert_frame_equal(saved_df, modified_df, check_dtype=True)

    def test_drop_columns(self):
        """Test dropping columns that exist in the DataFrame."""
        data = {
            "ORIGIN_CALL": ["A", "B"],
            "ORIGIN_STAND": [1, 2],
            "OTHER_COLUMN": ["X", "Y"],
        }
        df = pd.DataFrame(data)
        expected_data = {"OTHER_COLUMN": ["X", "Y"]}
        expected_df = pd.DataFrame(expected_data)
        result_df = self.data_preprocessor.drop_columns(df)
        pd.testing.assert_frame_equal(
            result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )


if __name__ == "__main__":
    unittest.main()
