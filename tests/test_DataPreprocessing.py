import numpy as np
import unittest
import pandas as pd
import ast

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


if __name__ == "__main__":
    unittest.main()
