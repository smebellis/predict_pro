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

    def test_extract_polyline(self):
        data = {
            "POLYLINE": [
                [-8.639847, 41.159826],
                [-8.640351, 41.159871],
                [-8.642196, 41.160114],
                [-8.644455, 41.160492],
                [-8.646921, 41.160951],
                [-8.649999, 41.161491],
                [-8.653167, 41.162031],
                [-8.656434, 41.16258],
                [-8.660178, 41.163192],
                [-8.663112, 41.163687],
                [-8.666235, 41.1642],
                [-8.669169, 41.164704],
                [-8.670852, 41.165136],
                [-8.670942, 41.166576],
                [-8.66961, 41.167962],
                [-8.668098, 41.168988],
                [-8.66664, 41.170005],
                [-8.665767, 41.170635],
                [-8.66574, 41.170671],
            ]
        }
        df = pd.DataFrame(data)

    def test_extract_start_location(self):
        pass

    def test_extract_end_location(self):
        pass

    def test_safe_convert_string_to_list(self):
        """Test that valid list strings are correctly converted to lists."""
        self.assertEqual(
            self.data_preprocessor.safe_convert_string_to_list("[1, 2, 3]"), [1, 2, 3]
        )
        self.assertEqual(
            self.data_preprocessor.safe_convert_string_to_list("['A', 'B', 'C']"),
            ["A", "B", "C"],
        )
        self.assertEqual(
            self.data_preprocessor.safe_convert_string_to_list("[True, False, True]"),
            [True, False, True],
        )
        self.assertEqual(
            self.data_preprocessor.safe_convert_string_to_list("[1.1, 2.2, 3.3]"),
            [1.1, 2.2, 3.3],
        )

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
