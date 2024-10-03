import numpy as np
import unittest
import pandas as pd

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
            "TIMESTAMP": [1727833970, 1727747570, 1727661170, 1727574770, 1727488370]
        }
        df = pd.DataFrame(data)
        cleaned_df = self.data_preprocessor.convert_timestamp(df)

        if isinstance(cleaned_df, pd.Series):
            cleaned_df = cleaned_df.to_frame()
        expected_df = pd.DataFrame(
            {
                "TIMESTAMP": pd.to_datetime(
                    [1727833970, 1727747570, 1727661170, 1727574770, 1727488370]
                )
            }
        )
        pd.testing.assert_frame_equal(cleaned_df, expected_df)

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
        start_location = df.iloc[0]["POLYLINE"]
        end_location = df.iloc[-1]["POLYLINE"]
        expected_start, expected_end = self.data_preprocessor.extract_polyline(df)
        self.assertEqual(start_location, expected_start)
        self.assertEqual(end_location, expected_end)


if __name__ == "__main__":
    unittest.main()
