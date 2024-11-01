import unittest
import tempfile
from pathlib import Path
import pandas as pd
import logging
from unittest import mock
from helper import (
    file_load,
)


class TestFileLoad(unittest.TestCase):
    def setUp(self):
        # Patch the logger to capture logging outputs
        self.log_patcher = mock.patch("logging.info")
        self.mock_info = self.log_patcher.start()

        self.error_patcher = mock.patch("logging.error")
        self.mock_error = self.error_patcher.start()

    def tearDown(self):
        self.log_patcher.stop()
        self.error_patcher.stop()

    def test_load_valid_csv(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
            tmp.write("col1,col2\n1,2\n3,4")
            tmp_path = Path(tmp.name)

        try:
            df = file_load(tmp_path)
            expected_df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
            pd.testing.assert_frame_equal(df, expected_df)
            self.mock_info.assert_called_with(f"Reading CSV file: {tmp_path}")
        finally:
            tmp_path.unlink()

    def test_load_valid_excel(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Create a sample Excel file
            df_expected = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})
            df_expected.to_excel(tmp_path, index=False)

        try:
            df = file_load(tmp_path)
            pd.testing.assert_frame_equal(df, df_expected)
            self.mock_info.assert_called_with(f"Reading Excel file: {tmp_path}")
        finally:
            tmp_path.unlink()

    def test_load_valid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as tmp:
            json_content = '{"col1": [9, 10], "col2": [11, 12]}'
            tmp.write(json_content)
            tmp_path = Path(tmp.name)

        try:
            df = file_load(tmp_path)
            expected_df = pd.DataFrame({"col1": [9, 10], "col2": [11, 12]})
            pd.testing.assert_frame_equal(df, expected_df)
            self.mock_info.assert_called_with(f"Reading JSON file: {tmp_path}")
        finally:
            tmp_path.unlink()

    def test_load_non_existing_file(self):
        non_existing_path = Path("non_existent_file.csv")
        with self.assertRaises(FileNotFoundError) as context:
            file_load(non_existing_path)
        self.mock_error.assert_called_with(f"File does not exist: {non_existing_path}")
        self.assertIn("File does not exist", str(context.exception))


if __name__ == "__main__":
    unittest.main()
