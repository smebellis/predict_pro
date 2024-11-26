# districts.py

import json
import sys
import logging
from pathlib import Path
from typing import Dict

from src.logger import get_logger

# Set up logging
logger = get_logger(__name__)


class DistrictLoadError(Exception):
    pass


def load_districts(districts_path: Path) -> Dict:
    """
    Loads district data from a JSON file.

    Args:
        districts_path (str): Path to the districts JSON file.

    Returns:
        dict: Districts data.

    Raises:
        SystemExit: If any error occurs during loading.
    """
    logger.info("Loading district boundary data...")
    try:
        with open(districts_path, "r") as file:
            districts = json.load(file)
        if not isinstance(districts, dict) or not districts:
            raise DistrictLoadError("Districts should be a non-empty dictionary")
        return districts
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(e)
        raise DistrictLoadError from e
