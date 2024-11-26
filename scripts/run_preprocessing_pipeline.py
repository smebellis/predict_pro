import os
import sys

import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.districts import load_districts
from src.helper import parse_arguments
from src.pipeline import preprocessing_pipeline
from src.logger import get_logger
from src.Preprocessing import Preprocessing

# Set up logging
logger = get_logger(__name__)


def main():
    try:
        args = parse_arguments()
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        return

    logger.info("Starting Preprocessing Pipeline.")

    # Load Districts Data
    logger.info("Loading district boundary data...")
    try:
        porto_districts = load_districts(args.districts_path)
    except porto_districts.DistrictLoadError:
        sys.exit(1)

    try:
        preprocessor = Preprocessing(districts=porto_districts)

        preprocessing_pipeline(
            args.input, args.output, district=porto_districts, preprocessor=preprocessor
        )

    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    logger.info("Preprocessing Pipeline Completed")


if __name__ == "__main__":
    main()
