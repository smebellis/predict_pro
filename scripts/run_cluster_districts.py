import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cluster_districts import (
    HDBSCAN_Clustering,
    cluster_trip_district,
    cluster_trip_time,
    determine_traffic_status,
    HDBSCAN_Clustering_Aggregated,
    HDBSCAN_Clustering_Aggregated_optimized,
    determine_traffic_status_by_quality,
)
from src.logger import get_logger
from src.utils.helper import (
    parse_arguments,
    read_csv_with_progress,
    save_dataframe_if_not_exists,
)

from src.districts import load_districts


# Set up logging
logger = get_logger(__name__)


def main():
    np.seterr(divide="ignore", invalid="ignore")
    try:
        args = parse_arguments()
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        return

    logger.info("Starting Clustering Pipeline.")

    # Load Districts Data
    logger.info("Loading district boundary data...")
    try:
        porto_districts = load_districts(args.districts_path)
    except porto_districts.DistrictLoadError:
        sys.exit(1)

    try:
        df = read_csv_with_progress(args.output)
        logger.info("CSV file read successfully.")
        clustered_df = cluster_trip_district(df, porto_districts)
        clustered_df = cluster_trip_time(clustered_df)

        clustered_df = HDBSCAN_Clustering_Aggregated_optimized(clustered_df)
        clustered_df = determine_traffic_status_by_quality(clustered_df)

        save_dataframe_if_not_exists(clustered_df, args.save)
        logger.info(f"Clustered DataFrame Save to {args.save}")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    logger.info("Completed Clustering Pipeline...")


if __name__ == "__main__":
    main()
