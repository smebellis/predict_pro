import os
import sys
import swifter
import ast

import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cluster_districts import (
    HDBSCAN_Clustering_Aggregated_optimized,
    cluster_trip_district,
    cluster_trip_time,
    determine_traffic_status_by_quality,
    determine_traffic_status,
    encode_geographical_context,
    aggregate_district_clusters,
    traffic_congestion_indicator,
    add_temporal_context,
    aggregate_historical_data,
)
from src.districts import load_districts
from src.helper import (
    parse_arguments,
    read_csv_with_progress,
    save_dataframe_if_not_exists,
    save_dataframe_overwrite,
)
from src.logger import get_logger

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
        df = read_csv_with_progress(args.input_processed)
        df = df.sample(n=5000, random_state=42)
        logger.info("CSV file read successfully.")

        # Check if 'DISTRICT_NAME' column exists
        if "DISTRICT_NAME" not in df.columns:
            logger.error(
                "The required 'DISTRICT_NAME' column is missing in the input CSV."
            )
            sys.exit(1)
        clustered_df = cluster_trip_district(df, porto_districts)
        clustered_df = cluster_trip_time(clustered_df)

        clustered_df = HDBSCAN_Clustering_Aggregated_optimized(clustered_df)
        clustered_df = determine_traffic_status(clustered_df)
        clustered_df = encode_geographical_context(clustered_df)
        clustered_df = aggregate_district_clusters(clustered_df)
        clustered_df = traffic_congestion_indicator(clustered_df)
        clustered_df = add_temporal_context(clustered_df)
        # clustered_df = aggregate_historical_data(clustered_df)
        save_dataframe_overwrite(clustered_df, args.save)
        logger.info(f"Clustered DataFrame Save to {args.save}")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    logger.info("Completed Clustering Pipeline...")


if __name__ == "__main__":
    main()
