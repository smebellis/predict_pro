import ast
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.districts import DistrictLoadError, load_districts
from src.helper import read_csv_with_progress, save_dataframe_if_not_exists
from src.logger import get_logger

logger = get_logger(__name__)


# Function to parse POLYLINE data
def parse_polyline(polyline_str):
    try:
        polyline = json.loads(polyline_str)
        if not isinstance(polyline, list) or len(polyline) < 2:
            raise ValueError("Polyline must be a list with at least two points.")
        return polyline
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Invalid POLYLINE format: {e}")
        return None


def determine_optimal_clusters(data, max_k=10):
    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

    plt.plot(range(1, max_k + 1), inertia_values, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method For Optimal k")
    plt.show()


def extract_features(polyline):
    """
    Extract meaningful features from a polyline for clustering.
    """
    polyline_points = np.array(polyline)
    if polyline_points.ndim != 2 or polyline_points.shape[1] != 2:
        return None
    mean_coords = polyline_points.mean(axis=0)
    total_length = np.sum(np.linalg.norm(np.diff(polyline_points, axis=0), axis=1))
    num_turns = len(polyline_points) - 2  # Simplistic turn count
    bounding_box = polyline_points.max(axis=0) - polyline_points.min(axis=0)
    variance = polyline_points.var(axis=0).sum()
    # New features
    curvature = np.sum(
        np.abs(np.diff(np.diff(polyline_points, axis=0), axis=0))
    )  # Curvature estimate
    segment_lengths = np.linalg.norm(np.diff(polyline_points, axis=0), axis=1)
    length_mean = np.mean(segment_lengths)
    length_variance = np.var(segment_lengths)

    return np.concatenate(
        [
            mean_coords,
            [
                total_length,
                num_turns,
                *bounding_box,
                variance,
                curvature,
                length_mean,
                length_variance,
            ],
        ]
    )


def cluster_trip_district(df: pd.DataFrame, districts: json) -> pd.DataFrame:
    """
    Cluster districts into predefined groups and map each district in the dataframe to its corresponding cluster.

    Args:
        df (pd.DataFrame): A dataframe containing a 'DISTRICT_NAME' column with district names.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'CLUSTER' column indicating the assigned cluster for each district.
    """
    district_boundaries = districts

    logger.info("Defining district centroids.")

    # Extract coordinates of district centroids
    centroids = []
    district_names = list(district_boundaries.keys())
    for district, bounds in district_boundaries.items():
        centroid_lat = (bounds["lower_lat"] + bounds["upper_lat"]) / 2
        centroid_long = (bounds["left_long"] + bounds["right_long"]) / 2
        centroids.append([centroid_lat, centroid_long])

    logger.info("Applying KMeans clustering to group districts into 7 clusters.")

    try:
        kmeans = KMeans(n_clusters=7, random_state=42)
        kmeans.fit(centroids)
        labels = kmeans.labels_
    except ValueError as e:
        logger.error(f"Error fitting KMeans model: {e}")
        raise
    except NotFittedError as e:
        logger.error(f"KMeans model not fitted: {e}")
        raise

    logger.info("Assigning cluster names.")

    # Create cluster names
    unique_labels = list(set(labels))
    cluster_names = [f"Cluster {i + 1}" for i in range(len(unique_labels))]

    # Map clusters to cluster names
    label_to_cluster_name = {
        unique_labels[i]: cluster_names[i] for i in range(len(unique_labels))
    }
    # Map districts to their respective clusters
    district_cluster_mapping = {
        district_names[i]: cluster_names[labels[i]] for i in range(len(district_names))
    }

    logger.info("Mapping each district in the dataframe to its corresponding cluster.")

    # Map each district in the dataframe to its cluster
    df["DISTRICT_CLUSTER"] = df["DISTRICT_NAME"].apply(
        lambda x: district_cluster_mapping.get(x, "Unknown_Cluster")
    )

    logger.info("District clustering completed.")

    return df


def cluster_trip_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster the time of trips into three distinct periods (morning, afternoon, night) based on the 'TIME' column.

    Args:
        df (pd.DataFrame): A dataframe containing a 'TIME' column with time values.

    Returns:
        pd.DataFrame: The input dataframe with additional 'TIME_CLUSTER' and 'TIME_PERIODS' columns indicating the time cluster and corresponding time period.
    """
    # Check if the input dataframe contains the required column
    if "TIME" not in df.columns:
        logger.error("The input dataframe must contain a 'TIME' column.")
        raise ValueError("The input dataframe must contain a 'TIME' column.")

    logger.info("Reshaping the TIME column for KMeans clustering.")

    # Reshape the TIME column for KMeans
    X = df["TIME"].values.reshape(-1, 1)

    logger.info(
        "Applying KMeans clustering with 3 clusters (morning, afternoon, night)."
    )

    # Apply KMeans clustering with 3 clusters (for morning, afternoon, night)
    try:
        kmeans = KMeans(n_clusters=3, random_state=0)
        df["TIME_CLUSTER"] = kmeans.fit_predict(X)
    except ValueError as e:
        logger.error(f"Error fitting KMeans model: {e}")
        raise
    except NotFittedError as e:
        logger.error(f"KMeans model not fitted: {e}")
        raise

    logger.info("Getting cluster centers and sorting them to define time intervals.")

    # Get the cluster centers (the centroids)
    centers = kmeans.cluster_centers_.flatten()

    # Sort the cluster centers to define the natural intervals
    sorted_centers = sorted(centers)
    logger.info(f"Cluster centers (time intervals): {sorted_centers}")

    # Map clusters to time of day based on sorted centers
    cluster_to_period = {
        centers.tolist().index(sorted_centers[0]): "Morning",
        centers.tolist().index(sorted_centers[1]): "Afternoon",
        centers.tolist().index(sorted_centers[2]): "Night",
    }

    df["TIME_PERIODS"] = df["TIME_CLUSTER"].map(cluster_to_period)

    logger.info("Time clustering completed.")

    return df


def determine_traffic_status(
    df: pd.DataFrame,
    dom_column: str = "DOM",
    light_threshold: float = 0.4,
    medium_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Determines the traffic status based on the DOM value and manual thresholds for light, medium, and high traffic.

    Args:
        df (pd.DataFrame): A dataframe containing the DOM values.
        dom_column (str): The column name for the DOM values.
        light_threshold (float, optional): The threshold below which traffic is considered 'Light'.
        medium_threshold (float, optional): The threshold above which traffic is considered 'High'. Values in between are 'Medium'.

    Returns:
        pd.DataFrame: The input dataframe with a new 'TRAFFIC_STATUS' column indicating 'Light', 'Medium', or 'Heavy' traffic.
    """
    # Check if the input dataframe contains the required column
    if dom_column not in df.columns:
        logger.error(f"The input dataframe must contain the '{dom_column}' column.")
        raise ValueError(f"The input dataframe must contain the '{dom_column}' column.")

    # Check if thresholds are valid
    if not (0 <= light_threshold < medium_threshold <= 1):
        logger.error(
            "Threshold values must be between 0 and 1, with light_threshold < medium_threshold."
        )
        raise ValueError(
            "Threshold values must be between 0 and 1, with light_threshold < medium_threshold."
        )

    logger.info("Determining traffic status based on DOM values and thresholds.")

    # Function to determine traffic status based on DOM value
    def traffic_status(dom_value: float) -> str:
        if dom_value < light_threshold:
            return "Light"
        elif dom_value < medium_threshold:
            return "Medium"
        else:
            return "Heavy"

    # Apply traffic status function to the dataframe
    df["TRAFFIC_STATUS"] = df[dom_column].apply(traffic_status)

    logger.info("Traffic status determination completed.")

    return df


def determine_traffic_status_by_quality(
    df: pd.DataFrame,
    quality_column: str = "MEMBERSHIP_QUALITY",
    cluster_column: str = "CLUSTER",
    light_threshold: float = 0.4,
    medium_threshold: float = 0.7,
    categories: List[str] = ["Light", "Medium", "Heavy"],
    default_category: str = "Unknown",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Determines the traffic status based on the MEMBERSHIP_QUALITY value and manual thresholds for Light, Medium, and Heavy traffic.

    Args:
        df (pd.DataFrame): A dataframe containing the MEMBERSHIP_QUALITY and CLUSTER values.
        quality_column (str): The column name for the MEMBERSHIP_QUALITY values.
        cluster_column (str): The column name for the CLUSTER values.
        light_threshold (float, optional): The threshold below which traffic is considered 'Light'.
        medium_threshold (float, optional): The threshold above which traffic is considered 'Heavy'. Values in between are 'Medium'.
        categories (List[str], optional): List of category labels corresponding to the conditions. Default is ["Light", "Medium", "Heavy"].
        default_category (str, optional): Label for values that do not meet any condition. Default is "Unknown".
        inplace (bool, optional): Whether to modify the original dataframe. If False, returns a new dataframe. Default is False.

    Returns:
        pd.DataFrame: The input dataframe with a new 'TRAFFIC_STATUS' column indicating 'Light', 'Medium', 'Heavy', or 'Unknown' traffic.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'WEEKDAY': ['Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'Saturday'],
        ...     'TIME': [14, 15, 22, 14, 10],
        ...     'CLUSTER': [0.0, 8.0, -1.0, 0.0, -1.0],
        ...     'DOM': [0.842566, 0.971883, 0.0, 0.908505, 0.0],
        ...     'OUTLIER_SCORE': [0.157434, 0.028117, 0.047706, 0.091495, 0.011821],
        ...     'MEMBERSHIP_QUALITY': [0.709918, 0.944557, 0.0, 0.825381, 0.0]
        ... }
        >>> df = pd.DataFrame(data)
        >>> result_df = determine_traffic_status_by_quality(df)
        >>> print(result_df)
            WEEKDAY  TIME  CLUSTER       DOM  OUTLIER_SCORE  MEMBERSHIP_QUALITY TRAFFIC_STATUS
        0     Sunday    14      0.0  0.842566       0.157434            0.709918          Heavy
        1  Thursday    15      8.0  0.971883       0.028117            0.944557          Heavy
        2    Tuesday    22     -1.0  0.000000       0.047706            0.000000        Unknown
        3  Wednesday    14      0.0  0.908505       0.091495            0.825381          Heavy
        4   Saturday    10     -1.0  0.000000       0.011821            0.000000        Unknown
    """
    # Input validation
    if quality_column not in df.columns:
        logger.error(f"The input dataframe must contain the '{quality_column}' column.")
        raise ValueError(
            f"The input dataframe must contain the '{quality_column}' column."
        )

    if cluster_column not in df.columns:
        logger.error(f"The input dataframe must contain the '{cluster_column}' column.")
        raise ValueError(
            f"The input dataframe must contain the '{cluster_column}' column."
        )

    if not (0 <= light_threshold < medium_threshold <= 1):
        logger.error(
            "Threshold values must be between 0 and 1, with light_threshold < medium_threshold."
        )
        raise ValueError(
            "Threshold values must be between 0 and 1, with light_threshold < medium_threshold."
        )

    if len(categories) != 3:
        logger.error("The 'categories' list must contain exactly three labels.")
        raise ValueError("The 'categories' list must contain exactly three labels.")

    logger.info(
        "Determining traffic status based on MEMBERSHIP_QUALITY and CLUSTER values."
    )

    # Create a copy if not inplace
    if not inplace:
        df = df.copy()

    # Ensure the quality column is numeric
    original_non_numeric = (
        df[quality_column].isna().sum()
    )  # Count NaNs including coercion
    df[quality_column] = pd.to_numeric(df[quality_column], errors="coerce")

    # Initialize TRAFFIC_STATUS with default_category
    df["TRAFFIC_STATUS"] = default_category

    # Assign 'Unknown' to rows where CLUSTER is -1.0
    unknown_mask = df[cluster_column] == -1.0
    df.loc[unknown_mask, "TRAFFIC_STATUS"] = default_category

    # Create a mask for valid clusters (CLUSTER != -1.0)
    valid_cluster_mask = ~unknown_mask

    # Define conditions and corresponding choices for valid clusters
    conditions = [
        df.loc[valid_cluster_mask, quality_column] < light_threshold,
        (df.loc[valid_cluster_mask, quality_column] >= light_threshold)
        & (df.loc[valid_cluster_mask, quality_column] < medium_threshold),
        df.loc[valid_cluster_mask, quality_column] >= medium_threshold,
    ]
    choices = categories

    # Assign traffic status using np.select for valid clusters
    df.loc[valid_cluster_mask, "TRAFFIC_STATUS"] = np.select(
        conditions,
        choices,
        default=default_category,  # Fallback to 'Unknown' if no condition matches
    )

    # Count 'Unknown' categories
    unknown_count = (df["TRAFFIC_STATUS"] == default_category).sum()
    if unknown_count > 0:
        logger.warning(
            f"{unknown_count} rows have '{default_category}' TRAFFIC_STATUS due to invalid or missing MEMBERSHIP_QUALITY values or being outliers."
        )

    logger.info("Traffic status determination completed.")

    return df


def HDBSCAN_Clustering_Aggregated_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply HDBSCAN clustering to polylines grouped by WEEKDAY and TIME, with aggregated membership probabilities.

    Parameters:
        df (pd.DataFrame):
            - Must contain the following columns:
                - 'WEEKDAY' (int or str): Represents the day of the week.
                - 'TIME' (str or appropriate time format): Represents the time slot.
                - 'POLYLINE' (str): String representation of a list of coordinate pairs, e.g., "[[x1, y1], [x2, y2], ...]".

    Returns:
        pd.DataFrame: Original DataFrame augmented with the following columns:
            - 'CLUSTER' (int): Assigned cluster label by HDBSCAN.
            - 'DOM' (float): Dominant membership probability.
            - 'PROBABILITY' (float): Aggregated membership probability.
            - 'OUTLIER_SCORE' (float): Outlier score assigned by HDBSCAN.
            - 'MEMBERSHIP_QUALITY' (float): Computed as DOM * (1 - OUTLIER_SCORE).

    Example:
        >>> df = pd.DataFrame({
        ...     'WEEKDAY': ['Monday', 'Monday'],
        ...     'TIME': ['Morning', 'Morning'],
        ...     'POLYLINE': ['[[0, 0], [1, 1], [2, 2]]', '[[10, 10], [11, 11], [12, 12]]']
        ... })
        >>> clustered_df = HDBSCAN_Clustering_Aggregated_optimized(df)
    """

    # Initialize cluster-related columns with NaN
    df["CLUSTER"] = np.nan
    df["DOM"] = np.nan
    df["OUTLIER_SCORE"] = np.nan

    # Group the dataframe by WEEKDAY and TIME
    grouped = df.groupby(["WEEKDAY", "TIME"])
    logger.info("Starting clustering process for grouped data.")

    for (weekday, time), group in grouped:
        logger.info(f"Processing group: WEEKDAY={weekday}, TIME={time}")

        # Extract polylines and preprocess
        polylines = []
        valid_indices = []
        features = []

        for index, row in group.iterrows():
            polyline_str = row.get("POLYLINE")
            if pd.isna(polyline_str):
                logger.debug(f"Skipping row {index} due to missing POLYLINE.")
                continue

            try:
                polyline = ast.literal_eval(polyline_str)
            except (ValueError, SyntaxError) as e:
                logger.warning(
                    f"Skipping row {index} due to invalid POLYLINE format: {e}"
                )
                continue

            # polyline = row.get("POLYLINE")
            # if pd.isna(polyline) or not isinstance(polyline, list):
            #     logger.debug(
            #         f"Skipping row {index} due to missing or invalid POLYLINE."
            #     )
            #     continue

            if not isinstance(polyline, list) or len(polyline) < 2:
                logger.debug(
                    f"Skipping row {index} due to insufficient points in POLYLINE."
                )
                continue
            feature = extract_features(polyline)
            if feature is None:
                logger.debug(
                    f"Skipping row {index} due to invalid polyline dimensions."
                )
                continue

            polylines.append(polyline)
            valid_indices.append(index)
            features.append(feature)

        if not features:
            logger.info(
                f"No valid polylines found for group WEEKDAY={weekday}, TIME={time}."
            )
            continue

        features = np.array(features)

        # Determine min_cluster_size based on group size
        group_size = len(features)
        min_cluster_size = max(round(math.log(group_size + 1)), 2)  # Avoid log(1)=0
        min_cluster_size = min(
            min_cluster_size, max(10, int(group_size * 0.1))
        )  # Ensure it's reasonable

        logger.debug(f"Group size: {group_size}, min_cluster_size: {min_cluster_size}")

        if min_cluster_size > group_size:
            logger.warning(
                f"min_cluster_size {min_cluster_size} is greater than group size {group_size}. Skipping group."
            )
            continue

        # Apply HDBSCAN clustering on the aggregated features
        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size, min_samples=5, metric="euclidean"
            )
            cluster_labels = clusterer.fit_predict(features)
            membership_probabilities = clusterer.probabilities_
            outlier_scores = clusterer.outlier_scores_

            # Assign clustering results to the DataFrame using vectorized operations
            df.loc[valid_indices, "CLUSTER"] = cluster_labels
            df.loc[valid_indices, "DOM"] = membership_probabilities
            df.loc[valid_indices, "OUTLIER_SCORE"] = outlier_scores

        except Exception as e:
            logger.warning(
                f"Clustering failed for group WEEKDAY={weekday}, TIME={time} with error: {e}"
            )
            continue

    # Compute MEMBERSHIP_QUALITY
    df["MEMBERSHIP_QUALITY"] = df["DOM"] * (1 - df["OUTLIER_SCORE"])

    logger.info("Clustering process completed.")
    return df


def encode_geographical_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode geographical context by clustering start and end coordinates.

    Args:
        df (pd.DataFrame): A dataframe containing 'START_LAT', 'START_LONG', 'END_LAT', 'END_LONG' columns.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'REGIONAL_CLUSTER' column indicating the geographical cluster.
    """
    logger.info(
        "Encoding geographical context by clustering start and end coordinates."
    )

    # Combine start and end coordinates into a single array
    coords = df[["START_LAT", "START_LONG", "END_LAT", "END_LONG"]].values

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    df["REGIONAL_CLUSTER"] = kmeans.fit_predict(coords)

    logger.info("Geographical context encoding completed.")

    return df


def aggregate_district_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate district and regional clusters into a composite feature.

    Args:
        df (pd.DataFrame): A dataframe containing 'DISTRICT_CLUSTER' and 'REGIONAL_CLUSTER' columns.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'COMBINED_CLUSTER' column.
    """
    logger.info("Aggregating district and regional clusters into a composite feature.")
    df["COMBINED_CLUSTER"] = (
        df["DISTRICT_CLUSTER"].astype(str) + "_" + df["REGIONAL_CLUSTER"].astype(str)
    )
    logger.info("District and regional cluster aggregation completed.")
    return df


def traffic_congestion_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate traffic congestion indicator based on travel time and trip distance.

    Args:
        df (pd.DataFrame): A dataframe containing 'TRAVEL_TIME' and 'TRIP_DISTANCE' columns.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'CONGESTION' column.
    """
    logger.info("Calculating Traffic congestion")
    # Drop rows where TRIP_DISTANCE is 0 or very small (e.g., less than 1e-5)
    # Create a new DataFrame copy to avoid SettingWithCopyWarning
    df = df[df["TRIP_DISTANCE"] > 1e-5].copy()

    # Calculate congestion indicator
    df["CONGESTION"] = df["TRAVEL_TIME"] / df["TRIP_DISTANCE"]

    logger.info("Traffic congestion indicator calculation completed.")
    return df


def add_temporal_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal context to the combined cluster feature.

    Args:
        df (pd.DataFrame): A dataframe containing 'COMBINED_CLUSTER' and 'TIME_PERIODS' columns.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'REGIONAL_TEMPORAL_CONTEXT' column.
    """
    logger.info("Adding temporal context to the combined cluster feature.")
    df["REGIONAL_TEMPORAL_CONTEXT"] = (
        df["COMBINED_CLUSTER"] + "_" + df["TIME_PERIODS"].astype(str)
    )
    logger.info("Temporal context added to combined cluster feature.")
    return df


def aggregate_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate historical data by grouping by regional temporal context and calculating averages.

    Args:
        df (pd.DataFrame): A dataframe containing 'REGIONAL_TEMPORAL_CONTEXT' column.

    Returns:
        pd.DataFrame: The input dataframe with merged aggregated historical features.
    """
    logger.info("Aggregating historical data by regional temporal context.")
    # Group by the new context and calculate averages
    regional_summary = (
        df.groupby("REGIONAL_TEMPORAL_CONTEXT")
        .agg(
            {
                "TRAVEL_TIME": "mean",
                "TRAFFIC_STATUS": lambda x: x.value_counts().idxmax(),  # Most common traffic status
                "AVG_SPEED": "mean",
            }
        )
        .reset_index()
    )

    # Merge aggregated features back to the main DataFrame
    df = df.merge(regional_summary, on="REGIONAL_TEMPORAL_CONTEXT", how="left")

    logger.info("Historical data aggregation completed.")
    return df


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    df = read_csv_with_progress(
        "/home/smebellis/ece5831_final_project/processed_data/preprocessed_dataset.csv"
    )

    sample_df = df.sample(n=50000, random_state=42)

    PORTO_DISTRICTS = "/home/smebellis/ece5831_final_project/data/porto_districts.json"
    # Load Districts Data
    logger.info("Loading district boundary data...")
    try:
        porto_districts = load_districts(PORTO_DISTRICTS)
    except DistrictLoadError:
        sys.exit(1)

    sample_df = cluster_trip_district(sample_df, porto_districts)
    sample_df = cluster_trip_time(sample_df)

    sample_df = HDBSCAN_Clustering_Aggregated_optimized(sample_df)
    sample_df = determine_traffic_status_by_quality(sample_df)
    sample_df = encode_geographical_context(sample_df)
    sample_df = aggregate_district_clusters(sample_df)
    sample_df = traffic_congestion_indicator(sample_df)
    sample_df = add_temporal_context(sample_df)
    # sample_df = aggregate_historical_data(sample_df)
    print(
        sample_df[
            [
                "WEEKDAY",
                "TIME",
                "DISTRICT_CLUSTER",
                "REGIONAL_CLUSTER",
                "COMBINED_CLUSTER",
                "REGIONAL_TEMPORAL_CONTEXT",
                "CONGESTION",
            ]
        ].head()
    )
    breakpoint()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style for seaborn
    sns.set(style="whitegrid")

    # Plot histogram for OUTLIER_SCORE
    plt.figure(figsize=(12, 6))
    sns.histplot(sample_df["OUTLIER_SCORE"], bins=50, kde=True, color="skyblue")
    plt.title("Distribution of OUTLIER_SCORE")
    plt.xlabel("OUTLIER_SCORE")
    plt.ylabel("Frequency")
    plt.show()

    # Plot histogram for DOM (Degree of Membership)
    plt.figure(figsize=(12, 6))
    sns.histplot(sample_df["DOM"], bins=50, kde=True, color="orange")
    plt.title("Distribution of DOM (Degree of Membership)")
    plt.xlabel("DOM")
    plt.ylabel("Frequency")
    plt.show()

    # Scatter plot to visualize relationship between OUTLIER_SCORE and DOM
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="OUTLIER_SCORE", y="DOM", data=sample_df, alpha=0.5)
    plt.title("Scatter Plot of OUTLIER_SCORE vs. DOM")
    plt.xlabel("OUTLIER_SCORE")
    plt.ylabel("DOM")
    plt.show()

    breakpoint()
    # # Boxplot by Cluster
    # plt.figure(figsize=(12, 8))
    # sns.boxplot(x="CLUSTER", y="MEMBERSHIP_QUALITY", data=df)
    # plt.title("Membership Quality by Cluster")
    # plt.xlabel("Cluster")
    # plt.ylabel("Membership Quality")
    # plt.show()
    # breakpoint()
    # df = determine_traffic_status(df)

    # save_dataframe_if_not_exists(
    #     df,
    #     "/home/smebellis/ece5831_final_project/processed_data/post_processing_clustered.csv",
    # )
