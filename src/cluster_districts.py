import ast
import math
from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
import sys
import json
from joblib import Parallel, delayed

from utils.helper import save_dataframe_if_not_exists

from logger import get_logger

logger = get_logger(__name__)


def cluster_trip_district(df: pd.DataFrame, DISTRICTS: json) -> pd.DataFrame:
    """
    Cluster districts into predefined groups and map each district in the dataframe to its corresponding cluster.

    Args:
        df (pd.DataFrame): A dataframe containing a 'DISTRICT_NAME' column with district names.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'CLUSTER' column indicating the assigned cluster for each district.
    """
    district_boundaries = DISTRICTS

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


def HDBSCAN_Clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply HDBSCAN clustering to polylines grouped by weekday and time.

    Parameters:
        df (pd.DataFrame): Input dataframe with columns 'WEEKDAY', 'TIME', and 'POLYLINE'.

    Returns:
        pd.DataFrame: Dataframe with added 'CLUSTER' and 'DOM' columns indicating cluster assignment and degree of membership.
    """
    # Initialize cluster and DOM columns with NaN
    df["CLUSTER"] = np.nan
    df["DOM"] = np.nan

    # Group the dataframe by WEEKDAY and TIME
    grouped = df.groupby(["WEEKDAY", "TIME"])
    logger.info("Starting clustering process for grouped data.")

    for (weekday, time), group in grouped:
        logger.info(f"Processing group: WEEKDAY={weekday}, TIME={time}")

        # Iterate through each row to extract polylines and apply clustering
        for index, row in group.iterrows():
            if pd.isna(row["POLYLINE"]):
                logger.debug(f"Skipping row {index} due to missing POLYLINE.")
                continue

            # Extract the list of coordinates from the POLYLINE column
            try:
                polyline: Any = ast.literal_eval(row["POLYLINE"])
            except (ValueError, SyntaxError) as e:
                logger.warning(
                    f"Skipping row {index} due to invalid POLYLINE format: {e}"
                )
                continue

            if len(polyline) < 2:
                logger.debug(
                    f"Skipping row {index} due to insufficient points in POLYLINE."
                )
                continue

            # Convert polyline to numpy array for clustering
            polyline_points = np.array(polyline)
            cnt = len(polyline_points)

            # Calculate MinPts as rounded log(cnt)
            min_pts = max(round(math.log(cnt)), 2)

            # If MinPts is less than or equal to 1, clustering is not feasible
            if min_pts <= 1:
                logger.debug(f"Skipping row {index} due to MinPts <= 1.")
                continue

            # Apply HDBSCAN clustering on the polyline
            try:
                clusterer = HDBSCAN(min_cluster_size=min_pts, min_samples=min_pts)
                cluster_labels = clusterer.fit_predict(polyline_points)
                membership_strengths = clusterer.outlier_scores_

                # Assign the results to the DataFrame for the current row
                df.at[index, "CLUSTER"] = (
                    cluster_labels[-1] if len(cluster_labels) > 0 else np.nan
                )
                df.at[index, "DOM"] = (
                    membership_strengths[-1]
                    if len(membership_strengths) > 0
                    else np.nan
                )
                # logger.debug(
                #     f"Assigned cluster {df.at[index, 'CLUSTER']} and DOM {df.at[index, 'DOM']} for row {index}."
                # )

            except ValueError as e:
                logger.warning(f"Clustering failed for row {index} with error: {e}")

    logger.info("Clustering process completed.")
    return df


# Extract 20 coordinate pairs from the polyline column
def extract_coordinate_pairs(polyline_str, num_pairs=20):
    # Convert the polyline string to a list of coordinate pairs
    coordinates = ast.literal_eval(polyline_str)

    # Ensure we do not exceed the number of available pairs
    num_pairs = min(len(coordinates), num_pairs)

    # Sample evenly spaced pairs if there are more than 20
    step = max(1, len(coordinates) // num_pairs)
    sampled_coordinates = coordinates[::step][:num_pairs]

    return sampled_coordinates


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    df = pd.read_csv(
        "/home/smebellis/ece5831_final_project/processed_data/clustered_taxi_data.csv"
    )
    PORTO_DISTRICTS = (
        "/home/smebellis/ece5831_final_project/src/utils/porto_districts.json"
    )
    try:
        with open(PORTO_DISTRICTS, "r") as FILE:
            DISTRICTS = json.load(FILE)
    except FileNotFoundError:
        logger.error(f"Districts file not found at {PORTO_DISTRICTS}.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Error decoding JSON from districts file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading districts: {e}")
        sys.exit(1)

    if not isinstance(DISTRICTS, dict) or not DISTRICTS:
        logger.error(
            "DISTRICTS should be a non-empty dictionary. Please provide valid district data."
        )
        sys.exit(1)

    df = cluster_trip_district(df, DISTRICTS)
    df = cluster_trip_time(df)
    df = HDBSCAN_Clustering(df)
    df = determine_traffic_status(df)

    save_dataframe_if_not_exists(
        df,
        "/home/smebellis/ece5831_final_project/processed_data/post_processing_clustered.csv",
    )
    # df["EXTRACTED"] = df["POLYLINE"].apply(lambda x: extract_coordinate_pairs(x))

    # for index, row in df.iterrows():
    #     extracted_pairs = row["EXTRACTED"]
    #     for coord in extracted_pairs:
    #         lat, lon = coord
    #         print(f"Row {index}: Latitude = {lat}, Longitude = {lon}")
