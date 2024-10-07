from helper import file_load_large_csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import folium
import webbrowser
from typing import Union
import sys


# Register tqdm with pandas
tqdm.pandas()


def haversine(
    lon1: float,
    lat1: float,
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    unit: str = "km",
) -> Union[float, np.ndarray]:
    """
    Calculate the great-circle distance between one point and multiple points on the Earth.

    Parameters:
    - lon1 (float): Longitude of the first point in decimal degrees.
    - lat1 (float): Latitude of the first point in decimal degrees.
    - lon2 (float or np.ndarray): Longitude(s) of the second point(s) in decimal degrees.
    - lat2 (float or np.ndarray): Latitude(s) of the second point(s) in decimal degrees.
    - unit (str, optional): Unit of distance ('km', 'miles', 'nmi'). Defaults to 'km'.

    Returns:
    - float or np.ndarray: Distance(s) between the first point and second point(s) in the specified unit.
    """
    # Validate unit
    units = {"km": 6371.0, "miles": 3956.0, "nmi": 3440.0}
    if unit not in units:
        raise ValueError("Unit must be one of 'km', 'miles', or 'nmi'.")

    # Convert decimal degrees to radians
    lon1_rad, lat1_rad = np.radians(lon1), np.radians(lat1)
    lon2_rad, lat2_rad = np.radians(lon2), np.radians(lat2)

    # Compute differences
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    # Calculate distance
    distance = c * units[unit]

    return distance


def assign_district(row: pd.Series, districts: pd.DataFrame) -> str:
    """
    Assigns a district to a given point using vectorized operations for performance.

    Parameters:
    - row (pd.Series): A row containing 'Long' and 'Lat' for the point.
    - districts (pd.DataFrame): DataFrame containing district boundaries and center coordinates.

    Returns:
    - str: The name of the closest district containing the point or "no district" if none found.
    """
    lon, lat = row["Long"], row["Lat"]

    # Correcting the bounding box logic if necessary
    # Adjust the inequality if this doesnt work.  The long is -8
    within_long = (districts["left_long"] >= lon) & (lon >= districts["right_long"])
    within_lat = (districts["lower_lat"] <= lat) & (lat <= districts["upper_lat"])
    filtered_districts = districts[within_long & within_lat]

    if filtered_districts.empty:
        return "no district"

    # Calculate distances to district centers
    distances = haversine(
        lon,
        lat,
        filtered_districts["center_long"].values,
        filtered_districts["center_lat"].values,
    )

    # Find the district with the minimum distance
    min_distance_idx = np.argmin(distances)
    closest_district = filtered_districts.iloc[min_distance_idx]["DISTRICT_NAME"]

    return closest_district


def load_districts(districts: dict) -> pd.DataFrame:
    """
    Convert a districts dictionary to a pandas DataFrame with calculated center coordinates.

    Parameters:
    - districts (dict): Dictionary containing district boundary information.

    Returns:
    - pd.DataFrame: Processed DataFrame with district boundaries and center coordinates.
    """
    districts_df = pd.DataFrame.from_dict(districts, orient="index").reset_index()
    districts_df = districts_df.rename(columns={"index": "DISTRICT_NAME"})
    districts_df["center_lat"] = (
        districts_df["lower_lat"] + districts_df["upper_lat"]
    ) / 2
    districts_df["center_long"] = (
        districts_df["left_long"] + districts_df["right_long"]
    ) / 2
    return districts_df


def assign_districts_to_taxi(
    taxi_df: pd.DataFrame,
    districts_df: pd.DataFrame,
    sample_size: int = 1000,
    use_sample: bool = True,
) -> pd.DataFrame:
    """
    Assign district names to taxi data points using the assign_district function.

    Parameters:
    - taxi_df (pd.DataFrame): DataFrame containing taxi trajectory data.
    - districts_df (pd.DataFrame): DataFrame containing district boundaries and centers.
    - sample_size (int, optional): Number of samples to process for testing. Defaults to 1000.
    - use_sample (bool, optional): Whether to process a sample or the entire dataset. Defaults to True.

    Returns:
    - pd.DataFrame: Taxi DataFrame with assigned district names.
    """
    # Rename columns for consistency
    taxi_df = taxi_df.rename(columns={"START_LAT": "Lat", "START_LONG": "Long"})

    # Initialize the 'DISTRICT_NAME' column with 'no district'
    taxi_df["DISTRICT_NAME"] = "no district"

    if use_sample:
        # Sample data for testing
        print(f"Sampling {sample_size} records for testing...")
        processed_df = taxi_df.sample(sample_size, random_state=42).copy()

        # Assign districts using the assign_district function
        tqdm.pandas(desc="Assigning districts to sample data")
        processed_df["DISTRICT_NAME"] = processed_df.progress_apply(
            assign_district, axis=1, args=(districts_df,)
        )

        return processed_df
    else:
        # Assign districts to the entire dataset
        print("Assigning districts to the entire dataset...")
        tqdm.pandas(desc="Assigning districts to entire data")
        taxi_df["DISTRICT_NAME"] = taxi_df.progress_apply(
            assign_district, axis=1, args=(districts_df,)
        )

        return taxi_df


def assign_district_vectorized(
    taxi_df: pd.DataFrame, districts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign district names to taxi data points using vectorized operations for better performance.

    Parameters:
    - taxi_df (pd.DataFrame): DataFrame containing taxi trajectory data.
    - districts_df (pd.DataFrame): DataFrame containing district boundaries and centers.

    Returns:
    - pd.DataFrame: Taxi DataFrame with assigned district names.
    """
    # Initialize 'DISTRICT_NAME' with 'no district'
    taxi_df["DISTRICT_NAME"] = "no district"

    # Iterate through each district and assign names where conditions are met
    for _, district in tqdm(
        districts_df.iterrows(), total=districts_df.shape[0], desc="Assigning districts"
    ):
        condition = (
            (taxi_df["Long"] >= district["left_long"])
            & (taxi_df["Long"] <= district["right_long"])
            & (taxi_df["Lat"] >= district["lower_lat"])
            & (taxi_df["Lat"] <= district["upper_lat"])
            & (taxi_df["DISTRICT_NAME"] == "no district")  # Prevent overwriting
        )
        taxi_df.loc[condition, "DISTRICT_NAME"] = district["DISTRICT_NAME"]

    return taxi_df


def main():

    # TODO:  Implement logging like in the other files.  This also needs to be renamed.  It is not
    # a visualization file.  This got a little out of hand during prototyping

    # Paths to data files
    TAXI_DATA_PATH = "/home/smebellis/ece5831_final_project/processed_data/update_taxi_trajectory.csv"

    # Porto, Portugal 18 Districts
    DISTRICTS = {
        "Sao Nicolau": {
            "lower_lat": 41.139357,
            "upper_lat": 41.143865,
            "left_long": -8.621299,
            "right_long": -8.611466,
        },
        "Se": {
            "lower_lat": 41.139193,
            "upper_lat": 41.148208,
            "left_long": -8.600352,
            "right_long": -8.611466,
        },
        "Vitoria": {
            "lower_lat": 41.144025,
            "upper_lat": 41.152879,
            "left_long": -8.611251,
            "right_long": -8.621298,
        },
        "Santo Ildefonso": {
            "lower_lat": 41.148047,
            "upper_lat": 41.164788,
            "left_long": -8.60056,
            "right_long": -8.611249,
        },
        "Cedofeita": {
            "lower_lat": 41.152879,
            "upper_lat": 41.165274,
            "left_long": -8.611247,
            "right_long": -8.628137,
        },
        "Miaragaia": {
            "lower_lat": 41.14274,
            "upper_lat": 41.152881,
            "left_long": -8.621298,
            "right_long": -8.627924,
        },
        "Bonfim and Campnha": {
            "lower_lat": 41.139827,
            "upper_lat": 41.165101,
            "left_long": -8.577693,
            "right_long": -8.600565,
        },
        "Massarelos": {
            "lower_lat": 41.14435,
            "upper_lat": 41.17252,
            "left_long": -8.627924,
            "right_long": -8.639681,
        },
        "Lordelo Do Ouro": {
            "lower_lat": 41.146765,
            "upper_lat": 41.172843,
            "left_long": -8.639467,
            "right_long": -8.661271,
        },
        "Foz Do Douro": {
            "lower_lat": 41.144509,
            "upper_lat": 41.17284,
            "left_long": -8.661271,
            "right_long": -8.703385,
        },
        "Matosinhos": {
            "lower_lat": 41.172523,
            "upper_lat": 41.210809,
            "left_long": -8.664262,
            "right_long": -8.732013,
        },
        "Perafita": {
            "lower_lat": 41.210875,
            "upper_lat": 41.262566,
            "left_long": -8.675873,
            "right_long": -8.73124,
        },
        "Maia": {
            "lower_lat": 41.210921,
            "upper_lat": 41.262463,
            "left_long": -8.585343,
            "right_long": -8.675948,
        },
        "Ramalde and Aldoar": {
            "lower_lat": 41.172683,
            "upper_lat": 41.210647,
            "left_long": -8.628359,
            "right_long": -8.664262,
        },
        "Paranhos": {
            "lower_lat": 41.164609,
            "upper_lat": 41.210949,
            "left_long": -8.577473,
            "right_long": -8.627925,
        },
        "Sao Pedro Fins": {
            "lower_lat": 41.210791,
            "upper_lat": 41.263121,
            "left_long": -8.540443,
            "right_long": -8.58532,
        },
        "Alfena": {
            "lower_lat": 41.21103,
            "upper_lat": 41.262843,
            "left_long": -8.492282,
            "right_long": -8.540443,
        },
        "Corujeiera and Roque": {
            "lower_lat": 41.140778,
            "upper_lat": 41.211122,
            "left_long": -8.491102,
            "right_long": -8.57768,
        },
        "Vila Nova De Gaia": {
            "lower_lat": 41.07919,
            "upper_lat": 41.136052,
            "left_long": -8.571571,
            "right_long": -8.67612,
        },
    }

    if not DISTRICTS:
        print(
            "Error: DISTRICTS dictionary is empty. Please provide district data.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load Taxi Data
    print("Loading taxi trajectory data...")
    try:
        taxi_df = file_load_large_csv(TAXI_DATA_PATH)
    except Exception as e:
        print(f"Error loading taxi data: {e}", file=sys.stderr)
        sys.exit(1)

    # Load and process Districts Data
    print("Processing district data...")
    districts_df = load_districts(DISTRICTS)

    # Parameters for district assignment
    SAMPLE_SIZE = 1000  # Number of records to sample
    USE_SAMPLE = False  # Set to False to process the entire dataset

    # Assign districts to taxi data
    print(f"Assigning districts to {'sample' if USE_SAMPLE else 'entire'} data...")
    assigned_df = assign_districts_to_taxi(
        taxi_df, districts_df, sample_size=SAMPLE_SIZE, use_sample=USE_SAMPLE
    )

    # TODO:  Use the function from DataPreprocessing.  All of those administrative functions
    # Need to go into a separate pipeline file.  Basically, turn the preprocessing method into its
    # on file.
    print(f"Saving file to csv.")
    assigned_df.to_csv(
        "/home/smebellis/ece5831_final_project/processed_data/taxi_trajectory_with_districts.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
    breakpoint()
