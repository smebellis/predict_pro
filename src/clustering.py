import math

import pandas as pd
from hdbscan import HDBSCAN
from tqdm import tqdm

from utils.helper import setup_logging

logger = setup_logging(__name__)


def cluster_hdbscan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform HDBSCAN clustering on a dataframe using the latitude and longitude of origin and destination.
    Returns a dataframe with Origin and Destination Cluster labels and DOM (Degree of Membership).

    Parameters:
    df (pd.DataFrame): Input DataFrame containing 'WEEKDAY', 'TIME', 'Long', 'Lat', 'END_LONG', 'END_LAT'.

    Returns:
    pd.DataFrame: DataFrame with added 'Origin_Cluster', 'Origin_DOM', 'Destination_Cluster', 'Destination_DOM'.
    """

    # Group the dataframe by WEEKDAY and TIME
    grouped = df.groupby(["WEEKDAY", "TIME"])

    # Make a list of sub-DataFrames from the grouped data
    sub_dfs = [group for _, group in grouped]

    # Initialize list to store updated sub-DataFrames
    updated_sub_dfs = []
    n_subframes = len(sub_dfs)

    # Iterate through each sub-DataFrame with progress bar
    logger.info((f"Clustering Sub-DataFrames"))
    for idx, sub_df in tqdm(
        enumerate(sub_dfs, start=1), total=n_subframes, desc="Clustering Sub-DataFrames"
    ):

        # Calculate the number of origin points
        origin_notnull = sub_df[["Long", "Lat"]].notnull().all(axis=1)
        Cnt_origin = origin_notnull.sum()
        MinPts_origin = math.ceil(math.log(Cnt_origin)) if Cnt_origin > 0 else 1

        # Calculate the number of destination points
        dest_notnull = sub_df[["END_LONG", "END_LAT"]].notnull().all(axis=1)
        Cnt_dest = dest_notnull.sum()
        MinPts_dest = math.ceil(math.log(Cnt_dest)) if Cnt_dest > 0 else 1

        # Apply HDBSCAN to Origin Points if MinPts_origin > 1 and sufficient points
        if MinPts_origin > 1 and Cnt_origin >= MinPts_origin:
            try:
                origin_clusterer = HDBSCAN(
                    min_samples=MinPts_origin, min_cluster_size=MinPts_origin
                )
                origin_coords = sub_df.loc[origin_notnull, ["Long", "Lat"]].values
                origin_labels = origin_clusterer.fit_predict(origin_coords)
                origin_DOM = origin_clusterer.probabilities_

                # Assign the cluster labels and DOM back to the sub-DataFrame
                sub_df.loc[origin_notnull, "Origin_Cluster"] = origin_labels
                sub_df.loc[origin_notnull, "Origin_DOM"] = origin_DOM
            except Exception as e:
                logger.error(f"Error clustering origin in group {idx}: {e}")
                sub_df.loc[origin_notnull, "Origin_Cluster"] = -1
                sub_df.loc[origin_notnull, "Origin_DOM"] = 0.0
        else:
            # Assign default values if clustering is not applicable
            sub_df["Origin_Cluster"] = -1
            sub_df["Origin_DOM"] = 0.0

        # Apply HDBSCAN to Destination Points if MinPts_dest > 1 and sufficient points
        if MinPts_dest > 1 and Cnt_dest >= MinPts_dest:
            try:
                dest_clusterer = HDBSCAN(
                    min_samples=MinPts_dest, min_cluster_size=MinPts_dest
                )
                dest_coords = sub_df.loc[dest_notnull, ["END_LONG", "END_LAT"]].values
                dest_labels = dest_clusterer.fit_predict(dest_coords)
                dest_DOM = dest_clusterer.probabilities_

                # Assign the cluster labels and DOM back to the sub-DataFrame
                sub_df.loc[dest_notnull, "Destination_Cluster"] = dest_labels
                sub_df.loc[dest_notnull, "Destination_DOM"] = dest_DOM
            except Exception as e:
                logger.error(f"Error clustering destination in group {idx}: {e}")
                sub_df.loc[dest_notnull, "Destination_Cluster"] = -1
                sub_df.loc[dest_notnull, "Destination_DOM"] = 0.0
        else:
            # Assign default values if clustering is not applicable
            sub_df["Destination_Cluster"] = -1
            sub_df["Destination_DOM"] = 0.0

        # Append the updated sub-DataFrame to the list within the loop
        updated_sub_dfs.append(sub_df)

    logger.info("\nClustering completed for all sub-DataFrames.")

    # Concatenate all updated sub-DataFrames into a single DataFrame
    clustered_df = pd.concat(updated_sub_dfs, ignore_index=True)

    return clustered_df


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/smebellis/ece5831_final_project/processed_data/taxi_data_processed.csv"
    )

    clustered_df = cluster_hdbscan(df)
