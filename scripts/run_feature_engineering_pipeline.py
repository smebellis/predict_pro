import pandas as pd
import torch
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from tqdm import tqdm
import ast
import pickle

import json


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.FeatureEngineering import FeatureEngineeringPipeline
from src.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 1000
# Paths
input_path = (
    "/home/smebellis/ece5831_final_project/processed_data/clustered_dataset.csv"
)
output_path = (
    "/home/smebellis/ece5831_final_project/processed_data/postprocessed_dataset.csv"
)


def convert_polyline_to_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the POLYLINE column from string representation to a list of coordinates.

    Args:
        df (pd.DataFrame): The dataframe containing the POLYLINE column.

    Returns:
        pd.DataFrame: The dataframe with POLYLINE converted to lists.
    """
    logger.info("Starting POLYLINE conversion to lists.")

    tqdm.pandas(desc="Converting POLYLINE to a list")
    logger.info("Converting POLYLINE strings to lists.")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(ast.literal_eval)
    logger.info("Conversion of POLYLINE to lists complete.")
    return df


def batch_process_pipeline(
    df: pd.DataFrame, pipeline: FeatureEngineeringPipeline, output_dir: str, prefix: str
):
    """
    Batch processes the dataframe using the feature engineering pipeline and saves the output tensors.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        pipeline (FeatureEngineeringPipeline): Feature engineering pipeline.
        output_dir (str): Directory to save batch output.
        prefix (str): Prefix for saving files.
    """

    logger.info(f"Batch processing {prefix} set with batch size of {BATCH_SIZE}.")
    for start_idx in range(0, len(df), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        logger.info(f"Processing batch {start_idx} to {end_idx} for {prefix} set.")

        # Convert polylines to images and additional features
        route_images_tensor, additional_features_tensor = pipeline.transform(batch_df)

        # Encode labels
        labels_tensor = torch.tensor(
            label_encoder.transform(batch_df["TRAFFIC_STATUS"]), dtype=torch.long
        )

        # Save the tensors to disk
        torch.save(
            route_images_tensor,
            os.path.join(
                output_dir, f"{prefix}_route_images_tensor_{start_idx}_{end_idx}.pt"
            ),
        )
        torch.save(
            additional_features_tensor,
            os.path.join(
                output_dir,
                f"{prefix}_additional_features_tensor_{start_idx}_{end_idx}.pt",
            ),
        )
        torch.save(
            labels_tensor,
            os.path.join(
                output_dir, f"{prefix}_labels_tensor_{start_idx}_{end_idx}.pt"
            ),
        )

        logger.info(f"Saved batch {start_idx} to {end_idx} for {prefix} set.")


def concatenate_and_save(prefix: str, output_dir: str):
    """
    Load all batch tensor files with the given prefix, concatenate them, and save as a single .pt file.
    Deletes the individual batch files after concatenation.

    Args:
        prefix (str): Prefix of the saved tensor files (e.g., "train", "val", "test").
        output_dir (str): Directory where the tensor batches are saved.
    """
    logger.info(f"Concatenating all {prefix} batch files into a single tensor.")
    batch_files = [
        f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".pt")
    ]
    batch_files.sort()  # Ensure files are loaded in the correct order

    tensors_route_images = []
    tensors_additional_features = []
    tensors_labels = []

    # Load each batch file and append to the corresponding list
    for batch_file in batch_files:
        tensor_path = os.path.join(output_dir, batch_file)
        if "route_images_tensor" in batch_file:
            tensors_route_images.append(torch.load(tensor_path))
        elif "additional_features_tensor" in batch_file:
            tensors_additional_features.append(torch.load(tensor_path))
        elif "labels_tensor" in batch_file:
            tensors_labels.append(torch.load(tensor_path))

    # Concatenate tensors
    concatenated_route_images = torch.cat(tensors_route_images, dim=0)
    concatenated_additional_features = torch.cat(tensors_additional_features, dim=0)
    concatenated_labels = torch.cat(tensors_labels, dim=0)

    # Save concatenated tensors as single .pt files
    torch.save(
        concatenated_route_images,
        os.path.join(output_dir, f"{prefix}_route_images_tensor.pt"),
    )
    torch.save(
        concatenated_additional_features,
        os.path.join(output_dir, f"{prefix}_additional_features_tensor.pt"),
    )
    torch.save(
        concatenated_labels, os.path.join(output_dir, f"{prefix}_labels_tensor.pt")
    )

    logger.info(
        f"Successfully saved concatenated {prefix} tensors as single .pt files."
    )

    # Delete batch files after successful concatenation
    logger.info(f"Deleting batch files for {prefix} set.")
    for batch_file in batch_files:
        batch_path = os.path.join(output_dir, batch_file)
        os.remove(batch_path)
        logger.info(f"Deleted batch file: {batch_path}")


if __name__ == "__main__":
    # Check if the postprocessed dataset exists
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            logger.info("Loaded postprocessed dataset from CSV.")
        except Exception as e:
            logger.error(f"Failed to load the postprocessed dataset: {e}")
            raise e
    else:
        # If the postprocessed file doesn't exist, process the original dataset
        try:
            df = pd.read_csv(input_path)
            logger.info("Loaded original dataset from CSV.")
        except FileNotFoundError as e:
            logger.error("CSV file not found. Please check the file path.")
            raise e

        # Test with a small sample, comment out the lines below to run on the whole dataset
        # df = df.sample(n=500000, random_state=42)

        # Convert the polyline column from string to list
        df = convert_polyline_to_list(df)

        # # Save the dataset to avoid reprocessing next time
        # try:
        #     df.to_csv(output_path, index=False)
        #     logger.info("Saved the polyline converted dataset as a CSV.")
        # except FileNotFoundError:
        #     logger.error(
        #         "The specified directory does not exist. Please check the file path."
        #     )
        # except PermissionError:
        #     logger.error(
        #         "Permission denied. Ensure the script has write access to the specified path."
        #     )
        # except OSError as e:
        #     logger.error(f"An OS error occurred: {e}")
        # except Exception as e:
        #     logger.error(f"An unexpected error occurred: {e}")

    # Split the dataset into training, validation, and test sets
    logger.info("Spliting the dataset into training, validation, and test sets.")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    logger.info(
        "Completed splitting the dataset into training, validation, and test sets."
    )

    # Initialize the feature engineering pipeline
    pipeline = FeatureEngineeringPipeline()

    # Fit the pipeline on the training set and transform all sets
    logger.info("Fitting the feature engineering pipeline on the training set.")
    pipeline.fit(train_df)

    # Create a directory to save preprocessed tensors if it doesn't exist
    output_dir = "preprocessed_tensors"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    # Encode the labels for training, validation, and test sets
    label_encoder = LabelEncoder()

    label_encoder.fit(train_df["TRAFFIC_STATUS"])

    # Create a directory to save pickle files for later processing
    pickle_dir = "pickle_files"
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
        logger.info(f"Created directory: {pickle_dir}")
    pickle_path = os.path.join(pickle_dir, "label_encoder.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"LabelEncoder saved at '{pickle_path}'.")

    batch_process_pipeline(train_df, pipeline, output_dir, prefix="train")
    batch_process_pipeline(val_df, pipeline, output_dir, prefix="val")
    batch_process_pipeline(test_df, pipeline, output_dir, prefix="test")

    logger.info("Batch processing complete. All tensors saved successfully.")

    # Concatenate batch files into single files for each set
    concatenate_and_save("train", output_dir)
    concatenate_and_save("val", output_dir)
    concatenate_and_save("test", output_dir)

    logger.info("All batches concatenated and saved as single .pt files successfully.")
