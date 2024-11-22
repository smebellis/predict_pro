import pandas as pd
import torch
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.FeatureEngineering import FeatureEngineeringPipeline
from src.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # Load the dataset
    try:
        df = pd.read_csv(
            "/home/smebellis/ece5831_final_project/processed_data/clustered_dataset.csv"
        )
        logger.info("Loaded dataset from CSV.")
    except FileNotFoundError as e:
        logger.error("CSV file not found. Please check the file path.")
        raise e

    # Test with a small sample, comment out lines below to run on whole dataset
    # df = df.sample(n=1000, random_state=42)

    # Split the dataset into training, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    logger.info("Split the dataset into training, validation, and test sets.")

    # Initialize the feature engineering pipeline
    pipeline = FeatureEngineeringPipeline()

    # Fit the pipeline on the training set and transform all sets
    logger.info("Fitting the feature engineering pipeline on the training set.")
    pipeline.fit(train_df)

    logger.info("Transforming the training, validation, and test sets.")
    train_route_images_tensor, train_additional_features_tensor = pipeline.transform(
        train_df
    )

    val_route_images_tensor, val_additional_features_tensor = pipeline.transform(val_df)
    test_route_images_tensor, test_additional_features_tensor = pipeline.transform(
        test_df
    )

    # Encode the labels for training, validation, and test sets
    label_encoder = LabelEncoder()
    train_labels_tensor = torch.tensor(
        label_encoder.fit_transform(train_df["TRAFFIC_STATUS"]), dtype=torch.long
    )
    val_labels_tensor = torch.tensor(
        label_encoder.transform(val_df["TRAFFIC_STATUS"]), dtype=torch.long
    )
    test_labels_tensor = torch.tensor(
        label_encoder.transform(test_df["TRAFFIC_STATUS"]), dtype=torch.long
    )

    # Create a directory to save preprocessed tensors if it doesn't exist
    output_dir = "preprocessed_tensors"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    # Save the preprocessed tensors to disk
    logger.info("Saving preprocessed tensors to disk.")
    torch.save(
        train_route_images_tensor,
        os.path.join(output_dir, "train_route_images_tensor.pt"),
    )
    torch.save(
        train_additional_features_tensor,
        os.path.join(output_dir, "train_additional_features_tensor.pt"),
    )
    torch.save(train_labels_tensor, os.path.join(output_dir, "train_labels_tensor.pt"))

    torch.save(
        val_route_images_tensor, os.path.join(output_dir, "val_route_images_tensor.pt")
    )
    torch.save(
        val_additional_features_tensor,
        os.path.join(output_dir, "val_additional_features_tensor.pt"),
    )
    torch.save(val_labels_tensor, os.path.join(output_dir, "val_labels_tensor.pt"))

    torch.save(
        test_route_images_tensor,
        os.path.join(output_dir, "test_route_images_tensor.pt"),
    )
    torch.save(
        test_additional_features_tensor,
        os.path.join(output_dir, "test_additional_features_tensor.pt"),
    )
    torch.save(test_labels_tensor, os.path.join(output_dir, "test_labels_tensor.pt"))

    logger.info("Preprocessed tensors saved successfully.")
