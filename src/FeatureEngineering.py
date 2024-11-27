# Imports
import ast
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import gc
import tracemalloc

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.helper import plot_route_images
from src.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineeringPipeline:
    """
    A pipeline for feature engineering that includes encoding categorical variables,
    scaling numerical features, and converting polyline data to image representations.

    Attributes:
        weekday_encoder (OneHotEncoder): Encoder for the 'WEEKDAY' feature.
        month_encoder (OneHotEncoder): Encoder for the 'MONTH' feature.
        cluster_encoder (OneHotEncoder): Encoder for 'DISTRICT_CLUSTER' and 'TIME_CLUSTER' features.
        scaler (MinMaxScaler): Scaler for numerical features.
    """

    def __init__(self):
        # Initialize encoders and scalers here
        self.weekday_encoder = OneHotEncoder()
        self.month_encoder = OneHotEncoder()

        self.cluster_encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = MinMaxScaler()

        # Initialize imputers
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.numerical_imputer = SimpleImputer(strategy="mean")
        logger.info("Initialized FeatureEngineeringPipeline with encoders and scaler.")

    def fit(self, dataframe):
        """
        Fits the encoders and scaler on the provided training data.

        Args:
            dataframe (pd.DataFrame): The training data containing features to fit.
        """
        logger.info("Fitting imputers, encoders, and scaler on the training data.")

        # Impute categorical features
        categorical_features = dataframe[
            ["WEEKDAY", "MONTH", "DISTRICT_CLUSTER", "TIME_CLUSTER"]
        ]
        self.categorical_imputer.fit(categorical_features)
        imputed_categorical = self.categorical_imputer.transform(categorical_features)

        # Impute numerical features
        numerical_features = dataframe[["DOM", "OUTLIER_SCORE", "MEMBERSHIP_QUALITY"]]
        self.numerical_imputer.fit(numerical_features)
        imputed_numerical = self.numerical_imputer.transform(numerical_features)

        # Fit encoders on imputed categorical features
        self.weekday_encoder.fit(imputed_categorical[:, 0].reshape(-1, 1))
        self.month_encoder.fit(imputed_categorical[:, 1].reshape(-1, 1))
        self.cluster_encoder.fit(imputed_categorical[:, 2:4])

        # Fit scaler on imputed numerical features
        self.scaler.fit(imputed_numerical)

        logger.info("Completed fitting encoders and scaler.")

    def transform(self, dataframe):
        """
        Transforms the data by encoding categorical features, scaling numerical features,
        and converting polyline data to image representations.

        Args:
            dataframe (pd.DataFrame): The data to transform.

        Returns:
            tuple: A tuple containing:
                - route_images_tensor (torch.Tensor): Tensor of route images.
                - additional_features_tensor (torch.Tensor): Tensor of additional engineered features.
        """

        logger.info("Starting transformation of the dataframe.")

        # Impute categorical features
        categorical_features = dataframe[
            ["WEEKDAY", "MONTH", "DISTRICT_CLUSTER", "TIME_CLUSTER"]
        ]
        imputed_categorical = self.categorical_imputer.transform(categorical_features)

        # One-Hot Encode categorical features
        logger.debug("Encoding categorical features.")
        weekday_encoded = self.weekday_encoder.transform(
            imputed_categorical[:, 0].reshape(-1, 1)
        ).toarray()
        month_encoded = self.month_encoder.transform(
            imputed_categorical[:, 1].reshape(-1, 1)
        ).toarray()
        cluster_encoded = self.cluster_encoder.transform(
            imputed_categorical[:, 2:4]
        ).toarray()

        # Impute numerical features
        numerical_features = dataframe[["DOM", "OUTLIER_SCORE", "MEMBERSHIP_QUALITY"]]
        imputed_numerical = self.numerical_imputer.transform(numerical_features)

        # Normalize numerical features
        logger.debug("Scaling numerical features.")
        scaled_features = self.scaler.transform(imputed_numerical)

        # Sin-Cosine Encoding for time feature
        logger.debug("Applying sin-cos encoding to the 'TIME' feature.")
        time = dataframe["TIME"].fillna(
            dataframe["TIME"].mean()
        )  # Handle missing 'TIME' if any
        time_sin = np.sin(2 * np.pi * time / 24)
        time_cos = np.cos(2 * np.pi * time / 24)
        time_encoded = np.column_stack((time_sin, time_cos))

        tracemalloc.start()

        # Convert POLYLINE to image representation
        logger.info("Converting POLYLINE to image representations.")
        route_images = [
            self.convert_route_to_image(polyline) for polyline in dataframe["POLYLINE"]
        ]

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        logger.debug("[ Top 10 Memory Consumers]")
        for stat in top_stats[:10]:
            logger.debug(stat)

        # Verify array types and dimensions before concatenation
        logger.debug("Verifying array types and dimensions before concatenation.")
        feature_arrays = {
            "weekday_encoded": weekday_encoded,
            "month_encoded": month_encoded,
            "cluster_encoded": cluster_encoded,
            "scaled_features": scaled_features,
            "time_encoded": time_encoded,
        }

        for name, array in feature_arrays.items():
            logger.debug(
                f"{name}: type={type(array)}, shape={array.shape}, ndim={array.ndim}"
            )
            if array.size == 0:
                logger.warning(f"Warning: {name} is empty!")
            elif array.ndim == 0:
                logger.warning(f"Warning: {name} is zero-dimensional!")

        # Concatenate all features with error handling
        try:
            logger.info("Concatenating all engineered features.")
            additional_features = np.concatenate(
                [
                    weekday_encoded,
                    month_encoded,
                    cluster_encoded,
                    scaled_features,
                    time_encoded,
                ],
                axis=1,
            )
            logger.debug(f"Concatenated features shape: {additional_features.shape}")
        except ValueError as e:
            logger.error("Error during concatenation of features.")
            for name, array in feature_arrays.items():
                logger.error(f"{name} shape: {array.shape}")
            logger.error(f"Error message: {e}")
            raise

        # Convert to torch tensors
        logger.info("Converting features to torch tensors.")
        route_images_tensor = torch.stack(
            [torch.tensor(img, dtype=torch.float32) for img in route_images]
        )
        additional_features_tensor = torch.tensor(
            additional_features, dtype=torch.float32
        )
        logger.info("Transformation complete.")
        return route_images_tensor, additional_features_tensor

    def convert_route_to_image_optimized(self, polyline: list) -> np.ndarray:
        """
        Optimized version of the function to convert a polyline to a grayscale image representation.

        Args:
            polyline (list): A list of [latitude, longitude] points.

        Returns:
            np.ndarray: A 32x32 grayscale image representing the route.
        """
        grid_size = 224
        image = np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Handle empty polyline
        if not polyline:
            logger.warning("Received an empty polyline. Returning a blank image.")
            return image

        # Parse and validate polyline points
        def parse_and_validate(polyline):
            cleaned_polyline = []
            for point in polyline:
                try:
                    # Parse strings if needed
                    if isinstance(point, str):
                        point = ast.literal_eval(point)
                    # Ensure point is a tuple/list of two floats
                    if (
                        isinstance(point, (list, tuple))
                        and len(point) == 2
                        and all(isinstance(coord, (int, float)) for coord in point)
                    ):
                        cleaned_polyline.append((float(point[0]), float(point[1])))
                except (ValueError, SyntaxError, TypeError) as e:
                    logger.warning(f"Invalid point in polyline: {point}. Error: {e}")
                    continue
            return cleaned_polyline

        polyline = parse_and_validate(polyline)

        # Downsample polyline to reduce processing load
        max_points = 1500
        if len(polyline) > max_points:
            indices = np.linspace(0, len(polyline) - 1, max_points, dtype=np.int32)
            polyline = [polyline[i] for i in indices]

        # Extract coordinates as NumPy arrays
        latitudes = np.array([point[0] for point in polyline], dtype=np.float32)
        longitudes = np.array([point[1] for point in polyline], dtype=np.float32)

        # Avoid division by zero by adding small variability if all points are the same
        if np.ptp(latitudes) == 0:
            latitudes[0] += 1e-3
        if np.ptp(longitudes) == 0:
            longitudes[0] += 1e-3

        # Calculate min and max
        lat_min, lat_max = latitudes.min(), latitudes.max()
        lon_min, lon_max = longitudes.min(), longitudes.max()

        # Normalize coordinates to grid size
        padding = 1e-3
        lat_norm = (latitudes - lat_min) / (lat_max - lat_min + padding)
        lon_norm = (longitudes - lon_min) / (lon_max - lon_min + padding)

        # Map to pixel indices
        x_coords = (lat_norm * (grid_size - 1)).astype(np.int32)
        y_coords = (lon_norm * (grid_size - 1)).astype(np.int32)

        # Stack coordinates and draw all lines at once
        coordinates = np.column_stack((y_coords, x_coords)).reshape(-1, 1, 2)
        cv2.polylines(image, [coordinates], isClosed=False, color=255, thickness=1)

        del latitudes, longitudes, lat_norm, lon_norm
        gc.collect()

        return image

    def convert_route_to_image(self, polyline: list) -> np.ndarray:
        """
        Converts a polyline to a grayscale image representation.

        Args:
            polyline (list): A list of [latitude, longitude] points.

        Returns:
            np.ndarray: A 32x32 grayscale image representing the route.
        """
        grid_size = 64
        image = np.zeros((grid_size, grid_size), dtype=np.uint8)

        if not polyline:
            logger.warning("Received an empty polyline. Returning a blank image.")
            return image

        # Extract latitude and longitude values
        latitudes = [point[0] for point in polyline]
        longitudes = [point[1] for point in polyline]

        # Handle case where there's no variability in coordinates to avoid division by zero
        if len(set(latitudes)) == 1:
            latitudes[0] += 1e-3
            logger.debug(
                "Added small variability to latitude due to lack of variability."
            )
        if len(set(longitudes)) == 1:
            longitudes[0] += 1e-3
            logger.debug(
                "Added small variability to longitude due to lack of variability."
            )
        # Normalize coordinates to fit in the grid
        lat_min, lat_max = min(latitudes), max(latitudes)
        lon_min, lon_max = min(longitudes), max(longitudes)

        # Add a small padding to the range to ensure proper spread across the grid
        padding = 1e-3
        lat_norm = (np.array(latitudes) - lat_min) / (lat_max - lat_min + padding)
        lon_norm = (np.array(longitudes) - lon_min) / (lon_max - lon_min + padding)

        # Map normalized coordinates to the grid and draw the route
        for i in range(1, len(lat_norm)):
            x1, y1 = int(lat_norm[i - 1] * (grid_size - 1)), int(
                lon_norm[i - 1] * (grid_size - 1)
            )
            x2, y2 = int(lat_norm[i] * (grid_size - 1)), int(
                lon_norm[i] * (grid_size - 1)
            )

            # Draw line on the image with increased thickness for better visibility
            if (
                0 <= x1 < grid_size
                and 0 <= y1 < grid_size
                and 0 <= x2 < grid_size
                and 0 <= y2 < grid_size
            ):
                cv2.line(image, (y1, x1), (y2, x2), color=1.0, thickness=1)

        # Debugging: Check if the image array contains any lines
        # print("Image array after drawing:", image)

        # Normalize image values to 0-255 for better visualization if needed
        image = (image * 255).astype(np.uint8)

        # Save image for debugging purposes
        # cv2.imwrite("debug_image.png", image)
        # logger.debug("Saved debug_image.png for route image visualization.")
        # cv2.imshow("Route Image", image)       # Uncomment if you want to display
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return image

    def fit_transform(self, dataframe: pd.DataFrame):
        """
        Fits the pipeline on the data and then transforms it.

        Args:
            dataframe (pd.DataFrame): The data to fit and transform.

        Returns:
            tuple: A tuple containing:
                - route_images_tensor (torch.Tensor): Tensor of route images.
                - additional_features_tensor (torch.Tensor): Tensor of additional engineered features.
        """
        logger.info("Starting fit_transform process.")
        self.fit(dataframe)
        transformed_data = self.transform(dataframe)
        logger.info("fit_transform process complete.")
        return transformed_data


# Testing
if __name__ == "__main__":

    df = pd.read_csv(
        "/home/smebellis/ece5831_final_project/processed_data/clustered_dataset.csv"
    )

    df = df.sample(n=10000, random_state=42)

    X = df.drop(columns=["TRAFFIC_STATUS"])
    y = df["TRAFFIC_STATUS"]

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Fit and transform features using your pipeline
    pipeline = FeatureEngineeringPipeline()
    route_images_tensor, additional_features_tensor = pipeline.fit_transform(X)

    # Convert the labels to torch tensor
    traffic_status_tensor = torch.tensor(y_encoded, dtype=torch.long)
    plot_route_images(route_images=route_images_tensor, num_images=5)
    breakpoint()
    torch.save(route_images_tensor, "route_images_tensor.pt")
    torch.save(additional_features_tensor, "additional_features_tensor.pt")
    torch.save(traffic_status_tensor, "traffic_status_tensor.pt")
