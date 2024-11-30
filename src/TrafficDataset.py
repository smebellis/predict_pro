import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from logger import get_logger
from typing import Tuple
import numpy as np
from PIL import Image

logger = get_logger(__name__)


def random_horizontal_flip(tensor, p=0.5):
    if torch.rand(1).item() < p:
        return tensor.flip(-1)  # Flip along the width axis
    return tensor


class TrafficDataset(Dataset):
    """
    Custom PyTorch Dataset for Traffic Data using the output of FeatureEngineeringPipeline.

    This dataset generates tensors for route images and additional features, suitable for training CNN models like AlexNet.

    Args:
        dataframe (pd.DataFrame): The dataframe containing raw traffic data.
        pipeline (FeatureEngineeringPipeline): The feature engineering pipeline used for transforming data.
    """

    def __init__(
        self,
        route_images_tensor: torch.tensor,
        additional_features_tensor: torch.tensor,
        labels_tensor: torch.tensor,
    ):
        # Load tensors from saved files
        logger.info("Loading preprocessed tensors.")
        assert (
            route_images_tensor.size(0)
            == additional_features_tensor.size(0)
            == labels_tensor.size(0)
        ), "Mismatch in number of samples between route images, additional features, and labels"
        self.route_images_tensor = route_images_tensor
        self.additional_features_tensor = additional_features_tensor
        self.labels_tensor = labels_tensor
        self.transforms = transforms
        logger.info("Loaded all tensors successfully.")
        logger.info(f"Number of samples: {route_images_tensor.size(0)}")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.route_images_tensor.size(0)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the feature tensor and corresponding route image tensor for the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the route image tensor and additional features tensor.
        """
        route_image = self.route_images_tensor[index]
        additional_features = self.additional_features_tensor[index]
        label = self.labels_tensor[index]

        # Ensure the image has a channel dimension
        if route_image.dim() == 2:
            route_image = route_image.unsqueeze(0)

        # Apply manual tensor-based augmentation
        route_image = random_horizontal_flip(route_image)
        return route_image, additional_features, label
