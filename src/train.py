import os
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from AlexnetTrafficCNN import AlexNetTrafficCNN
from logger import get_logger

logger = get_logger(__name__)
from FeatureEngineering import FeatureEngineeringPipeline


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
        return route_image, additional_features, label


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, additional_features, labels in dataloader:
        images, additional_features, labels = (
            images.to(device),
            additional_features.to(device),
            labels.to(device),
        )

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images, additional_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


if __name__ == "__main__":

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # Directory containing the preprocessed tensors
    preprocessed_dir = "preprocessed_tensors"

    # Load preprocessed tensors from the folder
    try:
        logger.info("Loading preprocessed tensors from disk.")
        train_route_images_tensor = torch.load(
            os.path.join(preprocessed_dir, "train_route_images_tensor.pt")
        )
        train_additional_features_tensor = torch.load(
            os.path.join(preprocessed_dir, "train_additional_features_tensor.pt")
        )
        train_labels_tensor = torch.load(
            os.path.join(preprocessed_dir, "train_labels_tensor.pt")
        )

        val_route_images_tensor = torch.load(
            os.path.join(preprocessed_dir, "val_route_images_tensor.pt")
        )
        val_additional_features_tensor = torch.load(
            os.path.join(preprocessed_dir, "val_additional_features_tensor.pt")
        )
        val_labels_tensor = torch.load(
            os.path.join(preprocessed_dir, "val_labels_tensor.pt")
        )

        test_route_images_tensor = torch.load(
            os.path.join(preprocessed_dir, "test_route_images_tensor.pt")
        )
        test_additional_features_tensor = torch.load(
            os.path.join(preprocessed_dir, "test_additional_features_tensor.pt")
        )
        test_labels_tensor = torch.load(
            os.path.join(preprocessed_dir, "test_labels_tensor.pt")
        )

    except FileNotFoundError as e:
        logger.error("One or more tensor files not found. Please check the file paths.")
        raise e

    # Verify tensor dimensions
    assert (
        train_route_images_tensor.size(0)
        == train_additional_features_tensor.size(0)
        == train_labels_tensor.size(0)
    ), "Mismatch in number of samples between training tensors"
    assert (
        val_route_images_tensor.size(0)
        == val_additional_features_tensor.size(0)
        == val_labels_tensor.size(0)
    ), "Mismatch in number of samples between validation tensors"
    assert (
        test_route_images_tensor.size(0)
        == test_additional_features_tensor.size(0)
        == test_labels_tensor.size(0)
    ), "Mismatch in number of samples between test tensors"

    # Create datasets
    train_dataset = TrafficDataset(
        train_route_images_tensor, train_additional_features_tensor, train_labels_tensor
    )
    val_dataset = TrafficDataset(
        val_route_images_tensor, val_additional_features_tensor, val_labels_tensor
    )
    test_dataset = TrafficDataset(
        test_route_images_tensor, test_additional_features_tensor, test_labels_tensor
    )

    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_classes = 3
    num_epochs = 10
    patience = 5  # Early stopping patience

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = AlexNetTrafficCNN(
        num_classes=num_classes,
        additional_features_dim=train_additional_features_tensor.shape[1],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(
            model, train_loader, criterion, optimizer, device
        )
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, additional_features, labels in val_loader:
                images, additional_features, labels = (
                    images.to(device),
                    additional_features.to(device),
                    labels.to(device),
                )

                # Forward pass through CNN
                outputs = model(images, additional_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "traffic_status_cnn_best.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model state for further use (e.g., testing or inference)
    model.load_state_dict(torch.load("traffic_status_cnn_best.pth"))
    print("Loaded the best model from training.")

    # Save the final model after completing the entire training loop (for comparison or future use)
    torch.save(model.state_dict(), "traffic_status_cnn_final.pth")
    print("Saved the final model after the training loop.")
