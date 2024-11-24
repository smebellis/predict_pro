import os
import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report  # Added for detailed metrics
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from AlexnetTrafficCNN import AlexNetTrafficCNN, BasicTrafficCNN
from TrafficStatus import TrafficStatusCNN
from logger import get_logger

from FeatureEngineering import FeatureEngineeringPipeline
from TrafficDataset import TrafficDataset

import argparse

# ============================
# Configuration and Setup
# ============================


# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)  # Set the seed

# Initialize logger
logger = get_logger(__name__)

# ============================
# Utility Functions
# ============================


def load_tensor(file_path: str) -> torch.Tensor:
    """
    Load a tensor from a given file path.

    Args:
        file_path (str): Path to the tensor file.

    Returns:
        torch.Tensor: Loaded tensor.
    """
    try:
        tensor = torch.load(file_path)
        logger.info(f"Loaded tensor from {file_path} with shape {tensor.shape}")
        return tensor
    except FileNotFoundError as e:
        logger.error(f"Tensor file not found: {file_path}")
        raise e


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float]: Average loss and accuracy.
    """
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, additional_features, labels in dataloader:
            # Adjust image dimensions if necessary
            if images.dim() == 3:
                images = images.unsqueeze(1)  # Assuming grayscale images
            images, additional_features, labels = (
                images.to(device),
                additional_features.to(device),
                labels.to(device),
            )

            outputs = model(images, additional_features)
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = loss_total / len(dataloader)
    accuracy = 100 * correct / total

    # Detailed classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["Class0", "Class1", "Class2"],
        zero_division=0,
    )
    logger.info(f"Classification Report:\n{report}")

    return avg_loss, accuracy


def load_and_concatenate_batches(prefix: str, preprocessed_dir: str) -> torch.Tensor:
    """
    Load all tensor batches with the given prefix from the specified directory and concatenate them.

    Args:
        prefix (str): Prefix of the saved tensor files (e.g., "train", "val", "test").
        preprocessed_dir (str): Directory where the tensor batches are saved.

    Returns:
        torch.Tensor: A concatenated tensor of all the batches with the given prefix.
    """
    batch_files = [f for f in os.listdir(preprocessed_dir) if f.startswith(prefix)]
    # Sort files to maintain order
    batch_files.sort()

    tensors = []
    for batch_file in batch_files:
        if batch_file.endswith(".pt"):
            tensor_path = os.path.join(preprocessed_dir, batch_file)
            tensor = torch.load(tensor_path)
            tensors.append(tensor)

    # Concatenate all tensors into a single tensor
    concatenated_tensor = torch.cat(tensors, dim=0)
    return concatenated_tensor


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    max_grad_norm = 1.0

    for batch_idx, (images, additional_features, labels) in enumerate(dataloader):
        images = images.unsqueeze(1)  # Add channel dimension
        images = images.to(device)
        additional_features = additional_features.to(device)
        labels = labels.long().to(device)
        # images = images.repeat(1, 16, 1, 1)  # Repeat to create 16 channels

        # Check for NaNs in inputs
        if torch.isnan(images).any():
            logger.error(f"NaN detected in images at batch {batch_idx}")
        if torch.isnan(additional_features).any():
            logger.error(f"NaN detected in additional features at batch {batch_idx}")
        if torch.isnan(labels).any():
            logger.error(f"NaN detected in labels at batch {batch_idx}")
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images, additional_features)

        # Check for NaNs in outputs
        # if torch.isnan(outputs).any():
        #     logger.error(f"NaN detected in outputs at batch {batch_idx}")

        loss = criterion(outputs, labels)
        # Check for NaNs in loss
        # if torch.isnan(loss).any():
        #     logger.error(f"NaN detected in loss at batch {batch_idx}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Log gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        logger.debug(f"Gradient Norm: {total_norm}")

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
            os.path.join(preprocessed_dir, "train_route_images_tensor.pt"),
            weights_only=False,
        )
        train_additional_features_tensor = torch.load(
            os.path.join(preprocessed_dir, "train_additional_features_tensor.pt"),
            weights_only=False,
        )
        train_labels_tensor = torch.load(
            os.path.join(preprocessed_dir, "train_labels_tensor.pt"), weights_only=False
        )

        val_route_images_tensor = torch.load(
            os.path.join(preprocessed_dir, "val_route_images_tensor.pt"),
            weights_only=False,
        )
        val_additional_features_tensor = torch.load(
            os.path.join(preprocessed_dir, "val_additional_features_tensor.pt"),
            weights_only=False,
        )
        val_labels_tensor = torch.load(
            os.path.join(preprocessed_dir, "val_labels_tensor.pt"), weights_only=False
        )

        test_route_images_tensor = torch.load(
            os.path.join(preprocessed_dir, "test_route_images_tensor.pt"),
            weights_only=False,
        )
        test_additional_features_tensor = torch.load(
            os.path.join(preprocessed_dir, "test_additional_features_tensor.pt"),
            weights_only=False,
        )
        test_labels_tensor = torch.load(
            os.path.join(preprocessed_dir, "test_labels_tensor.pt"), weights_only=False
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
    learning_rate = 1e-4
    num_classes = 3
    num_epochs = 100
    patience = 5  # Early stopping patience
    best_model_path = "checkpoints/traffic_status_cnn_best.pth"
    os.makedirs("checkpoints", exist_ok=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    # for i, (route_images, additional_features, labels) in enumerate(train_loader):
    #     logger.info(f"Batch {i + 1}:")
    #     logger.info(f"  Route images shape: {route_images.shape}")
    #     logger.info(f"  Additional features shape: {additional_features.shape}")
    #     logger.info(f"  Labels shape: {labels.shape}")

    #     # Stop after a few batches for debugging purposes
    #     if i >= 2:  # Adjust this value as needed to check more or fewer batches
    #         break
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Initialize model, loss function, and optimizer
    model = TrafficStatusCNN(
        num_additional_features=train_additional_features_tensor.size(1), device=device
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(
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
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model state for further use (e.g., testing or inference)
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded the best model from training.")

    # Save the final model after completing the entire training loop (for comparison or future use)
    torch.save(model.state_dict(), best_model_path)
    print("Saved the final model after the training loop.")
