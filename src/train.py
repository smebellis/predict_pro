import argparse
import os
import pickle
import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report  # Added for detailed metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from AlexnetTrafficCNN import AlexNetTrafficCNN, BasicTrafficCNN
from FeatureEngineering import FeatureEngineeringPipeline
from logger import get_logger
from src.TrafficStatusCNN import TrafficStatusCNN
from TrafficDataset import TrafficDataset

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
# Argument Parsing
# ============================

parser = argparse.ArgumentParser(description="Train TrafficStatusCNN model.")
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
)  # Adjusted learning rate
parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
parser.add_argument(
    "--preprocessed_dir",
    type=str,
    default="preprocessed_tensors",
    help="Directory of preprocessed tensors",
)
parser.add_argument(
    "--model_save_path",
    type=str,
    default="checkpoints/traffic_status_cnn_best.pth",
    help="Path to save the best model",
)
parser.add_argument(
    "--label_encoder_path",
    type=str,
    default="pickle_files/label_encoder.pkl",
    help="Path to the LabelEncoder if used",
)
args = parser.parse_args()

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

    # Verify unique labels
    unique_labels = np.unique(all_labels)
    unique_predictions = np.unique(all_predictions)
    logger.info(f"Unique labels in ground truth: {unique_labels}")
    logger.info(f"Unique labels in predictions: {unique_predictions}")

    # Get class names
    class_names = get_class_names(args.label_encoder_path, unique_labels)

    # Check if class_names length matches unique labels
    if len(class_names) != len(unique_labels):
        logger.warning(
            "Number of class names does not match number of unique labels. Omitting target_names."
        )
        report = classification_report(all_labels, all_predictions, zero_division=0)
    else:
        report = classification_report(
            all_labels, all_predictions, target_names=class_names, zero_division=0
        )

    # logger.info(f"Classification Report:\n{report}")

    return avg_loss, accuracy


def get_class_names(label_encoder_path: str, unique_labels: np.ndarray) -> list:
    """
    Retrieve class names from a LabelEncoder if available, else use label indices.

    Args:
        label_encoder_path (str): Path to the saved LabelEncoder.
        unique_labels (np.ndarray): Array of unique label indices.

    Returns:
        list: List of class names.
    """
    if os.path.exists(label_encoder_path):
        try:
            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
            class_names = label_encoder.classes_
            logger.info(f"Class names retrieved from LabelEncoder: {class_names}")
            return list(class_names)
        except Exception as e:
            logger.error(f"Failed to load LabelEncoder: {e}")
            logger.warning("Falling back to using label indices as class names.")
    else:
        logger.warning(
            f"LabelEncoder file not found at {label_encoder_path}. Using label indices as class names."
        )

    # Fallback to using label indices as class names
    return [f"Class{label}" for label in unique_labels]


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


# ============================
# Main Training Script
# ============================


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # Directory containing the preprocessed tensors
    preprocessed_dir = args.preprocessed_dir

    # Load preprocessed tensors from the folder
    try:
        logger.info("Loading preprocessed tensors from disk.")
        train_route_images_tensor = load_tensor(
            os.path.join(preprocessed_dir, "train_route_images_tensor.pt")
        )
        train_additional_features_tensor = load_tensor(
            os.path.join(preprocessed_dir, "train_additional_features_tensor.pt")
        )
        train_labels_tensor = load_tensor(
            os.path.join(preprocessed_dir, "train_labels_tensor.pt")
        )

        val_route_images_tensor = load_tensor(
            os.path.join(preprocessed_dir, "val_route_images_tensor.pt")
        )
        val_additional_features_tensor = load_tensor(
            os.path.join(preprocessed_dir, "val_additional_features_tensor.pt")
        )
        val_labels_tensor = load_tensor(
            os.path.join(preprocessed_dir, "val_labels_tensor.pt")
        )

        test_route_images_tensor = load_tensor(
            os.path.join(preprocessed_dir, "test_route_images_tensor.pt")
        )
        test_additional_features_tensor = load_tensor(
            os.path.join(preprocessed_dir, "test_additional_features_tensor.pt")
        )
        test_labels_tensor = load_tensor(
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
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_classes = 3
    num_epochs = args.num_epochs
    patience = args.patience  # Early stopping patience
    best_model_path = args.model_save_path
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,  # Changed drop_last to False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,  # Changed drop_last to False
    )

    # Initialize model, loss function, and optimizer
    model = TrafficStatusCNN(
        num_additional_features=train_additional_features_tensor.size(1), device=device
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    logger.info("Initialized optimizer and scheduler.")

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        max_grad_norm = 1.0

        for batch_idx, (images, additional_features, labels) in enumerate(train_loader):
            # Adjust image dimensions if necessary
            if images.dim() == 3:
                images = images.unsqueeze(1)  # Assuming grayscale images

            images = images.to(device)
            additional_features = additional_features.to(device)
            labels = labels.long().to(device)

            # Check for NaNs in inputs
            if torch.isnan(images).any():
                logger.error(f"NaN detected in images at batch {batch_idx}")
            if torch.isnan(additional_features).any():
                logger.error(
                    f"NaN detected in additional features at batch {batch_idx}"
                )
            if torch.isnan(labels).any():
                logger.error(f"NaN detected in labels at batch {batch_idx}")

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, additional_features)

            # Check for NaNs in outputs
            if torch.isnan(outputs).any():
                logger.error(f"NaN detected in outputs at batch {batch_idx}")

            loss = criterion(outputs, labels)

            # Check for NaNs in loss
            if torch.isnan(loss).any():
                logger.error(f"NaN detected in loss at batch {batch_idx}")

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Log gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            logger.debug(
                f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] - Gradient Norm: {total_norm:.4f}"
            )

            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        # Validation
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        logger.info(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Step the scheduler
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved with Validation Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement in Validation Loss. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # Load the best model state for further use (e.g., testing or inference)
    model.load_state_dict(torch.load(best_model_path))
    logger.info("Loaded the best model from training.")

    # ============================
    # Test Evaluation
    # ============================

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
# if __name__ == "__main__":

#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using Device: {device}")

#     # Directory containing the preprocessed tensors
#     preprocessed_dir = "preprocessed_tensors"

#     # Load preprocessed tensors from the folder
#     try:
#         logger.info("Loading preprocessed tensors from disk.")
#         train_route_images_tensor = torch.load(
#             os.path.join(preprocessed_dir, "train_route_images_tensor.pt"),
#             weights_only=False,
#         )
#         train_additional_features_tensor = torch.load(
#             os.path.join(preprocessed_dir, "train_additional_features_tensor.pt"),
#             weights_only=False,
#         )
#         train_labels_tensor = torch.load(
#             os.path.join(preprocessed_dir, "train_labels_tensor.pt"), weights_only=False
#         )

#         val_route_images_tensor = torch.load(
#             os.path.join(preprocessed_dir, "val_route_images_tensor.pt"),
#             weights_only=False,
#         )
#         val_additional_features_tensor = torch.load(
#             os.path.join(preprocessed_dir, "val_additional_features_tensor.pt"),
#             weights_only=False,
#         )
#         val_labels_tensor = torch.load(
#             os.path.join(preprocessed_dir, "val_labels_tensor.pt"), weights_only=False
#         )

#         test_route_images_tensor = torch.load(
#             os.path.join(preprocessed_dir, "test_route_images_tensor.pt"),
#             weights_only=False,
#         )
#         test_additional_features_tensor = torch.load(
#             os.path.join(preprocessed_dir, "test_additional_features_tensor.pt"),
#             weights_only=False,
#         )
#         test_labels_tensor = torch.load(
#             os.path.join(preprocessed_dir, "test_labels_tensor.pt"), weights_only=False
#         )

#     except FileNotFoundError as e:
#         logger.error("One or more tensor files not found. Please check the file paths.")
#         raise e

#     # Verify tensor dimensions
#     assert (
#         train_route_images_tensor.size(0)
#         == train_additional_features_tensor.size(0)
#         == train_labels_tensor.size(0)
#     ), "Mismatch in number of samples between training tensors"
#     assert (
#         val_route_images_tensor.size(0)
#         == val_additional_features_tensor.size(0)
#         == val_labels_tensor.size(0)
#     ), "Mismatch in number of samples between validation tensors"
#     assert (
#         test_route_images_tensor.size(0)
#         == test_additional_features_tensor.size(0)
#         == test_labels_tensor.size(0)
#     ), "Mismatch in number of samples between test tensors"

#     # Create datasets
#     train_dataset = TrafficDataset(
#         train_route_images_tensor, train_additional_features_tensor, train_labels_tensor
#     )
#     val_dataset = TrafficDataset(
#         val_route_images_tensor, val_additional_features_tensor, val_labels_tensor
#     )
#     test_dataset = TrafficDataset(
#         test_route_images_tensor, test_additional_features_tensor, test_labels_tensor
#     )

#     # Hyperparameters
#     batch_size = 16
#     learning_rate = 1e-4
#     num_classes = 3
#     num_epochs = 100
#     patience = 5  # Early stopping patience
#     best_model_path = "checkpoints/traffic_status_cnn_best.pth"
#     os.makedirs("checkpoints", exist_ok=True)
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
#     )
#     # for i, (route_images, additional_features, labels) in enumerate(train_loader):
#     #     logger.info(f"Batch {i + 1}:")
#     #     logger.info(f"  Route images shape: {route_images.shape}")
#     #     logger.info(f"  Additional features shape: {additional_features.shape}")
#     #     logger.info(f"  Labels shape: {labels.shape}")

#     #     # Stop after a few batches for debugging purposes
#     #     if i >= 2:  # Adjust this value as needed to check more or fewer batches
#     #         break
#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
#     )

#     # Initialize model, loss function, and optimizer
#     model = TrafficStatusCNN(
#         num_additional_features=train_additional_features_tensor.size(1), device=device
#     ).to(device)

#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # Early stopping variables
#     best_val_loss = float("inf")
#     patience_counter = 0

#     # Training loop
#     for epoch in range(num_epochs):
#         train_loss, train_accuracy = train(
#             model, train_loader, criterion, optimizer, device
#         )
#         logger.info(
#             f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
#         )

#         # Validation
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             for images, additional_features, labels in val_loader:
#                 images, additional_features, labels = (
#                     images.to(device),
#                     additional_features.to(device),
#                     labels.to(device),
#                 )

#                 # Forward pass through CNN
#                 outputs = model(images, additional_features)
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()

#         val_loss /= len(val_loader)
#         val_accuracy = 100 * val_correct / val_total
#         print(
#             f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
#         )

#         # Early stopping check
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             # Save the best model
#             torch.save(model.state_dict(), best_model_path)
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break

#     # Load the best model state for further use (e.g., testing or inference)
#     model.load_state_dict(torch.load(best_model_path))
#     print("Loaded the best model from training.")

#     # Save the final model after completing the entire training loop (for comparison or future use)
#     torch.save(model.state_dict(), best_model_path)
#     print("Saved the final model after the training loop.")
