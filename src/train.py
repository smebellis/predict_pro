import argparse
import os
import pickle
import random
import time
from collections import deque
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from AlexnetTrafficCNN import AlexNetTrafficCNN, BasicTrafficCNN
from FeatureEngineering import FeatureEngineeringPipeline
from logger import get_logger
from plotting import plot_metrics
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
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
)  # Adjusted learning rate
parser.add_argument(
    "--num_epochs", type=int, default=50, help="Number of training epochs"
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
        tensor = torch.load(file_path, weights_only=True)
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
    # Calculate other metrics
    metrics = {
        "Accuracy": accuracy_score(all_labels, all_predictions),
        "Precision": precision_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(all_labels, all_predictions, average="weighted"),
        "Confusion Matrix": confusion_matrix(all_labels, all_predictions),
    }
    # logger.info(f"Classification Report:\n{report}")

    return avg_loss, accuracy, metrics


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


# ============================
# Main Training Script
# ============================


def main():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    # Compute class weights to deal with class imbalance
    # class_weights = torch.tensor([1.0, 1.0, 5.0, 5.0], dtype=torch.float32).to(device)
    # Dynamic Class Weights
    labels = train_labels_tensor.cpu().numpy()
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    logger.info("Initialized optimizer and scheduler.")

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    # ============================
    # Save Metrics Incrementally
    # ============================

    # Parameters
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)
        logger.info(f"Created directory: {metrics_dir}")

    metrics_path = os.path.join(metrics_dir, "metrics_incremental.pkl")

    # Initialize or load existing metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
        logger.info("Loaded existing metrics for continuation.")
    else:
        metrics = {
            "train_accuracies": [],
            "val_accuracies": [],
            "train_losses": [],
            "val_losses": [],
        }

    # Parameters
    max_saved_models = 5  # Number of recent models to keep
    saved_models_queue = deque(maxlen=max_saved_models)

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
            if batch_idx % 10 == 0:
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
        metrics["train_losses"].append(epoch_loss)
        metrics["train_accuracies"].append(accuracy)
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        # Validation
        val_loss, val_accuracy, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        metrics["val_losses"].append(val_loss)
        metrics["val_accuracies"].append(val_accuracy)
        metrics["val_metrics"] = val_metrics  # Store validation metrics
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

        # Save metrics incrementally after every epoch
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)
        logger.info(f"Metrics incrementally saved at '{metrics_path}'.")

        # Save the current model with a timestamp
        os.makedirs("models", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M")
        current_model_path = os.path.join("models", f"model_{timestamp}.pth")
        torch.save(model.state_dict(), current_model_path)
        logger.info(f"Model saved with timestamp: {current_model_path}")

        # Rotate saved models
        if len(saved_models_queue) == max_saved_models:
            # If deque is full, the oldest model will be removed from the deque.
            # Get the oldest model path before appending the new one.
            oldest_model = saved_models_queue[0]
            if os.path.exists(oldest_model) and oldest_model != best_model_path:
                os.remove(oldest_model)
                logger.info(f"Oldest model removed from disk: {oldest_model}")

        # Append the new model path to the deque
        saved_models_queue.append(current_model_path)
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # Load the best model state for further use (e.g., testing or inference)
    model.load_state_dict(torch.load(best_model_path))
    logger.info("Loaded the best model from training.")

    # ============================
    # Test Evaluation
    # ============================

    test_loss, test_accuracy, test_metrics = evaluate(
        model, test_loader, criterion, device
    )
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Print the metrics similar to the Zero-Rule and Random Classifier example
    logger.info("\nTest Performance Metrics:\n")
    for metric, value in test_metrics.items():
        if metric == "Confusion Matrix":
            logger.info(f"{metric}:\n{value}\n")
        else:
            logger.info(f"{metric}: {value}")

    # Store metrics to a DataFrame for easier visualization if needed
    test_metrics_df = pd.DataFrame(test_metrics, index=[0])
    test_metrics_df.to_pickle("pickle_files/test_metrics.pkl")

    # ============================
    # Plot Training and Validation Metrics
    # ============================

    # Load the metrics
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    logger.info("Loaded the metrics for plotting.")
    plot_metrics(metrics)


if __name__ == "__main__":
    main()
