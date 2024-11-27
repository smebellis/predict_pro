import matplotlib.pyplot as plt
import pickle
import torch
from logger import get_logger

# from train import evaluate
from TrafficStatusCNN import TrafficStatusCNN

logger = get_logger(__name__)

# ============================
# Plot Training and Validation Metrics
# ============================


def plot_metrics(metrics):
    train_accuracies = metrics["train_accuracies"]
    val_accuracies = metrics["val_accuracies"]
    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Load the metrics
    with open("metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    plot_metrics(metrics)
