import matplotlib.pyplot as plt
import pickle
import torch
from logger import get_logger
import pandas as pd

# from train import evaluate
from TrafficStatusCNN import TrafficStatusCNN
from train import evaluate

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


def reload_and_evaluate_model(
    model_path, train_loader, val_loader, criterion, device, label_encoder
):
    # Reload the model
    model = TrafficStatusCNN(num_additional_features=10, device=device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Recompute metrics for training and validation sets
    train_loss, train_accuracy = evaluate(
        model, train_loader, criterion, device, label_encoder
    )
    val_loss, val_accuracy = evaluate(
        model, val_loader, criterion, device, label_encoder
    )

    logger.info(
        f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%"
    )
    return train_loss, train_accuracy, val_loss, val_accuracy


if __name__ == "__main__":
    # Load the metrics
    with open(
        "/home/smebellis/ece5831_final_project/metrics/Best_Metrics/metrics_2024-11-30_1054_best.pkl",
        "rb",
    ) as f:
        metrics = pickle.load(f)
    last_epoch_metrics = {
        "Accuracy": metrics["val_accuracies"][-1],
        "Precision": metrics["val_precision"][-1],
        "Recall": metrics["val_recall"][-1],
        "F1 Score": metrics["val_f1_score"][-1],
    }

    # Creating a DataFrame for easier visualization
    metrics_df = pd.DataFrame(last_epoch_metrics, index=["Validation Metrics"])
    print(metrics_df)
    breakpoint()
    plot_metrics(metrics)
