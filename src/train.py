import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
from AlexnetTrafficCNN import AlexNetTrafficCNN


class TrafficDataset(Dataset):
    def encode_weekday(self, weekday):
        # Convert weekday from string to a numerical representation
        weekdays = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        return weekdays.get(weekday, -1)

    def encode_cluster(self, cluster):
        # Convert cluster from string to a numerical representation
        clusters = {
            "Cluster 1": 1,
            "Cluster 2": 2,
            "Cluster 3": 3,
            "Cluster 4": 4,
            "Cluster 5": 5,
        }
        return clusters.get(cluster, -1)

    def encode_month(self, month):
        # Convert month from string to a numerical representation
        months = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }
        return months.get(month, -1)

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Create an image-like representation based on POLYLINE or other geographical features
        image = self.create_image_representation(row["POLYLINE"])
        if self.transform:
            image = self.transform(image)

        # Extract additional features (e.g., TIME, WEEKDAY, etc.)
        # Extract additional features (e.g., TIME, WEEKDAY, etc.)
        # Extract additional features (e.g., TIME, WEEKDAY, etc.)
        additional_features = torch.tensor(
            [
                self.encode_weekday(row["WEEKDAY"]),
                float(row["TIME"]),
                self.encode_month(row["MONTH"]),
                float(row["YEAR"]),
                self.encode_cluster(row["DISTRICT_CLUSTER"]),
                float(row["TIME_CLUSTER"]),
                float(row["DOM"]),
            ],
            dtype=torch.float,
        )
        label = torch.tensor(row["TRAFFIC_STATUS"], dtype=torch.long)
        return image, additional_features, label

    def create_image_representation(self, polyline):
        # Convert POLYLINE (list of coordinates) into an image-like representation
        # Create an empty grayscale image of size 224x224
        image = np.zeros((224, 224), dtype=np.uint8)

        # Normalize coordinates to fit into the 224x224 grid
        coordinates = (
            eval(polyline) if isinstance(polyline, str) else polyline
        )  # Convert string representation of list to actual list

        latitudes = [coord[0] for coord in coordinates]
        longitudes = [coord[1] for coord in coordinates]

        if len(latitudes) > 0 and len(longitudes) > 0:
            lat_min, lat_max = min(latitudes), max(latitudes)
            lon_min, lon_max = min(longitudes), max(longitudes)

            for lat, lon in coordinates:
                # Scale latitude and longitude to fit in the 224x224 grid
                x = int((lat - lat_min) / (lat_max - lat_min + 1e-5) * 223)
                y = int((lon - lon_min) / (lon_max - lon_min + 1e-5) * 223)
                # Draw the point on the image
                image = cv2.circle(image, (y, x), radius=1, color=255, thickness=-1)

        # Convert grayscale image to RGB by stacking the channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
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
    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 16
    num_classes = 3

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usinge Device: {device}")
    # Load dataset (replace with your actual dataset path)
    df = pd.read_csv(
        "/home/smebellis/ece5831_final_project/processed_data/clustered_dataset.csv"
    )  # Replace with actual path

    # Train-test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Create datasets and dataloaders
    train_dataset = TrafficDataset(train_df, transform=transform)
    val_dataset = TrafficDataset(val_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = AlexNetTrafficCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
                outputs = model(images)
                combined_features = torch.cat((outputs, additional_features), dim=1)
                loss = criterion(combined_features, labels)

                val_loss += loss.item()
                _, predicted = torch.max(combined_features, 1)
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

    # Save the final model
    torch.save(model.state_dict(), "traffic_status_cnn.pth")
