import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Sample CNN Model Definition
class TrafficStatusCNN(nn.Module):
    def __init__(
        self,
        num_additional_features,
        num_classes=3,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        super(TrafficStatusCNN, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self._initialize_weights()
        # Define CNN layers for route images with Batch Normalization and updated Dropout
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),  # Added Batch Normalization
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),  # Added Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),  # Added Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),  # Increased Dropout to 0.4
        ).to(device)

        # Fully connected layers for additional features
        self.fc_additional = nn.Sequential(
            nn.Linear(num_additional_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased Dropout to 0.4
        ).to(device)

        # Placeholder for dynamically computed flattened size
        self.flattened_size = None

        # Final classification layer (initialized later once flattened size is known)
        self.fc_combined = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, additional_features):
        # Ensure input images have the correct number of dimensions
        if images.ndim == 3:  # If there is no channel dimension
            images = images.unsqueeze(
                1
            )  # Add channel dimension, assuming single-channel grayscale

        # Forward pass through CNN
        image_features = self.cnn_layers(images)
        # print(
        #     "Shape after CNN layers:", image_features.shape
        # )  # Should be [batch_size, channels, height, width]

        # Dynamically calculate the flattened size if not already done
        if self.flattened_size is None:
            if len(image_features.shape) != 4:
                raise ValueError(
                    "Unexpected shape after CNN layers, expected 4 dimensions, got: {}".format(
                        image_features.shape
                    )
                )
            self.flattened_size = (
                image_features.size(1) * image_features.size(2) * image_features.size(3)
            )
            # Initialize fully connected layers with the calculated flattened size
            self.fc_combined = nn.Sequential(
                nn.Linear(self.flattened_size + 128, 256),
                nn.ReLU(),
                nn.Dropout(0.5),  # Increased dropout to 0.5
                nn.Linear(256, self.num_classes),
            ).to(self.device)

        # Flatten the image features
        image_features = image_features.view(image_features.size(0), -1)
        # print(
        #     "Shape after flattening image features:", image_features.shape
        # )  # Should be [batch_size, flattened_size]

        # Forward pass through fully connected layers for additional features
        additional_features_out = self.fc_additional(additional_features)
        # print(
        #     "Shape of additional features out:", additional_features_out.shape
        # )  # Should be [batch_size, 64]

        # Remove extra dimensions from additional_features_out if necessary
        if additional_features_out.ndim == 3 and additional_features_out.size(1) == 1:
            additional_features_out = additional_features_out.squeeze(1)
        # print(
        #     "Shape of additional features out after squeeze:",
        #     additional_features_out.shape,
        # )  # Should be [batch_size, 64]

        # Concatenate both sets of features
        combined_features = torch.cat((image_features, additional_features_out), dim=1)
        # print(
        #     "Shape after concatenation:", combined_features.shape
        # )  # Should be [batch_size, flattened_size + 64]

        # Pass through final classification layers
        output = self.fc_combined(combined_features)
        return output
