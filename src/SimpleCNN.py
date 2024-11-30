import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNWithAdditionalFeatures(nn.Module):
    def __init__(self, num_classes, num_additional_features):
        super(SimpleCNNWithAdditionalFeatures, self).__init__()
        # Convolutional layers for image features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2x2

        # Update flattened size based on the new output shape
        self.flattened_size = 32 * 32 * 32  # Adjust based on your actual input size

        # Fully connected layers for image features
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Fully connected layer for additional features
        self.additional_fc = nn.Linear(num_additional_features, 32)

        # Combined fully connected layer for final classification
        self.combined_fc = nn.Linear(64 + 32, num_classes)

    def forward(self, images, additional_features):
        # Process images through the convolutional layers
        x = F.relu(self.conv1(images))
        x = self.pool(F.relu(self.conv2(x)))
        # print(f"Shape after pooling: {x.shape}")  # Debugging output size
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer

        # Process image features through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Process additional features
        additional_features = F.relu(self.additional_fc(additional_features))

        # Concatenate image and additional features
        combined = torch.cat((x, additional_features), dim=1)

        # Final classification layer
        output = self.combined_fc(combined)
        return output
