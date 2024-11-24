import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetTrafficCNN(nn.Module):
    def __init__(self, num_classes=3, additional_features_dim=4):
        super(AlexNetTrafficCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                16, 64, kernel_size=11, stride=2, padding=2
            ),  # Changed input channels to 16
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Use the dynamically determined flattened size for the Linear layer
        self.flattened_size = self._get_flattened_size()

        # Combine the CNN output with the additional features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                self.flattened_size + additional_features_dim, 1024
            ),  # Reduced from 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def _get_flattened_size(self):
        # Create a dummy input to determine the size after feature extraction
        with torch.no_grad():
            dummy_input = torch.randn(
                1, 16, 224, 224
            )  # Example input with 16 channels and size 224x224
            features_out = self.features(dummy_input)
            print(
                f"Features output shape: {features_out.shape}"
            )  # Debugging line to show the output shape
            return features_out.view(1, -1).size(1)

    def forward(self, x, additional_features):
        # Pass through the CNN layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            print(f"Output shape after layer {i}: {x.shape}")

        # Flatten the features for concatenation
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch size
        print(f"Shape after flattening: {x.shape}")

        # Check the shape of additional features
        print(f"Shape of additional features: {additional_features.shape}")

        # Ensure batch sizes are identical
        if x.size(0) != additional_features.size(0):
            raise RuntimeError(
                f"Batch size mismatch: CNN output batch size {x.size(0)} does not match additional features batch size {additional_features.size(0)}"
            )

        # Concatenate the CNN output with additional features
        x = torch.cat((x, additional_features), dim=1)
        x = self.classifier(x)
        return x


class BasicTrafficCNN(nn.Module):
    def __init__(self, num_classes=3, additional_features_dim=4):
        super(BasicTrafficCNN, self).__init__()
        # Define a basic CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(
                16, 32, kernel_size=3, stride=1, padding=1
            ),  # Basic convolution layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Use the dynamically determined flattened size for the Linear layer
        self.flattened_size = self._get_flattened_size()

        # Combine the CNN output with the additional features
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size + additional_features_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def _get_flattened_size(self):
        # Create a dummy input to determine the size after feature extraction
        with torch.no_grad():
            dummy_input = torch.randn(
                1, 16, 64, 64
            )  # Example input with 16 channels and size 64x64
            features_out = self.features(dummy_input)
            return features_out.view(1, -1).size(1)

    def forward(self, x, additional_features):
        # Pass through the CNN layers
        x = self.features(x)
        # Flatten the features for concatenation
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch size
        # Ensure additional features are the same batch size
        if additional_features.size(0) != x.size(0):
            additional_features = additional_features.expand(x.size(0), -1)
        # Concatenate the CNN output with additional features
        x = torch.cat((x, additional_features), dim=1)
        # Pass through the classifier
        x = self.classifier(x)
        return x


# Example usage
if __name__ == "__main__":
    model = BasicTrafficCNN(num_classes=3)
    print(model)

    # Create a random tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(
        1, 16, 64, 64
    )  # Updated to match the new input channels and size
    additional_features = torch.randn(
        1, 4
    )  # Batch size of 1, with 4 additional features
    output = model(input_tensor, additional_features)
    print(output.shape)  # Expected output shape: (1, 3)
