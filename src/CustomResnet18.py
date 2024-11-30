import torch
import torch.nn as nn
from torchvision.models import resnet18


class CustomResNet18(nn.Module):
    def __init__(self, num_classes, num_additional_features):
        super(CustomResNet18, self).__init__()
        # Load the ResNet18 backbone
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify the fully connected layer to output a feature vector
        self.resnet.fc = nn.Linear(512, 128)  # Example: Reduce to 128 features

        # Add a branch for additional features
        self.additional_features_fc = nn.Linear(num_additional_features, 64)

        # Combine both branches
        self.combined_fc = nn.Linear(128 + 64, num_classes)

    def forward(self, images, additional_features):
        # Forward pass through ResNet backbone
        image_features = self.resnet(images)

        # Forward pass through additional features branch
        additional_features = self.additional_features_fc(additional_features)
        additional_features = torch.relu(additional_features)

        # Concatenate image and additional features
        combined_features = torch.cat((image_features, additional_features), dim=1)

        # Final classification
        outputs = self.combined_fc(combined_features)
        return outputs
