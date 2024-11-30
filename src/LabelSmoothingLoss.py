import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        """
        Label smoothing loss.

        Args:
            num_classes (int): Number of output classes.
            smoothing (float): Smoothing factor (0 means no smoothing).
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, predictions, targets):
        """
        Forward pass for label smoothing loss.

        Args:
            predictions (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        log_preds = torch.log_softmax(predictions, dim=-1)
        true_dist = torch.zeros_like(log_preds)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))
