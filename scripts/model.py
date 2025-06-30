"""
HaemaVision CNN Model Definition

This module defines the CNN architecture for blood cell classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BloodCellCNN(nn.Module):
    """
    Convolutional Neural Network for blood cell classification.
    
    Features:
    - Multiple convolutional layers with ReLU activations
    - Max pooling for spatial dimension reduction
    - Dropout layers for regularization
    - Batch normalization for training stability
    - Fully connected layers with softmax output
    """
    
    def __init__(self, num_classes=8, input_size=128):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes
            input_size (int): Input image size (square)
        """
        super(BloodCellCNN, self).__init__()
        
        # Calculate the size of features after convolutions and pooling
        # Each max pooling divides dimensions by 2
        self.feature_size = input_size // 8  # After 3 pooling layers: 128 -> 16
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * self.feature_size * self.feature_size, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image batch with shape [batch_size, 3, input_size, input_size]
            
        Returns:
            torch.Tensor: Logits for each class
        """
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout1(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class LightBloodCellCNN(nn.Module):
    """
    A lighter CNN model for blood cell classification.
    Useful for devices with limited computational resources.
    """
    
    def __init__(self, num_classes=8, input_size=128):
        super(LightBloodCellCNN, self).__init__()
        
        self.feature_size = input_size // 8
        
        # Simplified architecture with fewer filters
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64 * self.feature_size * self.feature_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def get_model(model_name="standard", num_classes=8, input_size=128):
    """
    Factory function to get the requested model.
    
    Args:
        model_name (str): Model variant to use ('standard' or 'light')
        num_classes (int): Number of output classes
        input_size (int): Input image size (square)
        
    Returns:
        nn.Module: Initialized model
    """
    if model_name.lower() == "light":
        return LightBloodCellCNN(num_classes, input_size)
    else:  # Default to standard model
        return BloodCellCNN(num_classes, input_size)
