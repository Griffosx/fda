"""
This script implements a complete workflow for training an image classification model using PyTorch.
It leverages a pre-trained ResNet34 architecture from torchvision, adapts it for a custom dataset,
and provides a structured training loop with validation and model saving capabilities.

The script is designed for datasets organized in a standard ImageFolder format, where
images are sorted into subdirectories corresponding to their respective classes.

Key functionalities include:
-   **Data Loading & Preprocessing:** Utilizes `ImageFolder` and `DataLoader` for efficient
    data handling, with configurable image resizing and tensor conversion.
-   **Dataset Splitting:** Automatically divides the dataset into training and validation sets.
-   **Model Initialization:** Supports loading a pre-trained ResNet34 model (recommended for
    transfer learning) and modifies its final layer to classify images into the
    specific number of classes in your dataset.
-   **Trainer Class:** Encapsulates the training and validation logic for better code
    organization and reusability. It handles forward passes, backward propagation,
    optimizer steps, and tracks loss and accuracy metrics per epoch.
-   **Model Saving:** Saves the model's state dictionary when a new best validation
    accuracy is achieved, allowing for easy resumption or deployment.
-   **Device Agnostic:** Automatically detects and uses a CUDA-enabled GPU if available,
    falling back to CPU otherwise.

To use this script, ensure your dataset is structured with subfolders for each class
(e.g., `data_dir/class1/img1.jpg`, `data_dir/class2/img2.jpg`), and update the `data_dir`
argument in the `if __name__ == "__main__":` block to point to your dataset's root directory.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
SAVE_PATH = "best_model.pth"
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        """
        A trainer class to handle the training and validation loops.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc

    def train(self, num_epochs, save_path=None):
        """Full training loop"""
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save the best model
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

            print("-" * 50)


def get_model(num_classes, use_pretrained=True):
    """
    Get a ResNet model with the final layer modified for classification.
    """
    if use_pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34()

    # Modify the final layer to match our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def main(
    data_dir,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    val_split=VAL_SPLIT,
    save_path=SAVE_PATH,
    use_pretrained=True,
):
    """
    Main function to train a ResNet model for image classification.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            # transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    # Create dataset using ImageFolder (handles class folders automatically)
    full_dataset = ImageFolder(root=data_dir, transform=transform)

    # Get class count and mapping
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")

    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    # Initialize model
    model = get_model(num_classes=num_classes, use_pretrained=use_pretrained)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    trainer.train(num_epochs=num_epochs, save_path=save_path)

    print("Training completed!")


if __name__ == "__main__":
    # Example usage
    main(data_dir="assets/train")
