"""
HaemaVision Model Training Script

This script handles the training process for the blood cell classification model:
- Loads and prepares datasets
- Sets up the model, loss function, and optimizer
- Implements the training loop
- Tracks and visualizes metrics
- Saves the best model weights
"""

import os
import json
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import get_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a blood cell classifier")
    parser.add_argument("--data_dir", type=str, default="../data", 
                        help="Path to the dataset directory")
    parser.add_argument("--model_dir", type=str, default="../models", 
                        help="Directory to save models")
    parser.add_argument("--model_type", type=str, default="standard", 
                        choices=["standard", "light"],
                        help="Model architecture to use")
    parser.add_argument("--img_size", type=int, default=128, 
                        help="Input image size (square)")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, 
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    return parser.parse_args()

def get_transforms(img_size):
    """Define data transforms for training and validation."""
    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def prepare_dataloaders(data_dir, train_transforms, val_transforms, batch_size, val_split, seed):
    """Create training and validation dataloaders."""
    # Create dataset with training transforms
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    
    # Get class to index mapping and class names
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)
    
    # Calculate sizes for train and validation sets
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Create train and validation splits
    train_dataset, val_indices = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create validation dataset with validation transforms
    val_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=val_transforms
    )
    
    # Use the same indices as the validation split
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices.indices]
    val_dataset.targets = [full_dataset.targets[i] for i in val_indices.indices]
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    dataset_info = {
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "train_size": train_size,
        "val_size": val_size
    }
    
    return train_loader, val_loader, dataset_info

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Update progress bar
        pbar.set_postfix({"batch_loss": loss.item(), 
                          "batch_acc": torch.sum(preds == labels.data).item() / inputs.size(0)})
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate(model, val_loader, criterion, device):
    """Validate the model performance on the validation set."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Disable gradient computation for validation
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def plot_training_history(train_loss, val_loss, train_acc, val_acc, save_path):
    """Plot and save training history graphs."""
    epochs = range(1, len(train_loss) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history plot saved to {save_path}")

def save_model(model, model_path, config_path, dataset_info, img_size, model_type):
    """Save the model and configuration."""
    # Save model weights
    torch.save(model.state_dict(), model_path)
    
    # Prepare configuration for saving
    config = {
        "model_type": model_type,
        "img_size": img_size,
        "num_classes": dataset_info["num_classes"],
        "class_names": dataset_info["class_names"],
        "class_to_idx": dataset_info["class_to_idx"],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    
    # Save configuration
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Model saved to {model_path}")
    print(f"Configuration saved to {config_path}")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create transforms
    train_transforms, val_transforms = get_transforms(args.img_size)
    
    # Prepare data loaders
    train_loader, val_loader, dataset_info = prepare_dataloaders(
        args.data_dir, train_transforms, val_transforms, 
        args.batch_size, args.val_split, args.seed
    )
    
    print(f"Dataset loaded: {dataset_info['train_size']} training samples, "
          f"{dataset_info['val_size']} validation samples")
    print(f"Classes: {dataset_info['class_names']}")
    
    # Initialize model
    model = get_model(args.model_type, dataset_info["num_classes"], args.img_size)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.model_dir, f"best_{args.model_type}_model.pth")
            config_path = os.path.join(args.model_dir, "model_config.json")
            
            save_model(model, best_model_path, config_path, dataset_info, args.img_size, args.model_type)
            print(f"New best validation accuracy: {best_val_acc:.4f}")
    
    # Total training time
    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot and save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.model_dir, f"training_history_{timestamp}.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, f"final_{args.model_type}_model.pth")
    save_model(model, final_model_path, config_path, dataset_info, args.img_size, args.model_type)

if __name__ == "__main__":
    main()
