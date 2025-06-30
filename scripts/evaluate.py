"""
HaemaVision Model Evaluation Script

This script evaluates a trained model on the test/validation set:
- Generates a confusion matrix
- Calculates accuracy, precision, recall, and F1-score
- Visualizes predictions on sample images
- Provides detailed performance metrics by class
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from model import get_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a blood cell classifier")
    parser.add_argument("--data_dir", type=str, default="../data", 
                        help="Path to the dataset directory")
    parser.add_argument("--model_dir", type=str, default="../models", 
                        help="Directory with saved models")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a specific model file (if not using best_model.pth)")
    parser.add_argument("--config_path", type=str, default="../models/model_config.json",
                        help="Path to the model configuration file")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sample images to visualize per class")
    return parser.parse_args()

def load_model_and_config(model_path, config_path):
    """Load the model and configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model with the right architecture
    model = get_model(
        model_name=config.get("model_type", "standard"),
        num_classes=config["num_classes"],
        input_size=config["img_size"]
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model, config

def get_data_loader(data_dir, config, batch_size):
    """Create a data loader for evaluation."""
    # Define transforms using the same normalization as training
    eval_transforms = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["normalization"]["mean"],
            std=config["normalization"]["std"]
        )
    ])
    
    # Create dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=eval_transforms)
    
    # Sanity check: verify that class mapping is consistent with training
    if dataset.class_to_idx != config["class_to_idx"]:
        print("WARNING: Class mapping in the dataset doesn't match the saved configuration.")
        print(f"Dataset classes: {dataset.class_to_idx}")
        print(f"Config classes: {config['class_to_idx']}")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return data_loader, dataset

def evaluate_model(model, data_loader, device):
    """Evaluate model and return predictions with true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save confusion matrix."""
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")
    return cm

def visualize_predictions(model, dataset, class_names, idx_to_class, config, num_samples, save_dir):
    """Visualize model predictions on sample images."""
    model.eval()
    device = next(model.parameters()).device
    
    # Create directories for visualization
    os.makedirs(save_dir, exist_ok=True)
    
    # Get samples by class
    samples_by_class = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        class_name = idx_to_class[label]
        
        if class_name not in samples_by_class:
            samples_by_class[class_name] = []
        
        if len(samples_by_class[class_name]) < num_samples:
            samples_by_class[class_name].append((img, i))
    
    # Define normalization for visualization
    mean = torch.tensor(config["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"]).view(3, 1, 1)
    
    # Visualize predictions for each class
    for class_name, samples in samples_by_class.items():
        fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
        
        for i, (img, idx) in enumerate(samples):
            # Get original image path for reference
            img_path = dataset.samples[idx][0]
            
            # Make prediction
            with torch.no_grad():
                input_tensor = img.unsqueeze(0).to(device)
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, pred_idx = torch.max(output, 1)
                pred_class = idx_to_class[pred_idx.item()]
                confidence = probs[0][pred_idx].item()
            
            # Denormalize image for display
            img_display = img * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # Display image with prediction
            if len(samples) > 1:
                ax = axes[i]
            else:
                ax = axes
            ax.imshow(img_display)
            color = 'green' if pred_class == class_name else 'red'
            title = f"Pred: {pred_class}\nConf: {confidence:.2f}"
            ax.set_title(title, color=color)
            ax.axis('off')
        
        plt.suptitle(f"Class: {class_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"predictions_{class_name}.png"))
        plt.close()
    
    print(f"Prediction visualizations saved to {save_dir}")

def print_metrics(y_true, y_pred, class_names):
    """Print detailed metrics for model evaluation."""
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Per-class precision, recall and f1-score
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Macro-averaged metrics
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    print("\nMacro-averaged Metrics:")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1-score: {f1_macro:.4f}")
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set default model path if not specified
    if args.model_path is None:
        args.model_path = os.path.join(args.model_dir, "best_standard_model.pth")
    
    # Load model and configuration
    model, config = load_model_and_config(args.model_path, args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get class information
    class_names = config["class_names"]
    class_to_idx = config["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print(f"Loaded model from {args.model_path}")
    print(f"Classes: {class_names}")
    
    # Create data loader
    data_loader, dataset = get_data_loader(args.data_dir, config, args.batch_size)
    print(f"Dataset: {len(dataset)} images")
    
    # Evaluate model
    print("Evaluating model...")
    predictions, true_labels = evaluate_model(model, data_loader, device)
    
    # Save confusion matrix
    cm_path = os.path.join(args.model_dir, "confusion_matrix.png")
    confusion_mat = plot_confusion_matrix(true_labels, predictions, class_names, cm_path)
    
    # Calculate and print metrics
    metrics = print_metrics(true_labels, predictions, class_names)
    
    # Save metrics to file
    metrics_path = os.path.join(args.model_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Visualize predictions
    vis_dir = os.path.join(args.model_dir, "prediction_samples")
    print(f"\nGenerating prediction visualizations for {args.num_samples} samples per class...")
    visualize_predictions(model, dataset, class_names, idx_to_class, config, args.num_samples, vis_dir)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
