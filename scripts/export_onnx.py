"""
HaemaVision Model ONNX Export Script

This script exports a trained PyTorch model to ONNX format.
ONNX allows for model deployment across different platforms and frameworks.
"""

import os
import json
import argparse
import torch
import numpy as np
from model import get_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export blood cell classifier to ONNX")
    parser.add_argument("--model_path", type=str, default="../models/best_standard_model.pth",
                        help="Path to trained model weights (.pth file)")
    parser.add_argument("--config_path", type=str, default="../models/model_config.json",
                        help="Path to model configuration file")
    parser.add_argument("--output_path", type=str, default="../models/blood_cell_classifier.onnx",
                        help="Path to save the exported ONNX model")
    parser.add_argument("--dynamic_batch", action="store_true",
                        help="Export model with dynamic batch dimension")
    return parser.parse_args()

def load_model(model_path, config_path):
    """Load the PyTorch model and configuration."""
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
    model.eval()  # Set to evaluation mode
    
    return model, config

def export_to_onnx(model, config, output_path, dynamic_batch=False):
    """Export the PyTorch model to ONNX format."""
    # Create a dummy input tensor for tracing
    img_size = config["img_size"]
    batch_size = 1  # Using batch size 1 for export
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Set dynamic axes if needed
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    # Export the model
    torch.onnx.export(
        model,                       # PyTorch model
        dummy_input,                 # Dummy input
        output_path,                 # Output file path
        export_params=True,          # Store the trained weights inside the model
        opset_version=14,            # ONNX version to export to
        do_constant_folding=True,    # Fold constants for optimization
        input_names=['input'],       # Model input name
        output_names=['output'],     # Model output name
        dynamic_axes=dynamic_axes,   # Dynamic axes if needed
        verbose=False
    )
    
    # Create metadata to store with the model
    metadata = {
        "input_size": img_size,
        "num_classes": config["num_classes"],
        "class_names": config["class_names"],
        "class_to_idx": config["class_to_idx"],
        "normalization": config["normalization"]
    }
    
    # Save metadata alongside ONNX model
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata

def verify_onnx_model(onnx_path):
    """Verify the exported ONNX model."""
    try:
        import onnx
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        # Check that the model is well-formed
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
        return True
    except ImportError:
        print("ONNX package not installed. Install with 'pip install onnx' to verify the model.")
        return False
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")
        return False

def main():
    """Main export function."""
    args = parse_args()
    
    # Load model and configuration
    print(f"Loading model from {args.model_path}")
    model, config = load_model(args.model_path, args.config_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting model to ONNX format: {args.output_path}")
    metadata = export_to_onnx(model, config, args.output_path, args.dynamic_batch)
    
    print(f"Model exported with metadata:")
    print(f"  - Input size: {metadata['input_size']}x{metadata['input_size']}")
    print(f"  - Number of classes: {metadata['num_classes']}")
    print(f"  - Classes: {', '.join(metadata['class_names'])}")
    
    # Verify the exported model
    verify_onnx_model(args.output_path)
    
    print(f"\nONNX model successfully exported to {args.output_path}")
    print(f"Metadata saved to {os.path.splitext(args.output_path)[0] + '_metadata.json'}")
    
    # Provide example usage
    print("\nExample usage in inference code:")
    print("```python")
    print("import onnxruntime as ort")
    print("import numpy as np")
    print("from PIL import Image")
    print("import json")
    print("")
    print("# Load metadata")
    print(f"with open('{os.path.splitext(args.output_path)[0]}_metadata.json', 'r') as f:")
    print("    metadata = json.load(f)")
    print("")
    print("# Initialize ONNX Runtime session")
    print(f"session = ort.InferenceSession('{args.output_path}')")
    print("")
    print("# Preprocess image")
    print("def preprocess_image(image_path, input_size):")
    print("    image = Image.open(image_path).convert('RGB')")
    print("    image = image.resize((input_size, input_size))")
    print("    # Convert to numpy array and normalize")
    print("    img_array = np.array(image).astype(np.float32) / 255.0")
    print("    # Apply normalization")
    print("    mean = np.array(metadata['normalization']['mean']).reshape(1, 1, 3)")
    print("    std = np.array(metadata['normalization']['std']).reshape(1, 1, 3)")
    print("    img_array = (img_array - mean) / std")
    print("    # HWC to CHW format (transpose)")
    print("    img_array = img_array.transpose(2, 0, 1)")
    print("    # Add batch dimension")
    print("    img_array = img_array[np.newaxis, ...]")
    print("    return img_array")
    print("")
    print("# Run inference")
    print("img_tensor = preprocess_image('path/to/image.jpg', metadata['input_size'])")
    print("ort_inputs = {session.get_inputs()[0].name: img_tensor}")
    print("ort_outputs = session.run(None, ort_inputs)")
    print("")
    print("# Process outputs")
    print("scores = ort_outputs[0][0]")
    print("predicted_idx = np.argmax(scores)")
    print("class_names = metadata['class_names']")
    print("print(f'Predicted class: {class_names[predicted_idx]}')")
    print("```")

if __name__ == "__main__":
    main()
