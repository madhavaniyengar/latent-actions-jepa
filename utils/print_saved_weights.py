import torch
import os
import argparse
from collections import OrderedDict
from typing import Union, Dict

def flatten_state_dict(state_dict: Union[Dict, OrderedDict], parent_key: str = '') -> Dict[str, torch.Tensor]:
    """
    Flatten a nested state dictionary into a single-level dictionary.
    
    Args:
        state_dict (Union[Dict, OrderedDict]): Potentially nested state dictionary
        parent_key (str): Key of parent dictionary (used for recursion)
        
    Returns:
        Dict[str, torch.Tensor]: Flattened dictionary containing only tensor values
    """
    items = []
    for k, v in state_dict.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, (dict, OrderedDict)):
            items.extend(flatten_state_dict(v, new_key).items())
        elif isinstance(v, torch.Tensor):
            items.append((new_key, v))
        else:
            print(f"Warning: Skipping non-tensor value at key '{new_key}' of type {type(v)}")
            
    return dict(items)

def load_and_print_weights(weights_path: str, detailed: bool = False) -> Dict[str, torch.Tensor]:
    """
    Load and print weights from a .pth file
    
    Args:
        weights_path (str): Path to the .pth weights file
        detailed (bool): If True, prints additional statistics for each tensor
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the loaded weights
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        
    # Load the weights
    try:
        state_dict = torch.load(weights_path, map_location='cpu')
    except Exception as e:
        raise Exception(f"Error loading weights: {str(e)}")
    
    # Handle different state dict formats
    if isinstance(state_dict, OrderedDict):
        nested_weights = state_dict
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        nested_weights = state_dict['state_dict']
    else:
        nested_weights = state_dict
        
    # Flatten the nested structure
    weights = flatten_state_dict(nested_weights)
        
    # Print summary
    print(f"\nLoaded weights from: {weights_path}")
    print(f"Number of parameter tensors: {len(weights)}\n")
    
    total_params = 0
    for name, tensor in weights.items():
        num_params = torch.numel(tensor)
        total_params += num_params
        
        print(f"Layer: {name}")
        print(f"Shape: {list(tensor.shape)}")
        print(f"Parameters: {num_params:,}")
        
        if detailed:
            print(f"Data type: {tensor.dtype}")
            print(f"Mean: {tensor.mean().item():.6f}")
            print(f"Std: {tensor.std().item():.6f}")
            print(f"Min: {tensor.min().item():.6f}")
            print(f"Max: {tensor.max().item():.6f}")
        print()
        
    print(f"Total parameters: {total_params:,}")
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and inspect PyTorch weights')
    parser.add_argument('weights_path', type=str, help='Path to the .pth weights file')
    parser.add_argument('--detailed', action='store_true', help='Print detailed tensor statistics')
    args = parser.parse_args()
    
    weights = load_and_print_weights(args.weights_path, args.detailed)