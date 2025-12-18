"""Device utility for Apple Silicon M2 MPS support"""
import torch


def get_device(use_mps=True, fallback_to_cpu=True):
    """
    Get the best available device for PyTorch operations.
    
    Args:
        use_mps: Whether to use Apple MPS if available
        fallback_to_cpu: Whether to fallback to CPU if MPS is not available
        
    Returns:
        torch.device: The device to use for computations
    """
    if use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders) acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU acceleration")
        return torch.device("cuda")
    elif fallback_to_cpu:
        print("Using CPU")
        return torch.device("cpu")
    else:
        raise RuntimeError("No suitable device found")


def move_to_device(data, device):
    """
    Move data to specified device.
    
    Args:
        data: Data to move (tensor, list of tensors, or dict)
        device: Target device
        
    Returns:
        Data moved to the device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data
