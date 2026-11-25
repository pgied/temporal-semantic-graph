
from omegaconf import DictConfig
import torch


def get_device():

    print(f'is cuda available: {torch.cuda.is_available()}')  # Should return True if CUDA is available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda' 
    elif torch.backends.mps.is_available():
        device = "mps"

    return device