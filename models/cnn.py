
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _cifar_feature_extractor():
    """
    Simplified CNN backbone: 4 convolutional layers.
    Architecture: Conv -> ReLU -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
    """
    return nn.Sequential(
        # Conv block 1: 3 -> 32
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        # Conv block 2: 32 -> 64
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


# After 2 maxpools on 32x32: 32 -> 16 -> 8, with 64 channels
CIFAR_FEATURE_DIM = 64 * 8 * 8
class SimpleCIFARCNN(nn.Module):
    """
    Simplified CNN for CIFAR experiments.
    
    Architecture:
        4 conv layers (2 blocks with maxpool)
        2 dense layers with dropout
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = _cifar_feature_extractor()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CIFAR_FEATURE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MultiHeadCIFARCNN(nn.Module):
    """
    Simplified CNN trunk shared across tasks with individual heads.
    4 conv layers + 2 dense layers with dropout.
    """
    
    def __init__(
        self,
        num_heads: int,
        device,
        head_output_sizes: Optional[List[int]] = None,
        dropout: float = 0.5
    ):
        super().__init__()
        self.features = _cifar_feature_extractor().to(device)
        self.shared_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CIFAR_FEATURE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ).to(device)
        
        if head_output_sizes is None:
            head_output_sizes = [2] * num_heads
        
        self.heads = nn.ModuleList([
            nn.Linear(256, out_dim) for out_dim in head_output_sizes
        ]).to(device)
        self.task_id = 0
        
        # Internal state to track the active task
        self.register_buffer("_active_task_id", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x = self.features(x)
        x = self.shared_classifier(x)
        # Classification (Active Task Head)
        t_id = self._active_task_id.item()
        return self.heads[t_id](x)

    @property
    def _shared_params(self) -> List[nn.Parameter]:
        """Returns the list of backbone parameters for the FOPNG projection."""
        return list(self.features.parameters()) + list(self.shared_classifier.parameters()) + list(self.heads.parameters())

    @property
    def num_shared_params(self) -> int:
        """Required for Fisher information estimation loops."""
        return sum(p.numel() for p in self._shared_params)

    def spawn(self, task_id: torch.Tensor | int):
        """
        Saves the task_id internally. 
        Gradients now naturally flow to the correct head during the next forward pass.
        """
        if torch.is_tensor(task_id):
            self._active_task_id.fill_(task_id.item())
        else:
            self._active_task_id.fill_(task_id)
