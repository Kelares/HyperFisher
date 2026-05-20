import torch
import torch.nn as nn
from torch.func import functional_call
from math import ceil
from typing import List
import torch.nn.functional as F
 
class MultiHeadMLP(nn.Module):
    """
    Multi-head version of TargetCNN that stores task state internally.
    Backbone (Conv1 -> FC1) is shared; FC2 is task-specific.
    """
    def __init__(self, num_tasks: int, device):
        super().__init__()
        # 1. Shared Backbone (Protected by FOPNG)
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        ).to(device)

        # 2. Task-specific Heads (Individual binary classifiers)
        self.heads = nn.ModuleList([
            nn.Linear(100, 2) for _ in range(num_tasks)
        ]).to(device)
        
        # Internal state to track the active task
        self.register_buffer("_active_task_id", torch.tensor(0, dtype=torch.long))

    @property
    def _shared_params(self) -> List[nn.Parameter]:
        """Returns the list of backbone parameters for the FOPNG projection."""
        return list(self.layers.parameters())

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the internally saved task_id.
        Matches the standard nn.Module interface.
        """
        # Feature Extraction (Backbone)
        x = self.layers(x)
        
        # Classification (Active Task Head)
        t_id = self._active_task_id.item()
        return self.heads[t_id](x)


class MLP(nn.Module):
    """
    3-layer MLP with:
    - Input: 784 (28x28)
    - Hidden: 100, 100
    - Output: num_classes (default 10)
    """
    
    def __init__(self, num_tasks, device, input_dim: int = 784, hidden_dim: int = 100, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc_out = nn.Linear(hidden_dim, num_classes).to(device)

    @property
    def _shared_params(self) -> List[nn.Parameter]:
        """Returns the list of backbone parameters for the FOPNG projection."""
        return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc_out.parameters())

    @property
    def num_shared_params(self) -> int:
        """Required for Fisher information estimation loops."""
        return sum(p.numel() for p in self._shared_params)

    def spawn(self, task_id: torch.Tensor | int):
        pass

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    