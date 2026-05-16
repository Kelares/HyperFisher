"""
FOPNG: Fisher-Orthogonal Projected Natural Gradient Descent
============================================================
Garg, Kolhe, Peng, Gopalam — UC Berkeley (ICML 2026)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from types import SimpleNamespace

class BottleneckTargetMLP(nn.Module):
    """
    An ultra-low capacity MLP designed to force a single-task capacity ceiling.
    Total parameter footprint is reduced to 12,730 parameters to isolate 
    the optimization behavior of hard projections vs soft regularizers.
    """
    def __init__(self):
        super().__init__()
        # Severely restricted hidden layer bottleneck
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image from (B, 1, 28, 28) to (B, 784)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    

class MultiHeadBottleneckTargetMLP(nn.Module):
    """
    Multi-head version of BottleneckTargetMLP that stores task state internally.
    Backbone (fc1) is shared and protected by FOPNG/eFOPNG; heads are task-specific.
    """
    def __init__(self, num_tasks: int, device, num_classes_per_task: int = 10):
        super().__init__()
        # 1. Shared Backbone Bottleneck (Protected via information geometry constraints)
        self.fc1 = nn.Linear(784, 16).to(device)

        # 2. Task-specific Heads (Individual 10-way classification suites per permutation)
        self.heads = nn.ModuleList([
            nn.Linear(16, num_classes_per_task) for _ in range(num_tasks)
        ]).to(device)
        
        # Internal buffer state to track the active task context
        self.register_buffer("_active_task_id", torch.tensor(0, dtype=torch.long))

    @property
    def _shared_params(self) -> List[nn.Parameter]:
        """Returns the list of backbone parameters for the FOPNG projection."""
        return list(self.fc1.parameters())

    @property
    def num_shared_params(self) -> int:
        """Required for Fisher information estimation loops."""
        return sum(p.numel() for p in self._shared_params)

    def spawn(self, task_id: torch.Tensor | int):
        """
        Saves the active task_id internally. 
        Gradients now naturally flow to the correct head during the next forward pass.
        """
        if torch.is_tensor(task_id):
            self._active_task_id.fill_(task_id.item())
        else:
            self._active_task_id.fill_(task_id)

    def update_task_specific_params(self, lr: float):
        """Manually updates the active head parameters using standard gradient descent steps."""
        with torch.no_grad():
            active_head = self.heads[self._active_task_id.item()]
            for p in active_head.parameters():
                if p.grad is not None:
                    p.data.add_(-lr * p.grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the internally saved task_id.
        Flattens the data and processes it through the shared bottleneck space.
        """
        # Flatten image from (B, 1, 28, 28) or flat layouts directly to (B, 784)
        x = x.view(x.size(0), -1)
        
        # Share feature mapping space
        x = torch.relu(self.fc1(x))
        
        # Route to the targeted task head realization
        t_id = self._active_task_id.item()
        return self.heads[t_id](x)

# ─────────────────────────────────────────────────────────────────────────────
# Updated Task Generator Interface
# ─────────────────────────────────────────────────────────────────────────────
class TaskGenerator:
    config = {
        "input_dim": 784,
        "num_classes": 10,
        "grads_per_task": 80,
        "max_directions": 400,
        "criterion": nn.CrossEntropyLoss(),
        "num_tasks": 10
    }
    config = SimpleNamespace(**config)

    # Exposed model classes to maintain structural pipeline consistency
    target_network = BottleneckTargetMLP()
    multihead = MultiHeadBottleneckTargetMLP

    @staticmethod
    def generate(task_id: int, batch_size: int = 64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        # Load Train and Test splits separately
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if task_id > 0:
            rng = torch.Generator().manual_seed(task_id)
            perm = torch.randperm(784, generator=rng)
            
            # Apply fixed random permutation to both splits based on unique task seeding
            train_dataset.data = train_dataset.data.view(-1, 784)[:, perm].view(-1, 28, 28)
            test_dataset.data = test_dataset.data.view(-1, 784)[:, perm].view(-1, 28, 28)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

