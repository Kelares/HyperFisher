from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from types import SimpleNamespace
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Permutation wrapper — applied per-sample, never modifies base dataset
# ─────────────────────────────────────────────────────────────────────────────
class PermutedDataset(Dataset):
    def __init__(self, base_dataset, perm: Optional[torch.Tensor] = None):
        self.base = base_dataset
        self.perm = perm  # None = task 0, original MNIST

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]   # img is [784] after the Lambda transform
        if self.perm is not None:
            img = img[self.perm]
        return img, label
class BottleneckTargetMLP(nn.Module):
    """
    An ultra-low capacity MLP designed to force a single-task capacity ceiling.
    Total parameter footprint is reduced to 12,730 parameters to isolate 
    the optimization behavior of hard projections vs soft regularizers.
    """
    def __init__(self, device, num_classes_per_task: int = 10):
        super().__init__()
        # Severely restricted hidden layer bottleneck
        self.fc1 = nn.Linear(784, 16).to(device)
        self.fc2 = nn.Linear(16, num_classes_per_task).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image from (B, 1, 28, 28) to (B, 784)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    @property
    def _shared_params(self) -> List[nn.Parameter]:
        """Returns the list of backbone parameters for the FOPNG projection."""
        return self.parameters()

    @property
    def num_shared_params(self) -> int:
        """Required for Fisher information estimation loops."""
        return sum(p.numel() for p in self._shared_params)

    def spawn(self, task_id: torch.Tensor | int):
        pass
    
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
# Multi-head target network
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadTarget(nn.Module):
    def __init__(self, num_tasks: int, device):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        ).to(device)

        # 10 outputs per head — Permuted-MNIST is always 10-class
        self.heads = nn.ModuleList([
            nn.Linear(100, 10) for _ in range(num_tasks)
        ]).to(device)

        self.register_buffer("_active_task_id", torch.tensor(0, dtype=torch.long))

    @property
    def _shared_params(self) -> List[nn.Parameter]:
        return list(self.layers.parameters())

    @property
    def num_shared_params(self) -> int:
        return sum(p.numel() for p in self._shared_params)

    def spawn(self, task_id: torch.Tensor | int):
        if torch.is_tensor(task_id):
            self._active_task_id.fill_(task_id.item())
        else:
            self._active_task_id.fill_(task_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # handle any stray spatial dims
        x = self.layers(x)
        return self.heads[self._active_task_id.item()](x)


# ─────────────────────────────────────────────────────────────────────────────
# Task generator
# ─────────────────────────────────────────────────────────────────────────────
class TaskGenerator:
    NUM_TASKS = 10   # raise to 15 if you want a longer sequence

    config = SimpleNamespace(
        input_dim=784,
        num_classes=10,
        num_tasks=NUM_TASKS,
        criterion=nn.CrossEntropyLoss(),
        grads_per_task=80,
        max_directions=400,
        task_classes=None,   # no class filtering for permuted
    )

    target_network = BottleneckTargetMLP
    multihead = MultiHeadBottleneckTargetMLP

    _train_data = None
    _test_data  = None

    @classmethod
    def _load(cls) -> None:
        if cls._train_data is not None:
            return
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten to [784]
        ])
        cls._train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=tf)
        cls._test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=tf)

    @classmethod
    def generate(cls, task_id: int, batch_size: int = 256):
        cls._load()

        # Task 0 = original pixels, task k>0 = fixed permutation seeded by k
        perm = None
        if task_id > 0:
            rng  = torch.Generator().manual_seed(task_id)
            perm = torch.randperm(784, generator=rng)

        train_loader = DataLoader(
            PermutedDataset(cls._train_data, perm),
            batch_size=batch_size, shuffle=True,
        )
        test_loader = DataLoader(
            PermutedDataset(cls._test_data, perm),
            batch_size=batch_size, shuffle=False,
        )
        return train_loader, test_loader