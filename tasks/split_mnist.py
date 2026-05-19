"""
Split-MNIST Task Generator — Multi-head
========================================
5 tasks, each a binary classification of two consecutive digit classes:
  Task 0: digits {0, 1}
  Task 1: digits {2, 3}
  Task 2: digits {4, 5}
  Task 3: digits {6, 7}
  Task 4: digits {8, 9}

Labels are kept as real digit indices (0-9), not remapped to {0, 1}.
This means standard CrossEntropyLoss and argmax over all 10 logits work
correctly with no changes needed in training loops or evaluate_accuracy.

The network has 10 output neurons. Each task trains only 2 of them
(the ones corresponding to its digit pair). The other 8 receive no
gradient signal for that task, so they are naturally preserved.

Interface is identical to permuted_mnist.TaskGenerator.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from utils import RemappedSubset
import torch.nn.functional as F


class MultiHeadTarget(nn.Module):
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
    
class TaskGenerator:

    TASK_CLASSES = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]

    config = SimpleNamespace(
        input_dim=784,
        num_classes=10,
        num_tasks=5,
        criterion=nn.CrossEntropyLoss(),
        grads_per_task=80,
        max_directions=400,
        task_classes=TASK_CLASSES,   # add this line
    )

    target_network = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    
    multihead = MLP

    _train_data: datasets.MNIST | None = None
    _test_data:  datasets.MNIST | None = None

    @classmethod
    def _load(cls) -> None:
        if cls._train_data is not None:
            return
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        cls._train_data = datasets.MNIST(
            root="./data", train=True,  download=True, transform=tf
        )
        cls._test_data = datasets.MNIST(
            root="./data", train=False, download=True, transform=tf
        )

    @classmethod
    def _make_split(cls, dataset, class_a, class_b, batch_size=256, shuffle=True):
        targets = dataset.targets
        mask = (targets == class_a) | (targets == class_b)
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        subset = RemappedSubset(dataset, [class_a, class_b], indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

    @classmethod
    def generate(
        cls,
        task_id: int,
        batch_size: int = 256,
    ) -> Tuple[DataLoader, DataLoader]:
        assert 0 <= task_id < cls.config.num_tasks, \
            f"task_id must be in [0, {cls.config.num_tasks}), got {task_id}"

        cls._load()
        class_a, class_b = cls.TASK_CLASSES[task_id]

        train_loader = cls._make_split(
            cls._train_data, class_a, class_b,
            batch_size=batch_size, shuffle=True,
        )
        test_loader = cls._make_split(
            cls._test_data, class_a, class_b,
            batch_size=batch_size, shuffle=False,
        )
        return train_loader, test_loader