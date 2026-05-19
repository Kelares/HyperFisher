"""
Split-CIFAR10 Task Generator
=============================
5 tasks, each a binary classification of two consecutive classes:
  Task 0: airplane (0) vs automobile (1)
  Task 1: bird (2)     vs cat (3)
  Task 2: deer (4)     vs dog (5)
  Task 3: frog (6)     vs horse (7)
  Task 4: ship (8)     vs truck (9)

Labels are kept as real class indices (0-9) so CrossEntropyLoss and
evaluate_accuracy work without modification, same as split_mnist.

Interface is identical to permuted_mnist.TaskGenerator.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from types import SimpleNamespace
from typing import List, Tuple, Optional
from tqdm import tqdm
from utils import RemappedSubset

TASK_CLASSES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


class DummyCIFARTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(1, 32), # BN before ReLU
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(1, 64), # BN before ReLU
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(1, 128), # BN before ReLU
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # New: Global Average Pooling to reduce 8192 features to 128
        self.gap = nn.AdaptiveAvgPool2d(1) 
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64), # Updated input dim
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.gap(x) # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1) # [batch, 128]
        return self.fc(x)
        
class TargetCNN(nn.Module):
    """
    Tiny CNN used as the *template* for the HyperNetwork.
    All parameters are overwritten at inference time via functional_call.

    Input : 3 × 32 × 32
    After conv1 + pool  →  16 × 16 × 16
    After conv2 + pool  →  32 ×  8 ×  8
    After conv3 + pool  →  32 ×  4 ×  4   (512 features)
    FC1  →  64
    FC2  →  num_classes (2 per task)

    Total parameters: ~47 300
    With chunk_size=256 → 185 chunks
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1   = nn.Linear(32 * 4 * 4, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # 16 × 16
        x = F.relu(F.max_pool2d(self.conv2(x), 2))   #  8 ×  8
        x = F.relu(F.max_pool2d(self.conv3(x), 2))   #  4 ×  4
        x = x.view(x.size(0), -1)                    # 512
        x = F.relu(self.fc1(x))                       #  64
        return self.fc2(x)                            #   2


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MultiHeadTargetCNN(nn.Module):
    """
    Multi-head version of TargetCNN that stores task state internally.
    Backbone (Conv1 -> FC1) is shared; FC2 is task-specific.
    """
    def __init__(self, num_tasks: int, device, num_classes_per_task: int = 2):
        super().__init__()
        # 1. Shared Backbone (Protected by FOPNG)
        self.conv1 = nn.Conv2d(3,  16, 3, padding=1).to(device)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1).to(device)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1).to(device)
        self.fc1   = nn.Linear(32 * 4 * 4, 64).to(device)

        # 2. Task-specific Heads (Individual binary classifiers)
        self.heads = nn.ModuleList([
            nn.Linear(64, num_classes_per_task) for _ in range(num_tasks)
        ]).to(device)
        
        # Internal state to track the active task
        self.register_buffer("_active_task_id", torch.tensor(0, dtype=torch.long))

    @property
    def _shared_params(self) -> List[nn.Parameter]:
        """Returns the list of backbone parameters for the FOPNG projection."""
        return list(self.conv1.parameters()) + \
               list(self.conv2.parameters()) + \
               list(self.conv3.parameters()) + \
               list(self.fc1.parameters())

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

    def update_task_specific_params(self, lr: float):
        """Manually updates the active head using standard SGD/Adam logic."""
        with torch.no_grad():
            active_head = self.heads[self._active_task_id.item()]
            for p in active_head.parameters():
                if p.grad is not None:
                    p.data.add_(-lr * p.grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the internally saved task_id.
        Matches the standard nn.Module interface.
        """
        # Feature Extraction (Backbone)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # Output: 16x16
        x = F.relu(F.max_pool2d(self.conv2(x), 2))   # Output: 8x8
        x = F.relu(F.max_pool2d(self.conv3(x), 2))   # Output: 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        # Classification (Active Task Head)
        t_id = self._active_task_id.item()
        return self.heads[t_id](x)

        
# ─────────────────────────────────────────────────────────────────────────────
# Task Generator
# ─────────────────────────────────────────────────────────────────────────────
class TaskGenerator:
    TASK_CLASSES = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    config = SimpleNamespace(
        input_dim=3072,
        num_classes=2,
        num_tasks=5,
        criterion=nn.CrossEntropyLoss(),
        task_classes=TASK_CLASSES,
        grads_per_task=150,
        max_directions=750,
    )
    # Continual Learning Task Configurations

    target_network = TargetCNN(num_classes=config.num_classes)

    # B. Multi-Head CIFAR CNN (Isolated visual classification pathways)
    multihead = MultiHeadTargetCNN

    _train_data: Optional[datasets.CIFAR10] = None
    _test_data:  Optional[datasets.CIFAR10] = None

    @classmethod
    def _load(cls) -> None:
        """Loads and normalizes the base CIFAR-10 datasets once."""
        if cls._train_data is not None:
            return
            
        print(" [TaskGenerator] Initializing base CIFAR-10 datasets...")
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
        
        # The download process itself has a built-in tqdm bar from torchvision
        cls._train_data = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=tf
        )
        cls._test_data = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=tf
        )

    @classmethod
    def _make_split(
        cls,
        dataset: datasets.CIFAR10,
        allowed_classes: List[int],
        batch_size: int = 64,
        shuffle: bool = True,
        desc: str = "Processing Task"
    ) -> DataLoader:
        """Creates a DataLoader for specific classes with remapped labels."""
        
        # 1. Fast indexing using .targets and tqdm
        # This avoids loading images during the search
        indices = []
        for i, label in enumerate(tqdm(dataset.targets, desc=f"  {desc}", leave=False)):
            if label in allowed_classes:
                indices.append(i)

        # 2. Pass those indices to the updated RemappedSubset
        subset = RemappedSubset(dataset, allowed_classes, indices=indices)

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )

    @classmethod
    def generate(
        cls,
        task_id: int,
        batch_size: int = 64,
    ) -> Tuple[DataLoader, DataLoader]:
        """Returns (train_loader, test_loader) for a specific task."""
        assert 0 <= task_id < cls.config.num_tasks, \
            f"task_id must be in [0, {cls.config.num_tasks}), got {task_id}"

        cls._load()
        allowed_classes = cls.TASK_CLASSES[task_id]

        # Use a descriptive prefix for the progress bar
        prefix = f"Task {task_id+1} ({allowed_classes})"
        
        train_loader = cls._make_split(
            cls._train_data, allowed_classes,
            batch_size=batch_size, shuffle=True,
            desc=f"{prefix} Train"
        )
        test_loader = cls._make_split(
            cls._test_data, allowed_classes,
            batch_size=batch_size, shuffle=False,
            desc=f"{prefix} Test"
        )
        
        return train_loader, test_loader