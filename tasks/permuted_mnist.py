"""
Permuted-MNIST Task Generator
===============================
Matches Garg et al. (2026) Table 1 setup exactly:
  - 5 tasks (configurable via NUM_TASKS)
  - MLP 784 → 100 → 100 → 10  (single shared output head, all tasks)
  - No MNIST normalisation — just transforms.ToTensor()
  - Per-task deterministic permutation seeded by task_id for reproducibility
  - Single-head: all tasks share the same 10 output neurons

For multi-head comparison experiments (user's own), MultiHeadTarget provides
separate 10-class heads per task with a shared backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from types import SimpleNamespace
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Permutation wrapper  (never modifies the base dataset in-place)
# ─────────────────────────────────────────────────────────────────────────────
class PermutedDataset(Dataset):
    """Wraps an MNIST dataset and applies a fixed pixel permutation per task."""

    def __init__(self, base_dataset, perm: Optional[torch.Tensor] = None):
        self.base = base_dataset
        self.perm = perm  # None → task 0, original MNIST

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]   # img: [784] after Lambda flatten
        if self.perm is not None:
            img = img[self.perm]
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# Canonical target network  (FOPNG paper: MLP 784→100→100→10, single head)
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """
    3-layer MLP matching Garg et al. (2026) and Farajtabar et al. (2020).
    All parameters are protected by the projection optimizer.
    Single shared output head — all permutation tasks use the same 10 neurons.
    """

    def __init__(self, num_tasks: int, device,
                 input_dim: int = 784, hidden_dim: int = 100,
                 num_classes: int = 10):
        super().__init__()
        self.fc1    = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2    = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc_out = nn.Linear(hidden_dim, num_classes).to(device)

    @property
    def _shared_params(self) -> List[nn.Parameter]:
        return (list(self.fc1.parameters()) +
                list(self.fc2.parameters()) +
                list(self.fc_out.parameters()))

    @property
    def num_shared_params(self) -> int:
        return sum(p.numel() for p in self._shared_params)

    def spawn(self, task_id) -> None:
        pass   # single head — no routing needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head target  (for user's own multi-head comparison experiments)
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadTarget(nn.Module):
    """
    Shared MLP backbone (784→100→100) with one 10-class head per task.
    Used for multi-head comparison experiments only — not FOPNG replication.
    Backbone is protected by the projection optimizer; heads are task-specific.
    """

    def __init__(self, num_tasks: int, device):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        ).to(device)

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

    def spawn(self, task_id) -> None:
        if torch.is_tensor(task_id):
            self._active_task_id.fill_(task_id.item())
        else:
            self._active_task_id.fill_(task_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return self.heads[self._active_task_id.item()](x)


# ─────────────────────────────────────────────────────────────────────────────
# Task Generator
# ─────────────────────────────────────────────────────────────────────────────
class TaskGenerator:
    NUM_TASKS = 5   # FOPNG paper uses 5; increase for longer-horizon experiments

    config = SimpleNamespace(
        input_dim    = 784,
        num_classes  = 10,
        num_tasks    = NUM_TASKS,
        criterion    = nn.CrossEntropyLoss(),
        task_classes = None,   # no class filtering — all 10 digits every task
        # Canonical hyperparameters from Table 1 (Permuted-MNIST):
        #   lr=1e-4 (Adam), 5e-3 (SGD), 1e-4 (FOPNG/eFOPNG)
        #   batch_size=10, epochs=5, grads_per_task=80, max_dirs=400
        grads_per_task  = 80,
        max_directions  = 400,
    )

    # For Hypernetwork
    target_network = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

    # For runs without hypernetwork
    solo_target = MLP

    _train_data = None
    _test_data  = None

    @classmethod
    def _load(cls) -> None:
        if cls._train_data is not None:
            return
        # No normalisation — matches FOPNG paper (transforms.ToTensor() only)
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten to [784]
        ])
        cls._train_data = datasets.MNIST(
            root="./data", train=True,  download=True, transform=tf)
        cls._test_data  = datasets.MNIST(
            root="./data", train=False, download=True, transform=tf)

    @classmethod
    def generate(cls, task_id: int, batch_size: int = 10):
        cls._load()

        # Task 0 → original pixels; task k > 0 → seeded fixed permutation
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