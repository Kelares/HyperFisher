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
from models.cnn import SimpleCIFARCNN, MultiHeadCIFARCNN
TASK_CLASSES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


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

    target_network = SimpleCIFARCNN(num_classes=config.num_classes)

    # B. Multi-Head CIFAR CNN (Isolated visual classification pathways)
    solo_target = MultiHeadCIFARCNN

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