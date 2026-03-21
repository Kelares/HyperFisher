"""
Split-MNIST Task Generator
==========================
5 tasks, each a binary classification of two consecutive digit classes:
  Task 0: digits {0, 1}
  Task 1: digits {2, 3}
  Task 2: digits {4, 5}
  Task 3: digits {6, 7}
  Task 4: digits {8, 9}

Labels are remapped to {0, 1} within each task so the target network
always has a 2-class output head, matching the standard Split-MNIST
benchmark used in EWC, OGD, and FOPNG papers.

Interface is identical to permuted_mnist.TaskGenerator so it can be
swapped in via --task split_mnist with no other changes.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class TaskGenerator:
    config = SimpleNamespace(
        input_dim=784,
        num_classes=2,        # binary per task
        num_tasks=5,
        criterion=nn.CrossEntropyLoss(),
        grads_per_task=80,
        max_directions=400,
    )

    # Same architecture as permuted_mnist but output dim = 2
    target_network = nn.Sequential(
        nn.Linear(784, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),    # binary output per task
    )

    # Which digit pair each task covers
    TASK_CLASSES = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]

    _train_data: datasets.MNIST | None = None
    _test_data:  datasets.MNIST | None = None

    @classmethod
    def _load(cls) -> None:
        """Download MNIST once and cache it."""
        if cls._train_data is not None:
            return
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),   # flatten to [784]
        ])
        cls._train_data = datasets.MNIST(
            root="./data", train=True,  download=True, transform=tf
        )
        cls._test_data = datasets.MNIST(
            root="./data", train=False, download=True, transform=tf
        )

    @classmethod
    def _make_split(
        cls,
        dataset: datasets.MNIST,
        class_a: int,
        class_b: int,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Filter dataset to only class_a / class_b samples and remap
        labels: class_a → 0, class_b → 1.
        """
        targets = dataset.targets
        mask    = (targets == class_a) | (targets == class_b)
        indices = mask.nonzero(as_tuple=True)[0].tolist()

        # Build a remapped dataset wrapper
        subset = Subset(dataset, indices)

        # Wrap in a collate function that remaps labels
        def collate(batch):
            xs, ys = zip(*batch)
            xs = torch.stack(xs)
            ys = torch.tensor(
                [0 if int(y) == class_a else 1 for y in ys],
                dtype=torch.long,
            )
            return xs, ys

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    @classmethod
    def generate(
        cls,
        task_id: int,
        batch_size: int = 256,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Returns (train_loader, test_loader) for the given task_id (0-indexed).
        """
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