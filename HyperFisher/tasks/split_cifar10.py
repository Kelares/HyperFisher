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

from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TASK_CLASSES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


class TaskGenerator:

    TASK_CLASSES = TASK_CLASSES

    config = SimpleNamespace(
        input_dim=3072,
        num_classes=10,
        num_tasks=5,
        criterion=nn.CrossEntropyLoss(),
        task_classes=TASK_CLASSES,
        grads_per_task=80,
        max_directions=400,
    )

    target_network = nn.Sequential(
        nn.Linear(3072, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )

    _train_data: datasets.CIFAR10 | None = None
    _test_data:  datasets.CIFAR10 | None = None

    @classmethod
    def _load(cls) -> None:
        if cls._train_data is not None:
            return
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten to [3072]
        ])
        cls._train_data = datasets.CIFAR10(
            root="./data", train=True,  download=True, transform=tf
        )
        cls._test_data = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=tf
        )

    @classmethod
    def _make_split(
        cls,
        dataset: datasets.CIFAR10,
        class_a: int,
        class_b: int,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> DataLoader:
        targets = torch.tensor(dataset.targets)
        mask    = (targets == class_a) | (targets == class_b)
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        subset  = Subset(dataset, indices)

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

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