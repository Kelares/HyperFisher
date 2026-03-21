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
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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
        nn.Linear(784, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 10),
    )

    _train_data: datasets.MNIST | None = None
    _test_data:  datasets.MNIST | None = None

    @classmethod
    def _load(cls) -> None:
        if cls._train_data is not None:
            return
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
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
        targets = dataset.targets
        mask    = (targets == class_a) | (targets == class_b)
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        subset  = Subset(dataset, indices)

        def collate(batch):
            xs, ys = zip(*batch)
            xs = torch.stack(xs)
            ys = torch.tensor([int(y) for y in ys], dtype=torch.long)
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