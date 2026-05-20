from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
# REMOVED: from utils import RemappedSubset
from models.mlp import MLP # Ensure your MLP has 10 output units

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
        task_classes=TASK_CLASSES,
    )

    # Use a standard MLP for a single-headed setup
    target_network = MLP
    solo_target = MLP 

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
        
        # Native Subset preserves the original labels (0-9)
        subset = Subset(dataset, indices)
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