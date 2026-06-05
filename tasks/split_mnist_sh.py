# split_mnist_sh.py — Scenario 1 SH (task-IL, shared 2-neuron head)

from __future__ import annotations
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import RemappedSubset          # back to RemappedSubset
from models.mlp import MLP

class TaskGenerator:

    TASK_CLASSES = [(0,1), (2,3), (4,5), (6,7), (8,9)]

    config = SimpleNamespace(
        input_dim   = 784,
        num_classes = 2,       # ← 2 not 10: one shared binary head
        num_tasks   = 5,
        criterion   = nn.CrossEntropyLoss(),
        grads_per_task  = 80,
        max_directions  = 400,
        task_classes    = TASK_CLASSES,
    )

    target_network = MLP    # MLP instantiated with num_classes=2 → fc_out: Linear(100, 2)
    solo_target    = MLP

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
        cls._train_data = datasets.MNIST("./data", train=True,  download=True, transform=tf)
        cls._test_data  = datasets.MNIST("./data", train=False, download=True, transform=tf)

    @classmethod
    def _make_split(cls, dataset, class_a, class_b, batch_size=256, shuffle=True):
        targets = dataset.targets
        mask    = (targets == class_a) | (targets == class_b)
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        # RemappedSubset: {class_a → 0, class_b → 1} for every task
        # Task 1: {0→0, 1→1}  Task 2: {2→0, 3→1}  etc.
        subset = RemappedSubset(dataset, [class_a, class_b], indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

    @classmethod
    def generate(cls, task_id: int, batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
        assert 0 <= task_id < cls.config.num_tasks
        cls._load()
        class_a, class_b = cls.TASK_CLASSES[task_id]
        return (
            cls._make_split(cls._train_data, class_a, class_b, batch_size, shuffle=True),
            cls._make_split(cls._test_data,  class_a, class_b, batch_size, shuffle=False),
        )