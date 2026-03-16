"""
FOPNG: Fisher-Orthogonal Projected Natural Gradient Descent
============================================================
Garg, Kolhe, Peng, Gopalam — UC Berkeley (ICML 2026)
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.func import functional_call
import torchvision
from torchvision import transforms
from types import SimpleNamespace


class TaskGenerator:
    config = {
        "input_dim": 784,
        "num_classes": 10,
        "grads_per_task": 80,
        "max_directions": 400,
        "criterion": nn.CrossEntropyLoss()
    }
    config = SimpleNamespace(**config)

    target_network = nn.Sequential(
        nn.Linear(config.input_dim, 100), 
        nn.ReLU(),
        nn.Linear(100, config.num_classes),
    )
    
    def generate(task_id, batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        # Load Train and Test splits separately
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if task_id > 0:
            rng = torch.Generator().manual_seed(task_id)
            perm = torch.randperm(784, generator=rng)
            
            # Apply permutation to both splits
            train_dataset.data = train_dataset.data.view(-1, 784)[:, perm].view(-1, 28, 28)
            test_dataset.data = test_dataset.data.view(-1, 784)[:, perm].view(-1, 28, 28)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


