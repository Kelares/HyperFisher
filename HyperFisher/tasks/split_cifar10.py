"""
FOPNG: Fisher-Orthogonal Projected Natural Gradient Descent
============================================================
Garg, Kolhe, Peng, Gopalam — UC Berkeley (ICML 2026)
"""
from __future__ import annotations

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from types import SimpleNamespace
from utils import SubsetByClass

class TaskGenerator:
    config = {
        "input_dim": 3072,
        "num_classes": 10,
        "grads_per_task": 80,
        "max_directions": 400,
        "criterion": nn.CrossEntropyLoss(),
        "num_tasks": 10
    }
    config = SimpleNamespace(**config)

    # target_network = nn.Sequential(
    #     nn.Linear(config.input_dim, 100), 
    #     nn.ReLU(),
    #     nn.Linear(100, config.num_classes),
    # )
    
    target_network = nn.Sequential(
        nn.Linear(3072, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )
    tasks = {}

    @classmethod
    def create_classes(cls, task_id, batch_size=64):
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), transforms.Lambda(lambda x: x.view(-1)) )])
        # Load Train and Test splits separately
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        default_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        for classes in default_groups:
            train_subset = SubsetByClass(train_dataset, classes)
            test_subset = SubsetByClass(test_dataset, classes)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
            cls.tasks[task_id] = (train_loader, test_loader)
            
        return train_loader, test_loader



    @classmethod
    def generate(cls, task_id, batch_size=64):
        if task_id not in cls.tasks:
            cls.create_classes(task_id, batch_size)
        return cls.tasks[task_id]

