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


TASK_CLASSES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_groups=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        # Changed BatchNorm to GroupNorm
        self.gn1 = nn.GroupNorm(num_groups, planes)
        
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_groups, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) # Changed here too
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_groups=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 64) # Changed BatchNorm to GroupNorm
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_groups=self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, num_groups=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_groups=num_groups)
    
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
# Helper Dataset for Multi-Head Remapping
# ─────────────────────────────────────────────────────────────────────────────
class RemappedSubset(Dataset):
    """Filters a dataset by classes and remaps them to 0...k-1."""
    def __init__(self, base_dataset, allowed_classes: List[int], indices: Optional[List[int]] = None):
        self.base = base_dataset
        # Mapping: e.g., {2: 0, 3: 1}
        self.class_to_new = {c: i for i, c in enumerate(sorted(allowed_classes))}
        
        if indices is not None:
            # Use the fast indices passed from the progress bar
            self.indices = indices
        else:
            # Fallback logic: Use .targets for speed instead of iterating the whole Dataset
            # iterating base_dataset (the images) is what causes the 'second of lag'
            targets = getattr(base_dataset, 'targets', None)
            if targets is not None:
                self.indices = [i for i, lbl in enumerate(targets) if int(lbl) in self.class_to_new]
            else:
                # Absolute fallback if .targets doesn't exist
                self.indices = [i for i, (_, lbl) in enumerate(base_dataset) if int(lbl) in self.class_to_new]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, lbl = self.base[self.indices[idx]]
        return img, self.class_to_new[int(lbl)]

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

    target_network = TargetCNN(num_classes=2) 
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