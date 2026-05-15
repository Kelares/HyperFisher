"""
Split-CIFAR100 Task Generator
==============================
10 tasks, each a 10-class classification of consecutive classes:
  Task 0: Classes 0-9
  Task 1: Classes 10-19
  ...
  Task 9: Classes 90-99

Interface is identical to the previous CIFAR-10 version.
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

class TargetCNN(nn.Module):
    """
    Adjusted for 10-class output per task.
    """
    def __init__(self, num_classes: int = 10): # Changed default to 10
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1   = nn.Linear(32 * 4 * 4, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MultiHeadTargetCNN(nn.Module):
    # Change num_classes_per_task default to 10
    def __init__(self, num_tasks: int, device, num_classes_per_task: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16, 3, padding=1).to(device)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1).to(device)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1).to(device)
        self.fc1   = nn.Linear(32 * 4 * 4, 64).to(device)

        self.heads = nn.ModuleList([
            nn.Linear(64, num_classes_per_task) for _ in range(num_tasks)
        ]).to(device)
        
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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
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
        self.in_planes = 16
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 16) # Changed BatchNorm to GroupNorm
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(128 * block.expansion, num_classes)

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

def ResNet18Slim(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_groups=16)
    
# ─────────────────────────────────────────────────────────────────────────────
# Task Generator for CIFAR-100
# ─────────────────────────────────────────────────────────────────────────────
class TaskGenerator:
    # Programmatically create 10 tasks with 10 classes each
    TASK_CLASSES = [list(range(i, i + 10)) for i in range(0, 100, 10)]

    config = SimpleNamespace(
        input_dim=3072,
        num_classes=10,      # 10 classes per task
        num_tasks=10,       # 10 tasks total
        criterion=nn.CrossEntropyLoss(),
        task_classes=TASK_CLASSES,
        grads_per_task=200,  # Increased slightly for higher complexity
        max_directions=2000, 
    )

    target_network = ResNet18Slim(num_classes=10) 
    multihead = MultiHeadTargetCNN

    _train_data: Optional[datasets.CIFAR100] = None
    _test_data:  Optional[datasets.CIFAR100] = None

    @classmethod
    def _load(cls) -> None:
        """Loads and normalizes the base CIFAR-100 datasets."""
        if cls._train_data is not None:
            return
            
        print(" [TaskGenerator] Initializing base CIFAR-100 datasets...")
        # CIFAR-100 specific normalization constants
        tf_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4), # CRITICAL: Allows the model to see shifted versions
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        cls._train_data = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=tf_train
        )
        cls._test_data = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=tf_test
        )

    @classmethod
    def _make_split(
        cls,
        dataset: datasets.CIFAR100,
        allowed_classes: List[int],
        batch_size: int = 64,
        shuffle: bool = True,
        desc: str = "Processing Task"
    ) -> DataLoader:
        indices = []
        # CIFAR100 also uses .targets (same as CIFAR10)
        for i, label in enumerate(tqdm(dataset.targets, desc=f"  {desc}", leave=False)):
            if label in allowed_classes:
                indices.append(i)

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
        assert 0 <= task_id < cls.config.num_tasks, \
            f"task_id must be in [0, {cls.config.num_tasks}), got {task_id}"

        cls._load()
        allowed_classes = cls.TASK_CLASSES[task_id]

        prefix = f"Task {task_id+1}"
        
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