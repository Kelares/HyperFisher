
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Low-level utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_grad_vector(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients into a single 1D tensor."""
    grads = []
    for p in model._shared_params:
        if p.grad is None:
            grads.append(torch.zeros_like(p.data).view(-1))
        else:
            grads.append(p.grad.view(-1))
    grads = torch.cat(grads)
    # grads /= model.num_of_chunks
    return grads

def _apply_flat_update(model: nn.Module, update: Tensor) -> None:
    """Add a flat update vector to model parameters in-place: θ ← θ + update."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.add_(update[offset: offset + n].view_as(p))
        offset += n

def evaluate_accuracy(model: nn.Module, loader, task_id, task_classes=None) -> float:
    """
    Multi-Head Evaluation: 
    Each task's 'spawn' produces a head with its own 2 outputs.
    Labels in the loader are already remapped to [0, 1].
    """
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device
    
    # Generate the task-specific head
    if hasattr(model, 'spawn'):
        model.spawn(task_id)
        
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) # Shape: [Batch, 2]
            
            # Simple argmax: model only sees classes for THIS task
            preds = logits.argmax(dim=1) 
            
            correct += (preds == y).sum().item()
            total   += y.size(0)
            
    return correct / total


def calc_bwt(results: dict, task_id: int):
    if task_id <= 1: return 0.0
    bwt = 0
    for i in range(1, task_id):
        # Difference between accuracy now vs when the task was first learned
        bwt += (results[task_id][i-1] - results[i][i-1])
    return bwt / (task_id - 1)

import gc

def stress_test_fopng_memory(num_params=5_000_000, max_directions=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # 1. Simulate the model's flattened gradients
    # Adjust num_params to roughly match your Permuted MNIST model size 
    # (e.g., MLP with hyper_hidden_dim=200 usually has a few million params)
    dummy_gradient = torch.randn(num_params, 1, device=device)
    
    print(f"Allocating FOPNG Buffer: {num_params} params x {max_directions} directions...")
    try:
        # 2. Simulate a completely full FOPNG basis buffer (U)
        U = torch.randn(num_params, max_directions, device=device)
        
        # Normalize columns to simulate an orthogonal basis
        U = U / torch.norm(U, dim=0, keepdim=True) 
        
        print(f"Buffer allocated. VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 3. Perform the projection step: g_proj = g - U @ (U^T @ g)
        print("Executing projection step...")
        
        # Step A: U^T @ g (Shape: M x 1)
        projection_coeffs = torch.matmul(U.T, dummy_gradient) 
        
        # Step B: U @ coeffs (Shape: P x 1)
        projected_component = torch.matmul(U, projection_coeffs) 
        
        # Step C: Final subtraction
        g_proj = dummy_gradient - projected_component
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✅ Success! Peak VRAM during projection: {peak_memory:.2f} GB")
        
    except torch.cuda.OutOfMemoryError:
        print("❌ OOM ERROR: The projection matrices are too large for your GPU.")
    finally:
        # Cleanup
        del dummy_gradient, U, projection_coeffs, projected_component, g_proj
        torch.cuda.empty_cache()
        gc.collect()


def plot_overlap(matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=labels, yticklabels=labels)
    plt.title("Fisher Overlap (Task Similarity)")
    return plt


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