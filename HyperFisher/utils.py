
import torch
import torch.nn as nn
from torch import Tensor
# ─────────────────────────────────────────────────────────────────────────────
# Low-level utilities
# ─────────────────────────────────────────────────────────────────────────────

def _flat_grad(model: nn.Module) -> Tensor:
    """Flatten all parameter .grad fields into a single vector [D]."""
    parts = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().view(-1))
        else:
            parts.append(p.data.new_zeros(p.numel()))
    return torch.cat(parts)


def _apply_flat_update(model: nn.Module, update: Tensor) -> None:
    """Add a flat update vector to model parameters in-place: θ ← θ + update."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.add_(update[offset: offset + n].view_as(p))
        offset += n

def calc_bwt(results: dict, task_id: int):
    # If task_id is 0 (the first task), return 0.0 to prevent division by zero
    if task_id <= 0: return 0.0
    
    bwt = 0.0
    for i in range(0, task_id):
        # results[current_task][past_task] - results[past_task][past_task]
        bwt += (results[task_id][i] - results[i][i])
        
    # Divide by the number of past tasks (which is exactly task_id when 0-indexed)
    return bwt / task_id

def evaluate_accuracy(model: nn.Module, loader, task_id) -> float:
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device
    
    if hasattr(model, 'spawn'):
        model.spawn(task_id)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total
    

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

