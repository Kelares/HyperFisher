
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

def calc_bwt(results: dict):
    bwt = 0
    T = len(results)
    if T <= 1: return 0.0
    for i in range(1, T):
        bwt += (results[T][i-1] - results[i][i-1])
    return bwt / (T - 1)

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
    