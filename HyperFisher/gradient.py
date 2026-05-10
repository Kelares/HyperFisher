import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from tqdm import tqdm
from utils import get_grad_vector



def set_grad_vector(model: nn.Module, grad_vector: torch.Tensor):
    """Set model.grad tensors from a single 1D gradient vector."""
    idx = 0
    for p in model.parameters():
        numel = p.data.numel()
        g = grad_vector[idx:idx+numel].view_as(p.data)
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        p.grad.copy_(g)
        idx += numel

class GradientMemory:
    def __init__(self, mode: str = 'raw', max_directions: int = 2000):
        self.mode = mode
        self.max_directions = max_directions
        # Store all directions as columns in a single 2D tensor [D, K]
        # Initially None to clearly distinguish from an empty state
        self.basis: Optional[torch.Tensor] = None 

    @torch.no_grad()
    def add(self, v: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Adds direction(s) to memory and ensures they are stored as matrix columns.
        Handles both a single vector [D] and a list of vectors [[D], [D], ...].
        """
        # 1. Convert input to a 2D column block [D, K_new]
        if isinstance(v, list):
            if not v: return
            # Stack list of 1D vectors as columns
            new_data = torch.stack([vec.detach().view(-1) for vec in v], dim=1)
        else:
            # Turn single vector into a column
            new_data = v.detach().view(-1, 1)

        # 2. Expand the matrix
        if self.basis is None:
            self.basis = new_data
        else:
            # Concatenate along the column dimension (dim=1)
            # Result: [D, K_old + K_new]
            self.basis = torch.cat([self.basis, new_data], dim=1)

        # 3. Check budget and compress if needed
        if self.basis.size(1) > self.max_directions:
            self.compress()

    @torch.no_grad()
    def compress(self):
        """Reduces the matrix to max_directions using SVD (Orthonormal Basis)."""
        if self.basis is None: return
        print(f"    Reducing the gradient size from {len(self)} to {self.max_directions} via SVD")
        
        # full_matrices=False ensures U is [D, K]
        U, S, _ = torch.linalg.svd(self.basis, full_matrices=False)
        
        # Retain only the most significant orthogonal directions
        # Re-scale U by the singular values to keep the physical magnitude of the gradients
        self.basis = U[:, :self.max_directions] @ torch.diag(S[:self.max_directions])

    @property
    def matrix(self) -> Optional[torch.Tensor]:
        """Returns the single Tensor object representing the subspace."""
        return self.basis

    def __len__(self) -> int:
        """Returns the number of stored directions (columns)."""
        return self.basis.size(1) if self.basis is not None else 0

    def clear(self):
        """Deletes the tensor and frees GPU memory."""
        self.basis = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Gradient Collection Strategies
# =============================================================================

class GradientCollector(ABC):
    """Abstract base class for gradient collection strategies."""

    def __init__(self, grads_per_task):
        self.grads_per_task = grads_per_task

    def subset(self, loader):
        # If batch_size is specified, use only that many samples total for estimation
        if self.grads_per_task is not None:
            dataset = loader.dataset
            print(self.grads_per_task)
            # Take only the first batch_size samples
            limited_dataset = Subset(dataset, range(min(self.grads_per_task, len(dataset))))
            subset_loader = DataLoader(limited_dataset, batch_size=len(limited_dataset), shuffle=False)
        else:
            subset_loader = loader
        return subset_loader

    @abstractmethod
    def collect(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        """Collect gradient directions from a task."""
        pass
    
    def collect_prefisher(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        fisher_matrix: torch.Tensor,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        # Create a temporary memory buffer for raw gradients
        temp_memory = GradientMemory(mode=memory.mode, max_directions=self.grads_per_task)
        
        # Collect raw gradients into temporary buffer
        self.collect(temp_memory, model, dataloader, device, multihead, task_id)
        
        for grad_vec in temp_memory.vectors:
            if isinstance(fisher_matrix, torch.Tensor):
                # Diagonal Fisher
                if fisher_matrix.dim() == 1:
                    prefisher_grad = fisher_matrix * grad_vec
                else:
                    prefisher_grad = fisher_matrix @ grad_vec
            else:
                prefisher_grad = grad_vec
            memory.add(prefisher_grad)
    
    def collect_empirical_fisher_preconditioned(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        """
        Collect gradients pre-multiplied by empirical Fisher using associative property.
        
        Key idea: F = sum_i(g_i * g_i^T), so F*g = sum_i(g_i * (g_i^T * g))
        For each gradient g, we compute F*g by accumulating contributions from all 
        collected gradients without storing the n x n Fisher matrix.
        
        First pass: collect all raw gradients
        Second pass: for each collected gradient, compute F*g and add to memory
        """
        # First pass: collect all raw gradients
        print(f"Collecting empirical Fisher-preconditioned gradients (task {task_id})...")
        temp_memory = GradientMemory(mode=memory.mode, max_directions=num_directions)
        
        self.collect(temp_memory, model, dataloader, num_directions, device, multihead, task_id)
        
        if not temp_memory.vectors:
            return
        
        raw_gradients = temp_memory.vectors  # List of gradient vectors
        gradients = []
        # Second pass: for each collected gradient, compute F*g on-the-fly
        for g in raw_gradients:
            # Compute F*g = sum_i(g_i * (g_i^T * g))
            Fg = torch.zeros_like(g)
            for g_i in raw_gradients:
                # g_i^T * g: scalar dot product
                dot_prod = torch.dot(g_i, g)
                # g_i * (g_i^T * g): accumulate scaled gradient
                Fg = Fg + g_i * dot_prod
            gradients.append(Fg)
            # Add the empirical Fisher preconditioned gradient to memory
        memory.add(gradients)


class GTLCollector(GradientCollector):
    """
    Ground-Truth Logit gradient collector (OGD-GTL).
    Computes gradients with respect to the ground-truth class logit.
    """
    
    def collect(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
        task_id: int,
        multihead: bool = False,
    ):
        model.eval()
        collected = 0
        
        desc = "Collecting GTL gradients"
        if task_id is not None:
            desc += f" (task {task_id})"
        iterator = tqdm(dataloader, desc=desc, leave=False)
        gradients = []
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            
            batch_size = x.size(0)
            for i in range(batch_size):
                if collected >= num_directions:
                    break
                
                model.zero_grad()
                model.spawn(task_id)
                xi = x[i:i+1]
                yi = y[i:i+1]
                
                if multihead:
                    logits = model(xi, task_id=task_id)
                else:
                    logits = model(xi)
                
                # Ground truth logit
                gt_logit = logits[0, yi.item()]
                gt_logit.backward()
                
                grad_vec = get_grad_vector(model).detach()
                gradients.append(grad_vec)
                collected += 1
        memory.add(gradients)
        
        print(f"  Collected {collected} GTL directions (total: {len(memory)})")


class AVECollector(GradientCollector):
    """
    Average logit gradient collector (OGD-AVE).
    Computes gradients with respect to the average of all logits.
    """
    
    def collect(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        task_id: Optional[int],

        multihead: bool = False,
    ):
        model.eval()
        collected = 0
        
        dataloader = self.subset(dataloader)

        desc = "Collecting AVE gradients"
        if task_id is not None:
            desc += f" (task {task_id.item()})"
        pbar = tqdm(total=self.grads_per_task, desc=desc, leave=False)
        
        gradients = []
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            for i in range(x.size(0)):
                model.zero_grad()
                model.spawn(task_id)
                w = model.w
                if multihead:
                    output = model(x[i:i+1], task_id=task_id)
                else:
                    output = model(x[i:i+1])
                
                # Average of all logits
                avg_logit = output.mean()
                (g_w,) = torch.autograd.grad(avg_logit, w)

                # ── Step 2: translate g_w → g_θ = Jᵀg_w ──────────────
                model.zero_grad()
                model.spawn(task_id)                # fresh graph
                model.w.backward(g_w.detach())      # θ.grad = Jᵀ g_w

                g_theta = torch.cat([
                    p.grad.view(-1) for p in model._shared_params if p.grad is not None
                ])
                gradients.append(g_theta.detach())
                model.zero_grad()

                # grad_vec = get_grad_vector(model).detach()
                # gradients.append(g_w)
                collected += 1
                pbar.update(1)
        memory.add(gradients)

        print(f"  Collected {collected} AVE directions (total: {len(memory)})")

