import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from typing import List, Optional, Tuple, Union, Literal
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
    def __init__(self, mode: str = 'raw', max_directions: int = 2000, normalization: bool = False, compression: Literal["svd", "fifo", "stop"] = "svd"):
        self.mode = mode
        self.max_directions = max_directions
        # Store all directions as columns in a single 2D tensor [D, K]
        # Initially None to clearly distinguish from an empty state
        self.basis: Optional[torch.Tensor] = None 
        self.debug = True
        self.normalization = normalization
        self.compression = compression

    @torch.no_grad()
    def add(self, v: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Adds direction(s) to memory and ensures they are stored as matrix columns.
        Handles both a single vector [D] and a list of vectors [[D], [D], ...].
        """
        if self.compression == "stop":
            if self.basis and self.basis.size(1) >= self.max_directions:
                if self.debug:
                    print(f"  [DEBUG compress] STOP — truncated to first {self.max_directions} cols, new directions discarded")
                return None
            
        # 1. Convert input to a 2D column block [D, K_new]
        if isinstance(v, list):
            new_vecs = torch.stack([vec.detach().view(-1) for vec in v], dim=1)
        else:
            # Turn single vector into a column
            if v.dim() == 1:
                new_vecs = v.detach().view(-1, 1)
            else:
                new_vecs = v.detach() # Already a [D, K] matrix
                
        # # # Normalize columns to unit Euclidean norm
        if self.normalization:
            # norms = new_vecs.norm(dim=0, keepdim=True) + 1e-8
            # new_vecs = new_vecs / norms
            Q, _ = torch.linalg.qr(new_vecs)
            new_vecs = Q
            # U, S, _ = torch.linalg.svd(new_vecs , full_matrices=False)
            # new_vecs = U #@ torch.diag(S)

        # 2. Expand the matrix
        if self.basis is None:
            self.basis = new_vecs
        else:
            # Concatenate along the column dimension (dim=1)
            # Result: [D, K_old + K_new]
            self.basis = torch.cat([self.basis, new_vecs], dim=1)

        # 3. Check budget and compress if needed
        if self.basis.size(1) > self.max_directions:
            self.compress()

         # 🔍 DEBUG: Print norms of newly added vectors
        if self.debug:
            print(f" [DEBUG add] Added vectors are length: {new_vecs.norm(dim=0).mean():.4f}")
            print(f"  [DEBUG add] New vec: min={new_vecs.min():.3f}, max={new_vecs.max():.3f}")
    

    @torch.no_grad()
    def compress(self):
        """Reduces the matrix to max_directions using SVD (Orthonormal Basis)."""
        if self.basis is None: return

        match self.compression:
            case "svd":
                print(f"    Reducing the gradient size from {len(self)} to {self.max_directions} via SVD")
                
                # full_matrices=False ensures U is [DEBUG add] Added vectors are unit length: 1.0000 [D, K]
                # weighted_G = F_old.view(-1, 1) * self.basis
                U, S, _ = torch.linalg.svd(self.basis , full_matrices=False)
                if self.basis.size(1) > self.max_directions:
                    if self.normalization:
                        U = U[:, :self.max_directions]
                        S = S[:self.max_directions]
                
                if self.normalization:
                    self.basis = U
                else:
                    self.basis = U @ torch.diag(S)

                # DEBUG: Verify U columns are unit-norm
                if self.debug:
                    U_norms = self.basis.norm(dim=0)
                    print(f"  [DEBUG compress] Post-SVD column norms: min={U_norms.min():.6f}, max={U_norms.max():.6f}")

            case "fifo":
                n = self.basis.size(1)                        
                evicted = n - self.max_directions             #
                print(f"    Reducing the gradient size from {n} to {self.max_directions} via FIFO")
                self.basis = self.basis[:, -self.max_directions:]
                if self.debug:
                    print(f"  [DEBUG compress] FIFO evicted cols 0:{evicted}, kept cols {evicted}:{n}")
  
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

    @property
    def vectors(self):
        return self.basis.unbind(dim=1)
    
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
        task_id: Optional[int] = None,
    ):
        # Create a temporary memory buffer for raw gradients
        temp_memory = GradientMemory(mode=memory.mode, max_directions=self.grads_per_task*10)
        
        # Collect raw gradients into temporary buffer
        self.collect(temp_memory, model, dataloader, device, task_id)
        
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
        task_id: Optional[int] = None,
        normalize: bool = False
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
        # 1. Collect raw gradients into a temporary memory
        temp_memory = GradientMemory(mode=memory.mode, max_directions=num_directions, normalization=normalize)
        self.collect(temp_memory, model, dataloader, device, task_id)
        G = temp_memory.basis  # This is your [D, K] matrix
        if G is None:
            return

        # 2. Vectorized Fisher Preconditioning
        # We use the associative trick: B @ (B.T @ B)
        # This is mathematically identical to sum_i(g_i * (g_i.T @ g))
        
        # Step A: Compute the Gram matrix [K, K] (all pairs of dot products)
        gram_matrix = G.T @ G
        
        # Step B: Project back onto the basis [D, K]
        # This effectively computes F*g for all gradients at once
        Fg_matrix = G @ gram_matrix

        # 3. Add to the permanent memory (handles matrix input automatically)
        memory.add(Fg_matrix)

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
        device: str,
        task_id: int,
    ):
        dataloader = self.subset(dataloader)
        model.eval()
        collected = 0
        
        desc = "Collecting GTL gradients"
        if task_id is not None:
            desc += f" (task {task_id})"
        iterator = tqdm(dataloader, desc=desc, leave=False)
        gradients = []
        model.spawn(task_id)

        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            
            batch_size = x.size(0)
            for i in range(batch_size):
                model.zero_grad()
                xi = x[i:i+1]
                yi = y[i:i+1]
                
                logits = model(xi)
                
                # Ground truth logit
                gt_logit = logits[0, yi.item()]
                gt_logit.backward(retain_graph=True)
                
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

    ):
        model.eval()
        collected = 0
        
        dataloader = self.subset(dataloader)

        desc = "Collecting AVE gradients"
        if task_id is not None:
            desc += f" (task {task_id.item()})"
        pbar = tqdm(total=self.grads_per_task, desc=desc, leave=False)
        
        gradients = []

        model.spawn(task_id)
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            for i in range(x.size(0)):
                model.zero_grad()
                
                output = model(x[i:i+1])
                
                # Average of all logits
                avg_logit = output.mean()
                avg_logit.backward(retain_graph=True)
                
                grad_vec = get_grad_vector(model).detach()
                gradients.append(grad_vec)
                collected += 1
                pbar.update(1)
        memory.add(gradients)

        print(f"  Collected {collected} AVE directions (total: {len(memory)})")

class BoundaryCollector(GradientCollector):
    """
    Specifically designed for Split (2-class) tasks.
    Protects the exact decision boundary by collecting the gradient of the logit difference.
    """
    def collect(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        task_id: Optional[int],
        F_old,
    ):
        model.eval()
        collected = 0
        
        dataloader = self.subset(dataloader)

        desc = "Collecting Boundary gradients"
        if task_id is not None:
            desc += f" (task {task_id})"
        pbar = tqdm(total=self.grads_per_task, desc=desc, leave=False)
        
        gradients = []
        model.spawn(task_id)

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            for i in range(x.size(0)):
                model.zero_grad()
                
                logits = model(x[i:i+1])
                
                # The single degree of freedom for a 2-class head
                # We protect the exact distance between the two classes
                diff_logit = logits[0, 0] - logits[0, 1]
                diff_logit.backward(retain_graph=True)
                
                grad_vec = get_grad_vector(model).detach()
                gradients.append(grad_vec)
                
                collected += 1
                pbar.update(1)
                
        memory.add(gradients, F_old)
        if hasattr(memory, "debug") and memory.debug:
            print(f"  Collected {collected} Boundary directions (total: {memory.basis.size(1) if memory.basis is not None else 0})")