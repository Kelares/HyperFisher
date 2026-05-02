from __future__ import annotations
from typing import Callable, List, Optional, Literal
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import wandb
from utils import calc_bwt, evaluate_accuracy, plot_overlap
import matplotlib.pyplot as plt
import numpy as np
import gc
from math import inf

# ─────────────────────────────────────────────────────────────────────────────
# Generalized FOPNG Class
# ─────────────────────────────────────────────────────────────────────────────
class FOPNG:
    def __init__(
        self,
        lr: float = 1e-3,
        lam: float = 1e-3,
        damping: float = 0.2,
        alpha: float = 0.5,
        grads_per_task: int = 80,
        max_directions: int = 400,
        fisher_samples: int = 1024,
        device_mode: Literal["cpu", "gpu", "hybrid"] = "hybrid",
    ):
        self.lr = lr
        self.lam = lam
        self.damping = damping
        self.alpha = alpha
        self.grads_per_task = grads_per_task
        self.max_directions = max_directions
        self.fisher_samples = fisher_samples
        self.device_mode = device_mode
        
        self.F_old: Optional[Tensor] = None
        self.G: Optional[Tensor] = None
        self._F_new: Optional[Tensor] = None
        self._A_inv: Optional[Tensor] = None
        self._device: Optional[torch.device] = None
        self.debug = 1
        self.fisher_after_task = {}
        self.quantile = 0.95

    def _get_shared_params(self, model: nn.Module):
        """Generalized helper to extract the parameters to be protected."""
        if hasattr(model, "_shared_params"):
            return model._shared_params
        return list(model.parameters())

    def _is_hypernet(self, model: nn.Module) -> bool:
        """Determines if the model follows the HyperNetwork 'spawn' pattern."""
        return hasattr(model, "spawn")

    def _fopng_update(self, g, G, F_old, F_new, A_inv, eps=1e-8):
        """Computes the projected natural-gradient update vector[cite: 3]."""
        F_new_inv = 1.0 / (F_new + self.damping) 
        comp_dev = G.device
        g_comp = g.to(comp_dev)

        # Riemannian Projection logic[cite: 3]
        GtFg = G.t() @ (F_new_inv * g_comp)
        coeff = A_inv @ GtFg
        correction = (G @ coeff) 
        Pg = g_comp - correction

        # Natural Gradient Step[cite: 3]
        v_raw = F_new_inv * Pg
        
        # Mandatory Stability Clipping
        max_norm = 0.5
        v_norm = torch.norm(v_raw)
        v_star = v_raw * (max_norm / v_norm) if v_norm > max_norm else v_raw
        
        # Metric calculation for logging
        F_sqrt = F_old.clamp(min=0).sqrt()
        weighted_rho = ((F_sqrt * Pg).norm() / ((F_sqrt * g_comp).norm() + eps)).item()
        
        return -(self.lr * v_star), weighted_rho, correction.norm().item(), (Pg.norm() / (g_comp.norm() + eps)).item()

    def compute_fisher_diag(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable, device: torch.device) -> Tensor:
        """Computes diagonal Fisher for both HyperNets and standard architectures[cite: 3]."""
        target_dev = self._get_target_device(device)
        model.eval()
        shared = self._get_shared_params(model)
        D_theta = sum(p.numel() for p in shared)
        fisher = torch.zeros(D_theta, device=target_dev)
        is_hyper = self._is_hypernet(model)
        n_seen = 0

        with torch.enable_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                model.zero_grad()

                if is_hyper:
                    model.spawn(task_id)
                    w = model.w
                    loss = criterion(model(x), y)
                    (g_w,) = torch.autograd.grad(loss, w)
                    model.zero_grad()
                    model.spawn(task_id)
                    model.w.backward(g_w.detach()) # Translate to theta-space
                else:
                    loss = criterion(model(x), y)
                    loss.backward()

                g_theta = torch.cat([p.grad.view(-1) for p in shared if p.grad is not None])
                fisher.add_(g_theta.detach().to(target_dev).pow(2))
                n_seen += x.size(0)
                if n_seen >= self.fisher_samples: break

        # Stability: Clip outliers before normalizing[cite: 3]
        fisher_nonzero = fisher[fisher > 0]
        if len(fisher_nonzero) > 0:
            p = torch.quantile(fisher_nonzero, self.quantile)
            fisher = fisher.clamp(max=p.item())
        if fisher.max() > 0:
            fisher = (fisher / fisher.max()) + 1e-4
            fisher = fisher / fisher.max()
        
        model.train()
        return fisher

    def prepare_epoch(self, F_new: Tensor) -> None:
        self._F_new = F_new
        self._A_inv = self._build_A_inv(self.G, self._F_new)

    def _build_A_inv(self, G, F_new):
        """Constructs the projection inversion matrix using the variable lambda[cite: 3]."""
        scale = 1.0 / (F_new + self.damping) 
        scaled_G = scale.unsqueeze(1) * G
        A = G.t() @ scaled_G
        # Shield rigidity fix: use self.lam instead of hardcoded values[cite: 3]
        A = A + self.lam * torch.eye(A.shape[0], device=A.device)
        return torch.linalg.pinv(A)

    def step(self, model: nn.Module, task_id, g_input: Tensor) -> float:
        """
        Generalized step: g_input is g_w for HyperNets, or the loss Tensor for standard nets[cite: 3].
        """
        model_device = next(model.parameters()).device
        target_dev = self._get_target_device(model_device)
        compute_dev = self._get_compute_device(model_device)
        shared = self._get_shared_params(model)
        is_hyper = self._is_hypernet(model)

        model.zero_grad()
        if is_hyper:
            model.spawn(task_id)
            model.w.backward(g_input) # Translate g_w -> g_theta
        else:
            g_input.backward() # g_input was the loss

        g_theta = torch.cat([p.grad.view(-1) for p in shared if p.grad is not None])

        # Step 2: FOPNG projection logic[cite: 3]
        v_star, w_rho, corr, r_rho = self._fopng_update(g_theta.to(target_dev), self.G, self.F_old, self._F_new, self._A_inv)
        
        # Step 3: Manual Apply
        v_star = v_star.to(compute_dev)
        pointer = 0
        with torch.no_grad():
            for p in shared:
                n = p.numel()
                p.data.add_(v_star[pointer : pointer + n].view_as(p))
                pointer += n
        return w_rho, corr, r_rho

    def _collect_gradients(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> Tensor:
        """Builds the G matrix with NaN/Inf filtering[cite: 3]."""
        target_dev = self._get_target_device(next(model.parameters()).device)
        grads: List[Tensor] = []
        model.eval()
        shared = self._get_shared_params(model)
        is_hyper = self._is_hypernet(model)

        with torch.enable_grad():
            while len(grads) < self.grads_per_task:
                for x, y in loader:
                    if len(grads) >= self.grads_per_task: break
                    x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                    model.zero_grad()
                    
                    if is_hyper:
                        model.spawn(task_id)
                        loss = criterion(model(x), y)
                        (g_w,) = torch.autograd.grad(loss, model.w)
                        model.zero_grad(); model.spawn(task_id)
                        model.w.backward(g_w.detach())
                    else:
                        loss = criterion(model(x), y)
                        loss.backward()

                    g_theta = torch.cat([p.grad.view(-1) for p in shared if p.grad is not None])
                    
                    # 🛑 THE BOUNCER: NaN/Inf Filter[cite: 3]
                    if torch.isnan(g_theta).any() or torch.isinf(g_theta).any():
                        continue

                    grads.append(g_theta.detach().to(target_dev))

        model.train()
        return torch.stack(grads, dim=1)

    def after_task(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        """End of task cleanup and memory SVD compression[cite: 3]."""
        device = next(model.parameters()).device
        target_dev = self._get_target_device(device)

        F_new = self.compute_fisher_diag(model, task_id, loader, criterion, device)
        
        if self.F_old is None:
            self.F_old = F_new.detach().to(target_dev)
        else:
            # Arithmetic Mean Fisher update[cite: 3]
            n = task_id.item() + 1
            self.F_old = ((n - 1) / n) * self.F_old + (1.0 / n) * F_new.detach().to(target_dev)

        new_cols = self._collect_gradients(model, task_id, loader, criterion)
        self.G = new_cols if self.G is None else torch.cat([self.G, new_cols], dim=1)

        # Memory SVD Compression[cite: 3]
        if self.G.shape[1] > self.max_directions:
            # Nuclear Sanitizer: Drop any columns that somehow got a NaN
            valid_mask = ~(torch.isnan(self.G).any(dim=0) | torch.isinf(self.G).any(dim=0))
            self.G = self.G[:, valid_mask]
            U, S, Vh = torch.linalg.svd(self.G, full_matrices=False)
            self.G = U[:, :self.max_directions]

        self.fisher_after_task[task_id.item()] = F_new 
        torch.cuda.empty_cache()
        gc.collect()

    def _get_target_device(self, dev): return torch.device("cpu") if self.device_mode in ["cpu", "hybrid"] else dev
    def _get_compute_device(self, dev): return dev if self.device_mode in ["gpu", "hybrid"] else torch.device("cpu")
    def _cosine_similarity(self, a, b): return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()
    def compute_overlap_matrix(self):
        keys = list(self.fisher_after_task.keys())
        n = len(keys)
        # Initialize a symmetric matrix with 1s on the diagonal
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = self.frechet(
                    self.fisher_after_task[keys[i]], 
                    self.fisher_after_task[keys[j]]
                )
                matrix[i, j] = overlap
                matrix[j, i] = overlap  # Symmetry
                
        return matrix, keys

# ─────────────────────────────────────────────────────────────────────────────
# Generalized FOPNG+ (Jacobian Refresher)
# ─────────────────────────────────────────────────────────────────────────────
class FOPNGPlus(FOPNG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_history: List[tuple] = []

    def after_task(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        device = next(model.parameters()).device
        target_dev = self._get_target_device(device)
        F_new = self.compute_fisher_diag(model, task_id, loader, criterion, device)

        if self.F_old is None: self.F_old = F_new.detach().to(target_dev)
        else:
            n = task_id.item() + 1
            self.F_old = ((n - 1) / n) * self.F_old + (1.0 / n) * F_new.detach().to(target_dev)

        self._task_history.append((task_id.clone(), loader))

        # FOPNG+: Jacobian Refresh Loop[cite: 3]
        print(f"  [FOPNG+] Refreshing G for {len(self._task_history)} task(s)...")
        all_cols = [self._collect_gradients(model, t_id, t_ld, criterion) for t_id, t_ld in self._task_history]
        G_fresh = torch.cat(all_cols, dim=1)

        if G_fresh.shape[1] > self.max_directions:
            valid_mask = ~(torch.isnan(G_fresh).any(dim=0) | torch.isinf(G_fresh).any(dim=0))
            U, S, Vh = torch.linalg.svd(G_fresh[:, valid_mask], full_matrices=False)
            G_fresh = U[:, :self.max_directions]

        self.G = G_fresh
        self.fisher_after_task[task_id.item()] = F_new 
        gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# Training Engine
# ─────────────────────────────────────────────────────────────────────────────
def train_fopng_generic(model, train_loaders, test_loaders, criterion, **kwargs):
    lr = kwargs.get('lr', 1e-2)
    device = next(model.parameters()).device
    method = kwargs.get('method', 'fopng')
    
    fopng = FOPNGPlus(**kwargs) if method == 'fopng_plus' else FOPNG(**kwargs)
    is_hyper = hasattr(model, "spawn")
    results = {}

    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)
        best_loss = inf; epoch = 0
        
        if t == 0:
            opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
            while best_loss >= 0.2 and epoch < kwargs.get('max_epochs', 400):
                # Standard training loop...
                pass
            fopng.after_task(model, task_id, loader, criterion)
        else:
            # 1. ASYMMETRIC WARMUP[cite: 3]
            if hasattr(model, "task_emb"):
                for p in fopng._get_shared_params(model): p.requires_grad = False
                # Warmup with high LR (0.1)[cite: 3]
                opt_warm = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
                for _ in range(15):
                    for x, y in loader:
                        opt_warm.zero_grad()
                        if is_hyper: model.spawn(task_id)
                        loss = criterion(model(x.to(device)), y.to(device))
                        loss.backward(); opt_warm.step()
                for p in fopng._get_shared_params(model): p.requires_grad = True

            # 2. CONSTRAINED TRAINING
            while best_loss >= 0.15 and epoch < kwargs.get('max_epochs', 400):
                F_new = fopng.compute_fisher_diag(model, task_id, loader, criterion, device)
                fopng.prepare_epoch(F_new)
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if is_hyper:
                        model.spawn(task_id)
                        loss = criterion(model(x), y)
                        (g_w,) = torch.autograd.grad(loss, model.w)
                        fopng.step(model, task_id, g_w.detach())
                    else:
                        loss = criterion(model(x), y)
                        fopng.step(model, task_id, loss)
                epoch += 1
            fopng.after_task(model, task_id, loader, criterion)
    return results