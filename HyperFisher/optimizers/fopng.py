from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import wandb
from utils import _flat_grad, _apply_flat_update, calc_bwt, evaluate_accuracy
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FOPNG Class
# ─────────────────────────────────────────────────────────────────────────────
class FOPNG:
    def __init__(
        self,
        lr: float = 1e-3,
        lam: float = 1e-3,
        alpha: float = 0.5,
        grads_per_task: int = 80,
        max_directions: int = 400,
        fisher_samples: int = 1024,
    ):
        self.lr            = lr
        self.lam           = lam
        self.alpha         = alpha
        self.grads_per_task = grads_per_task
        self.max_directions = max_directions
        self.fisher_samples = fisher_samples

        self.F_old: Optional[Tensor] = None
        self.G:     Optional[Tensor] = None
        self._F_new: Optional[Tensor] = None
        self._A_inv: Optional[Tensor] = None
        self._device: Optional[torch.device] = None

    def compute_fisher(self, model: nn.Module, loader: DataLoader, criterion: Callable) -> Tensor:
        return self.compute_fisher_diag(model, loader, criterion, self._device, self.fisher_samples)
    
    def compute_fisher_diag(
        self,
        hyper_network: nn.Module,
        task_id, 
        loader: DataLoader,
        criterion: Callable,
        device: torch.device,
        max_samples: int = 1024,
    ) -> Tensor:
        hyper_network.eval()
        D = sum(p.numel() for p in hyper_network.parameters())
        fisher = torch.zeros(D, device=device)
        n_seen = 0
        n_batches = 0 # <--- NEW: Track batches

        hyper_network.eval()
        hyper_network.spawn(task_id)

        with torch.enable_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                hyper_network.zero_grad()
                
                output = hyper_network(x)

                loss = criterion(output, y)
                loss.backward(retain_graph=True)

                g = _flat_grad(hyper_network)
                fisher.add_(g.pow(2))

                n_seen += x.size(0)
                n_batches += 1
                if n_seen >= max_samples:
                    break

        hyper_network.zero_grad()
        hyper_network.train()
        return fisher / max(n_batches, 1)
    
    def prepare_epoch(self, F_new: Tensor) -> None:
        assert self.F_old is not None, "Call after_task() after task 1 before training task 2."
        self._F_new = F_new
        self._A_inv = self._build_A_inv(self.G, self.F_old, F_new, self.lam)

    def step(self, model: nn.Module) -> float:
        assert self._A_inv is not None, "Call prepare_epoch(F_new) before step()."
        g = _flat_grad(model)
        v_star, rho = self._fopng_update(
            g=g, G=self.G, F_old=self.F_old, F_new=self._F_new,
            A_inv=self._A_inv, lr=self.lr, lam=self.lam,
        )
        _apply_flat_update(model, v_star)
        model.zero_grad()
        return rho

    def after_task(self, hyper_network: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        device = next(hyper_network.parameters()).device
        self._device = device

        F_new = self.compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
        if self.F_old is None:
            self.F_old = F_new.clone()
        else:
            self.F_old = (1.0 - self.alpha) * self.F_old + self.alpha * F_new

        new_cols = self._collect_gradients(hyper_network, task_id, loader, criterion)
        self.G   = new_cols if self.G is None else torch.cat([self.G, new_cols], dim=1)

        if self.G.shape[1] > self.max_directions:
            # Uniformly sample indices across the entire chronological history
            # This ensures every task gets an equal slice of the max_directions budget
            indices = torch.linspace(
                0, self.G.shape[1] - 1, 
                steps=self.max_directions, 
                dtype=torch.long, 
                device=self._device
            )
            self.G = self.G[:, indices]


        wandb.log({
            "fopng/fisher/min": self.F_old.min().item(),
            "fopng/fisher/max": self.F_old.max().item(),
            "fopng/fisher/mean": self.F_old.mean().item(),
            "fopng/memory/G_cols": self.G.shape[1],
            "task_completed": task_id.item() + 1
        })

    def _collect_gradients(self, hyper_network: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> Tensor:
        grads: List[Tensor] = []
        hyper_network.eval()
        hyper_network.spawn(task_id)

        with torch.enable_grad():
            for x, y in loader:
                if len(grads) >= self.grads_per_task:
                    break
                x, y = x.to(self._device), y.to(self._device)
                hyper_network.zero_grad()

                output = hyper_network(x)
                loss = criterion(output, y)
                loss.backward(retain_graph=True)
                grads.append(_flat_grad(hyper_network).clone())
        hyper_network.zero_grad()
        hyper_network.train()
        return torch.stack(grads, dim=1)

    def _fopng_update(
        self,
        g: Tensor,      # [D]   current task gradient
        G: Tensor,      # [D, m] gradient memory
        F_old: Tensor,  # [D]
        F_new: Tensor,  # [D]
        A_inv: Tensor,  # [m, m]
        lr: float,
        lam: float,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Compute v*  (Theorem 1, eq. 5).

        Step 1 — project g:
            Pg = g  −  F_old G A⁻¹ Gᵀ F_old g

        Step 2 — unit natural gradient *descent* step in F_new metric:
            v* = -η · F_new⁻¹ Pg / sqrt( Pgᵀ F_new⁻¹ Pg )

        The minus sign is required because g points uphill (gradient of the loss),
        so F_new⁻¹ Pg also points uphill.  We negate to descend.
        Applied as  θ ← θ + v*  (i.e. θ ← θ - η · normalised_natural_grad).
        """
        # ── projection ────────────────────────────────────────────────────
        F_old_g  = F_old * g                          # [D]    F_old · g
        GtFg     = G.t() @ F_old_g                    # [m]    Gᵀ F_old g
        coeff    = A_inv @ GtFg                        # [m]    A⁻¹ Gᵀ F_old g
        Pg       = g - F_old * (G @ coeff)             # [D]    Pg = g − F_old G A⁻¹ Gᵀ F_old g

        # Calculate norms
        g_norm = torch.norm(g)
        Pg_norm = torch.norm(Pg)

        # Calculate ratio (add epsilon to avoid division by zero)
        rho = (Pg_norm / (g_norm + 1e-8)).item()

        # ── unit natural gradient ──────────────────────────────────────────
        F_new_inv    = 1.0 / (F_new + lam)            # [D]
        F_new_inv_Pg = F_new_inv * Pg                  # [D]    F_new⁻¹ Pg
        fisher_norm  = torch.sqrt((Pg * F_new_inv_Pg).sum() + eps)   # scalar

        return -lr * F_new_inv_Pg / fisher_norm, rho      # [D]  negative = descent

    def _build_A_inv(
        self,
        G: Tensor,      # [D, m]
        F_old: Tensor,  # [D]
        F_new: Tensor,  # [D]
        lam: float,
    ) -> Tensor:
                
        # ─────────────────────────────────────────────────────────────────────────────
        # Core math
        # ─────────────────────────────────────────────────────────────────────────────

        """
        A  = Gᵀ F_old F_new⁻¹ F_old G  +  λ I      [m × m]

        With diagonal Fishers, row i of (F_old F_new⁻¹ F_old G) is:

            F_old[i]² / F_new[i]  ×  G[i, :]

        Precomputed once per epoch.  Returns A⁻¹  [m × m].
        """
        F_new_inv = 1.0 / (F_new + lam)                 # [D]
        scale     = (F_old ** 2) * F_new_inv             # [D]   F_old² / F_new
        scaled_G  = scale.unsqueeze(1) * G               # [D, m]
        A         = G.t() @ scaled_G                     # [m, m]
        A         = A + lam * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.pinv(A)                      # [m, m]



def train_fopng(
    hyper_network: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    lam: float = 1e-3,
    alpha: float = 0.5,
    grads_per_task: int = 80,
    max_directions: int = 400,
    fisher_samples: int = 1024,
    epochs: int = 5,
    first_task_optimizer_cls=torch.optim.Adam,
    verbose: bool = True,
) -> FOPNG:
    device = next(hyper_network.parameters()).device
    fopng = FOPNG(
        lr=lr, lam=lam, alpha=alpha,
        grads_per_task=grads_per_task,
        max_directions=max_directions,
        fisher_samples=fisher_samples,
    )
    results = {}
    global_epoch = 0

    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)
        
        if t == 0:
            if verbose: print(f"[FOPNG] Task 1 – {first_task_optimizer_cls.__name__}")
            opt = first_task_optimizer_cls(hyper_network.parameters(), lr=lr)
            for epoch in range(epochs):
                total_loss = 0.0
                hyper_network.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    hyper_network.spawn(task_id)
                    output = hyper_network(x)
                    loss = criterion(output, y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                wandb.log({"fopng/train/loss": avg_loss, "fopng/global_epoch": global_epoch, "task": t+1})
                global_epoch += 1
                if verbose: print(f"  epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")
            fopng.after_task(hyper_network, task_id, loader, criterion)

        else:
            if verbose: print(f"\n[FOPNG] Task {t+1}")
            for epoch in range(epochs):
                F_new = fopng.compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
                fopng.prepare_epoch(F_new)
                total_loss, total_rho = 0.0, 0.0
                hyper_network.train()
                
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    hyper_network.spawn(task_id)
                    output = hyper_network(x)
                    loss = criterion(output, y)
                    loss.backward()
                    total_loss += loss.item()
                    
                    rho = fopng.step(hyper_network)
                    total_rho += rho
                    
                avg_loss = total_loss / len(loader)
                avg_rho = total_rho / len(loader)
                
                wandb.log({
                    "fopng/train/loss": avg_loss, 
                    "fopng/train/rho_avg": avg_rho,
                    "fopng/global_epoch": global_epoch, 
                    "task": t+1
                })
                global_epoch += 1
                
                if verbose: print(f"  epoch {epoch+1}/{epochs} loss={avg_loss:.4f} rho={avg_rho:.4f}")
            fopng.after_task(hyper_network, task_id, loader, criterion)
                
        # ── Evaluate on ALL tasks using TEST loaders ───────────────────
        results[t+1] = []
        eval_metrics = {"task_completed": t+1}
        
        # CHANGED: Iterate over every single task, seen or unseen!
        for i in range(len(test_loaders)): 
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            acc = evaluate_accuracy(hyper_network, test_loaders[i], eval_task_id)
            results[t+1].append(acc)
            eval_metrics[f"fopng/eval/acc_task_{i+1}"] = acc
            if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")
            
        if t != 0:
            bwt = calc_bwt(results)
            eval_metrics["fopng/eval/bwt"] = bwt
            if verbose: print(f"BWT for task {t+1}: {bwt:.4f}")
            
        wandb.log(eval_metrics)

    tasks_completed = sorted(list(results.keys())) # [1, 2, 3]
    num_eval_tasks = len(test_loaders)

    # 1. Log the overlapping FOPNG chart
    plt.figure(figsize=(10, 6))
    
    # Define a clean, distinct color palette
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        # Force solid line (linestyle='-') and cycle through colors
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title("FOPNG Hypernetwork: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    # Log the cleanly colored plot directly to W&B
    wandb.log({"FOPNG Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results