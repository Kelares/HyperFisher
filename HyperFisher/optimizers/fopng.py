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
        self.debug = 1

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
        """
        Compute diagonal Fisher in generated weight space for the ACTIVE head.
        Then scatter it to the GLOBAL parameter dimension.
        """
        hyper_network.eval()
        
        # 1. Spawn and get the specific size for this task's active parameters
        hyper_network.spawn(task_id)
        D_task = hyper_network.w.shape[0]
        fisher_task = torch.zeros(D_task, device=device)
        
        n_seen = 0
        with torch.enable_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                hyper_network.zero_grad()

                # Fresh graph each batch
                hyper_network.spawn(task_id)   
                w = hyper_network.w
                output = hyper_network(x, task_id=task_id.item())
                
                # Auto-remap labels to binary if using a 2-class head
                if output.shape[1] == 2:
                    y = y % 2

                loss = criterion(output, y)
                (g_w,) = torch.autograd.grad(loss, w)
                fisher_task.add_(g_w.detach().pow(2))

                n_seen += x.size(0)
                if n_seen >= max_samples:
                    break

        hyper_network.zero_grad()
        hyper_network.train()

        # 2. Scatter Task-Fisher into Global-Fisher
        fisher_global = torch.zeros(hyper_network.num_target_params, device=device)
        active_indices = hyper_network.get_active_indices(task_id.item())
        fisher_global[active_indices] = fisher_task

        # 3. Global Normalization & Power Smoothing
        with torch.no_grad():
            if fisher_global.max() > 0:
                fisher_global = fisher_global / fisher_global.max()
            
            # Inflation factor. 0.5 works well for CIFAR-10 complexity
            fisher_global = torch.pow(fisher_global, 0.5) 

        return fisher_global
    
    def prepare_epoch(self, F_new: Tensor) -> None:
        assert self.F_old is not None, "Call after_task() after task 1 before training task 2."
        self._F_new = F_new
        # Build A_inv using the GLOBAL matrices
        self._A_inv = self._build_A_inv(self.G, self.F_old, F_new, self.lam)

    def step(self, model: nn.Module, task_id, g_w: Tensor) -> float:
        assert self._A_inv is not None, "Call prepare_epoch(F_new) before step()."

        # ── 1. Slice global matrices to match current active task parameters ──
        active_idx = model.get_active_indices(task_id.item())
        F_old_task = self.F_old[active_idx]
        F_new_task = self._F_new[active_idx]
        G_task = self.G[active_idx, :]

        # ── 2. Project in task-specific w-space ───────────────────────────────
        v_star_w, rho = self._fopng_update(
            g=g_w, G=G_task, F_old=F_old_task, F_new=F_new_task,
            A_inv=self._A_inv, lr=self.lr, lam=self.lam,
        )

        # ── 3. Map v_star_w to θ-space via Jᵀ ─────────────────────────────────
        model.zero_grad()
        model.spawn(task_id)
        model.w.backward(v_star_w.detach())

        # ── 4. Normalise to cancel Jacobian amplification ─────────────────────
        jt_v = torch.cat([
            p.grad.view(-1) for p in model.parameters() if p.grad is not None
        ])
        jt_v_norm  = jt_v.norm()
        v_star_norm = v_star_w.norm()

        scale = (v_star_norm / (jt_v_norm + 1e-8)).item()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad * scale)

        model.zero_grad()
        return rho

    def after_task(self, hyper_network: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        device = next(hyper_network.parameters()).device
        self._device = device

        # F_new is now Global sized
        F_new = self.compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
        
        if self.F_old is not None:
            cosine_sim = self._cosine_similarity(self.F_old, F_new)
            pearson_corr = self._pearson_correlation(self.F_old, F_new)
            topk_iou = self._calculate_topk_iou(self.F_old, F_new)
        else:
            cosine_sim, pearson_corr, topk_iou = 1.0, 1.0, 1.0

        if self.F_old is None:
            self.F_old = F_new.clone()
        else:
            self.F_old = (1.0 - self.alpha) * self.F_old + self.alpha * F_new

        # ── Collect and Scatter Gradients (G) ─────────────────────────────────
        new_cols_task = self._collect_gradients(hyper_network, task_id, loader, criterion)
        
        new_cols_global = torch.zeros(hyper_network.num_target_params, self.grads_per_task, device=device)
        active_idx = hyper_network.get_active_indices(task_id.item())
        new_cols_global[active_idx, :] = new_cols_task

        self.G = new_cols_global if self.G is None else torch.cat([self.G, new_cols_global], dim=1)

        # ── Prune G if it exceeds max directions ──────────────────────────────
        if self.G.shape[1] > self.max_directions:
            indices = torch.linspace(
                0, self.G.shape[1] - 1, 
                steps=self.max_directions, 
                dtype=torch.long, 
                device=self._device
            )
            self.G = self.G[:, indices]

        logs = {
            "fopng/fisher/min": self.F_old.min().item(),
            "fopng/fisher/max": self.F_old.max().item(),
            "fopng/fisher/mean": self.F_old.mean().item(),
            "fopng/memory/G_cols": self.G.shape[1],
            "fopng/fisher_overlap/cosine": cosine_sim,
            "fopng/fisher_overlap/pearson": pearson_corr,
            "fopng/fisher_overlap/topk_iou": topk_iou,
            "task_completed": task_id.item() + 1
        }
        print(logs)
        wandb.log(logs)

    def _collect_gradients(self, hyper_network: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> Tensor:
        grads: List[Tensor] = []
        hyper_network.eval()

        with torch.enable_grad():
            while len(grads) < self.grads_per_task:
                for x, y in loader:
                    if len(grads) >= self.grads_per_task:
                        break
                    x, y = x.to(self._device), y.to(self._device)
                    hyper_network.zero_grad()

                    hyper_network.spawn(task_id)
                    w = hyper_network.w
                    output = hyper_network(x, task_id=task_id.item())
                    
                    if output.shape[1] == 2:
                        y = y % 2

                    loss = criterion(output, y)
                    (g_w,) = torch.autograd.grad(loss, w)
                    grads.append(g_w.detach().clone())

        hyper_network.zero_grad()
        hyper_network.train()
        return torch.stack(grads, dim=1)

    def _fopng_update(
        self,
        g: Tensor,
        G: Tensor,
        F_old: Tensor,
        F_new: Tensor,
        A_inv: Tensor,
        lr: float,
        lam: float,
        eps: float = 1e-8,
    ) -> Tensor:
        F_old_g  = F_old * g
        GtFg     = G.t() @ F_old_g
        coeff    = A_inv @ GtFg
        Pg       = g - F_old * (G @ coeff)

        g_norm = torch.norm(g)
        Pg_norm = torch.norm(Pg)
        rho = (Pg_norm / (g_norm + 1e-8)).item()

        F_new_inv    = 1.0 / (F_new + lam)
        F_new_inv_Pg = F_new_inv * Pg
        fisher_norm = torch.sqrt((Pg * F_new_inv_Pg).sum() + eps)

        return -lr * F_new_inv_Pg / fisher_norm, rho

    def _build_A_inv(self, G: Tensor, F_old: Tensor, F_new: Tensor, lam: float) -> Tensor:
        F_new_inv = 1.0 / (F_new + lam)
        scale     = (F_old ** 2) * F_new_inv
        scale = torch.clamp(scale, max=1e4) 
        scaled_G  = scale.unsqueeze(1) * G
        A         = G.t() @ scaled_G
        A         = A + lam * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.pinv(A)
    
    def _cosine_similarity(self, F_a, F_b):
        F_a_flat = F_a.view(-1)
        F_b_flat = F_b.view(-1)
        dot_product = torch.dot(F_a_flat, F_b_flat)
        norm_a = torch.norm(F_a_flat, p=2)
        norm_b = torch.norm(F_b_flat, p=2)
        return (dot_product / (norm_a * norm_b + 1e-8)).item()

    def _pearson_correlation(self, F_a, F_b):
        F_a_flat = F_a.view(-1)
        F_b_flat = F_b.view(-1)
        mean_a = torch.mean(F_a_flat)
        mean_b = torch.mean(F_b_flat)
        A_centered = F_a_flat - mean_a
        B_centered = F_b_flat - mean_b
        covariance = torch.sum(A_centered * B_centered)
        var_a = torch.sum(A_centered ** 2)
        var_b = torch.sum(B_centered ** 2)
        pearson_r = covariance / (torch.sqrt(var_a * var_b) + 1e-8)
        return pearson_r.item()

    def _calculate_topk_iou(self, F_a, F_b, k_fraction=0.10):
        F_a = F_a.view(-1)
        F_b = F_b.view(-1)
        total_params = F_a.numel()
        k = int(total_params * k_fraction)
        if k == 0: return 0.0
        _, indices_a = torch.topk(F_a, k)
        _, indices_b = torch.topk(F_b, k)
        combined_indices = torch.cat((indices_a, indices_b))
        _, counts = combined_indices.unique(return_counts=True)
        intersection_size = (counts > 1).sum().item()
        union_size = (2 * k) - intersection_size
        return intersection_size / (union_size + 1e-8)


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
    task_classes: Optional[list] = None,
    verbose: bool = True,
) -> dict:
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
                    output = hyper_network(x, task_id=t)
                    
                    if output.shape[1] == 2:
                        y = y % 2
                        
                    loss = criterion(output, y)
                    loss.backward()
                    
                    # Optional clipping to prevent initial HNet explosions
                    torch.nn.utils.clip_grad_norm_(hyper_network.parameters(), max_norm=1.0)
                    
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
                    hyper_network.zero_grad()

                    hyper_network.spawn(task_id)
                    w = hyper_network.w
                    output = hyper_network(x, task_id=t)
                    
                    if output.shape[1] == 2:
                        y = y % 2
                        
                    loss = criterion(output, y)
                    total_loss += loss.item()

                    (g_w,) = torch.autograd.grad(loss, w)
                    rho = fopng.step(hyper_network, task_id, g_w.detach())
                    total_rho += rho

                avg_loss = total_loss / len(loader)
                avg_rho  = total_rho  / len(loader)

                wandb.log({
                    "fopng/train/loss":    avg_loss,
                    "fopng/train/rho_avg": avg_rho,
                    "fopng/global_epoch":  global_epoch,
                    "task":                t + 1,
                })
                global_epoch += 1

                if verbose: print(f"  epoch {epoch+1}/{epochs} loss={avg_loss:.4f} rho={avg_rho:.4f}")
            fopng.after_task(hyper_network, task_id, loader, criterion)
                
        # ── Evaluate on ALL tasks using TEST loaders ───────────────────
        results[t+1] = []
        eval_metrics = {"task_completed": t+1}
        tc = task_classes[t] if task_classes is not None else None
        acc = evaluate_accuracy(hyper_network, test_loaders[t], task_id, task_classes=tc)
        results[t+1].append(acc)
        eval_metrics[f"fopng/eval/acc_task_{t+1}"] = acc
        if verbose: print(f"  Task {t+1} Acc: {acc*100:.1f}%")

        # for i in range(len(test_loaders)): 
        #     eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
        #     tc = task_classes[i] if task_classes is not None else None
        #     # IMPORTANT: Ensure evaluate_accuracy in utils.py handles task_id=eval_task_id.item()
        #     # and applies y = y % 2 if output.shape[1] == 2!
        #     acc = evaluate_accuracy(hyper_network, test_loaders[i], eval_task_id, task_classes=tc)
        #     results[t+1].append(acc)
        #     eval_metrics[f"fopng/eval/acc_task_{i+1}"] = acc
        #     if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")
            
        if t != 0:
            bwt = calc_bwt(results, task_id=t+1)
            eval_metrics["fopng/eval/bwt"] = bwt
            if verbose: print(f"BWT for task {t+1}: {bwt:.4f}")
            
        wandb.log(eval_metrics)

    tasks_completed = sorted(list(results.keys()))
    num_eval_tasks = len(test_loaders)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title("FOPNG Hypernetwork: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    wandb.log({"FOPNG Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results