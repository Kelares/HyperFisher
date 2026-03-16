"""
FOPNG: Fisher-Orthogonal Projected Natural Gradient Descent
============================================================
Garg, Kolhe, Peng, Gopalam — UC Berkeley (ICML 2026)
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.func import functional_call
import torchvision
from torchvision import transforms
import wandb


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


# ─────────────────────────────────────────────────────────────────────────────
# Diagonal Fisher estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_fisher_diag(
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

    with torch.enable_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            hyper_network.zero_grad()
            
            hyper_network.spawn(task_id)
            output = hyper_network(x)

            loss = criterion(output, y)
            loss.backward()

            g = _flat_grad(hyper_network)
            fisher.add_(g.pow(2))

            n_seen += x.size(0)
            n_batches += 1
            if n_seen >= max_samples:
                break

    hyper_network.zero_grad()
    hyper_network.train()
    return fisher / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def _build_A_inv(
    G: Tensor, F_old: Tensor, F_new: Tensor, lam: float,
) -> Tensor:
    # <--- NEW: Cast to Float64 to prevent catastrophic underflow!
    G_64 = G.to(torch.float64)
    F_old_64 = F_old.to(torch.float64)
    F_new_64 = F_new.to(torch.float64)
    
    fisher_eps = 1e-3  
    F_new_inv = 1.0 / (F_new_64 + fisher_eps)
    scale     = (F_old_64 ** 2) * F_new_inv
    scaled_G  = scale.unsqueeze(1) * G_64
    A         = G_64.t() @ scaled_G
    
    A_inv = torch.linalg.pinv(A, rcond=1e-4)
    return A_inv.to(G.dtype) # <--- Cast back to Float32

def _fopng_update(
    g: Tensor, G: Tensor, F_old: Tensor, F_new: Tensor,
    A_inv: Tensor, lr: float, lam: float, eps: float = 1e-8,
) -> tuple[Tensor, float]:
    fisher_eps = 1e-3  

    # <--- NEW: Cast inputs to Float64 for the projection
    g_64 = g.to(torch.float64)
    G_64 = G.to(torch.float64)
    F_old_64 = F_old.to(torch.float64)
    A_inv_64 = A_inv.to(torch.float64)

    # ── projection ──
    F_old_g  = F_old_64 * g_64
    GtFg     = G_64.t() @ F_old_g
    coeff    = A_inv_64 @ GtFg
    Pg_64    = g_64 - F_old_64 * (G_64 @ coeff)
    
    Pg = Pg_64.to(g.dtype) # <--- Convert back

    g_norm = torch.norm(g)
    Pg_norm = torch.norm(Pg)
    rho = (Pg_norm / (g_norm + eps)).item()

    # ── unit natural gradient ──
    F_new_inv    = 1.0 / (F_new + fisher_eps)
    F_new_inv_Pg = F_new_inv * Pg
    fisher_norm  = torch.sqrt((Pg * F_new_inv_Pg).sum() + eps)
    fisher_norm  = torch.clamp(fisher_norm, min=1.0)

    return -lr * F_new_inv_Pg / fisher_norm, rho


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
        return compute_fisher_diag(model, loader, criterion, self._device, self.fisher_samples)

    def prepare_epoch(self, F_new: Tensor) -> None:
        assert self.F_old is not None, "Call after_task() after task 1 before training task 2."
        self._F_new = F_new
        self._A_inv = _build_A_inv(self.G, self.F_old, F_new, self.lam)

    def step(self, model: nn.Module) -> float:
        assert self._A_inv is not None, "Call prepare_epoch(F_new) before step()."
        g = _flat_grad(model)
        v_star, rho = _fopng_update(
            g=g, G=self.G, F_old=self.F_old, F_new=self._F_new,
            A_inv=self._A_inv, lr=self.lr, lam=self.lam,
        )
        _apply_flat_update(model, v_star)
        model.zero_grad()
        return rho

    def after_task(self, hyper_network: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        device = next(hyper_network.parameters()).device
        self._device = device

        F_new = compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
        if self.F_old is None:
            self.F_old = F_new.clone()
        else:
            self.F_old = (1.0 - self.alpha) * self.F_old + self.alpha * F_new

        new_cols = self._collect_gradients(hyper_network, task_id, loader, criterion)
        self.G   = new_cols if self.G is None else torch.cat([self.G, new_cols], dim=1)

        if self.G.shape[1] > self.max_directions:
            self.G = self.G[:, -self.max_directions:]
            
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

        with torch.enable_grad():
            for x, y in loader:
                if len(grads) >= self.grads_per_task:
                    break
                x, y = x.to(self._device), y.to(self._device)
                hyper_network.zero_grad()
                hyper_network.spawn(task_id)

                output = hyper_network(x)
                loss = criterion(output, y)
                loss.backward()
                grads.append(_flat_grad(hyper_network).clone())
        hyper_network.zero_grad()
        hyper_network.train()
        return torch.stack(grads, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience utilities
# ─────────────────────────────────────────────────────────────────────────────

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
                F_new = compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
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

    fopng_lines = []
    keys = []

    # Format the data for W&B's line_series
    for i in range(num_eval_tasks):
        fopng_lines.append([results[t][i] for t in tasks_completed])
        keys.append(f"Task {i+1} Acc")

    # 1. Log the overlapping FOPNG chart
    wandb.log({
        "FOPNG Overlapping Accuracies": wandb.plot.line(
            xs=tasks_completed,
            ys=fopng_lines,
            keys=keys,
            title="FOPNG: All Tasks",
            xname="task_completed"
        )
    })
    return fopng


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test & Main Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    wandb.init(
        project="FOPNG-Experiments",
        config={
            "lr": 1e-3,           # Adjusted to safe natural gradient range
            "lam": 1e-3,          # Reverted to safe paper default
            "alpha": 0.5,
            "grads_per_task": 40, # Shrunk to fit VRAM
            "max_directions": 120,
            "epochs": 5,
            "num_tasks": 3,
            "input_dim": 784,
            "embedding_dim": 4,
            "num_classes": 10
        }
    )
    config = wandb.config

    # Tell W&B to use 'task_completed' as the x-axis for all eval metrics
    wandb.define_metric("task_completed")
    wandb.define_metric("fopng/eval/*", step_metric="task_completed")
    wandb.define_metric("baseline/eval/*", step_metric="task_completed")

    torch.manual_seed(0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(DEVICE)

    class HyperNetwork(nn.Module):
        def __init__(self, device, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_network = nn.Sequential(
                nn.Linear(config.input_dim, 100), 
                nn.ReLU(),
                nn.Linear(100, config.num_classes),
            ).to(device)

            for param in self.target_network.parameters():
                param.requires_grad = False
                
            num_target_params = sum(p.numel() for p in self.target_network.parameters())

            self.task_emb = nn.Embedding(num_embeddings=config.num_tasks, embedding_dim=config.embedding_dim).to(device)
            shared_dim = 32
            self.shared_context = nn.Parameter(torch.randn(1, shared_dim)).to(device)

            hyper_input_dim = config.embedding_dim + shared_dim
            self.layers = nn.Sequential(
                nn.Linear(hyper_input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, num_target_params)
            ).to(device)

            with torch.no_grad():
                torch.nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.01)
                torch.nn.init.normal_(self.layers[-1].bias, mean=0.0, std=0.1)

            self.target_params = None

        def spawn(self, task_id):
            t_vec = self.task_emb(task_id).to(device)
            shared = self.shared_context.expand(t_vec.shape[0], -1).to(device)
            x = torch.cat([t_vec, shared], dim=1).to(device)
            target_params = self.layers(x).squeeze().to(device)
            self.target_params = self.get_params_dict(target_params)

        def forward(self, x):
            return functional_call(self.target_network, self.target_params, x)

        def get_params_dict(self, flat_params):
            param_dict = {}
            pointer = 0
            for name, param in self.target_network.named_parameters():
                num_param = param.numel()
                param_dict[name] = flat_params[pointer:pointer + num_param].view_as(param)
                pointer += num_param
            return param_dict
        
    hyper_network = HyperNetwork(device)
    criterion = nn.CrossEntropyLoss()

    def make_permuted_mnist_task(task_id, batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        # Load Train and Test splits separately
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if task_id > 0:
            rng = torch.Generator().manual_seed(task_id)
            perm = torch.randperm(784, generator=rng)
            
            # Apply permutation to both splits
            train_dataset.data = train_dataset.data.view(-1, 784)[:, perm].view(-1, 28, 28)
            test_dataset.data = test_dataset.data.view(-1, 784)[:, perm].view(-1, 28, 28)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    # Unpack the returned tuples into separate lists
    datasets = [make_permuted_mnist_task(task_id=t) for t in range(config.num_tasks)]
    train_loaders = [d[0] for d in datasets]
    test_loaders = [d[1] for d in datasets]

    print("\n--- Starting FOPNG Training ---")
    fopng = train_fopng(
        hyper_network, train_loaders, test_loaders, criterion,
        lr=config.lr, lam=config.lam, alpha=config.alpha,
        grads_per_task=config.grads_per_task, max_directions=config.max_directions,
        epochs=config.epochs, verbose=True,
    )

    print("\n" + "=" * 60)
    print("baseline COMPARISON")
    print("=" * 60)

    # Simplified Baseline Model (No hypernetwork, just direct parameters)
    model_baseline = nn.Sequential(
        nn.Linear(config.input_dim, 100), nn.ReLU(),
        nn.Linear(100, config.num_classes),
    ).to(device)
    
    baseline = torch.optim.Adam(model_baseline.parameters(), lr=config.lr)
    results_baseline = {}
    global_epoch_baseline = 0
    
    for t, loader in enumerate(train_loaders):
        for epoch in range(config.epochs):
            total_loss = 0.0
            model_baseline.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                baseline.zero_grad()
                loss = criterion(model_baseline(x), y)
                loss.backward()
                baseline.step()
                total_loss += loss.item()

            avg_loss = total_loss/len(loader)
            wandb.log({"baseline/train/loss": avg_loss, "baseline/global_epoch": global_epoch_baseline, "task": t+1})
            global_epoch_baseline += 1
            print(f"  epoch {epoch+1}/{config.epochs}  loss={avg_loss:.4f}")

        # ── Evaluate on ALL tasks using TEST loaders ───────────────────
        results_baseline[t+1] = []
        eval_metrics_baseline = {"task_completed": t+1}
        
        # CHANGED: Iterate over every single task, seen or unseen!
        for i in range(len(test_loaders)):
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            acc = evaluate_accuracy(model_baseline, test_loaders[i], eval_task_id)
            results_baseline[t+1].append(acc)
            eval_metrics_baseline[f"baseline/eval/acc_task_{i+1}"] = acc
            print(f"  Task {i+1}: {acc*100:.1f}%")
            
        if t != 0:
            bwt_baseline = calc_bwt(results_baseline)
            eval_metrics_baseline["baseline/eval/bwt"] = bwt_baseline
            print(f"BWT for task {t+1}: {bwt_baseline:.4f}")
            
        wandb.log(eval_metrics_baseline)

    # ─────────────────────────────────────────────────────────────────────────────
    # Force W&B to generate overlapping Custom Charts
    # ─────────────────────────────────────────────────────────────────────────────
    tasks_completed = sorted(list(results_baseline.keys())) # [1, 2, 3]
    num_eval_tasks = len(test_loaders)

    baseline_lines = []
    keys = []

    # Format the data for W&B's line_series
    for i in range(num_eval_tasks):
        baseline_lines.append([results_baseline[t][i] for t in tasks_completed])
        keys.append(f"Task {i+1} Acc")

    # 1. Log the overlapping Baseline chart
    wandb.log({
        "Baseline Overlapping Accuracies": wandb.plot.line_series(
            xs=tasks_completed,
            ys=baseline_lines,
            keys=keys,
            title="Baseline (Adam): All Tasks",
            xname="task_completed"
        )
    })

    wandb.finish()