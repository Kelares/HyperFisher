from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import numpy as np

from utils import _flat_grad, _apply_flat_update, calc_bwt, evaluate_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# EWC Class  —  drop-in replacement / comparison baseline for FOPNG
# ─────────────────────────────────────────────────────────────────────────────
class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Interface is intentionally parallel to FOPNG so the two can be swapped
    with minimal changes to your training script.

    Key differences from FOPNG
    ──────────────────────────
    • No gradient memory matrix G.
    • No natural-gradient projection.  A plain SGD step is taken on
      loss_task + ewc_penalty, where the penalty anchors weights toward
      the previous task's optimum using the diagonal Fisher as importance.
    • `step()` here expects the *total* loss (task + penalty) to already
      have been back-propped, and simply calls the inner Adam/SGD update.

    Fisher computation
    ──────────────────
    Uses the true diagonal Fisher:
        F_i = E_{x, y~p(y|x)} [(d log p(y|x) / d theta_i)^2]
    Computed per-sample (not batch-mean) and normalised by n_samples.
    criterion argument is kept for interface compatibility but not used
    internally — NLL with model-sampled labels is used instead.

    Penalty
    ───────
    Summed independently per task (not via EMA accumulation) to prevent
    the penalty from growing unboundedly as more tasks are seen:
        Ω = (λ/2) Σ_t Σ_i F^t_i (θ_i − θ*^t_i)²

    Usage
    ─────
        ewc = EWC(lr=1e-3, lam=100)

        # Task 0 — standard training, then register anchor
        for epoch in ...:
            loss.backward(); ewc.step(model)
        ewc.after_task(model, task_id, loader, criterion)

        # Task t>0 — add penalty to loss before backward
        for epoch in ...:
            loss = criterion(model(x), y) + ewc.penalty(model)
            loss.backward()
            ewc.step(model)
        ewc.after_task(model, task_id, loader, criterion)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        lam: float = 100.0,         # EWC regularisation strength
        alpha: float = 0.5,         # EMA weight for Fisher (diagnostics only)
        fisher_samples: int = 1024,
        optimizer_cls=torch.optim.Adam,
    ):
        self.lr             = lr
        self.lam            = lam
        self.alpha          = alpha
        self.fisher_samples = fisher_samples
        self.optimizer_cls  = optimizer_cls

        # Per-task Fisher and anchors — independent per task to avoid
        # penalty snowballing with EMA accumulation
        self.fishers: Dict[int, Tensor] = {}   # task_id -> diagonal Fisher [D]
        self.anchors: Dict[int, Tensor] = {}   # task_id -> theta* [D]

        # EMA Fisher kept only for diagnostics / overlap logging (mirrors FOPNG)
        self.F_accum: Optional[Tensor] = None

        self.all_fishers: List[Tensor] = []
        self._device: Optional[torch.device] = None
        self._opt: Optional[torch.optim.Optimizer] = None

    # ── public helpers ────────────────────────────────────────────────────────

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create (or recreate) the inner parameter optimizer."""
        self._opt = self.optimizer_cls(model.parameters(), lr=self.lr)
        return self._opt

    def penalty(self, model: nn.Module) -> Tensor:
        """
        EWC quadratic penalty — summed independently over all seen tasks:
            Ω = (λ/2) Σ_t Σ_i F^t_i (θ_i − θ*^t_i)²

        Returns a scalar tensor. Returns 0 before the first after_task call.
        """
        if not self.fishers:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device)

        theta = self._get_flat_params(model)
        loss  = torch.tensor(0.0, device=theta.device)
        for tid in self.fishers:
            diff = theta - self.anchors[tid]
            loss = loss + (self.fishers[tid] * diff.pow(2)).sum()
        return (self.lam / 2.0) * loss

    def step(self, model: nn.Module) -> None:
        """
        Apply one gradient step.  The caller is responsible for calling
        loss.backward() (including the EWC penalty) before this.

        Mirrors FOPNG.step() — returns nothing (no rho analogue).
        """
        assert self._opt is not None, "Call build_optimizer(model) before step()."
        self._opt.step()
        self._opt.zero_grad()

    def after_task(
        self,
        hyper_network: nn.Module,
        task_id,
        loader: DataLoader,
        criterion: Callable,
    ) -> None:
        """
        Called once after a task finishes.  Computes the diagonal Fisher,
        saves the current parameters as the new anchor θ*, and accumulates
        the EMA Fisher for logging.

        Signature matches FOPNG.after_task() exactly.
        """
        device = next(hyper_network.parameters()).device
        self._device = device

        # Fully sever any lingering computation graph from the training loop
        hyper_network.zero_grad(set_to_none=True)

        F_new = self.compute_fisher_diag(
            hyper_network, task_id, loader, criterion, device
        )
        self.all_fishers.append(F_new.cpu())

        # ── Fisher overlap diagnostics (same as FOPNG) ────────────────────
        if self.F_accum is not None:
            cosine_sim   = self._cosine_similarity(self.F_accum, F_new)
            pearson_corr = self._pearson_correlation(self.F_accum, F_new)
            topk_iou     = self._calculate_topk_iou(self.F_accum, F_new)
        else:
            cosine_sim = pearson_corr = topk_iou = 1.0

        # ── EMA Fisher for diagnostics only ───────────────────────────────
        if self.F_accum is None:
            self.F_accum = F_new.clone()
        else:
            self.F_accum = (1.0 - self.alpha) * self.F_accum + self.alpha * F_new

        # ── Store per-task Fisher and anchor ──────────────────────────────
        tid = task_id.item() if isinstance(task_id, Tensor) else int(task_id)
        self.fishers[tid] = F_new.clone()
        self.anchors[tid] = self._get_flat_params(hyper_network).detach().clone()

        # ── Logging ───────────────────────────────────────────────────────
        logs = {
            "ewc/fisher/min":              self.F_accum.min().item(),
            "ewc/fisher/max":              self.F_accum.max().item(),
            "ewc/fisher/mean":             self.F_accum.mean().item(),
            "ewc/fisher_overlap/cosine":   cosine_sim,
            "ewc/fisher_overlap/pearson":  pearson_corr,
            "ewc/fisher_overlap/topk_iou": topk_iou,
            "task_completed":              tid + 1,
        }
        print(logs)
        wandb.log(logs)

    # ── Fisher computation ────────────────────────────────────────────────────

    def compute_fisher_diag(
        self,
        model: nn.Module,
        task_id,
        loader: DataLoader,
        criterion: Callable,    # kept for interface compatibility, not used
        device: torch.device,
        max_samples: Optional[int] = None,
    ) -> Tensor:
        """
        True diagonal Fisher via per-sample gradients:
            F_i = E_{x, y~p(y|x)} [(d log p(y|x) / d theta_i)^2]

        Two forward passes per sample:
          1. torch.no_grad() pass to sample y_hat from model's distribution
          2. Fresh grad-enabled pass to compute NLL loss and backprop

        This guarantees each iteration has a fully independent computation
        graph, avoiding the 'backward through graph a second time' error.
        Normalised by n_samples (not n_batches).
        """
        max_samples = max_samples or self.fisher_samples
        model.eval()
        model.zero_grad(set_to_none=True)

        D      = sum(p.numel() for p in model.parameters())
        fisher = torch.zeros(D, device=device)
        n_seen = 0

        if hasattr(model, "spawn"):
            model.spawn(task_id)

        for x, _ in loader:
            x = x.to(device)
            # Don't overshoot max_samples
            x = x[:max_samples - n_seen]

            for xi in x:
                # ── Pass 1: sample label from model (no graph) ────────────
                with torch.no_grad():
                    logits_det = model(xi.unsqueeze(0))
                    y_hat      = torch.distributions.Categorical(
                                     logits=logits_det
                                 ).sample()

                # ── Pass 2: independent forward + backward for Fisher ─────
                # xi.detach().clone() gives this sample its own graph root
                model.zero_grad(set_to_none=True)
                logits = model(xi.detach().clone().unsqueeze(0))
                loss   = F.nll_loss(F.log_softmax(logits, dim=1), y_hat)
                loss.backward()

                g = _flat_grad(model)
                fisher.add_(g.pow(2))
                n_seen += 1

            if n_seen >= max_samples:
                break

        model.zero_grad(set_to_none=True)
        model.train()
        # Normalise by n_samples, not n_batches
        return fisher / max(n_seen, 1)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_flat_params(model: nn.Module) -> Tensor:
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def _cosine_similarity(self, F_a: Tensor, F_b: Tensor) -> float:
        a, b   = F_a.view(-1), F_b.view(-1)
        dot    = torch.dot(a, b)
        norm_a = torch.norm(a, p=2)
        norm_b = torch.norm(b, p=2)
        return (dot / (norm_a * norm_b + 1e-8)).item()

    def _pearson_correlation(self, F_a: Tensor, F_b: Tensor) -> float:
        a, b  = F_a.view(-1).float(), F_b.view(-1).float()
        a_c   = a - a.mean()
        b_c   = b - b.mean()
        cov   = (a_c * b_c).sum()
        denom = torch.sqrt((a_c**2).sum() * (b_c**2).sum()) + 1e-8
        return (cov / denom).item()

    def _calculate_topk_iou(
        self, F_a: Tensor, F_b: Tensor, k_fraction: float = 0.10
    ) -> float:
        a, b = F_a.view(-1), F_b.view(-1)
        k    = max(1, int(a.numel() * k_fraction))
        _, ia     = torch.topk(a, k)
        _, ib     = torch.topk(b, k)
        combined  = torch.cat([ia, ib])
        _, counts = combined.unique(return_counts=True)
        inter     = (counts > 1).sum().item()
        union     = 2 * k - inter
        return inter / union


# ─────────────────────────────────────────────────────────────────────────────
# train_ewc  —  mirrors train_fopng() exactly
# ─────────────────────────────────────────────────────────────────────────────
def train_ewc(
    hyper_network: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    lam: float = 100.0,
    alpha: float = 0.5,
    fisher_samples: int = 1024,
    epochs: int = 5,
    first_task_optimizer_cls=torch.optim.Adam,
    verbose: bool = True,
) -> Dict:
    device = next(hyper_network.parameters()).device
    ewc = EWC(
        lr=lr,
        lam=lam,
        alpha=alpha,
        fisher_samples=fisher_samples,
        optimizer_cls=first_task_optimizer_cls,
    )
    ewc.build_optimizer(hyper_network)

    results      = {}
    global_epoch = 0

    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)

        if t == 0:
            # ── Task 0: plain supervised training (no penalty yet) ────────
            if verbose:
                print(f"[EWC] Task 1 – {first_task_optimizer_cls.__name__}")

            for epoch in range(epochs):
                total_loss = 0.0
                hyper_network.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    ewc._opt.zero_grad()

                    if hasattr(hyper_network, "spawn"):
                        hyper_network.spawn(task_id)

                    output = hyper_network(x)
                    loss   = criterion(output, y)
                    loss.backward()
                    ewc._opt.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                wandb.log({
                    "ewc/train/loss":   avg_loss,
                    "ewc/global_epoch": global_epoch,
                    "task":             t + 1,
                })
                global_epoch += 1
                if verbose:
                    print(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

            ewc.after_task(hyper_network, task_id, loader, criterion)

        else:
            # ── Task t>0: training with EWC penalty ───────────────────────
            if verbose:
                print(f"\n[EWC] Task {t+1}")

            for epoch in range(epochs):
                total_loss    = 0.0
                total_penalty = 0.0
                hyper_network.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    ewc._opt.zero_grad()

                    if hasattr(hyper_network, "spawn"):
                        hyper_network.spawn(task_id)

                    output      = hyper_network(x)
                    task_loss   = criterion(output, y)
                    ewc_penalty = ewc.penalty(hyper_network)
                    loss        = task_loss + ewc_penalty

                    loss.backward()
                    ewc._opt.step()

                    total_loss    += task_loss.item()
                    total_penalty += ewc_penalty.item()

                avg_loss    = total_loss    / len(loader)
                avg_penalty = total_penalty / len(loader)

                wandb.log({
                    "ewc/train/loss":        avg_loss,
                    "ewc/train/ewc_penalty": avg_penalty,
                    "ewc/global_epoch":      global_epoch,
                    "task":                  t + 1,
                })
                global_epoch += 1

                if verbose:
                    print(
                        f"  epoch {epoch+1}/{epochs}  "
                        f"loss={avg_loss:.4f}  penalty={avg_penalty:.4f}"
                    )

            ewc.after_task(hyper_network, task_id, loader, criterion)

        # ── Evaluate on ALL tasks (same structure as train_fopng) ─────────
        results[t + 1] = []
        eval_metrics   = {"task_completed": t + 1}

        for i in range(len(test_loaders)):
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            acc = evaluate_accuracy(hyper_network, test_loaders[i], eval_task_id)
            results[t + 1].append(acc)
            eval_metrics[f"ewc/eval/acc_task_{i+1}"] = acc
            if verbose:
                print(f"  Task {i+1} Acc: {acc*100:.1f}%")

        if t != 0:
            bwt = calc_bwt(results, task_id=t + 1)
            eval_metrics["ewc/eval/bwt"] = bwt
            if verbose:
                print(f"BWT for task {t+1}: {bwt:.4f}")

        wandb.log(eval_metrics)

    # ── Final accuracy plot ───────────────────────────────────────────────────
    tasks_completed = sorted(results.keys())
    num_eval_tasks  = len(test_loaders)

    plt.figure(figsize=(10, 6))
    cmap   = plt.get_cmap("gist_rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]

    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        plt.plot(
            tasks_completed, accs,
            marker="o", linestyle="-", linewidth=2.5,
            color=colors[i % len(colors)], label=f"{i+1}",
        )

    plt.title("EWC Hypernetwork: All Tasks", fontsize=14, fontweight="bold")
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    wandb.log({"EWC Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results