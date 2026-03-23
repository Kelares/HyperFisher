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

from utils import calc_bwt, evaluate_accuracy


def _is_hypernetwork(model: nn.Module) -> bool:
    return hasattr(model, "spawn") and hasattr(model, "task_emb") and hasattr(model, "layers")


class EWC:
    def __init__(
        self,
        lr: float = 1e-3,
        lam: float = 1e5,
        fisher_samples: int = 1024,
        optimizer_cls=torch.optim.Adam,
    ):
        self.lr             = lr
        self.lam            = lam
        self.fisher_samples = fisher_samples
        self.optimizer_cls  = optimizer_cls

        self.fishers:  Dict[int, Tensor] = {}  # task_id -> diagonal Fisher [D]
        self.anchors:  Dict[int, Tensor] = {}  # task_id -> flat params / generated weights
        self._tid_tensors: Dict[int, Tensor] = {}  # task_id -> task_id tensor (hypernetwork only)
        self._opt: Optional[torch.optim.Optimizer] = None

    def build_optimizer(self, model: nn.Module) -> None:
        self._opt = self.optimizer_cls(model.parameters(), lr=self.lr)

    # ── penalty ───────────────────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> Tensor:
        if not self.fishers:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        if _is_hypernetwork(model):
            return self._penalty_hypernetwork(model)
        else:
            return self._penalty_mlp(model)

    def _penalty_mlp(self, model: nn.Module) -> Tensor:
        # θ space: compare current flat params vs saved anchors
        theta = torch.cat([p.view(-1) for p in model.parameters()])
        loss  = torch.tensor(0.0, device=theta.device)
        for tid in self.fishers:
            diff = theta - self.anchors[tid]
            loss = loss + (self.fishers[tid] * diff.pow(2)).sum()
        return (self.lam / 2.0) * loss

    def _penalty_hypernetwork(self, model: nn.Module) -> Tensor:
        # Generated weight space: regenerate w for each past task and compare vs saved anchor
        device = next(model.parameters()).device
        loss   = torch.tensor(0.0, device=device)
        for tid, task_id_tensor in self._tid_tensors.items():
            # Regenerate current weights for this task — keep in graph so
            # grad flows back through model.layers into θ
            t_vec = model.task_emb(task_id_tensor)
            w     = model.layers(t_vec).squeeze()          # [D_target], grad intact
            diff  = w - self.anchors[tid]                  # anchor is detached
            loss  = loss + (self.fishers[tid] * diff.pow(2)).sum()
        return (self.lam / 2.0) * loss / len(self.fishers)

    # ── after_task ────────────────────────────────────────────────────────────

    def after_task(
        self,
        model: nn.Module,
        task_id: int,
        loader: DataLoader,
        task_id_tensor: Optional[Tensor] = None,  # required for hypernetwork
    ) -> None:
        device = next(model.parameters()).device
        model.eval()
        model.zero_grad(set_to_none=True)

        if _is_hypernetwork(model):
            assert task_id_tensor is not None, \
                "task_id_tensor is required for HyperNetwork after_task"
            self._after_task_hypernetwork(model, task_id, task_id_tensor, loader, device)
        else:
            self._after_task_mlp(model, task_id, loader, device)

        model.train()

    def _after_task_mlp(
        self, model: nn.Module, task_id: int, loader: DataLoader, device: torch.device
    ) -> None:
        D      = sum(p.numel() for p in model.parameters())
        fisher = torch.zeros(D, device=device)
        n_seen = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for xi, yi in zip(x, y):
                if n_seen >= self.fisher_samples:
                    break

                model.zero_grad(set_to_none=True)
                logits = model(xi.unsqueeze(0))
                loss   = F.cross_entropy(logits, yi.unsqueeze(0))
                loss.backward()

                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().view(-1))
                    else:
                        grads.append(torch.zeros(p.numel(), device=device))
                fisher += torch.cat(grads).pow(2)
                n_seen += 1

            if n_seen >= self.fisher_samples:
                break

        fisher /= max(n_seen, 1)
        model.zero_grad(set_to_none=True)

        self.fishers[task_id] = fisher
        self.anchors[task_id] = torch.cat(
            [p.data.detach().view(-1) for p in model.parameters()]
        )
        print(f"[EWC/MLP] task {task_id} Fisher — "
              f"min: {fisher.min():.2e}  max: {fisher.max():.2e}  mean: {fisher.mean():.2e}")

    def _after_task_hypernetwork(
        self,
        model: nn.Module,
        task_id: int,
        task_id_tensor: Tensor,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        # Anchor = generated weight vector for this task, detached
        with torch.no_grad():
            t_vec  = model.task_emb(task_id_tensor)
            anchor = model.layers(t_vec).squeeze().clone()
        self.anchors[task_id]      = anchor
        self._tid_tensors[task_id] = task_id_tensor

        # Fisher in generated weight space via autograd.grad(loss, w)
        D_target = anchor.numel()
        fisher   = torch.zeros(D_target, device=device)
        n_seen   = 0

        for x, _ in loader:
            x = x.to(device)
            for xi in x:
                if n_seen >= self.fisher_samples:
                    break

                # Pass 1: sample label
                with torch.no_grad():
                    model.spawn(task_id_tensor)
                    logits = model(xi.unsqueeze(0))
                    y_hat  = torch.distributions.Categorical(logits=logits).sample()

                # Pass 2: fresh graph — w is the differentiable leaf we want Fisher for
                from torch.func import functional_call
                model.zero_grad(set_to_none=True)
                t_vec  = model.task_emb(task_id_tensor)
                w      = model.layers(t_vec).squeeze()          # [D_target], in graph
                logits = functional_call(
                    model.target_network,
                    model.get_params_dict(w),
                    xi.unsqueeze(0),
                )
                loss = F.cross_entropy(logits, y_hat)
                (w_grad,) = torch.autograd.grad(loss, w)
                fisher   += w_grad.detach().pow(2)
                n_seen   += 1

            if n_seen >= self.fisher_samples:
                break

        fisher /= max(n_seen, 1)
        model.zero_grad(set_to_none=True)

        self.fishers[task_id] = fisher
        print(f"[EWC/HyperNet] task {task_id} Fisher — "
              f"min: {fisher.min():.2e}  max: {fisher.max():.2e}  mean: {fisher.mean():.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# train_ewc
# ─────────────────────────────────────────────────────────────────────────────

def train_ewc(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    lam: float = 1e5,
    fisher_samples: int = 1024,
    epochs: int = 5,
    optimizer_cls=torch.optim.Adam,
    task_classes: Optional[list] = None,
    verbose: bool = True,
) -> Dict:
    device   = next(model.parameters()).device
    is_hyper = _is_hypernetwork(model)
    ewc      = EWC(lr=lr, lam=lam, fisher_samples=fisher_samples, optimizer_cls=optimizer_cls)
    ewc.build_optimizer(model)

    results      = {}
    global_epoch = 0

    for t, loader in enumerate(train_loaders):
        task_id_tensor = torch.tensor([t], dtype=torch.long, device=device)
        if verbose:
            print(f"\n[EWC] Task {t+1}")

        for epoch in range(epochs):
            model.train()
            total_loss    = 0.0
            total_penalty = 0.0

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                ewc._opt.zero_grad()

                if is_hyper:
                    model.spawn(task_id_tensor)

                output      = model(x)
                task_loss   = criterion(output, y)
                ewc_penalty = ewc.penalty(model)
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
                print(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  penalty={avg_penalty:.4f}")

        # Register anchor and Fisher for this task
        if is_hyper:
            ewc.after_task(model, t, loader, task_id_tensor=task_id_tensor)
        else:
            ewc.after_task(model, t, loader)

        # ── Evaluate on all tasks ─────────────────────────────────────────
        results[t + 1] = []
        eval_metrics   = {"task_completed": t + 1}

        for i in range(len(test_loaders)):
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            acc = evaluate_accuracy(model, test_loaders[i], eval_task_id if is_hyper else None, task_classes=tc)
            results[t + 1].append(acc)
            eval_metrics[f"ewc/eval/acc_task_{i+1}"] = acc
            if verbose:
                print(f"  Task {i+1} Acc: {acc*100:.1f}%")

        if t != 0:
            bwt = calc_bwt(results, task_id=t + 1)
            eval_metrics["ewc/eval/bwt"] = bwt
            if verbose:
                print(f"  BWT: {bwt:.4f}")

        wandb.log(eval_metrics)

    # ── Plot ──────────────────────────────────────────────────────────────────
    tasks_completed = sorted(results.keys())
    num_eval_tasks  = len(test_loaders)
    plt.figure(figsize=(10, 6))
    cmap   = plt.get_cmap("gist_rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        plt.plot(tasks_completed, accs, marker="o", linestyle="-",
                 linewidth=2.5, color=colors[i], label=str(i + 1))
    plt.title("EWC: All Tasks", fontsize=14, fontweight="bold")
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    wandb.log({"EWC Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results