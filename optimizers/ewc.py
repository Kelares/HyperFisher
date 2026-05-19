from __future__ import annotations

from math import inf
from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import wandb

from utils import calc_bwt, evaluate_accuracy
from hyper_network import HyperRegulizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_protected_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Returns the parameters EWC should anchor and penalise.

    - HyperNetwork  → model._shared_params  (layers + chunk_emb; excludes task_emb)
    - MultiHeadCNN  → model._shared_params  (backbone conv/fc; excludes heads)
    - Plain MLP     → model.parameters()    (all params)
    """
    if hasattr(model, "_shared_params"):
        return list(model._shared_params)
    return list(model.parameters())


def _flat(params: List[nn.Parameter]) -> Tensor:
    return torch.cat([p.view(-1) for p in params])


def get_magnitude_decay_lr(current_lr: float) -> float:
    """1e-2 → 5e-3 → 1e-3 → 5e-4 → …"""
    sci_str       = f"{current_lr:.1e}"
    mantissa, exp = sci_str.split("e")
    mantissa, exp = float(mantissa), int(exp)
    if mantissa >= 4.9:
        return 1.0 * (10 ** exp)
    return 5.0 * (10 ** (exp - 1))


# ─────────────────────────────────────────────────────────────────────────────
# EWC
# ─────────────────────────────────────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation.

    Works uniformly for:
      • Chunked HyperNetwork  (has .spawn, ._shared_params)
      • MultiHeadTargetCNN    (has .spawn, ._shared_params)
      • Plain MLP             (neither attribute → falls back to all params)

    The Fisher diagonal and anchor are always computed over _get_protected_params,
    so task_emb rows (HyperNet) and task-specific heads (MultiHead) are never
    included — they cannot cause cross-task interference.
    """

    __name__ = "EWC"

    def __init__(
        self,
        lr:             float = 1e-3,
        lam:            float = 10,
        fisher_samples: int   = 1024,
        optimizer_cls         = torch.optim.Adam,
    ):
        self.lr             = lr
        self.lam            = lam
        self.fisher_samples = fisher_samples
        self.optimizer_cls  = optimizer_cls

        self.fishers: Dict[int, Tensor] = {}   # task_id → diagonal Fisher [D_shared]
        self.anchors: Dict[int, Tensor] = {}   # task_id → flat anchor     [D_shared]

    # ── Penalty ───────────────────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> Tensor:
        if not self.fishers:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        theta = _flat(_get_protected_params(model))
        loss  = torch.tensor(0.0, device=theta.device)
        for tid in self.fishers:
            diff = theta - self.anchors[tid].to(theta.device)
            loss = loss + (self.fishers[tid].to(theta.device) * diff.pow(2)).sum()
        return (self.lam / 2.0) * loss

    # ── After-task: compute Fisher diagonal + anchor ──────────────────────────

    def after_task(
        self,
        model:     nn.Module,
        task_id:   int,
        task_id_t: Tensor,
        loader:    DataLoader,
        device:    torch.device,
    ) -> None:
        has_spawn = hasattr(model, "spawn")
        params    = _get_protected_params(model)
        D         = sum(p.numel() for p in params)
        fisher    = torch.zeros(D, device=device)
        n_seen    = 0

        model.eval()
        model.zero_grad(set_to_none=True)

        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            for xi in x_batch:
                if n_seen >= self.fisher_samples:
                    break

                model.zero_grad(set_to_none=True)
                if has_spawn:
                    model.spawn(task_id_t)

                logits = model(xi.unsqueeze(0))
                y_hat  = torch.distributions.Categorical(logits=logits).sample()
                loss   = F.cross_entropy(logits, y_hat)
                loss.backward()

                g = torch.cat([
                    p.grad.view(-1) if p.grad is not None
                    else torch.zeros(p.numel(), device=device)
                    for p in params
                ])
                fisher += g.pow(2)
                n_seen += 1

            if n_seen >= self.fisher_samples:
                break

        fisher /= max(n_seen, 1)
        model.zero_grad(set_to_none=True)
        model.train()

        self.fishers[task_id] = fisher
        self.anchors[task_id] = _flat(params).detach().clone()

        print(
            f"[EWC] task {task_id} Fisher — "
            f"min: {fisher.min():.2e}  max: {fisher.max():.2e}  mean: {fisher.mean():.2e}"
        )

    def _normalise_fishers(self) -> None:
        """Normalise each stored Fisher to unit max so earlier tasks
        (which converge to near-zero gradients) are not swamped by later ones."""
        for tid in self.fishers:
            f       = self.fishers[tid]
            max_val = f.max()
            if max_val > 0:
                self.fishers[tid] = f / max_val


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  (mirrors run_continual_method)
# ─────────────────────────────────────────────────────────────────────────────

def run_ewc(
    model:         nn.Module,
    train_loaders: List[DataLoader],
    test_loaders:  List[DataLoader],
    criterion:     Callable,
    config:        dict,
):
    """Unified entry point for EWC — mirrors run_continual_method."""

    return train_ewc(
        model          = model,
        train_loaders  = train_loaders,
        test_loaders   = test_loaders,
        criterion      = criterion,
        regulizer      = regulizer_instance,
        lr             = config.get("lr",             1e-3),
        lam            = config.get("lam",            1e5),
        fisher_samples = config.get("fisher_samples", 1024),
        epochs         = config.get("epochs",         5),
        max_epochs     = config.get("max_epochs",     None),
        task_classes   = config.get("task_classes",   None),
        verbose        = config.get("verbose",        True),
        saved          = config.get("saved",          False),
        warmup         = config.get("warmup",         False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training engine  (mirrors projections.py → train())
# ─────────────────────────────────────────────────────────────────────────────

def train_ewc(
    model:          nn.Module,
    train_loaders:  List[DataLoader],
    test_loaders:   List[DataLoader],
    criterion:      Callable,
    regulizer:      Optional[HyperRegulizer] = None,
    *,
    lr:             float          = 1e-3,
    lam:            float          = 1e5,
    fisher_samples: int            = 1024,
    epochs:         int            = 5,
    max_epochs:     Optional[int]  = None,
    optimizer_cls                  = torch.optim.Adam,
    first_task_optimizer_cls       = torch.optim.AdamW,
    task_classes:   Optional[list] = None,
    verbose:        bool           = True,
    saved:          bool           = False,
    warmup:         bool           = False,
) -> Dict:
    device      = next(model.parameters()).device
    has_spawn   = hasattr(model, "spawn")
    ewc         = EWC(lr=lr, lam=lam, fisher_samples=fisher_samples, optimizer_cls=optimizer_cls)
    name        = ewc.__name__   # "EWC" — used as wandb key prefix
    regulizer = HyperRegulizer() if regulizer else None

    results         = {"acc": {}}
    global_epoch    = 0
    loss_to_achieve = 0.1
    _max_epochs     = max_epochs if max_epochs else epochs
    base_lr         = lr
    save_path       = "weights/ewc_first_run_weights.pt"

    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)

        best_loss           = inf
        loss_repeat         = 0
        lr_patience_counter = 0
        best_parameters     = None
        ewc.lr              = base_lr      # reset LR at the start of every task
        epoch               = 0

        # ── Task 0 ────────────────────────────────────────────────────────────
        if t == 0:
            if not saved:
                if verbose:
                    print(f"\n[{name}] Task 1 – {first_task_optimizer_cls.__name__}")

                opt = first_task_optimizer_cls(
                    model.parameters(), lr=1e-3, weight_decay=1e-4
                )

                while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
                    total_loss = 0.0
                    model.train()

                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        model.zero_grad()
                        if has_spawn:
                            model.spawn(task_id)
                        output = model(x)
                        loss   = criterion(output, y)
                        loss.backward()
                        opt.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(loader)
                    wandb.log({
                        f"{name}/train/loss":   avg_loss,
                        f"{name}/global_epoch": global_epoch,
                        "task":                 t + 1,
                    })
                    global_epoch += 1

                    if verbose:
                        print(f"  epoch {epoch+1}/{_max_epochs}  loss={avg_loss:.4f}")

                    if avg_loss < best_loss:
                        best_loss           = avg_loss
                        lr_patience_counter = 0
                        loss_repeat         = 0
                        best_parameters     = model.state_dict()
                    else:
                        lr_patience_counter += 1
                        loss_repeat         += 1

                    if lr_patience_counter == 3:
                        for g in opt.param_groups:
                            g["lr"] = get_magnitude_decay_lr(g["lr"])
                        lr_patience_counter = 0
                        if verbose:
                            print(f"    [Scheduler] Lowering LR to {opt.param_groups[0]['lr']}")

                    epoch += 1

                model.load_state_dict(best_parameters)
                reason = (
                    f"best_loss: {best_loss}"     if best_loss  < loss_to_achieve else
                    f"loss_repeat: {loss_repeat}" if loss_repeat >= 10            else
                    f"epoch: {epoch}"
                )
                print(f"Task 1 Finished: {reason}")

            else:
                model.load_state_dict(torch.load(save_path, weights_only=True))

        # ── Tasks 1+ ──────────────────────────────────────────────────────────
        else:
            if verbose:
                print(f"\n[{name}] Task {t+1}")

            # ── Optional warmup: freeze shared params, warm up task embedding ─
            if warmup:
                for param in model._shared_params:
                    param.requires_grad = False

                active_params = filter(lambda p: p.requires_grad, model.parameters())
                opt_warmup    = first_task_optimizer_cls(active_params, lr=0.1, weight_decay=1e-4)
                warmup_n      = 5

                for i in range(warmup_n):
                    total_loss = 0.0
                    total_reg  = 0.0
                    model.train()

                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt_warmup.zero_grad()
                        if has_spawn:
                            model.spawn(task_id)
                        output = model(x)
                        loss   = criterion(output, y)
                        if regulizer:
                            w_penalty  = regulizer.loss(model, task_id)
                            total_reg += w_penalty.item()
                            loss       = loss + w_penalty
                        loss.backward()
                        opt_warmup.step()
                        total_loss += loss.item()

                    avg_loss = total_loss / len(loader)
                    avg_reg  = total_reg  / len(loader)
                    if verbose:
                        print(f"  warmup {i+1}/{warmup_n}  loss={avg_loss:.4f}  reg={avg_reg:.4f}")

                for param in model._shared_params:
                    param.requires_grad = True

            # ── Main EWC loop ─────────────────────────────────────────────────
            opt = ewc.optimizer_cls(model.parameters(), lr=ewc.lr)

            while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
                total_loss    = 0.0
                total_penalty = 0.0
                total_reg     = 0.0
                model.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    model.zero_grad()

                    if has_spawn:
                        model.spawn(task_id)

                    output      = model(x)
                    task_loss   = criterion(output, y)
                    ewc_penalty = ewc.penalty(model)
                    loss        = task_loss + ewc_penalty

                    if regulizer:
                        w_penalty  = regulizer.loss(model, task_id)
                        total_reg += w_penalty.item() if torch.is_tensor(w_penalty) else w_penalty
                        loss       = loss + w_penalty

                    loss.backward()
                    opt.step()
                    total_loss    += task_loss.item()
                    total_penalty += ewc_penalty.item()

                n_batches   = len(loader)
                avg_loss    = total_loss    / n_batches
                avg_penalty = total_penalty / n_batches
                avg_reg     = total_reg     / n_batches

                if avg_loss < best_loss:
                    best_loss           = avg_loss
                    lr_patience_counter = 0
                    loss_repeat         = 0
                    best_parameters     = model.state_dict()
                else:
                    lr_patience_counter += 1
                    loss_repeat         += 1

                if lr_patience_counter == 3:
                    ewc.lr = get_magnitude_decay_lr(ewc.lr)
                    for pg in opt.param_groups:
                        pg["lr"] = ewc.lr
                    lr_patience_counter = 0
                    if verbose:
                        print(f"    [Scheduler] Loss stalled. Lowering LR to {ewc.lr}")

                wandb.log({
                    f"{name}/train/loss":        avg_loss,
                    f"{name}/train/ewc_penalty": avg_penalty,
                    f"{name}/train/reg_penalty": avg_reg,
                    f"{name}/global_epoch":      global_epoch,
                    "task":                      t + 1,
                })
                global_epoch += 1

                if verbose:
                    print(
                        f"  epoch {epoch+1}/{_max_epochs}  "
                        f"loss={avg_loss:.4f}  ewc={avg_penalty:.4f}  lr={ewc.lr}"
                    )
                if regulizer:
                    print(f"  reg_loss={avg_reg:.4f}")

                epoch += 1

            model.load_state_dict(best_parameters)

        # ── Compute Fisher + anchor, normalise ────────────────────────────────
        ewc.after_task(model, t, task_id, loader, device)
        ewc._normalise_fishers()

        # ── Snapshot generated weights for the regulizer ──────────────────────
        # Only meaningful for HyperNetwork (has .w after spawn); safe no-op otherwise.
        if regulizer is not None and hasattr(model, "w"):
            model.spawn(task_id)
            regulizer.old_weights[task_id.item()] = model.w.detach()

        # ── Evaluate on ALL tasks using test loaders ──────────────────────────
        results["acc"][t + 1] = []
        eval_metrics = {"task_completed": t + 1}

        for i in range(len(test_loaders)):
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc           = task_classes[i] if task_classes is not None else None
            acc          = evaluate_accuracy(model, test_loaders[i], eval_task_id, task_classes=tc)
            results["acc"][t + 1].append(acc)
            eval_metrics[f"{name}/eval/acc_task_{i+1}"] = acc
            if verbose:
                print(f"  Task {i+1} Acc: {acc*100:.1f}%")

        if t != 0:
            bwt = calc_bwt(results["acc"], task_id=t + 1)
            eval_metrics[f"{name}/eval/bwt"] = bwt
            results["bwt"] = bwt
            if verbose:
                print(f"BWT at task {t+1}: {bwt:.4f}")

        wandb.log(eval_metrics)

    # ── Final plot ────────────────────────────────────────────────────────────
    tasks_completed = sorted(results["acc"].keys())
    num_eval_tasks  = len(test_loaders)

    plt.figure(figsize=(10, 6))
    cmap   = plt.get_cmap("gist_rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]

    for i in range(num_eval_tasks):
        accs = [results["acc"][t][i] for t in tasks_completed]
        plt.plot(
            tasks_completed, accs,
            marker="o", linestyle="-", linewidth=2.5,
            color=colors[i % len(colors)], label=str(i + 1),
        )

    plt.title(f"{name}  Hypernetwork: All Tasks", fontsize=14, fontweight="bold")
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    wandb.log({f"{name}  Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results