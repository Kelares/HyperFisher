from __future__ import annotations

from typing import Callable, Dict, List, Optional
from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Assuming calc_bwt and evaluate_accuracy are implemented in utils
from utils import calc_bwt, evaluate_accuracy


def _is_hypernetwork(model: nn.Module) -> bool:
    return hasattr(model, "spawn") and hasattr(model, "chunk_emb") and hasattr(model, "layers")


def get_magnitude_decay_lr(current_lr: float) -> float:
    """
    Decays LR perfectly through magnitudes:
    1e-2 -> 5e-3 -> 1e-3 -> 5e-4 -> 1e-4 ...
    """
    sci_str = f"{current_lr:.1e}"
    mantissa, exp = sci_str.split('e')
    mantissa = float(mantissa)
    exp = int(exp)
    
    if mantissa >= 4.9:
        return 1.0 * (10 ** exp)
    else:
        return 5.0 * (10 ** (exp - 1))


class EWC:
    def __init__(
        self,
        lam: float = 1e5,
        fisher_samples: int = 1024,
    ):
        self.lam            = lam
        self.fisher_samples = fisher_samples

        self.fishers:  Dict[int, Tensor] = {}  # task_id -> diagonal Fisher [D]
        self.anchors:  Dict[int, Tensor] = {}  # task_id -> flat params / generated weights
        self._tid_tensors: Dict[int, Tensor] = {}  # task_id -> task_id tensor (hypernetwork only)

    # ── penalty ───────────────────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> Tensor:
        if not self.fishers:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        if _is_hypernetwork(model):
            return self._penalty_hypernetwork(model)
        else:
            return self._penalty_mlp(model)

    def _penalty_mlp(self, model: nn.Module) -> Tensor:
        theta = torch.cat([p.view(-1) for p in model.parameters()])
        loss  = torch.tensor(0.0, device=theta.device)
        for tid in self.fishers:
            diff = theta - self.anchors[tid]
            loss = loss + (self.fishers[tid] * diff.pow(2)).sum()
        return (self.lam / 2.0) * loss

    def _penalty_hypernetwork(self, model: nn.Module) -> Tensor:
            device = next(model.parameters()).device
            loss   = torch.tensor(0.0, device=device)
            
            for tid, task_id_tensor in self._tid_tensors.items():
                t_emb = model.task_emb(task_id_tensor).view(-1)
                t_vec = t_emb.repeat(model.num_of_chunks, 1)
                
                chunk_ids = torch.arange(model.num_of_chunks, device=device)
                c_vec = model.chunk_emb(chunk_ids)
                
                x = torch.cat([t_vec, c_vec], dim=1)
                flat_w = model.layers(x).view(-1)
                w = flat_w[:model.num_target_params] 
                
                diff  = w - self.anchors[tid]         
                
                # THE FIX: Use .mean() instead of .sum() to prevent the 
                # 102,602 parameter sum from causing a massive gradient explosion.
                loss  = loss + (self.fishers[tid] * diff.pow(2)).mean()
                
            return (self.lam / 2.0) * loss / len(self.fishers)

    # ── after_task ────────────────────────────────────────────────────────────

    def after_task(
        self,
        model: nn.Module,
        task_id: int,
        loader: DataLoader,
        task_id_tensor: Optional[Tensor] = None, 
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
        with torch.no_grad():
            t_emb = model.task_emb(task_id_tensor).view(-1)
            t_vec = t_emb.repeat(model.num_of_chunks, 1)
            
            chunk_ids = torch.arange(model.num_of_chunks, device=device)
            c_vec = model.chunk_emb(chunk_ids)
            
            x = torch.cat([t_vec, c_vec], dim=1)
            flat_w = model.layers(x).view(-1)
            anchor = flat_w[:model.num_target_params].clone()
            
        self.anchors[task_id]      = anchor
        self._tid_tensors[task_id] = task_id_tensor

        D_target = anchor.numel()
        fisher   = torch.zeros(D_target, device=device)
        n_seen   = 0

        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            for xi in x_batch:
                if n_seen >= self.fisher_samples:
                    break

                with torch.no_grad():
                    model.spawn(task_id_tensor)
                    logits = model(xi.unsqueeze(0))
                    y_hat  = torch.distributions.Categorical(logits=logits).sample()

                from torch.func import functional_call
                model.zero_grad(set_to_none=True)
                
                t_emb = model.task_emb(task_id_tensor).view(-1)
                t_vec = t_emb.repeat(model.num_of_chunks, 1)
                c_vec = model.chunk_emb(torch.arange(model.num_of_chunks, device=device))
                
                x_cat = torch.cat([t_vec, c_vec], dim=1)
                w = model.layers(x_cat).view(-1)[:model.num_target_params]

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
    max_epochs: int = None,
    first_task_optimizer_cls=torch.optim.SGD,
    task_classes: Optional[list] = None,
    verbose: bool = True,
) -> Dict:
    device   = next(model.parameters()).device
    is_hyper = _is_hypernetwork(model)
    ewc      = EWC(lam=lam, fisher_samples=fisher_samples)

    results      = {}
    global_epoch = 0

    for t, loader in enumerate(train_loaders):
        task_id_tensor = torch.tensor([t], dtype=torch.long, device=device)
        best_loss = inf
        loss_repeat = 0
        _max_epochs = max_epochs if max_epochs else epochs
        lr_patience_counter = 0
        best_parameters = None
        base_lr = lr
        epoch = 0

        # ── Task 1 ─────────────────────────────────────────────────────────
        if t == 0:
            if verbose: print(f"[EWC] Task 1 – {first_task_optimizer_cls.__name__}")
            opt = first_task_optimizer_cls(model.parameters(), lr=base_lr, weight_decay=1e-4)
            loss_to_achieve = 0.2
            
            while best_loss >= loss_to_achieve and loss_repeat < 8 and epoch < _max_epochs:
                total_loss = 0.0
                model.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    if is_hyper: model.spawn(task_id_tensor)
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                wandb.log({"ewc/train/loss": avg_loss, "ewc/global_epoch": global_epoch, "task": t+1})
                global_epoch += 1
                if verbose: print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f}")

                # REDUCE LR ON PLATEAU
                if lr_patience_counter >= 5:
                    for g in opt.param_groups:
                        g['lr'] = get_magnitude_decay_lr(g['lr'])
                    lr_patience_counter = 0 
                    if verbose: print(f"    [Scheduler] Loss stalled. Halving LR to {g['lr']}")
    
                if best_loss < avg_loss:
                    loss_repeat += 1
                    lr_patience_counter += 1
                else:
                    loss_repeat = 0
                    lr_patience_counter = 0
                    best_loss = avg_loss
                    best_parameters = model.state_dict()

                epoch += 1
            
            reason = f"best_loss: {best_loss}" if best_loss < loss_to_achieve else f"loss_repeat: {loss_repeat}" if loss_repeat < 10 else f"epoch: {epoch}"
            print(f"Task 1 Finished: {reason}")
            
            model.load_state_dict(best_parameters)
            if is_hyper:
                ewc.after_task(model, t, loader, task_id_tensor=task_id_tensor)
            else:
                ewc.after_task(model, t, loader)

        # ── Tasks > 1 ──────────────────────────────────────────────────────
        else:
            if verbose: print(f"\n[EWC] Task {t+1}")
            
            # WARMUP FOR HYPERNETWORK EMBEDDINGS
            if is_hyper:
                for param in model._shared_params:
                    param.requires_grad = False
                
                active_params = filter(lambda p: p.requires_grad, model.parameters())
                opt = first_task_optimizer_cls(active_params, lr=0.1, weight_decay=1e-4)
                warmup_n = 15
                
                for i in range(warmup_n):
                    total_loss = 0.0
                    model.train()
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        model.spawn(task_id_tensor)
                        output = model(x)
                        loss = criterion(output, y)
                        loss.backward()
                        opt.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(loader)
                    if verbose: print(f"  embedding layer warm up {i+1}/{warmup_n} loss={avg_loss:.4f}")

                for param in model._shared_params:
                    param.requires_grad = True

            # MAIN TRAINING LOOP FOR TASK > 1
            loss_to_achieve = 0.15 
            opt = first_task_optimizer_cls(model.parameters(), lr=base_lr, weight_decay=1e-4)
            
            while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
                total_loss = 0.0
                total_penalty = 0.0
                model.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()

                    if is_hyper: model.spawn(task_id_tensor)

                    output      = model(x)
                    task_loss   = criterion(output, y)
                    ewc_penalty = ewc.penalty(model)
                    loss        = task_loss + ewc_penalty

                    loss.backward()
                    opt.step()

                    total_loss    += task_loss.item()
                    total_penalty += ewc_penalty.item()

                n_batches = len(loader)
                avg_loss    = total_loss    / n_batches
                avg_penalty = total_penalty / n_batches

                # REDUCE LR ON PLATEAU
                if lr_patience_counter >= 5:
                    for g in opt.param_groups:
                        g['lr'] = get_magnitude_decay_lr(g['lr'])
                    lr_patience_counter = 0 
                    if verbose: print(f"    [Scheduler] Loss stalled. Halving LR to {g['lr']}")
    
                if best_loss < avg_loss:
                    loss_repeat += 1
                    lr_patience_counter += 1
                else:
                    loss_repeat = 0
                    lr_patience_counter = 0
                    best_loss = avg_loss
                    best_parameters = model.state_dict()

                wandb.log({
                    "ewc/train/loss":        avg_loss,
                    "ewc/train/ewc_penalty": avg_penalty,
                    "ewc/global_epoch":      global_epoch,
                    "task":                  t + 1,
                })
                global_epoch += 1

                if verbose: print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f} penalty={avg_penalty:.4f} lr={opt.param_groups[0]['lr']}")
                epoch += 1

            # Restore the best performing parameters and calculate Fisher
            model.load_state_dict(best_parameters) 
            if is_hyper:
                ewc.after_task(model, t, loader, task_id_tensor=task_id_tensor)
            else:
                ewc.after_task(model, t, loader)

        # ── Normalise ─────────────────────────────────────────────────────
        # Applied after Fisher is captured, matching the old EWC script location
        for tid in ewc.fishers:
            f = ewc.fishers[tid]
            max_val = f.max()
            if max_val > 0:
                ewc.fishers[tid] = f / max_val
        
        # ── Evaluate on ALL tasks ─────────────────────────────────────────
        results[t+1] = []
        eval_metrics = {"task_completed": t+1}
        
        for i in range(len(test_loaders)): 
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            acc = evaluate_accuracy(model, test_loaders[i], eval_task_id if is_hyper else None, task_classes=tc)
            results[t+1].append(acc)
            eval_metrics[f"ewc/eval/acc_task_{i+1}"] = acc
            if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")
            
        if t != 0:
            bwt = calc_bwt(results, task_id=t+1)
            eval_metrics["ewc/eval/bwt"] = bwt
            if verbose: print(f"BWT at task {t+1}: {bwt:.4f}")

        wandb.log(eval_metrics)

    # ── Plotting ──────────────────────────────────────────────────────────────
    tasks_completed = sorted(list(results.keys())) 
    num_eval_tasks = len(test_loaders)
    
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title("EWC Hypernetwork: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    wandb.log({"EWC Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results