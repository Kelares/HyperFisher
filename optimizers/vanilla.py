import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Optional, Union
from torch.utils.data import DataLoader
from math import inf
from utils import get_grad_vector, evaluate_accuracy, calc_bwt
from models.hyper_network import HyperRegulizer

def get_magnitude_decay_lr(current_lr: float) -> float:
    sci_str = f"{current_lr:.1e}"  
    mantissa, exp = sci_str.split('e')
    mantissa = float(mantissa)
    exp = int(exp)
    if mantissa >= 4.9:
        return 1.0 * (10 ** exp)
    else:               
        return 5.0 * (10 ** (exp - 1))

# Map method names to their respective classes
METHOD_MAP = {
    "sgd": torch.optim.SGD,
    "adam":  torch.optim.Adam
}

def train_vanilla(
    method: str,
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    regulizer: bool = False,
    *,
    lr: float = 1e-3,
    epochs: int = 5,
    max_epochs: int = None,
    task_classes: Optional[list] = None,
    verbose: bool = True,
    warmup: bool = False,
    beta: float = 0.1,
):
    """
    Standard training baseline refactored to match the Projection Train structure.
    """
    optimizer_cls = METHOD_MAP[method]
    optimizer_name = optimizer_cls.__name__.upper()

    device = next(model.parameters()).device
    results = {"acc": {}}
    global_epoch = 0
    loss_to_achieve = 0.1
    _max_epochs = max_epochs if max_epochs else epochs
    base_lr = lr
    regulizer_instance = HyperRegulizer(beta=beta) if regulizer else None


    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)
        best_loss = inf
        loss_repeat = 0
        lr_patience_counter = 0
        best_parameters = None
        epoch = 0
        
        # Initialize standard optimizer for the task
        opt = optimizer_cls(model.parameters(), lr=base_lr)

        # ── Task 1 (Standard Initialization) ──────────────────────────────────
        if t == 0:
            if verbose: print(f"[{optimizer_name}] Task 1 – {optimizer_cls.__name__}")
            
            while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
                total_loss = 0.0
                model.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    model.spawn(task_id)
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                wandb.log({
                    f"{optimizer_name}/train/loss": avg_loss, 
                    f"{optimizer_name}/global_epoch": global_epoch, 
                    "task": t + 1
                })
                global_epoch += 1
                if verbose: print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    loss_repeat = 0
                    best_parameters = model.state_dict()
                    lr_patience_counter = 0

                else:
                    loss_repeat += 1
                    lr_patience_counter += 1

                if lr_patience_counter == 3:
                    for g in opt.param_groups:
                        g['lr'] = get_magnitude_decay_lr(g['lr'])
                    lr_patience_counter = 0
                    if verbose: print(f"    [Scheduler] Lowering LR to {opt.param_groups[0]['lr']}")
                    
                epoch += 1

            model.load_state_dict(best_parameters)
            reason = f"best_loss: {best_loss}" if best_loss < loss_to_achieve else f"loss_repeat: {loss_repeat}"
            print(f"Task 1 Finished: {reason}")

        # ── Tasks > 1 (Continual Learning Phase) ──────────────────────────────
        else:
            if verbose: print(f"\n[{optimizer_name}] Task {t+1}")
            
            if warmup:
                # FREEZING SHARED_PARAMS FOR TASK EMBEDDING WARMUP
                for param in model._shared_params:
                    param.requires_grad = False
                
                warmup_opt = optimizer_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
                warmup_n = 5
                for i in range(warmup_n):
                    total_loss = 0.0
                    model.train()
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        warmup_opt.zero_grad()
                        model.spawn(task_id)
                        loss = criterion(model(x), y)
                        loss.backward()
                        warmup_opt.step()
                        total_loss += loss.item()
                    if verbose: print(f"  embedding warm up {i+1}/{warmup_n} loss={total_loss/len(loader):.4f}")

                for param in model._shared_params:
                    param.requires_grad = True

            # Main Training loop for Task > 1
            while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
                total_loss = 0.0
                total_reg = 0.0
                model.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    model.spawn(task_id)
                    output = model(x)
                    loss = criterion(output, y)

                    if regulizer_instance:
                        w_penalty =  regulizer_instance.loss(model, task_id)
                        total_reg += w_penalty.item()
                        loss += w_penalty

                    loss.backward()
                    opt.step()
                    total_loss += loss.item()

                n_batches = len(loader)
                avg_loss = total_loss / n_batches
                avg_reg = total_reg / n_batches


                # Intelligent Tracking & Scheduler
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    lr_patience_counter = 0
                    loss_repeat = 0
                    best_parameters = model.state_dict()
                else:
                    lr_patience_counter += 1
                    loss_repeat += 1

                if lr_patience_counter == 3:
                    for g in opt.param_groups:
                        g['lr'] = get_magnitude_decay_lr(g['lr'])
                    lr_patience_counter = 0
                    if verbose: print(f"    [Scheduler] Lowering LR to {opt.param_groups[0]['lr']}")

                wandb.log({
                    f"{optimizer_name}/train/loss": avg_loss,
                    f"{optimizer_name}/global_epoch": global_epoch,
                    "task": t + 1,
                })
                global_epoch += 1

                if verbose: print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f} lr={opt.param_groups[0]['lr']}")
                if regulizer_instance: print(f" reg_loss={avg_reg:.4f}")

                epoch += 1

            model.load_state_dict(best_parameters)

        # ── End of Task Management ──────────────────────────────────────────
        if regulizer_instance:
            model.spawn(task_id)
            regulizer_instance.old_weights[task_id.item()] = model.w.detach()

        # ── Evaluation ──────────────────────────────────────────────────────
        results["acc"][t+1] = []
        eval_metrics = {"task_completed": t+1}
        
        for i in range(len(test_loaders)): 
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            acc = evaluate_accuracy(model, test_loaders[i], eval_task_id, task_classes=tc)
            results["acc"][t+1].append(acc)
            eval_metrics[f"{optimizer_name}/eval/acc_task_{i+1}"] = acc
            if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")
            
        if t != 0:
            bwt = calc_bwt(results["acc"], task_id=t+1)
            eval_metrics[f"{optimizer_name}/eval/bwt"] = bwt
            if verbose: print(f"BWT at task {t+1}: {bwt:.4f}")
            results["bwt"] = bwt

        wandb.log(eval_metrics)

    # ── Final Plotting ───────────────────────────────────────────────────
    tasks_completed = sorted(list(results["acc"].keys()))
    num_eval_tasks = len(test_loaders)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results["acc"][t][i] for t in tasks_completed]
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title(f"{optimizer_name} Baseline: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    wandb.log({f"{optimizer_name} Overlapping Accuracies": wandb.Image(plt)})
    plt.close()
    results["bwt"] = 0
    return results