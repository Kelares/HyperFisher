import torch
import torch.nn as nn
import wandb
from typing import Callable, List, Optional
from torch.utils.data import DataLoader
from utils import evaluate_accuracy, calc_bwt
import matplotlib.pyplot as plt
import numpy as np
from math import inf
from hyper_network import HyperRegulizer

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

def train_vanilla(
    hyper_network: nn.Module, 
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    epochs: int = 5,
    max_epochs: int = None,
    optim=torch.optim.SGD,
    task_classes: Optional[list] = None,
    verbose: bool = True
):
    device = next(hyper_network.parameters()).device
    results_baseline = {}
    global_epoch_baseline = 0
    regulizer = HyperRegulizer()
    
    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)
        best_loss = inf
        loss_repeat = 0
        _max_epochs = max_epochs if max_epochs else epochs
        lr_patience_counter = 0
        best_parameters = None
        base_lr = lr
        epoch = 0
        opt = optim(hyper_network.parameters(), lr=base_lr, weight_decay=1e-4)

        # ── Task 1 ─────────────────────────────────────────────────────────
        if t == 0:
            if verbose: print(f"[Vanilla] Task 1 – {optim.__name__}")
            loss_to_achieve = 0.2
            
            while best_loss >= loss_to_achieve and loss_repeat < 8 and epoch < _max_epochs:
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
                wandb.log({"baseline/train/loss": avg_loss, "baseline/global_epoch": global_epoch_baseline, "task": t+1})
                global_epoch_baseline += 1
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
                    best_parameters = hyper_network.state_dict()

                epoch += 1
            
            reason = f"best_loss: {best_loss}" if best_loss < loss_to_achieve else f"loss_repeat: {loss_repeat}" if loss_repeat < 10 else f"epoch: {epoch}"
            print(f"Task 1 Finished: {reason}")
            
            # Load the best state dict before moving to evaluation or next task
            hyper_network.load_state_dict(best_parameters)

        # ── Tasks > 1 ──────────────────────────────────────────────────────
        else:
            # if verbose: print(f"\n[Vanilla] Task {t+1}")
            
            # # FREEZING SHARED_PARAMS SO THE TASK EMBEDDING GETS AN EARLY START
            # for param in hyper_network._shared_params:
            #     param.requires_grad = False
            
            # active_params = filter(lambda p: p.requires_grad, hyper_network.parameters())
            # warmup_n = 15
            
            # for i in range(warmup_n):
            #     total_loss = 0.0
            #     hyper_network.train()
            #     for x, y in loader:
            #         x, y = x.to(device), y.to(device)
            #         opt.zero_grad()
            #         hyper_network.spawn(task_id)
            #         output = hyper_network(x)
            #         if regulizer:
            #             loss = regulizer.loss(hyper_network, task_id, criterion, output, y)
            #         else:
            #             loss = criterion(output, y)
                                        
            #         loss.backward()
            #         opt.step()
            #         total_loss += loss.item()
                
            #     avg_loss = total_loss / len(loader)
            #     if verbose: print(f"  embedding layer warm up {i+1}/{warmup_n} loss={avg_loss:.4f}")

            # # UNFREEZING SHARED_PARAMS
            # for param in hyper_network._shared_params:
            #     param.requires_grad = True

            # MAIN TRAINING LOOP
            loss_to_achieve = 0.15 
            
            while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
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

                n_batches = len(loader)
                avg_loss = total_loss / n_batches

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
                    best_parameters = hyper_network.state_dict()

                wandb.log({
                    "baseline/train/loss": avg_loss,
                    "baseline/global_epoch": global_epoch_baseline,
                    "task": t + 1,
                })
                global_epoch_baseline += 1

                if verbose: print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f} lr={opt.param_groups[0]['lr']}")
                epoch += 1

            hyper_network.load_state_dict(best_parameters) 
        
        # ── Evaluate on ALL tasks using TEST loaders ───────────────────
        results_baseline[t+1] = []
        eval_metrics_baseline = {"task_completed": t+1}
        
        for i in range(len(test_loaders)): 
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            acc = evaluate_accuracy(hyper_network, test_loaders[i], eval_task_id, task_classes=tc)
            results_baseline[t+1].append(acc)
            eval_metrics_baseline[f"baseline/eval/acc_task_{i+1}"] = acc
            if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")
            
        if t != 0:
            bwt_baseline = calc_bwt(results_baseline, task_id=t+1)
            eval_metrics_baseline["baseline/eval/bwt"] = bwt_baseline
            if verbose: print(f"BWT at task {t+1}: {bwt_baseline:.4f}")

        wandb.log(eval_metrics_baseline)

    # ── Plotting ─────────────────────────────────────────────────────────────
    tasks_completed = sorted(list(results_baseline.keys())) 
    num_eval_tasks = len(test_loaders)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results_baseline[t][i] for t in tasks_completed]
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title("Baseline Hypernetwork: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    wandb.log({"Baseline Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results_baseline