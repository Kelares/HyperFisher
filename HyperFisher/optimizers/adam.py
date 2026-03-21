import torch
import torch.nn as nn
import wandb
from typing import Callable, List, Optional
from torch.utils.data import DataLoader
from utils import evaluate_accuracy, calc_bwt
import matplotlib.pyplot as plt
import numpy as np


def train_adam(
    hyper_network: nn.Module, 
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion,
    lr,
    epochs,
    task_classes: Optional[list] = None,
):
    device = next(hyper_network.parameters()).device
    optimizer = torch.optim.Adam(hyper_network.parameters(), lr=lr)

    results_baseline = {}
    global_epoch_baseline = 0
    
    for t, loader in enumerate(train_loaders):
        # 2. FIXED: Create the task_id tensor for the baseline
        task_id = torch.tensor([t], dtype=torch.long, device=device)
        
        for epoch in range(epochs):
            total_loss = 0.0
            hyper_network.train()
            
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                # 3. FIXED: Spawn the target parameters
                hyper_network.spawn(task_id)
                output = hyper_network(x)
                
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss/len(loader)
            wandb.log({"baseline/train/loss": avg_loss, "baseline/global_epoch": global_epoch_baseline, "task": t+1})
            global_epoch_baseline += 1
            print(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        # ── Evaluate on ALL tasks using TEST loaders ───────────────────
        results_baseline[t+1] = []
        eval_metrics_baseline = {"task_completed": t+1}
        
        for i in range(len(test_loaders)):
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            # 4. FIXED: Evaluate the baseline hypernetwork, not the MLP
            acc = evaluate_accuracy(hyper_network, test_loaders[i], eval_task_id, task_classes=tc)
            results_baseline[t+1].append(acc)
            eval_metrics_baseline[f"baseline/eval/acc_task_{i+1}"] = acc
            print(f"  Task {i+1}: {acc*100:.1f}%")
            
        if t != 0:
            bwt_baseline = calc_bwt(results_baseline, task_id=t+1)
            eval_metrics_baseline["baseline/eval/bwt"] = bwt_baseline
            print(f"BWT for task {t+1}: {bwt_baseline:.4f}")
            
        wandb.log(eval_metrics_baseline)

    # ─────────────────────────────────────────────────────────────────────────────
    # Force W&B to generate overlapping Custom Charts with SOLID COLORS
    # ─────────────────────────────────────────────────────────────────────────────
    
    tasks_completed = sorted(list(results_baseline.keys())) 
    num_eval_tasks = len(test_loaders)

    # 5. FIXED: Use matplotlib to strictly enforce colored solid lines
    plt.figure(figsize=(10, 6))
    
    # Define a clean, distinct color palette
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results_baseline[t][i] for t in tasks_completed]
        # Force solid line (linestyle='-') and cycle through colors
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title("Baseline (Adam Hypernetwork): All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    # Log the cleanly colored plot directly to W&B
    wandb.log({"Baseline Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()