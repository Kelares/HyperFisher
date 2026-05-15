import argparse
import wandb
import torch
import torch.nn as nn
import copy
import importlib

from hyper_network import HyperNetwork
from optimizers.ewc import train_ewc
# Import the unified launcher
from optimizers.projections import run_continual_method 
from utils import stress_test_fopng_memory
from optimizers.vanilla import train_vanilla

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Experiments CLI")
    
    # Core parameters
    parser.add_argument("--model", type=str, default="HyperNetwork", choices=["HyperNetwork", "TargetNetwork"])
    parser.add_argument("--task", type=str, required=True, choices=["permuted_mnist", "split_mnist", "split_cifar10", "split_cifar100"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=None)

    # Method selection
    parser.add_argument(
        "--methods", 
        nargs='+', 
        required=False,
        default=["fopng", "adam"],
        choices=["sgd", "adam", "ogd", "ognd", "fopng", "fopng_prefisher", "fng", "efopng", "ewc"],
    )

    # Optimization/Fisher parameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--fisher_samples", type=int, default=1024)
    parser.add_argument("--fisher_clipping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fisher_normalization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grads_per_task", type=int, default=40)
    parser.add_argument("--max_directions", type=int, default=80)
    
    # Model/Hypernet Specific
    parser.add_argument("--hyper_hidden_dim", type=int, default=16)
    parser.add_argument("--task_embedding_dim", type=int, default=4)
    parser.add_argument("--chunk_embedding_dim", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--regulizer", action=argparse.BooleanOptionalAction, default=True)

    # Infrastructure
    parser.add_argument("--check_vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device_mode", type=str, default="hybrid", choices=["cpu", "gpu", "hybrid"])
    parser.add_argument("--saved", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()

    # Task and Data Setup
    task_module = importlib.import_module(f"tasks.{args.task}")
    Task = task_module.TaskGenerator
    task_config = Task.config
    criterion = nn.CrossEntropyLoss() if task_config.criterion is None else task_config.criterion
    target_network = Task.target_network

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Define the grouping logic based on model type and regularizer state
    if args.model == "TargetNetwork":
        config_identity = "Target_Network"
    else:
        config_identity = f"HyperNet_Reg_{args.regulizer}"

    wandb.init(
        project="HyperFisher", 
        config=vars(args),
        group=args.task,          # Grouping by Task (e.g., split_mnist)
        job_type=config_identity, # Sub-grouping within that task
    )

    config = wandb.config
    config.update({"num_tasks": task_config.num_tasks, "task_classes": getattr(task_config, 'task_classes', None)})

    if config.check_vram:
        stress_test_fopng_memory()

    datasets = [Task.generate(task_id=t, batch_size=config.batch_size) for t in range(config.num_tasks)]
    train_loaders = [d[0] for d in datasets]
    test_loaders = [d[1] for d in datasets]

    wandb.define_metric("task_completed")
    # Model Initialization
    model = HyperNetwork(
        target_network_template=target_network,
        device=device, 
        config=config
    ) if config.model == "HyperNetwork" else Task.multihead(Task.config.num_tasks, device)
    print(model)

    initial_state = copy.deepcopy(model.state_dict())
    best_acc = -1
    best_bwt = -1

    # --- Unified Method Loop ---
    for method in args.methods:
        # Restart model state
        model.load_state_dict(initial_state)
        if config.model == "HyperNetwork":
            model.target_params = None 
            model.w = None

        print(f"\n--- Starting {method.upper()} Training ---")
        wandb.define_metric(f"{method}/eval/*", step_metric="task_completed")
       

        if method == "ewc":
            pass

        elif method == "sgd" or method == "adam":
            results = train_vanilla(
                method=method,
                model=model,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                criterion=criterion,
                lr=config.get("lr", 1e-3),
                epochs=config.get("epochs", 5),
                max_epochs=config.get("max_epochs"),
                task_classes=config.get("task_classes"),
                verbose=config.get("verbose", True),
                warmup=config.get("warmup", False), 
            )

        else:
            # Execute unified launcher
            results = run_continual_method(
                method=method,
                model=model,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                criterion=criterion,
                config=config
            )

        # Standardized Post-Training Logging
        if results:
            final_task_id = max(results["acc"].keys())
            final_accuracies = results["acc"][final_task_id]
            average_accuracy = sum(final_accuracies) / len(final_accuracies)
            
            wandb.log({f"{method}/eval/average_accuracy": average_accuracy})
            
            if average_accuracy >= best_acc:
                best_acc = average_accuracy

            final_bwt = results["bwt"]

            if final_bwt >= best_bwt:
                best_bwt = final_bwt

    wandb.log({"best/average_accuracy": best_acc})
    wandb.log({"best/bwt": best_bwt})