import argparse
import wandb
from hyper_network import HyperNetwork
from mlp_base import MLP
import copy


import torch
import torch.nn as nn
import wandb
from optimizers.projections import train_fopng, train_OGND, train_PreFopng, train_efopng, train_OGD#, train_fopng_plus
from optimizers.vanilla import train_vanilla
from optimizers.ewc import train_ewc

import importlib
from utils import stress_test_fopng_memory

# THE INITIAL STABILITY OF THE TRAINING (FIRST TASK) SEEMS VERY IMPORTANT FOR SETTING A STABLE SPACE FOR CL ARCHITECTURES TO MOVE IN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Experiments CLI")
    
    
    # ------------------------------
    # Core parameters
    # ------------------------------
    parser.add_argument("--model", type=str, default="HyperNetwork", 
                        choices=["HyperNetwork", "MLP"])

    # TASK SPECIFIC
    parser.add_argument("--task", type=str, required=True,
                        choices=["permuted_mnist", "split_mnist", "split_cifar10"]) #, "rotated_mnist", "", "split_cifar10", "split_cifar100"
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--max_epochs", type=int, default=None)

    # ------------------------------

    parser.add_argument(
        "--methods", 
        nargs='+', 
        required=False,
        default=["fopng", "adam"],
        choices=["sgd", "adam", "ogd", "ognd", "fopng", "fopng_plus", "fopng_prefisher", "fng", "efopng", "ewc"],
    )
    # LEARNING SPECIFIC
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.01)

    parser.add_argument("--alpha", type=float, default=0.3)
    
    parser.add_argument("--fisher_samples", type=int, default=1024)
    parser.add_argument("--fisher_clipping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fisher_normalization", action=argparse.BooleanOptionalAction, default=False)


    parser.add_argument("--grads_per_task", type=int, default=40)
    parser.add_argument("--max_directions", type=int, default=80)
    # ------------------------------
    
    # MODEL SPECIFIC
    parser.add_argument("--hyper_hidden_dim", type=int, default=16)
    parser.add_argument("--task_embedding_dim", type=int, default=4)
    parser.add_argument("--chunk_embedding_dim", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--regulizer", action=argparse.BooleanOptionalAction, default=True)

    
    # ------------------------------

    parser.add_argument("--check_vram", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device_mode", type=str, default="hybrid", choices=["cpu", "gpu", "hybrid"])
    
    # TRAINING
    parser.add_argument("--saved", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=False)
    # ------------------------------
    args = parser.parse_args()

    task = args.task
    methods = args.methods
    epochs = args.epochs

    lr = args.lr
    lam = args.lam
    alpha = args.alpha
    fisher_samples = args.fisher_samples

    seed = args.seed
    grads_per_task = args.grads_per_task
    max_directions = args.max_directions

    hyper_hidden_dim = args.hyper_hidden_dim

    # Task specific configs
    task_module = importlib.import_module(f"tasks.{task}")
    Task = task_module.TaskGenerator
    task_config = Task.config
    criterion = nn.CrossEntropyLoss() if task_config.criterion is None else task_config.criterion
    target_network = Task.target_network
    multihead = Task.multihead
    ########################

    if methods is None:
        methods = ["fopng", "adam"]


    torch.manual_seed(0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(DEVICE)

    wandb.init(
        project="HyperFisher",
        config=vars(args)
    )

    config = wandb.config
    config.update({"num_tasks": task_config.num_tasks})
    if config.check_vram:
        stress_test_fopng_memory()

    # Unpack the returned tuples into separate lists
    datasets = [Task.generate(task_id=t, batch_size=config.batch_size) for t in range(config.num_tasks)]
    
    train_loaders = [d[0] for d in datasets]
    test_loaders = [d[1] for d in datasets]


    # Tell W&B to use 'task_completed' as the x-axis for all eval metrics
    wandb.define_metric("task_completed")

    # 1. Create the model once before the loop
    model = HyperNetwork(
        target_network_template=target_network, 
        device=device, 
        config=config
    ) if config.model == "HyperNetwork" else multihead(5, device) #MLP(target_network, device=device)

    config.update({"architecture": model})
    if config.model == "HyperNetwork":
        config.update({"num_of_chunks": model.num_of_chunks})
        print(model.num_of_chunks)
    print(model)

    # 2. Save the "Fresh" state (Deep copy of weights)
    initial_state = copy.deepcopy(model.state_dict())
    best_acc = -1
    print(config)
    for method in methods:
        wandb.define_metric(f"{method}/eval/*", step_metric="task_completed")
       
        # 3. RESTART: Load the fresh state back into the model
        model.load_state_dict(initial_state)
        # Ensure any internal buffers (like model.w or model.target_params) are cleared
        if config.model == "HyperNetwork":
            model.target_params = None 
            model.w = None

        match method:
            case "efopng":
                print("\n--- Starting eFOPNG Training ---")
                # task1_lr = config.lr * 5 if config.task == "split_cifar10" else config.lr
                results = train_efopng(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, lam=config.lam, alpha=config.alpha,
                    grads_per_task=config.grads_per_task, max_directions=config.max_directions,
                    epochs=config.epochs, max_epochs=config.max_epochs, verbose=True,
                    fisher_samples=config.fisher_samples,
                    task_classes = getattr(task_config, 'task_classes', None),
                    device_mode = config.device_mode,
                    saved = config.saved,
                    warmup = config.warmup,
                    fisher_clipping = config.fisher_clipping,
                    fisher_normalization = config.fisher_normalization,
                    regulizer = config.regulizer
                )
                final_task_id = max(results.keys())
                final_accuracies = results[final_task_id]
                average_accuracy = sum(final_accuracies) / len(final_accuracies)
                
                wandb.log({"eFOPNG/eval/average_accuracy": average_accuracy})
            
            case "fopng":
                print("\n--- Starting FOPNG Training ---")
                # task1_lr = config.lr * 5 if config.task == "split_cifar10" else config.lr
                results = train_fopng(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, lam=config.lam, alpha=config.alpha,
                    grads_per_task=config.grads_per_task, max_directions=config.max_directions,
                    epochs=config.epochs, max_epochs=config.max_epochs, verbose=True,
                    fisher_samples=config.fisher_samples,
                    task_classes = getattr(task_config, 'task_classes', None),
                    device_mode = config.device_mode,
                    saved = config.saved,
                    warmup = config.warmup,
                    fisher_clipping = config.fisher_clipping,
                    fisher_normalization = config.fisher_normalization,
                    regulizer = config.regulizer
                )
                final_task_id = max(results.keys())
                final_accuracies = results[final_task_id]
                average_accuracy = sum(final_accuracies) / len(final_accuracies)
                
                wandb.log({"FOPNG/eval/average_accuracy": average_accuracy})
            
            case "ognd":
                print("\n--- Starting OGND Training ---")
                # task1_lr = config.lr * 5 if config.task == "split_cifar10" else config.lr
                results = train_OGND(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, lam=config.lam, alpha=config.alpha,
                    grads_per_task=config.grads_per_task, max_directions=config.max_directions,
                    epochs=config.epochs, max_epochs=config.max_epochs, verbose=True,
                    fisher_samples=config.fisher_samples,
                    task_classes = getattr(task_config, 'task_classes', None),
                    device_mode = config.device_mode,
                    saved = config.saved,
                    warmup = config.warmup,
                    fisher_clipping = config.fisher_clipping,
                    fisher_normalization = config.fisher_normalization,
                    regulizer = config.regulizer

                )
                final_task_id = max(results.keys())
                final_accuracies = results[final_task_id]
                average_accuracy = sum(final_accuracies) / len(final_accuracies)
                
                wandb.log({"OGND/eval/average_accuracy": average_accuracy})
            
            case "ogd":
                print("\n--- Starting OGD Training ---")
                # task1_lr = config.lr * 5 if config.task == "split_cifar10" else config.lr
                results = train_OGD(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, lam=config.lam, alpha=config.alpha,
                    grads_per_task=config.grads_per_task, max_directions=config.max_directions,
                    epochs=config.epochs, max_epochs=config.max_epochs, verbose=True,
                    fisher_samples=config.fisher_samples,
                    task_classes = getattr(task_config, 'task_classes', None),
                    device_mode = config.device_mode,
                    saved = config.saved,
                    warmup = config.warmup,
                    fisher_clipping = config.fisher_clipping,
                    fisher_normalization = config.fisher_normalization,
                    regulizer = config.regulizer

                )
                final_task_id = max(results.keys())
                final_accuracies = results[final_task_id]
                average_accuracy = sum(final_accuracies) / len(final_accuracies)
                
                wandb.log({"OGD/eval/average_accuracy": average_accuracy})

            case "fopng_prefisher":
                print("\n--- Starting fopng_prefisher Training ---")
                # task1_lr = config.lr * 5 if config.task == "split_cifar10" else config.lr
                results = train_PreFopng(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, lam=config.lam, alpha=config.alpha,
                    grads_per_task=config.grads_per_task, max_directions=config.max_directions,
                    epochs=config.epochs, max_epochs=config.max_epochs, verbose=True,
                    fisher_samples=config.fisher_samples,
                    task_classes = getattr(task_config, 'task_classes', None),
                    device_mode = config.device_mode,
                    saved = config.saved,
                    warmup = config.warmup,
                    fisher_clipping = config.fisher_clipping,
                    fisher_normalization = config.fisher_normalization,
                    regulizer = config.regulizer

                )
                final_task_id = max(results.keys())
                final_accuracies = results[final_task_id]
                average_accuracy = sum(final_accuracies) / len(final_accuracies)
                
                wandb.log({"preFOPNG/eval/average_accuracy": average_accuracy})

            # case "fopng_plus":
            #     print("--- Starting FOPNG+ Training ---")
            #     results = train_fopng_plus(
            #         model, train_loaders, test_loaders, criterion,
            #         lr=config.lr, lam=config.lam, damping=config.damping, alpha=config.alpha,
            #         grads_per_task=config.grads_per_task, max_directions=config.max_directions,
            #         epochs=config.epochs, max_epochs=config.max_epochs, verbose=True,
            #         fisher_samples=config.fisher_samples,
            #         task_classes=getattr(task_config, 'task_classes', None)
            #     )
            #     final_task_id = max(results.keys())
            #     final_accuracies = results[final_task_id]
            #     average_accuracy = sum(final_accuracies) / len(final_accuracies)
            #     wandb.log({"fopng_plus/eval/average_accuracy": average_accuracy})

            case "ewc":
                print("\n--- Starting EWC Training ---")

                results = train_ewc(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, lam=1e3, epochs=config.epochs,
                    task_classes = getattr(task_config, 'task_classes', None)
                )
                final_task_id = max(results.keys())
                final_accuracies = results[final_task_id]
                average_accuracy = sum(final_accuracies) / len(final_accuracies)
                
                wandb.log({"ewc/eval/average_accuracy": average_accuracy})

            case "adam":
                print("\n" + "=" * 60)
                print("BASELINE COMPARISON (Adam)")
                print("=" * 60)

                train_vanilla(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, epochs=config.epochs, 
                    task_classes = getattr(task_config, 'task_classes', None),
                    optim=torch.optim.Adam

                )

            case "sgd":
                print("\n" + "=" * 60)
                print("BASELINE COMPARISON (SGD)")
                print("=" * 60)
                results = train_vanilla(
                    model, train_loaders, test_loaders, criterion,
                    lr=config.lr, epochs=config.epochs, 
                    max_epochs=config.max_epochs,
                    task_classes = getattr(task_config, 'task_classes', None),
                    optim=torch.optim.SGD
                )
                final_task_id = max(results.keys())
                final_accuracies = results[final_task_id]
                average_accuracy = sum(final_accuracies) / len(final_accuracies)
                
                wandb.log({"ewc/eval/average_accuracy": average_accuracy})
        if average_accuracy >= best_acc:
            best_acc = average_accuracy
    wandb.log({"best/average_accuracy": best_acc})