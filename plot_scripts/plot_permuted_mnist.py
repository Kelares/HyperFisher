import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Initialize API
api = wandb.Api()
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"
task = "permuted_mnist"
def plotJob(job_type):
    # 2. Fetch runs
    runs = api.runs(f"{entity}/{project}", filters={"jobType": job_type, "group": task, "state": "finished"})

    data_list = []

    print(f"Found {len(runs)} runs. Extracting trajectory data...")

    for run in runs:
        if run.id == "3vojbbtn":
            print(run.id)
            print(run.summary)
            continue

        # Try multiple possible keys where we stored the structured results
        results = run.summary.get("best/results") or run.summary.get("results")
        
        if not results or "acc" not in results:
            # Check if it's stored under method/results (e.g., ifopng/results)
            method_name = run.config.get("methods", [""])[0].lower()
            results = run.summary.get(f"{method_name}/results")
            
        print(run.config.get("methods", [""])[0].lower())
        if not results or "acc" not in results:
            continue

        method = run.config.get("methods")[0].upper()
        seed = run.config.get("seed")
        id = run.id
        acc_matrix = results["acc"]

        # Calculate average accuracy at each task step t
        # acc_matrix looks like: {"1": [task1_acc, task2_acc, ...], "2": [...]}
        for t_str, acc_list in acc_matrix.items():
            t = int(t_str)
            # Average accuracy of tasks 1 through t after training task t
            current_accs = acc_list[:t]
            avg_acc = np.mean(current_accs)
            
            data_list.append({
                "id": id,
                "Method": method,
                "Seed": seed,
                "Tasks Trained (t)": t,
                "Average Accuracy": avg_acc
            })

    df = pd.DataFrame(data_list)

    # 3. Create the Plot
    plt.figure(figsize=(10, 6))

    # Standard academic style
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", len(df["Method"].unique()))

    # Plotting with lineplot automatically averages across seeds and adds a 95% CI shadow
    line_plot = sns.lineplot(
        data=df, 
        x="Tasks Trained (t)", 
        y="Average Accuracy", 
        hue="Method", 
        style="Method",
        markers=True, 
        markersize=8,
        linewidth=2.5,
        errorbar=('ci', 0.95),  # "sd" for standard deviation, "ci" for confidence interval
        palette=palette
    )

    # Formatting
    plt.title(f"Average Accuracy Trajectory: {task}", fontsize=15, fontweight='bold')
    plt.xlabel("Number of Tasks Learned", fontsize=12)
    plt.ylabel("Average Accuracy (tasks 1 to t)", fontsize=12)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.ylim(0.5, 1.0) # Adjusted based on CIFAR-10 expected ranges

    # Add information about Hypernetworks and Regularizer
    match job_type:
        case "HyperNet_Reg_True":
            model =  "Chunked Hypernetwork"
            regulizer = "Regularizer: Hypernetwork Output Regularizer (Reg=True)"

        case "HyperNet_Reg_False":
            model =  "Chunked Hypernetwork"
            regulizer = "Regularizer: Hypernetwork Output Regularizer (Reg=False)"

        case "Target_Network":
            model =  "TargetNetwork"
            regulizer = ""
        
    plt.text(0.98, 0.02, f"Model: {model}\n{regulizer}", 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend(title="Optimization Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save for Thesis
    name = f"average_acc_{job_type}_{task}"
    path = f"visualizations/{name}"
    plt.savefig(f"{path}.pdf", dpi=300)
    plt.savefig(f"{path}.png", dpi=300)

    print(f"\nSuccess! Generated '{name}'.")
    print(f"Summary of methods plotted: {df['Method'].unique()}")

plotJob("HyperNet_Reg_True")
# plotJob("HyperNet_Reg_False")
# plotJob("Target_Network")
