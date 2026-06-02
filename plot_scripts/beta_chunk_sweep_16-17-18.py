import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import numpy as np
import os

# ==============================================================================
# TYPESETTING — NeurIPS/ICML single-column style
# ==============================================================================
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.edgecolor": "#bbbbbb",
    "figure.dpi": 300,
})

API     = wandb.Api()
ENTITY  = "michalowski-jb-tilburg-university"
PROJECT = "HyperFisher"

def fetch_sweep_data():
    print("Fetching data from W&B...")
    runs = API.runs(f"{ENTITY}/{PROJECT}", filters={
        "state": "finished",
        "config.experiment_id": {"$in": [701, 702, 703]}
    })
    
    rows = []
    for run in runs:
        cfg = run.config
        task = cfg.get("task")
        beta = cfg.get("beta")
        chunk_size = cfg.get("chunk_size")
        
        summary = run.summary
        acc = summary.get("best/average_accuracy")
        bwt = summary.get("best/bwt")
        
        if acc is not None and bwt is not None:
            rows.append({
                "task": task,
                "beta": float(beta),
                "chunk_size": float(chunk_size),
                "acc": float(acc),
                "bwt": float(bwt)
            })
            
    print(f"Found {len(rows)} valid runs.")
    return pd.DataFrame(rows)

def main():
    df = fetch_sweep_data()
    if df.empty:
        print("No data found. Exiting.")
        return

    # Averages over seeds if there happen to be multiple
    df = df.groupby(['task', 'chunk_size', 'beta'])[['acc', 'bwt']].mean().reset_index()

    tasks = ['split_mnist_sh', 'split_cifar10', 'split_cifar100']
    task_titles = ['(a) Sweep A: split-MNIST', '(b) Sweep B: split-CIFAR-10', '(c) Sweep C: split-CIFAR-100']

    # Define Shapes for Chunk Sizes and Colors for Betas dynamically
    unique_chunks = sorted(df['chunk_size'].unique())
    unique_betas = sorted(df['beta'].unique())
    
    # Standard marker shapes
    AVAILABLE_MARKERS = ['o', 's', '^', 'D', 'v', '*', 'p']
    
    # ==============================================================================
    # COLORMAP FIX: Map discrete unique betas to evenly spaced gradient points
    # This prevents values like 0.01 and 0.05 from being identical shades of yellow.
    # ==============================================================================
    colors_list = ["#FFEA00", "#FF8C00", "#FF0000", "#800080", "#00008B"]
    cmap = mcolors.LinearSegmentedColormap.from_list("yellow_to_purple", colors_list)
    
    n_betas = len(unique_betas)
    if n_betas > 1:
        color_indices = np.linspace(0, 1, n_betas)
    else:
        color_indices = [0.5]
        
    beta_to_color = {b: cmap(c) for b, c in zip(unique_betas, color_indices)}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    fig.suptitle("Performance vs Backward Transfer Trade-off", fontsize=13, y=1.05)

    for i, task in enumerate(tasks):
        task_df = df[df['task'] == task].copy()
        ax = axes[i]
        
        if task_df.empty: 
            ax.set_title(f"{task_titles[i]}\n(No Data)")
            ax.axis('off')
            continue
        
        # Plot each point individually to map shape and color
        for _, row in task_df.iterrows():
            m_idx = unique_chunks.index(row['chunk_size']) % len(AVAILABLE_MARKERS)
            marker = AVAILABLE_MARKERS[m_idx]
            color = beta_to_color[row['beta']]
            
            ax.scatter(
                row['bwt'], row['acc'], 
                color=color, marker=marker, 
                s=80, alpha=0.85, edgecolors='black', linewidth=0.8
            )
            
        ax.set_title(task_titles[i], fontsize=11)
        ax.set_xlabel("Backward Transfer (BWT)")
        if i == 0:
            ax.set_ylabel("Best Average Accuracy")
            
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.axvline(0, color='red', linestyle=':', alpha=0.5, zorder=0)

    # ==============================================================================
    # CUSTOM LEGEND CONSTRUCTION (Fixed LaTeX Rendering)
    # ==============================================================================
    legend_elements = []
    
    # 1. Add Beta (Color) indicators
    legend_elements.append(mlines.Line2D([], [], color='none', label='Beta ($\\beta$)'))
    for b in unique_betas:
        legend_elements.append(mlines.Line2D(
            [0], [0], marker='o', color='w', label=f"{b}",
            markerfacecolor=beta_to_color[b], markeredgecolor='black', markersize=8
        ))
        
    # Spacer
    legend_elements.append(mlines.Line2D([], [], color='none', label=''))
    
    # 2. Add Chunk Size (Shape) indicators
    legend_elements.append(mlines.Line2D([], [], color='none', label='Chunk Size ($c$)'))
    for idx, c in enumerate(unique_chunks):
        m = AVAILABLE_MARKERS[idx % len(AVAILABLE_MARKERS)]
        legend_elements.append(mlines.Line2D(
            [0], [0], marker=m, color='w', label=f"{int(c)}",
            markerfacecolor='gray', markeredgecolor='black', markersize=8
        ))

    # Place legend outside the plots on the right
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
               framealpha=1.0, title="Hyperparameters", title_fontsize=10)

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig('plots/beta-chunk-sweep_16-17-18.png', bbox_inches='tight')
    plt.savefig('plots/beta-chunk-sweep_16-17-18.pdf', bbox_inches='tight')
    print("Saved plots as 'plots/beta-chunk-sweep_16-17-18.*'.")

if __name__ == "__main__":
    main()