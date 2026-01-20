import json
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = 'hopper/runs/transformer_medium_Loss_0.00046/benchmarks/benchmark.json'
OUTPUT_IMAGE = 'robustness_results.png'

def main():
    # 1. Load the Data
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 2. Parse the Structure
    # Structure is: data['seeds'][seed_id][occlusion_level] = {reward, steps, reason}
    seeds_data = data['seeds']
    
    # Auto-detect the occlusion levels from the first seed
    first_seed = next(iter(seeds_data.values()))
    # Convert keys "0", "15" to integers and sort them
    levels = sorted([int(k) for k in first_seed.keys()])
    
    print(f"Found occlusion levels: {levels}")

    # Initialize storage
    stats = {
        lvl: {
            'rewards': [], 
            'success_count': 0, 
            'total_count': 0
        } 
        for lvl in levels
    }

    # 3. Aggregate Data
    for seed_id, seed_runs in seeds_data.items():
        for lvl_str, result in seed_runs.items():
            lvl = int(lvl_str)
            
            # Store Reward
            stats[lvl]['rewards'].append(result['reward'])
            
            # Check Success
            # Success is defined as TIMEOUT (survived full duration)
            if "TIMEOUT" in result['reason']:
                stats[lvl]['success_count'] += 1
            
            stats[lvl]['total_count'] += 1

    # 4. Calculate Metrics (Means, Stds, Rates)
    x_levels = []
    success_rates = []
    mean_rewards = []
    std_rewards = []

    for lvl in levels:
        x_levels.append(lvl)
        
        # Success Rate Calculation
        total = stats[lvl]['total_count']
        successes = stats[lvl]['success_count']
        rate = (successes / total) * 100 if total > 0 else 0
        success_rates.append(rate)
        
        # Reward Calculation
        rewards = stats[lvl]['rewards']
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))

    # 5. Plotting
    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Success Rate (The Survival Curve) ---
    ax1.plot(x_levels, success_rates, 'o-', color='#1f77b4', linewidth=2, label='Success Rate')
    
    # Styling
    ax1.set_title("Robustness to Sensor Occlusion (Survival Rate)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Occlusion Length (Steps)", fontsize=12)
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(levels)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate points with values
    for x, y in zip(x_levels, success_rates):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # --- Plot 2: Mean Reward (The Degradation Curve) ---
    # Use error bars to show standard deviation
    ax2.errorbar(x_levels, mean_rewards, yerr=std_rewards, fmt='o-', color='#ff7f0e', 
                 linewidth=2, capsize=5, label='Mean Reward')
    
    # Styling
    ax2.set_title("Performance Degradation (Mean Reward)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Occlusion Length (Steps)", fontsize=12)
    ax2.set_ylabel("Average Reward", fontsize=12)
    ax2.set_xticks(levels)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add a global title
    plt.suptitle(f"Agent Robustness Analysis (N={len(seeds_data)} Seeds)", fontsize=16)
    plt.tight_layout()
    
    # 6. Save and Show
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ Graph saved to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    main()