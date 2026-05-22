import wandb
import re
import pandas as pd
import collections
import numpy as np

api = wandb.Api()
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"
job_type = "Target_Network"

# 1. Fetch all runs
RUN_IDS = ["h002e9fd", "ux4omlt4"] 

runs = api.runs(f"{entity}/{project}", filters={"name": {"$in": RUN_IDS}})

print(f"Found {len(runs)} runs. Restructuring...")

for run in runs:
    try:
        # 2. Download the log file
        log_file = run.file("output.log")
        log_file.download(replace=True, root=f"./logs/{run.id}")
        log_path = f"./logs/{run.id}/output.log"
        
        # 3. Parse Log for Accuracies
        # We need to map: Trained_Up_To -> [Acc_Task1, Acc_Task2, ..., Acc_Task5]
        acc_over_time = collections.defaultdict(lambda: [0.0] * 5)
        current_training_task = 0
        
        with open(log_path, "r") as f:
            for line in f:
                # Identify which task was just finished training
                # Matches patterns like "[eFOPNG] Task 2" or "[SGD] Task 2"
                task_trained_match = re.search(r"\[.*\] Task (\d+)", line)
                if task_trained_match and "Acc:" not in line:
                    current_training_task = int(task_trained_match.group(1))
                
                # Extract Evaluated Task and Accuracy
                # Matches patterns like "  Task 1 Acc: 93.8%"
                acc_match = re.search(r"Task (\d+) Acc: (\d+\.\d+)%", line)
                if acc_match and current_training_task > 0:
                    eval_task_id = int(acc_match.group(1))
                    accuracy = float(acc_match.group(2)) / 100.0 # Convert to decimal (0.938)
                    
                    # Store in the list (Task 1 -> Index 0)
                    acc_over_time[current_training_task][eval_task_id - 1] = accuracy

        # 4. Construct the Results Dictionary
        # Matches your internal 'results' structure from projections.py
        # Convert integer keys to strings for W&B compatibility
        results = {"acc": {str(k): v for k, v in acc_over_time.items()}}
        # Alternative: Average accuracy only after the final task is completed
        final_task = max(acc_over_time.keys())
        run.summary["best/average_accuracy"] = float(np.mean(acc_over_time[final_task][:final_task]))

        bwt = run.summary["OGD/eval/bwt"]
        results["bwt"] = bwt
        print(results)
        method = run.config["methods"][0]
        # 5. Update W&B Summary
        run.summary["best/bwt"] = bwt
        run.summary["best/results"] = results
        run.summary[f"{method.lower()}/results"] = results

        run.update()
        
        print(f"  Successfully updated summary for run {run.id} ({run.config.get('methods')})")

    except Exception as e:
        print(f"  Failed to update run {run.id}: {e}")

print("\nDone! All runs now have the 'results' dict in their summary.")