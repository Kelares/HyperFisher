import wandb
import re
import os

# 1. Set your project details
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"
api = wandb.Api()

# 2. Filter for finished vanilla runs
runs = api.runs(
    f"{entity}/{project}", 
    filters={"config.methods": {"$in": [["adam"], ["sgd"]]}, "state": "finished",
             "$or": [                                    # Catch 0 or missing
            {"summary_metrics.best/bwt": 0},
            {"summary_metrics.best/bwt": {"$exists": False}}
        ]
    }
)

print(f"Found {len(runs)} runs.")

# Regex to capture the float value after "BWT at task X: "
# \s* handles potential spacing, and ([-0-9.]+) captures the negative/positive float
bwt_pattern = re.compile(r"BWT at task \d+:\s*([-0-9.]+)")

for run in runs:
    try:
        print(f"\nConnecting to run {run.id} ({run.name})")
        
        # Safely fetch the current metric
        current_bwt = run.summary.get("best/bwt")
        
        # Check if the run needs fixing
        if current_bwt == 0 or current_bwt is None:
            print(f"  - Needs fixing. Current best/bwt: {current_bwt}")
            
            # W&B automatically saves standard console output to 'output.log'
            log_file = run.file("output.log")
            log_file.download(replace=True, root=".")
            
            with open("output.log", "r") as f:
                logs = f.read()
            
            # Find all occurrences of the BWT print statement
            matches = bwt_pattern.findall(logs)
            
            if matches:
                # Grab the final match from the end of the log
                final_bwt = float(matches[-1])
                print(f"  - Found final BWT in logs: {final_bwt}")
                
                # Overwrite the summary and push to W&B servers
                run.summary["best/bwt"] = final_bwt
                run.summary.update()
                print("  ✅ Successfully updated W&B summary.")
            else:
                print("  ⚠️ Could not find the BWT string in output.log.")
                
            # Clean up the downloaded file so your local directory stays clean
            if os.path.exists("output.log"):
                os.remove("output.log")
        else:
            print(f"  - Skipping. best/bwt is already populated with: {current_bwt}")
            
    except Exception as e:
        print(f"❌ Failed to process run {run.id}. Error: {e}")

print("\nFinished processing all runs.")