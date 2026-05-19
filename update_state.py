import wandb

# 1. Set your project details
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"

# 2. Add the specific run IDs you want to mark as finished
RUN_IDS = ["h002e9fd", "ux4omlt4"] 

for run_id in RUN_IDS:
    try:
        print(f"Connecting to run {run_id}...")
        
        # Force resume the specific run ID
        run = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume="must",
            settings=wandb.Settings(quiet=True) # Keeps the console logs clean
        )
        
        # Explicitly finish with exit_code=0 to guarantee a 'finished' state
        run.finish(exit_code=0)
        
        print(f"✅ Run {run_id} successfully changed to 'finished'.\n")
        
    except Exception as e:
        print(f"❌ Failed to update run {run_id}. Error: {e}\n")