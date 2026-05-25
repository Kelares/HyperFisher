import wandb

# 1. Set your project details
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"
api = wandb.Api()

# 2. Add the specific run IDs you want to mark as finished
runs = api.runs(f"{entity}/{project}", filters={"group": "garbage_405"})

for run in runs:
    try:
        print(f"Connecting to run {run.id}, {run.config['experiment_id']}...")
        
        run.config["experiment_id"] = None
        run.update()
        print(run.config["experiment_id"])
    except Exception as e:
        print(f"❌ Failed to update run {run.id}. Error: {e}\n")