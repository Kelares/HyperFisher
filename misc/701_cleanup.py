import wandb

# 1. Set your project details
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"
api = wandb.Api()

# 2. Add the specific run IDs you want to mark as finished
runs = api.runs(f"{entity}/{project}", filters={"group": "move_to_701"})

for run in runs:
    try:
        print(f"Connecting to run {run.id}, {run.config['experiment_id']}...")
        
        run.config["experiment_id"] = 701
        run.update()
        print(run.config["experiment_id"])
    except Exception as e:
        print(f"❌ Failed to update run {run.id}. Error: {e}\n")