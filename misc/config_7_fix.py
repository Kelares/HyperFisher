import wandb

# Initialize the API
api = wandb.Api()

# Replace with your actual W&B username/entity and project name
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"

# Define the MongoDB-style filter
filters = {
    "$and": [
        {"jobType": "HyperNet_Reg_True"},
        {"group": "Split_mnist_sh"}, 
        {"config.hyper_hidden_dim": 8},
    ]
}
print(filters)

# Fetch the runs
runs = api.runs(path=f"{entity}/{project}", filters=filters)

print(f"Found {len(runs)} matching runs.")

# Iterate through the results
for run in runs:
    print(run.id, run.config)

    run.config["experiment_id"] = 407
    print(run.id, run.config["experiment_id"])

    run.update()