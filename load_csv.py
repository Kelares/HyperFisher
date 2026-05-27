import pandas as pd
import wandb
api = wandb.Api()
IDS = [405] #[401, 402, 405, 406, 412, 413]
# Project is specified by <entity/project-name>
runs = api.runs("michalowski-jb-tilburg-university/HyperFisher", filters={"config.experiment_id": {"$in": IDS}})
print(f"Found {len(runs)} runs.")

d = {}

for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    exp_id = run.config["experiment_id"]
    if not exp_id:
        print(exp_id, run.id)
        continue
    if exp_id not in d:
        d[exp_id] = []

    temp = {}
    temp["summary"] = run.summary._json_dict

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    temp["config"] = {k: v for k,v in run.config.items() if not k.startswith('_')}

    print(exp_id, run.id)
    d[exp_id].append(temp)

for experiment_id in d:
    runs_df = pd.DataFrame(d[experiment_id])
    runs_df.to_csv(f"results/{experiment_id}.csv")