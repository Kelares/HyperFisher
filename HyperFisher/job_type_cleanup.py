import wandb

# Initialize the Public API
api = wandb.Api()

# Specify your W&B details
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"
old_job_type = "HyperNet_Reg_False"
new_job_type = "Target_Network"

# Fetch runs in the project
runs = api.runs(f"{entity}/{project}", filters={"jobType": old_job_type})

for run in runs:
    print(f"Updating run {run.id} from '{old_job_type}' to '{new_job_type}'...")
    
    # Resume the run and set the new job type
    updated_run = wandb.init(
        entity=entity,
        project=project,
        id=run.id,
        resume="must",
        job_type=new_job_type
    )
    
    # Finish the run to close it properly
    updated_run.finish()
