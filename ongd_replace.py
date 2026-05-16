import wandb

api = wandb.Api()
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"
job_type = "Target_Network"

runs = api.runs(f"{entity}/{project}", filters={"jobType": job_type})

for run in runs:
    methods = run.config.get("methods", [])
    
    if any(m in ["ognd", "odng"] for m in methods):
        print(f"Renaming {run.id}...")
        
        # 1. Update the Config list
        new_methods = ["ong" if m in ["ognd", "odng"] else m for m in methods]
        run.config["methods"] = new_methods
        
        # 2. Extract and Deep Copy the results into PLAIN dicts
        # We look for the data in either 'ognd/results' or 'odng/results'
        old_key = "ognd/results" if "ognd/results" in run.summary else "odng/results"
        raw_data = run.summary.get(old_key)
        
        if raw_data and "acc" in raw_data:
            # Reconstruct everything as plain Python types
            # This is the "Clean Break" that prevents the json_encode error
            clean_results = {
                "acc": {str(k): v for k, v in raw_data["acc"].items()},
                "bwt": raw_data.get("bwt")
            }
            
            # 3. Assign the plain dict to the new key
            run.summary["ong/results"] = clean_results
            
            # 4. Clear the old "bad" key by setting it to None
            run.summary[old_key] = None
        
        # 5. Push all changes
        run.update()
        print(f"  Successfully migrated to 'ong'")

print("\nCleanup complete.")