import wandb

# 1. Set your project details
entity = "michalowski-jb-tilburg-university"
project = "HyperFisher"

# --- 1. String Replacement Helper ---
def rename_str(text):
    """Replaces both lowercase and camelCase variations."""
    if isinstance(text, str):
        return text.replace('efopng', 'ifopng').replace('eFOPNG', 'iFOPNG')
    return text

# --- 2. Deep Recursion & Sanitization ---
def deep_replace(val, changed_flag):
    """
    Recursively digs through lists and dicts, sanitizes W&B objects, 
    and replaces strings anywhere they hide.
    """
    if hasattr(val, 'keys'):  # Catches standard dicts AND W&B SummarySubDicts
        new_dict = {}
        for k, v in val.items():
            new_k = rename_str(k)
            if new_k != k:
                changed_flag[0] = True
            new_dict[new_k] = deep_replace(v, changed_flag)
        return new_dict
        
    elif isinstance(val, list):
        return [deep_replace(v, changed_flag) for v in val]
        
    elif isinstance(val, str):
        new_val = rename_str(val)
        if new_val != val:
            changed_flag[0] = True
        return new_val
        
    else:
        return val # Ints, floats, bools, etc. pass through unchanged

# --- 3. Main Logic ---
api = wandb.Api()

# Your specific nested list filter
run_filter = {
    "config.methods": {
        "$in": [
            ["efopng"], ["efopng_ema"], ["efopng_prefisher"], 
            ["ifopng"], ["ifopng_ema"], ["ifopng_prefisher"]
        ]
    }
}

runs = api.runs(f"{entity}/{project}", filters=run_filter)
print(f"Found {len(runs)} runs matching the filter. Starting deep migration...")

for run in runs:
    needs_update = False
    
    # --- Pass over the Config ---
    new_config = {}
    config_keys_to_delete = []
    
    for key, value in run.config.items():
        changed = [False] # Pass-by-reference flag to detect deep changes
        new_key = rename_str(key)
        new_val = deep_replace(value, changed)
        
        # If the key changed, or a value deep inside changed
        if new_key != key or changed[0]:
            new_config[new_key] = new_val
            if new_key != key:
                config_keys_to_delete.append(key)
                
    if new_config:
        run.config.update(new_config)
        for k in config_keys_to_delete:
            del run.config[k]
        needs_update = True

    # --- Pass over the Summary ---
    new_summary = {}
    summary_keys_to_delete = []
    
    for key, value in run.summary.items():
        changed = [False]
        new_key = rename_str(key)
        new_val = deep_replace(value, changed)
        
        if new_key != key or changed[0]:
            new_summary[new_key] = new_val
            if new_key != key:
                summary_keys_to_delete.append(key)

    if new_summary:
        run.summary.update(new_summary)
        for k in summary_keys_to_delete:
            del run.summary[k]
        needs_update = True

    # --- Push Updates to W&B ---
    if needs_update:
        run.update()
        print(f"Updated run: {run.name} ({run.id})")
        print(f"  -> Config updates: {len(new_config)} | Deletions: {len(config_keys_to_delete)}")
        print(f"  -> Summary updates: {len(new_summary)} | Deletions: {len(summary_keys_to_delete)}")

print("Deep migration complete!")