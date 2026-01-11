
import os
import importlib
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from pathlib import Path

from enum import Enum
class ModelArch(Enum):
    TRANSFORMER = "transformer"
    SSM = "ssm"

class AgentLevel(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    EXPERT = "expert"

class Gyms(Enum):
    HOPPER = "hopper"


# --- EXPERIMENT SELECTION ---
@dataclass
class ExperimentConfig:
    gym: Gyms
    level: AgentLevel
    model: ModelArch

    @property
    def dataset_id(self) -> str:
        return f"mujoco/{self.gym.value}/{self.level.value}-v0"

# --- TRUE CONFIGURATIONNNNN ---
CURRENT_CONFIG = ExperimentConfig(
    gym=Gyms.HOPPER,
    level=AgentLevel.MEDIUM,
    model=ModelArch.TRANSFORMER
)

print(CURRENT_CONFIG.dataset_id)

# --- LEARNING CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- load environment and dataset ---
gym = importlib.import_module(CURRENT_CONFIG.gym.value)
print(gym)
dataset, loader = gym.loadDataset(CURRENT_CONFIG)
# PREFIX = F"{CURRENT_CONFIG.gym}_{ARCHITECTURE}_"


# --- 3. MODEL SETUP ---

model = importlib.import_module(CURRENT_CONFIG.model.value)
print(model)
match CURRENT_CONFIG.model.value:
    case "ssm":
        LEARNING_RATE = 8e-4
    case "transformer":
        LEARNING_RATE = 1e-3

actor = model.create_actor(DEVICE)
optimizer = torch.optim.AdamW(actor.parameters(), lr=LEARNING_RATE)

#LOSS_ACHIEVED = "0.00036"
#RUN_DIR = f"{CURRENT_CONFIG.gym.value}/runs/{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_Loss_{LOSS_ACHIEVED}"
#PATH_OF_SAVE = f"{RUN_DIR}/agent.pt"
#actor.load_state_dict(torch.load(PATH_OF_SAVE, map_location=DEVICE))

# --- 4. TRAINING LOOP --- 

best_save_path = f"{CURRENT_CONFIG.gym.value}/runs/{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_best.pt"
try:
    print(f"Starting training on {DEVICE}...")
    actor.train()

    best_loss = float('inf')
    patience_counter = 0
    PATIENCE_LIMIT = 3  # Stop if no improvement for 3 epochs
    EPOCHS = 1000

    for epoch in range(EPOCHS):
        total_loss = 0
        for i, batch in enumerate(loader):
            s = batch['states'].to(DEVICE)
            a = batch['actions'].to(DEVICE)
            r = batch['rtg'].to(DEVICE)
            m = batch['mask'].to(DEVICE)
            # Predict action
            pred_action = actor(s, a, r)

            # Loss: (Predicted Action - Real Action)^2
            loss = F.mse_loss(pred_action, a, reduction='none')
            # Apply mask (ignore padding)
            loss = (loss.mean(dim=-1) * m).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.5f}")

        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(actor.state_dict(), best_save_path)
            print(f"   ✅ New Best Model Saved (Loss: {best_loss:.5f})")
        else:
            patience_counter += 1
            print(f"   No improvement. Patience: {patience_counter}/{PATIENCE_LIMIT}")
            
        if patience_counter >= PATIENCE_LIMIT:
            print("🛑 Early Stopping triggered. Model has converged.")
            break

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current state...")

except Exception as e:
    print(f"\nTraining stopped due to error: {e}")

finally:
    # 1. Calculate Average Loss (safely)
    current_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    # 2. Construct the Main Directory Path
    # Structure: hopper/runs/ssm_medium_Loss_0.12345
    folder_name = f"{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_Loss_{current_loss:.5f}"
    run_dir = Path(CURRENT_CONFIG.gym.value) / "runs" / folder_name

    # 3. Create Directories
    # parents=True creates 'hopper/runs' if they don't exist
    # exist_ok=True prevents errors if the folder already exists
    (run_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)

    # 4. Save the Agent File inside the main folder
    save_path = run_dir / "agent.pt"
    torch.save(actor.state_dict(), save_path)

    print(f"✅ Run environment created at: {run_dir}")
    print(f"💾 Model saved as: {save_path}")






    # PREFIX = F"{CURRENT_CONFIG.gym}_{CURRENT_CONFIG.model}_"CURRENT_CONFIG
    # filename, suffix = find_biggest_suffix('saves', PREFIX)
    # print
    # if filename:
    #     index = suffix
    # else:
    #     index = 0
'''# --- Progress save and indexing ---
import re
from pathlib import Path

def find_biggest_suffix(directory_path, prefix):
    # 1. Define the directory
    dir_path = Path(directory_path)
    
    # 2. Define the pattern: prefix + underscore + digits + end of string
    # "smth_(\d+)$" means: match 'smth_', capture digits, ensure it ends there.
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+).pt$")
    
    max_num = -1
    max_file = None

    # 3. Iterate through files
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                # Extract the number (group 1) and convert to int
                num = int(match.group(1))
                
                if num > max_num:
                    max_num = num
                    max_file = file_path.name

    return max_file, max_num

'''
