
import os
import importlib
from dataclasses import dataclass
import torch
import torch.nn.functional as F

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
    model=ModelArch.SSM
)

print(CURRENT_CONFIG.dataset_id)

# --- LEARNING CONFIGURATION ---
LEARNING_RATE = 1e-4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- load environment and dataset ---
gym = importlib.import_module(CURRENT_CONFIG.gym.value)
print(gym)
dataset, loader = gym.loadDataset(CURRENT_CONFIG)
# PREFIX = F"{CURRENT_CONFIG.gym}_{ARCHITECTURE}_"


# --- 3. MODEL SETUP ---

model = importlib.import_module(CURRENT_CONFIG.model.value)
print(model)

actor = model.create_actor(DEVICE)
optimizer = torch.optim.AdamW(actor.parameters(), lr=LEARNING_RATE)


# --- Progress save and indexing ---
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


# --- 4. TRAINING LOOP ---
try:
    print(f"Starting training on {DEVICE}...")
    actor.train()

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
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.5f}")
except:
    print("Training stopped")
finally:
    
    # PREFIX = F"{CURRENT_CONFIG.gym}_{CURRENT_CONFIG.model}_"CURRENT_CONFIG
    # filename, suffix = find_biggest_suffix('saves', PREFIX)
    # print
    # if filename:
    #     index = suffix
    # else:
    #     index = 0
    save_path =  f"{CURRENT_CONFIG.gym.value}/runs/{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_Loss_{total_loss/len(loader):.5f}.pt"
    torch.save(actor.state_dict(), save_path)
    print(f"Done. Model saved as {save_path}")

