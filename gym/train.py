
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
    context_length: int
    
    @property
    def dataset_id(self) -> str:
        return f"mujoco/{self.gym.value}/{self.level.value}-v0"

# --- TRUE CONFIGURATIONNNNN ---
CURRENT_CONFIG = ExperimentConfig(
    gym=Gyms.HOPPER,
    level=AgentLevel.MEDIUM,
    model=ModelArch.SSM,
    context_length=64
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
actor = model.create_actor(DEVICE)

def train():
    # --- 1. SETUP ---
    LEARNING_RATE = 8e-4 if CURRENT_CONFIG.model == ModelArch.SSM else 1e-3
    optimizer = torch.optim.AdamW(actor.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)



    LOSS_ACHIEVED = "0.04324"

    RUN_DIR = f"{CURRENT_CONFIG.gym.value}/runs/{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_Loss_{LOSS_ACHIEVED}"
    PATH_OF_SAVE = f"{RUN_DIR}/agent.pt"
    actor.load_state_dict(torch.load(PATH_OF_SAVE, map_location=DEVICE))

    best_save_path = Path(f"{CURRENT_CONFIG.gym.value}/runs/{CURRENT_CONFIG.model.value}_best.pt")
    best_save_path.parent.mkdir(parents=True, exist_ok=True)

    avg_loss = 0.0 # Initialize for finally block

    try:
        actor.train()
        best_loss = float('inf')
        patience_counter = 0
        PATIENCE_LIMIT = 3  # Stop if no improvement for 3 epochs
        EPOCHS = 10 # Increased slightly for Mamba convergence
        total_loss = 0
        
        for epoch in range(EPOCHS):
            total_loss = 0.0
            
            for batch in loader:
                s = batch['states'].to(DEVICE)
                a = batch['actions'].to(DEVICE)
                r = batch['rtg'].to(DEVICE)
                m = batch['mask'].to(DEVICE)

                # Forward
                pred_action = actor(s, a, r)

                # Loss: (B, T, A) -> (B, T)
                # No reduction='mean' yet, we do it manually with the mask
                loss = F.mse_loss(pred_action, a, reduction='none').mean(dim=-1)
                masked_loss = (loss * m).sum() / (m.sum() + 1e-8)

                optimizer.zero_grad()
                masked_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                total_loss += masked_loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.6f}")


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

    except Exception as e:
        print(f"Error during training: {e}")
        raise e
    finally:
        # Standardize folder naming
        folder_name = f"{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_Loss_{avg_loss:.5f}"
        run_dir = Path(CURRENT_CONFIG.gym.value) / "runs" / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(actor.state_dict(), run_dir / "agent.pt")
        print(f"Final model saved to {run_dir}")

if __name__ == "__main__":
    train()

