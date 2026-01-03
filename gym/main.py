import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchrl.modules import DTActor, DecisionTransformer
import minari
import numpy as np
import os

# --- CONFIGURATION ---
BATCH_SIZE = 64
CONTEXT_LEN = 20      # The model looks at the last 20 steps
RTG_SCALE = 1000.0    # Scale returns so 3000 becomes 3.0
LEARNING_RATE = 1e-4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX = "dt_hopper_SIMPLE"

# --- 1. DATASET CLASS (The "Slicer") ---
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, context_len):
        self.context_len = context_len
        self.indices = []
        self.trajectories = trajectories
        
        # Index every valid timestep in every trajectory
        for i, traj in enumerate(trajectories):
            # We can start predicting from step 0 up to end-1
            for t in range(len(traj['rewards'])):
                self.indices.append((i, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, end_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        # Calculate window start (handle negative indices for padding)
        start_idx = end_idx - self.context_len + 1
        real_start = max(0, start_idx)
        real_end = end_idx + 1
        seq_len = real_end - real_start
        
        # Create buffers (Context_Len, Dim)
        states = np.zeros((self.context_len, 11), dtype=np.float32)
        actions = np.zeros((self.context_len, 3), dtype=np.float32)
        rtg = np.zeros((self.context_len, 1), dtype=np.float32)
        mask = np.zeros((self.context_len,), dtype=np.float32)

        # Fill buffers (Right-Aligned)
        buf_start = self.context_len - seq_len
        states[buf_start:] = traj['observations'][real_start:real_end]
        actions[buf_start:] = traj['actions'][real_start:real_end]
        rtg[buf_start:] = traj['rtg'][real_start:real_end].reshape(-1, 1)
        mask[buf_start:] = 1.0

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rtg": torch.from_numpy(rtg),
            "mask": torch.from_numpy(mask)
        }

# --- 2. PREPARE DATA ---
print("Loading Minari Dataset (downloading if needed)...")
# "Medium" is best for training (contains mix of success/fail)
md = minari.load_dataset("mujoco/hopper/simple-v0", download=True)

trajectories = []
all_obs = []

# Convert Minari format to simple list of dicts
for episode in md.iterate_episodes():
    # Calculate Reward-to-Go (Reverse Cumulative Sum)
    rewards = episode.rewards
    rtg = np.cumsum(rewards[::-1])[::-1] / RTG_SCALE
    
    trajectories.append({
        "observations": episode.observations[:-1], # Drop terminal state
        "actions": episode.actions,
        "rewards": rewards,
        "rtg": rtg
    })
    all_obs.append(episode.observations[:-1])

# Normalize States (CRITICAL for Transformers)
all_obs = np.concatenate(all_obs, axis=0)
mean = np.mean(all_obs, axis=0)
std = np.std(all_obs, axis=0) + 1e-6

for traj in trajectories:
    traj["observations"] = (traj["observations"] - mean) / std

# Save stats for later inference
np.savez("normalizations/hopper_normalization_simple.npz", mean=mean, std=std)

dataset = TrajectoryDataset(trajectories, CONTEXT_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. MODEL SETUP ---
dt_config = DecisionTransformer.default_config()
dt_config.n_embd = 128
dt_config.n_layer = 3
dt_config.n_head = 1
dt_config.n_positions = 1000 # Max episode steps

actor = DTActor(
    state_dim=11, 
    action_dim=3, 
    transformer_config=dt_config, 
    device=DEVICE
).to(DEVICE)

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
    filename, suffix = find_biggest_suffix('saves', PREFIX)
    print
    if filename:
        index = suffix
    else:
        index = 0

    torch.save(actor.state_dict(), f"saves/{PREFIX}_{index + 1}.pt")
    print(f"Done. Model saved as saves/{PREFIX}_{index + 1}.pt")

