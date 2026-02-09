import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from ssm import create_actor, STATE_DIM, ACT_DIM
import torch.nn.functional as F
import re
import os
from pathlib import Path


env = gym.make("MiniGrid-MemoryS13Random-v0", render_mode=None)
env = FlatObsWrapper(env)
device = "cuda" if torch.cuda.is_available() else "cpu"

    
def load_dataset(file_name):
    with open(file_name, "rb") as f:
        dataset = pickle.load(f)
    return dataset

sieve = load_dataset("perfect_expert_data+.pkl")

class MiniGridTrajectoryDataset(Dataset):
    def __init__(self, trajectories, context_len=20):
        self.trajectories = trajectories
        self.context_len = context_len
        
        # Calculate start indices for sampling sub-sequences
        self.indices = []
        for traj_idx, traj in enumerate(trajectories):
            traj_len = len(traj['observations'])
            # We want to be able to sample every possible starting point
            for i in range(traj_len):
                self.indices.append((traj_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_step = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        # Determine the end of our sub-sequence
        end_step = start_step + self.context_len
        traj_total_len = len(traj['observations'])
        
        # Slice the data (handle cases where trajectory ends before context_len)
        real_end = min(end_step, traj_total_len)
        s = traj['observations'][start_step:real_end]
        a = traj['actions'][start_step:real_end]
        r = traj['rewards'][start_step:real_end]
        rtg = traj['rtg'][start_step:real_end]
        
        # 1. Padding: Convert to torch and pad with zeros if sequence is too short
        curr_len = len(s)
        pad_len = self.context_len - curr_len
        
        # Observations (B, K, 147)
        states = torch.tensor(np.array(s), dtype=torch.float32)

        curr_len, feature_dim = states.shape # feature_dim should be 147
        
        if pad_len > 0:
            # print(f"DEBUG: feature_dim {feature_dim}, pad_len: {pad_len}")
            padding = torch.zeros((pad_len, feature_dim))
            states = torch.cat([states, padding], dim=0)        

        # Actions (B, K) - Long for CrossEntropy
        actions = torch.tensor(np.array(a), dtype=torch.long)
        if pad_len > 0:
            actions = torch.cat([actions, torch.zeros((pad_len,), dtype=torch.long)], dim=0)
            
        # RTGs (B, K, 1)
        returns = torch.tensor(np.array(rtg), dtype=torch.float32).reshape(-1, 1)
        if pad_len > 0:
            returns = torch.cat([returns, torch.zeros((pad_len, 1))], dim=0)
            
        # 2. Masking: 1 for real data, 0 for padding
        mask = torch.cat([torch.ones(curr_len), torch.zeros(pad_len)], dim=0)
        
        return {
            'states': states,
            'actions': actions,
            'rtgs': returns,
            'mask': mask
        }

# --- How to use with DataLoader ---
dataset = MiniGridTrajectoryDataset(sieve, context_len=30)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(dataloader)

def train_step(actor, optimizer, batch, device):
    # Unpack batch
    states = batch['states'].to(device)    # (B, T, 147)
    actions = batch['actions'].to(device)  # (B, T) - Long integers
    rtgs = batch['rtgs'].to(device)        # (B, T, 1)
    mask = batch['mask'].to(device)        # (B, T) - 1s for data, 0s for padding

    # 1. Forward pass
    # actor returns logits: (B, T, 7)
    logits = actor(states, actions, rtgs)
    
    # 2. Reshape for CrossEntropy
    # Flatten B and T dimensions
    logits = logits.view(-1, ACT_DIM)      # (B*T, 7)
    targets = actions.view(-1)             # (B*T)
    
    # 3. Compute Loss with Manual Masking
    # 'reduction=none' allows us to zero out the padding loss before averaging
    loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Apply the mask
    masked_loss = (loss * mask.view(-1)).sum() / mask.sum()
    
    # 4. Backward pass
    optimizer.zero_grad()
    masked_loss.backward()
    optimizer.step()
    
    return masked_loss.item()
# Run it


actor = create_actor(device)

################################################################
LOSS_ACHIEVED = "0.23"
index = 1
FOLDER_PATH = f"runs/{index}_{LOSS_ACHIEVED}"

actor = create_actor(device)

actor.load_state_dict(torch.load(f"{FOLDER_PATH}/agent.pt", map_location=device, weights_only=True))
actor.eval()
################################################################



# Define learning rate - 1e-4 is a safe starting point for Mamba/DT
learning_rate = 1e-4
weight_decay = 0.1

# Initialize Optimizer
optimizer = torch.optim.AdamW(
    actor.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay
)

num_epochs = 1  # Start with this for MiniGrid-Memory
best_loss = float('inf')

repeat_counter = 0
for epoch in range(num_epochs):
    epoch_loss = 0
    actor.train() # Set to training mode
    
    for batch in dataloader:
        loss_val = train_step(actor, optimizer, batch, device)
        epoch_loss += loss_val
        
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
    # Save the 'best' model if loss improved
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(actor.state_dict(), "mamba_maze_best.pt")
        repeat_counter = 0
    else:
        repeat_counter += 1

    if repeat_counter == 5:
        print("repeat_counter limit reached.")
        break


def get_dynamic_session_id(path="./runs"):
    if not os.path.exists(path):
        os.makedirs(path)
        return 1
    
    # Find all folders matching 'train_N'
    folders = [f for f in os.listdir(path) if re.match(r'^(\d+)_', f)]
    if not folders:
        return 1
    print(folders)
    # Extract numbers and find the max
    ids = [int(f.split('_')[0]) for f in folders]
    return max(ids) + 1



folder_name = f"{get_dynamic_session_id()}_{best_loss}"

run_dir = "runs" / Path(folder_name) 
run_dir.mkdir(parents=True, exist_ok=True)

print(f"SAVING IN {run_dir}")
torch.save(actor.state_dict(), f"{run_dir}/agent.pt")

