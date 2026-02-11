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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def load_dataset(file_name):
    with open(file_name, "rb") as f:
        dataset = pickle.load(f)
    return dataset

dataset = load_dataset("rtg+S9_dataset.pickle")

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
        
        # Slice the data (seq_len, 3, 84, 84)
        real_end = min(end_step, traj_total_len)
        s = traj['observations'][start_step:real_end]
        a = traj['actions'][start_step:real_end]
        r = traj['rewards'][start_step:real_end]
        rtg = traj['rtg'][start_step:real_end]
        
        # 1. Padding Logic
        curr_len = len(s)
        pad_len = self.context_len - curr_len
        
        # Observations: Shape (curr_len, 3, 84, 84)
        states = torch.tensor(np.array(s), dtype=torch.float32)
        if states.max() > 1.0:
            print(states.max(), "state out of bounds")
        # Correctly capture dimensions for padding
        # Unpack the 4 dimensions: sequence, channels, height, width
        _, c, h, w = states.shape 
        
        if pad_len > 0:
            # Create padding with the same spatial dimensions
            padding = torch.zeros((pad_len, c, h, w), dtype=torch.float32)
            states = torch.cat([states, padding], dim=0)        

        # Actions: Long for CrossEntropy
        actions = torch.tensor(np.array(a), dtype=torch.long).flatten()
        if pad_len > 0:
            actions = torch.cat([actions, torch.zeros((pad_len,), dtype=torch.long)], dim=0)
            
        # RTGs: Shape (curr_len, 1)
        returns = torch.tensor(np.array(rtg), dtype=torch.float32).reshape(-1, 1)
        if pad_len > 0:
            returns = torch.cat([returns, torch.zeros((pad_len, 1))], dim=0)
            
        # Masking: 1 for real data, 0 for padding
        mask = torch.cat([torch.ones(curr_len), torch.zeros(pad_len)], dim=0)
        
        return {
            'states': states,    # (context_len, 3, 84, 84)
            'actions': actions,  # (context_len,)
            'rtgs': returns,     # (context_len, 1)
            'mask': mask         # (context_len,)
        }

# --- How to use with DataLoader ---
trajectory_dataset = MiniGridTrajectoryDataset(dataset)
dataloader = DataLoader(trajectory_dataset, batch_size=64, shuffle=True)
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

    # 3a. Apply a "Reward Weight" 
    # This makes the model learn more from steps that have a high RTG
    # We take the RTG at each step and use it as a multiplier
    reward_weights = rtgs.view(-1) 
    
    # 3b. We add 0.1 so that even 0-reward samples have a tiny bit of influence
    weighted_loss = loss * (reward_weights + 0.1)

    # 4. Apply the mask
    masked_loss = (weighted_loss * mask.view(-1)).sum() / mask.sum()
    
    # 5. Backward pass
    optimizer.zero_grad()
    masked_loss.backward()
    optimizer.step()
    
    return masked_loss.item()


actor = create_actor(device)

################################################################
# LOSS_ACHIEVED = "0.1336"
# index = 9
# FOLDER_PATH = Path(f"runs/{index}_{LOSS_ACHIEVED}")
# FOLDER_PATH.mkdir(parents=True, exist_ok=True)
# actor = create_actor(device)

# actor.load_state_dict(torch.load(f"{FOLDER_PATH}/agent.pt", map_location=device))
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

num_epochs = 500  # Start with this for MiniGrid-Memory
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

    if repeat_counter == 50:
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

