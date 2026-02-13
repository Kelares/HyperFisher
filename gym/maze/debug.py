import torch
import pickle
import numpy as np
from ssm import create_actor
from pathlib import Path

# 1. SETUP
device = "cuda" if torch.cuda.is_available() else "cpu"
actor = create_actor(device)

LOSS_ACHIEVED = "0.05603680570431529"
index = 13
FOLDER_PATH = Path(f"runs/{index}_{LOSS_ACHIEVED}")
actor.load_state_dict(torch.load(f"{FOLDER_PATH}/agent.pt", map_location=device))
actor.eval()

# 2. LOAD TRAJECTORY
with open("dataset_rtg_S9_80%.pickle", "rb") as f:
    dataset = pickle.load(f)

traj = dataset[0] 

# 3. PREPARE TENSORS
ctx_len = 20
s = traj['observations'][:ctx_len]
a = traj['actions'][:ctx_len]
r = traj['rtg'][:ctx_len]

# --- FIX 1: Normalize States ---
states = torch.tensor(np.array(s), dtype=torch.float32)
if states.max() > 1.0:
    states = states / 255.0  

# --- FIX 2: Correct Dimensions for Actions ---
# Must be (Batch, Time). Not (Batch, Time, 1).
# Flatten first to kill the extra '1' from the dataset list of lists.
actions = torch.tensor(np.array(a), dtype=torch.long).flatten()
actions = actions.unsqueeze(0).to(device) # Shape: (1, T)

# --- FIX 3: Correct Dimensions for RTGs ---
# This is the tricky part. 
# The Linear layer NEEDS (1, T, 1) input, but that produces (1, T, 1, Hidden) output.
# We must edit ssm.py to fix this permanently, OR we can try to rely on ssm.py squeezing it.
# Let's provide the exact shape train.py provides: (1, T, 1)
rtgs = torch.tensor(np.array(r), dtype=torch.float32).flatten()
rtgs = rtgs.reshape(1, -1, 1).to(device) # Shape: (1, T, 1)

# Add Batch Dim to States
states = states.unsqueeze(0).to(device)   # (1, T, 3, 84, 84)

# 4. DEBUG SHAPES
print(f"States:  {states.shape}")
print(f"Actions: {actions.shape}")
print(f"RTGs:    {rtgs.shape}")

# 5. EXECUTION & CRITICAL FIX FOR SSM.PY
try:
    with torch.no_grad():
        logits = actor(states, actions, rtgs)
except RuntimeError as e:
    if "stack expects each tensor to be equal size" in str(e):
        print("\n!!! DETECTED SSM.PY BUG !!!")
        print("Your ssm.py is failing to squeeze the RTG embedding.")
        print("You must open 'ssm.py' and find the forward function.")
        print("Change this line:")
        print("   rtg_embeddings = self.rtg_embed(rtgs)")
        print("To this:")
        print("   rtg_embeddings = self.rtg_embed(rtgs).squeeze(-2)")
        print("Then run this script again.")
        exit()
    else:
        raise e

# 6. COMPARE PREDICTIONS
print("\n--- TEACHER FORCING RESULTS ---")
print(f"{'Step':<5} | {'Expert':<10} | {'Model Pred':<10} | {'Confidence':<10} | {'Match?'}")
print("-" * 55)

actual_len = actions.shape[1]
for t in range(actual_len):
    expert_act = actions[0, t].item()
    model_logits = logits[0, t, :]
    model_act = torch.argmax(model_logits).item()
    confidence = torch.softmax(model_logits, dim=0)[model_act].item()
    match = "✅" if expert_act == model_act else "❌"
    print(f"{t:<5} | {expert_act:<10} | {model_act:<10} | {confidence:.4f}     | {match}")