from ssm import create_actor, STATE_DIM, ACT_DIM
import torch
import gymnasium as gym
import random
from pathlib import Path
import pickle
import numpy as np

# --- CRITICAL IMPORT ---
# We MUST use create_env to match the exact wrapper stack used in generate_dataset.py
from utils import create_env 

device = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN = 20

LOSS_ACHIEVED = "0.05603680570431529"
index = 13
FOLDER_PATH = Path(f"runs/{index}_{LOSS_ACHIEVED}")
FOLDER_PATH.mkdir(parents=True, exist_ok=True)
actor = create_actor(device)

actor.load_state_dict(torch.load(f"runs/big/agent.pt", map_location=device)) # Note: Fixed path to use FOLDER_PATH
actor.eval()

# --- INITIALIZE ENVIRONMENT EXACTLY LIKE TRAINING ---
state_dict, config = pickle.load(open("./minigrid.nn", "rb"))
config["environment"]["name"] = "MiniGrid-MemoryS9-v0"

# This guarantees the images are the right way up!
env = create_env(config["environment"], render=True)

random_seed = random.randint(0, 1_000_000)

obs = env.reset()
done = False

# Normalize state just in case (to ensure 0.0-1.0 range)
state_tensor = torch.from_numpy(obs).float()
if state_tensor.max() > 1.0:
    state_tensor /= 255.0

# 2. Initialize sequence buffers
# Shape: (Batch=1, Seq=1, C=3, H=84, W=84)
states = state_tensor.unsqueeze(0).unsqueeze(0).to(device)
NULL_ACTION = 6 
actions = torch.full((1, 1), NULL_ACTION, dtype=torch.long).to(device)

target_return = 0.96 # The "Expert" goal
rtgs = torch.tensor([target_return]).float().reshape(1, 1, 1).to(device)

total_reward = 0
c = 0

while not done:
    with torch.no_grad():
        logits = actor(states, actions, rtgs)
        action = torch.argmax(logits[:, -1, :], dim=-1).item()

    # --- EXECUTE ---
    # create_env expects a list for the action!
    step_result = env.step(action) 
    
    # Gracefully handle Gym version differences (4 vs 5 returns)
    if len(step_result) == 4:
        obs, reward, done, info = step_result
    else:
        obs, reward, done, trunc, info = step_result
        if trunc: done = True
        
    print(f"Step: {c} | Action: {action} | Reward: {reward}")

    # --- CRITICAL FIX START ---
    # Update the buffer: The action we just used to transition FROM the current state
    # must be recorded in the current timestep's slot.
    actions[:, -1] = torch.tensor([[action]]).to(device)
    # --- CRITICAL FIX END ---
    
    total_reward += reward

    # 3. PREPARE NEXT STEP
    cur_state = torch.from_numpy(obs).float()
    if cur_state.max() > 1.0:
        cur_state /= 255.0
    cur_state = cur_state.unsqueeze(0).unsqueeze(0).to(device)
    
    # Fix RTG: Keep it constant!
    cur_rtg = torch.tensor([target_return]).float().reshape(1, 1, 1).to(device)

    # Create a new placeholder for the NEXT action
    next_action_placeholder = torch.full((1, 1), NULL_ACTION, dtype=torch.long).to(device)
    
    # 5. Concatenate
    states = torch.cat([states, cur_state], dim=1)
    rtgs = torch.cat([rtgs, cur_rtg], dim=1)
    actions = torch.cat([actions, next_action_placeholder], dim=1)

    # 6. Sliding Window (Crucial)
    if states.shape[1] > CONTEXT_LEN:
        states = states[:, -CONTEXT_LEN:, ...]
        actions = actions[:, -CONTEXT_LEN:]
        rtgs = rtgs[:, -CONTEXT_LEN:, :]
    
    c += 1

print(f"Steps taken: {c}")
print(f"Episode finished with reward: {total_reward}")