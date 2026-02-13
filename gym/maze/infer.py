from ssm import create_actor, STATE_DIM, ACT_DIM
import torch
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import random
from pathlib import Path
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN=20
import torch.nn.functional as F

LOSS_ACHIEVED = "0.05603680570431529"
index = 13
FOLDER_PATH = Path(f"runs/{index}_{LOSS_ACHIEVED}")
FOLDER_PATH.mkdir(parents=True, exist_ok=True)
actor = create_actor(device)

actor.load_state_dict(torch.load(f"{FOLDER_PATH}/agent.pt", map_location=device))
actor.eval()


# env = gym.make("MiniGrid-MemoryS9-v0", render_mode="rgb_array")
# env = RecordVideo(env, video_folder=FOLDER_PATH / "videos", name_prefix="") # Record every episode
# env = FlatObsWrapper(env)

from recurrent_ppo_truncated_bptt.environments.minigrid_env import Minigrid
import numpy as np

env = Minigrid(env_name = "MiniGrid-MemoryS9-v0", realtime_mode = True)
# env = RecordVideo(env, video_folder=FOLDER_PATH / "videos", name_prefix="") # Record every episode


random_seed = random.randint(0,1_000_000)
# random_seed = 297
# env.set_wrapper_attr("name_prefix", random_seed)

obs = env.reset()
done = False

# 2. Initialize sequence buffers
# Shape: (Batch=1, Seq=1, C=3, H=84, W=84)
states = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
NULL_ACTION = 6 
actions = torch.full((1, 1), NULL_ACTION, dtype=torch.long).to(device)

target_return = 0.96 # The "Expert" goal
rtgs = torch.tensor([target_return]).float().reshape(1, 1, 1).to(device)

done = False
current_target = target_return

total_reward = 0
c = 0
while not done:
    with torch.no_grad():
        # print("\n=== INFERENCE INPUT DEBUG ===")
        # print(f"State Range: {states.min().item():.4f} to {states.max().item():.4f}")
        # print(f"State Mean:  {states.mean().item():.4f}")
        # print(f"Action Unique Values: {torch.unique(actions).tolist()}")
        # print(f"Shape: {states.shape}")
        # print("============================\n")
            
        # model expects (B, L, C, H, W) for states
        # logits shape: (B, L, act_dim)
        logits = actor(states, actions, rtgs)
        # Take the action from the very last timestep
        # temperature = 1.0

        # probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        # action = torch.multinomial(probabilities, num_samples=1).item()

        # print(probabilities, action)
        print(logits[:, -1, :])
        action = torch.argmax(logits[:, -1, :], dim=-1).item()

    obs, reward, done, info = env.step(action)
    print(obs, action)

    print(obs, obs.shape, reward, action, actions.shape)
    print(f"Action Unique Values: {torch.unique(actions).tolist()}")

    # --- CRITICAL FIX START ---
    # Update the buffer: The action we just used to transition FROM the current state
    # must be recorded in the current timestep's slot.
    actions[:, -1] = torch.tensor([[action]]).to(device)
    # --- CRITICAL FIX END ---
    
    total_reward += reward

    # 3. PREPARE NEXT STEP
    cur_state = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Fix RTG: Keep it constant! (Remove the -= decay line)
    cur_rtg = torch.tensor([target_return]).float().reshape(1, 1, 1).to(device)

    # Create a new placeholder for the NEXT action (initialized to 0)
    next_action_placeholder = torch.full((1, 1), NULL_ACTION, dtype=torch.long).to(device)
    
    # 5. Concatenate
    states = torch.cat([states, cur_state], dim=1)
    rtgs = torch.cat([rtgs, cur_rtg], dim=1)
    actions = torch.cat([actions, next_action_placeholder], dim=1) # Append placeholder

    # 6. Sliding Window (Crucial)
    # Ensure this matches the 'CONTEXT_LEN' used in TrajectoryDataset
    if states.shape[1] > CONTEXT_LEN:
        states = states[:, -CONTEXT_LEN:, ...]
        actions = actions[:, -CONTEXT_LEN:]
        rtgs = rtgs[:, -CONTEXT_LEN:, :]
    
    c += 1

env.render()
print(f"Steps taken: {c}")
print(f"Episode finished with reward: {total_reward}")
env.close()