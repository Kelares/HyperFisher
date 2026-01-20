import torch
import numpy as np
from torchrl.modules import DTActor, DecisionTransformer
from enum import Enum
from dataclasses import dataclass
import importlib
import random

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
    record: bool

    @property
    def dataset_id(self) -> str:
        return f"mujoco/{self.gym.value}/{self.level.value}-v0"


CURRENT_CONFIG = ExperimentConfig(
    gym=Gyms.HOPPER,
    level=AgentLevel.MEDIUM,
    model=ModelArch.SSM,
    record=True
)
# LOSS_ACHIEVED = "0.01577"
# LOSS_ACHIEVED = "0.00576" #ssm
# LOSS_ACHIEVED = "0.00326" # PERFECT SSM
LOSS_ACHIEVED = "0.00046" # PERFECT TRANSFORMER

RUN_DIR = f"{CURRENT_CONFIG.gym.value}/runs/{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_Loss_{LOSS_ACHIEVED}"
PATH_OF_SAVE = f"{RUN_DIR}/agent.pt"

# --- CONFIGURATION ---
# Must match your training config exactly!
CONTEXT_LEN = 20
RTG_SCALE = 1000.0
TARGET_RETURN = 3600.0  # We ask the model for an "Expert" performance
DEVICE = "cuda"          # Inference is fast enough on CPU

# --- 1. SETUP ENVIRONMENT & MODEL ---
gym = importlib.import_module(CURRENT_CONFIG.gym.value)
print(gym)
env, state_dim, action_dim, state_mean, state_std, act_mean, act_std = gym.liveEnv(CURRENT_CONFIG, DEVICE, RUN_DIR)

# Initialize Model Architecture
model = importlib.import_module(CURRENT_CONFIG.model.value)
print(model)
actor = model.create_actor(DEVICE)

# Load Weights
print(f"LOADING {PATH_OF_SAVE} WEIGHTS !!!!!!!!!!!!!!")
actor.load_state_dict(torch.load(PATH_OF_SAVE, map_location=DEVICE))
actor.eval()

# --- 2. INFERENCE UTILITIES ---
def get_action(states, actions, rewards, rtg_target):
    # 1. Pad to Context Length (if history is short, e.g. start of episode)
    # We always need exactly CONTEXT_LEN steps for the model
    T = states.shape[1]
    
    if T < CONTEXT_LEN:
        # Create padding
        pad_len = CONTEXT_LEN - T
        
        # Pad States (zeros)
        s_pad = torch.zeros(1, pad_len, state_dim, device=DEVICE)
        s_input = torch.cat([s_pad, states], dim=1)
        
        # Pad Actions (zeros)
        a_pad = torch.zeros(1, pad_len, action_dim, device=DEVICE)
        a_input = torch.cat([a_pad, actions], dim=1)
        
        # Pad RTG (zeros)
        r_pad = torch.zeros(1, pad_len, 1, device=DEVICE)
        r_input = torch.cat([r_pad, rewards], dim=1)
    else:
        # Crop to last CONTEXT_LEN steps
        s_input = states[:, -CONTEXT_LEN:, :]
        a_input = actions[:, -CONTEXT_LEN:, :]
        r_input = rewards[:, -CONTEXT_LEN:, :]

    # 2. Forward Pass
    with torch.no_grad():
        # DTActor output shape: [Batch, Time, Action_Dim]
        # We only want the LAST action prediction
        action_pred = actor(s_input, a_input, r_input)
    
    return action_pred[0, -1] # Return the last action

# --- 3. RUNNING THE LOOP ---
print(f"Targeting Return: {TARGET_RETURN}")
random_seed = random.randint(0,1_000_000)
env.set_wrapper_attr("name_prefix", random_seed)

obs, _ = env.reset(seed=random_seed)
print(env.unwrapped.np_random_seed)

done = False
total_reward = 0
OCCLUSION_LENGTH = 0#25#32 #20 # Add blind frames  #32 SEEMS TO BE A GOOD BREAKING POINT FOR A 62 CONTEXT WINDOW SSM MEMORY OBSTRUCTION
step_counter = 0
GLITCH_START = 500 
GLITCH_END = GLITCH_START + OCCLUSION_LENGTH


# Buffers to hold history (Start empty)
# Dimensions: [Batch=1, Time, Dim]
history_states = torch.tensor(obs, device=DEVICE).float().view(1, 1, state_dim)
history_states = (history_states - state_mean) / state_std # NORMALIZE IMMEDIATELY

history_actions = torch.zeros((1, 1, action_dim), device=DEVICE) # Dummy action for t=0
history_rtg = torch.tensor([[[TARGET_RETURN / RTG_SCALE]]], device=DEVICE).float()

while not done:
    # --- PRE A. APPLY OCCLUSION ---
    # If we are in the glitch window, the model sees ZERO (or last frame)
    if GLITCH_START <= step_counter < GLITCH_END:
        # OPTION 1: Total Blackout (Zeros) - Hardest test
        # We replace the *input* to the neural net with zeros
        # (But we keep the real history buffer intact for when sensors come back)
        current_state_input = torch.zeros_like(history_states)
        # OPTION 2: Frozen Frame (Last known value) - More realistic
        # current_state_input = history_states.clone()
        # current_state_input[:, -1, :] = history_states[:, GLITCH_START-1, :]
    else:
        current_state_input = history_states

    # A. Ask Model for Action
    action = get_action(current_state_input, history_actions, history_rtg, TARGET_RETURN)
    
    # --- STEP 3 FIX: UN-NORMALIZE FOR ENVIRONMENT ---
    # We must scale the action back to raw physics units using the loaded stats
    action_raw = (action * act_std) + act_mean

    action_np = action_raw.cpu().numpy()

    # B. Step Environment
    next_obs, reward, terminated, truncated, _ = env.step(action_np)

    if terminated or truncated:
        print(f"\n🛑 Episode Ended!")
        print(f"   Steps taken: {len(history_actions[0])}")
        print(f"   Reason: {'💀 DIED (Terminated)' if terminated else '⏰ TIMEOUT (Truncated)'}")

    done = terminated or truncated
    total_reward += reward
    step_counter += 1

    # C. Update History Buffers
    # 1. Normalize and Append New State
    next_obs_t = torch.tensor(next_obs, device=DEVICE).float().view(1, 1, state_dim)
    next_obs_t = (next_obs_t - state_mean) / state_std
    history_states = torch.cat([history_states, next_obs_t], dim=1)

    # 2. Append Action Taken
    action_t = action.view(1, 1, action_dim)
    history_actions = torch.cat([history_actions, action_t], dim=1)

    # 3. Calculate and Append New Return-to-Go
    # RTG decreases as we collect reward (We need less future reward to hit target)
    current_rtg = history_rtg[0, -1, 0].item()
    new_rtg_val = current_rtg - (reward / RTG_SCALE)
    new_rtg_t = torch.tensor([[[new_rtg_val]]], device=DEVICE).float()
    history_rtg = torch.cat([history_rtg, new_rtg_t], dim=1)

print(f"Episode Finished. Total Reward: {total_reward:.2f}")
env.close()