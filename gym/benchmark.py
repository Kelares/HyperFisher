
import torch
import numpy as np
from torchrl.modules import DTActor, DecisionTransformer
from enum import Enum
from dataclasses import dataclass
import importlib
import json

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
    model=ModelArch.TRANSFORMER,
    record=False
)
# LOSS_ACHIEVED = "0.00576"
# PATH_OF_SAVE = f"{CURRENT_CONFIG.gym.value}/runs/{CURRENT_CONFIG.model.value}_{CURRENT_CONFIG.level.value}_Loss_{LOSS_ACHIEVED}.pt"
# LOSS_ACHIEVED = "0.00326" #PERFECT SSM
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
env, state_dim, action_dim, state_mean, state_std = gym.liveEnv(CURRENT_CONFIG, DEVICE, LOSS_ACHIEVED)

# Initialize Model Architecture
model = importlib.import_module(CURRENT_CONFIG.model.value)
print(model)
actor = model.create_actor(DEVICE)

# Load Weights
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

# --- 3. EVALUATION LOOP (Multiple Episodes) ---
NUM_EPISODES = 1#25

print(f"🚀 Starting Evaluation for {NUM_EPISODES} episodes...")
print(f"Targeting Return: {TARGET_RETURN}")

# Create a deterministic list of seeds so your thesis results are reproducible
SEEDS = [200 + i for i in range(NUM_EPISODES)]
seed_rewards = []
succesful_runs = 0
steps_taken = []
reasons = []

### OCCLUSIONS
OCCLUSION_LENGTHS = [0, 15, 30, 45, 60]
# OCCLUSION_LENGTH = 32 #32 #20 # Add blind frames  #32 SEEMS TO BE A GOOD BREAKING POINT FOR A 62 CONTEXT WINDOW SSM MEMORY OBSTRUCTION
GLITCH_START = 500 
record = {}

for seed in SEEDS:
    record[seed] = {}
    for gap in OCCLUSION_LENGTHS:
        GLITCH_END = GLITCH_START + gap

        obs, _ = env.reset(seed=seed) 
        
        done = False
        episode_reward = 0
        step_counter = 0 # occlusion

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
            action_np = action.cpu().numpy()

            # B. Step Environment
            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            if terminated or truncated:
                print(f"\n🛑 Episode Ended!")
                print(f"   Steps taken: {len(history_actions[0])}")
                print(f"   Reason: {'💀 DIED (Terminated)' if terminated else '⏰ TIMEOUT (Truncated)'}")
                # steps_taken.append(len(history_actions[0]))
                # reasons.append(f"Reason: {'💀 DIED (Terminated)' if terminated else '⏰ TIMEOUT (Truncated)'}")
                record[seed][gap] = {
                    "reward": episode_reward,
                    "steps": len(history_actions[0]),
                    "reason": f"Reason: {'💀 DIED (Terminated)' if terminated else '⏰ TIMEOUT (Truncated)'}"
                    }
            done = terminated or truncated
            episode_reward += reward
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
            
        print(f"Seed {seed}: Score {episode_reward}")
        if done==truncated:
            succesful_runs += 1
        seed_rewards.append(episode_reward)


env.close()
mean_score = np.mean(seed_rewards)
std_score = np.std(seed_rewards)
min_score = np.min(seed_rewards)
max_score = np.max(seed_rewards)

print("\n" + "="*30)
print(f"📊 FINAL RESULTS ({NUM_EPISODES} Episodes)")
print(f"Mean: {mean_score:.2f} ± {std_score:.2f}")
print(f"Range: [{min_score:.2f}, {max_score:.2f}]")
print("="*30)

print(f"success rate: {(succesful_runs/(len(SEEDS)*len(OCCLUSION_LENGTHS)))*100}%\n")

# with open(f"{RUN_DIR}/benchmarks/benchmark.txt", "w+") as f:
#     f.write("="*30 +"\n")
#     f.write(f"📊 FINAL RESULTS ({NUM_EPISODES} Episodes)\n")
#     f.write(f"Mean: {mean_score:.2f} ± {std_score:.2f}\n")
#     f.write(f"Range: [{min_score:.2f}, {max_score:.2f}]\n")
#     f.write("="*30 + "\n")
#     f.write(f"success rate: {(succesful_runs/(len(SEEDS)*len(OCCLUSION_LENGTHS)))*100}%\n")
#     f.write("RECORDS:::\n")
#     for seed in record:
#         f.write(f"Seed: {seed}\n")
#         f.write("="*30 + "\n")
#         for gap in record[seed]:
#             f.write(f"Gap length: {gap}\n")
#             f.write(f"  reward: {record[seed][gap]['reward']}\n")
#             f.write(f"  steps_taken: {record[seed][gap]['steps']}\n")
#             f.write(f"  reason: {record[seed][gap]['reason']}\n")
#         f.write("="*30 + "\n")

with open(f"{RUN_DIR}/benchmarks/benchmark.json", "w+") as f:
    json.dump(record, f)