import torch
from torch.utils.data import Dataset, DataLoader
import minari
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


MODULE_DIR = Path(__file__).parent.resolve()




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

# --- DATA PREPARATION ---
def loadDataset(CURRENT_CONFIG):
    # --- HOPPER CONFIGURATION ---
    BATCH_SIZE = 64
    print(CURRENT_CONFIG.model.value)
    
    CONTEXT_LEN = CURRENT_CONFIG.context_length.value
    print(f"CONTEXT LENGTH: {CONTEXT_LEN}")

    RTG_SCALE = 1000.0    # Scale returns so 3000 becomes 3.0

    print(f"Loading Minari Dataset: {CURRENT_CONFIG.dataset_id} (downloading if needed)...")
    # "Medium" is best for training (contains mix of success/fail)
    md = minari.load_dataset(CURRENT_CONFIG.dataset_id, download=True)

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

    # ... existing observation normalization ...
    all_obs = np.concatenate(all_obs, axis=0)
    obs_mean = np.mean(all_obs, axis=0)
    obs_std = np.std(all_obs, axis=0) + 1e-6


    # --- ACTION SCALING FIX ---
    # Hopper-v5 actions are physically limited to [-1, 1].
    # Instead of Mean/Std normalization, we ensure the data strictly 
    # maps to the Tanh range [-1, 1].
    for traj in trajectories:
        # State normalization (Mean/Std is fine for states)
        traj["observations"] = (traj["observations"] - obs_mean) / obs_std
        
        # Action Scaling: Ensure they are clamped to [-1, 1]
        # (Minari data should already be here, but we enforce it for the model)
        traj["actions"] = np.clip(traj["actions"], -1.0, 1.0)

    # Save stats (We still need obs stats for inference)
    np.savez(
        f"{MODULE_DIR}/normalizations/{CURRENT_CONFIG.level.value}_{CONTEXT_LEN}.npz", 
        obs_mean=obs_mean, obs_std=obs_std,
        act_mean=np.zeros(3), # No longer needed for scaling
        act_std=np.ones(3)     # No longer needed for scaling
    )

    dataset = TrajectoryDataset(trajectories, CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataset, loader

# --- LIVE ENVIRONMENT ---
def liveEnv(CURRENT_CONFIG, DEVICE, RUN_DIR):
    CONTEXT_LEN = CURRENT_CONFIG.context_length.value
    print(f"CONTEXT LENGTH: {CONTEXT_LEN}")

    env = gym.make("Hopper-v5", render_mode="rgb_array", max_episode_steps=1000) # "human" to see it, "rgb_array" for headless

    if CURRENT_CONFIG.record:
        env = RecordVideo(
            env, 
            video_folder=f"{RUN_DIR}/videos",
            episode_trigger=lambda episode_id: True,
            name_prefix=""
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load Normalization Stats
    stats = np.load(f"{MODULE_DIR}/normalizations/{CURRENT_CONFIG.level.value}_{CONTEXT_LEN}.npz")
    state_mean = torch.from_numpy(stats['obs_mean']).to(DEVICE).float()
    state_std = torch.from_numpy(stats['obs_std']).to(DEVICE).float()
    
    # FORCE ACTION STATS TO NEUTRAL
    # This prevents run.py from distorting the Tanh output (-1 to 1)
    act_mean = torch.zeros(action_dim, device=DEVICE)
    act_std = torch.ones(action_dim, device=DEVICE)
    return env, state_dim, action_dim, state_mean, state_std, act_mean, act_std