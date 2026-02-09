import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from sb3_contrib import RecurrentPPO
import pickle
import numpy as np

def collect_perfect_expert_data(model_path, env_id, target_episodes=1000):
    env = FlatObsWrapper(gym.make(env_id))
    model = RecurrentPPO.load(model_path)
    
    expert_trajectories = []
    attempts = 0

    print(f"Filtering for {target_episodes} successful episodes...")

    while len(expert_trajectories) < target_episodes:
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        traj = {'observations': [], 'actions': [], 'rewards': []}
        done = False
        total_reward = 0
        
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, 
                                               episode_start=episode_starts, 
                                               deterministic=True)
            
            traj['observations'].append(obs)
            traj['actions'].append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            traj['rewards'].append(reward)
            total_reward += reward
            
            done = terminated or truncated
            episode_starts = np.array([done])

        attempts += 1
        # CRITICAL: Only save if the agent actually succeeded
        if total_reward > 0:
            expert_trajectories.append(traj)
            if len(expert_trajectories) % 50 == 0:
                print(f"Collected {len(expert_trajectories)}/{target_episodes}")

    with open("perfect_expert_data.pkl", "wb") as f:
        pickle.dump(expert_trajectories, f)
    print(f"Success! Filtered from {attempts} total attempts.")

collect_perfect_expert_data("ppo_minigrid_expert_v2", "MiniGrid-MemoryS13Random-v0")