import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import numpy as np
import pickle
from sb3_contrib import RecurrentPPO

def generate_minigrid_dataset(env_id, num_episodes=1000):
    # Use FlatObsWrapper to make it compatible with Decision Transformer inputs
    env = gym.make(env_id, render_mode=None)
    env = FlatObsWrapper(env)
    
    dataset = []

    print(f"Starting data collection for {env_id}...")
    model = RecurrentPPO.load("oracle_S7")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }

        while not (terminated or truncated):
            # --- ORACLE LOGIC ---
            # For Minigrid-Memory, the 'Oracle' can be a simple BFS or 
            # the internal 'env.unwrapped.actions' sequence if you use 
            # a scripted bot. Here, we'll assume a 'Perfect' solver.
            # --------------------
            
            # Note: For Memory tasks, you can use the 'minigrid' built-in bot:
            # Here we simulate the oracle action selection:
            action, _ = model.predict(obs)
            
            # Record current state
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            
            # Step the env
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_data['rewards'].append(reward)
            episode_data['terminals'].append(terminated)

        dataset.append(episode_data)
        
        if (ep + 1) % 100 == 0:
            print(f"Collected {ep + 1}/{num_episodes} episodes")

    save_path = f"{env_id}.pickle"
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")

# Run it
generate_minigrid_dataset("MiniGrid-MemoryS7-v0")