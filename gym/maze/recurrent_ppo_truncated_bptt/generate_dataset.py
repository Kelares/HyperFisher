import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper, OneHotPartialObsWrapper, ImgObsWrapper
import numpy as np
import pickle

from model import ActorCriticModel
import pickle
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import torch 
from utils import create_env
import numpy as np

device = "cpu"
torch.set_default_tensor_type("torch.FloatTensor")

# --- CONFIGURATION ---
# 0.2 means 20% of the time we force a random mistake.
# This creates the "Noisy Expert" data needed to cure the loops.
EPSILON = 0.2  
# ---------------------

def generate_minigrid_dataset(env_id, num_episodes=1000):
    dataset = []

    print(f"Starting NOISY data collection for {env_id} with EPSILON={EPSILON}...")
    
    state_dict, config = pickle.load(open("./minigrid.nn", "rb"))

    config["environment"]["name"] = env_id
    print(config["environment"])

    env = create_env(config["environment"], render=False)
    print(env)
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,))

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


    for ep in range(num_episodes):
        hxs, cxs = model.init_recurrent_cell_states(1, device)
        if config["recurrence"]["layer_type"] == "gru":
            recurrent_cell = hxs
        elif config["recurrence"]["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)


        obs = env.reset()
        done = False
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
        }

        while not done:
            # --- 1. RUN EXPERT (Update Memory) ---
            # CRITICAL: We MUST run the model forward pass even if we plan to 
            # do a random move. The LSTM/GRU 'recurrent_cell' needs to see 
            # the current observation to update its internal map. 
            # If we skipped this, the expert would have "memory gaps".
            policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)), recurrent_cell, device, 1)
            
            # Extract the Expert's proposed action
            expert_action = []
            for action_branch in policy:
                expert_action.append(action_branch.sample().item())

            # --- 2. APPLY NOISE (Epsilon-Greedy) ---
            if np.random.rand() < EPSILON:
                # NOISE: Force a random action from the environment
                # We wrap it in a list to match the expert_action structure [int]
                # This causes collisions and forces the expert to learn recovery on the next step.
                action = [env.action_space.sample()]
            else:
                # EXPERT: Do the optimal move
                action = expert_action
            
            # --- 3. EXECUTE ---
            obs, reward, done, info = env.step(action)
            
            # Record current state
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            
        dataset.append(episode_data)
        
        if (ep + 1) % 5 == 0:
            print(f"Collected {ep + 1}/{num_episodes} episodes")

    save_path = f"{env_id}_S9_PRE.pickle"
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Noisy Expert Dataset saved to {save_path}")

# Run it
generate_minigrid_dataset("MiniGrid-MemoryS9-v0")