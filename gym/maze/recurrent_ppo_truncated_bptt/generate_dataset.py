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


def generate_minigrid_dataset(env_id, num_episodes=1000):
    dataset = []

    print(f"Starting data collection for {env_id}...")
    
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
            # --- ORACLE LOGIC ---
            # For Minigrid-Memory, the 'Oracle' can be a simple BFS or 
            # the internal 'env.unwrapped.actions' sequence if you use 
            # a scripted bot. Here, we'll assume a 'Perfect' solver.
            # --------------------
            
            # Note: For Memory tasks, you can use the 'minigrid' built-in bot:
            # Here we simulate the oracle action selection:

            policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)), recurrent_cell, device, 1)
            # Sample action
            action = []
            # print(policy)
            for action_branch in policy:
                # print(action_branch)
                action.append(action_branch.sample().item())
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Record current state
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            # print(action, reward, done)
        dataset.append(episode_data)
        
        if (ep + 1) % 5 == 0:
            print(f"Collected {ep + 1}/{num_episodes} episodes")

    save_path = f"{env_id}_S9_PRE.pickle"
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")

# Run it
generate_minigrid_dataset("MiniGrid-MemoryS9-v0")