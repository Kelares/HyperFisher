from model import ActorCriticModel
import pickle
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import torch 
from utils import create_env
import numpy as np

device = "cpu"
torch.set_default_tensor_type("torch.FloatTensor")


state_dict, config = pickle.load(open("./minigrid.nn", "rb"))

config["environment"]["name"] = "MiniGrid-MemoryS17Random-v0"
print(config["environment"])

env = create_env(config["environment"], render=True)
print(env)
model = ActorCriticModel(config, env.observation_space, (env.action_space.n,))

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Run and render episode
done = False
episode_rewards = []

# Init recurrent cell
hxs, cxs = model.init_recurrent_cell_states(1, device)
if config["recurrence"]["layer_type"] == "gru":
    recurrent_cell = hxs
elif config["recurrence"]["layer_type"] == "lstm":
    recurrent_cell = (hxs, cxs)

obs = env.reset()
while not done:
    # Render environment
    env.render()
    # Forward model
    policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)), recurrent_cell, device, 1)
    # Sample action
    action = []
    for action_branch in policy:
        action.append(action_branch.sample().item())
    # Step environment
    obs, reward, done, info = env.step(action)
    episode_rewards.append(reward)
    print(action, reward, done)

# After done, render last state
env.render()

print("Episode length: " + str(info["length"]))
print("Episode reward: " + str(info["reward"]))

env.close()