

import mikasa_robo_suite
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info
from tqdm.notebook import tqdm
import torch
import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode

env_name = "RememberColor9-v0"
obs_mode = "rgb" # or "state"
num_envs = 1
seed = 42

env = gym.make(env_name, num_envs=num_envs, obs_mode=obs_mode, render_mode="all")

state_wrappers_list, episode_timeout = env_info(env_name)
print(f"Episode timeout: {episode_timeout}")
for wrapper_class, wrapper_kwargs in state_wrappers_list:
    env = wrapper_class(env, **wrapper_kwargs)

env = RecordEpisode(env, f"{env_name}/{seed}", max_steps_per_video=episode_timeout)

obs, _ = env.reset(seed=seed)
print(obs.keys())
for i in tqdm(range(episode_timeout)):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(torch.from_numpy(action))

env.close()


