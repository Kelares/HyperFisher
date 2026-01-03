import minari

dataset = minari.load_dataset("mujoco/hopper/medium-v0")

first = dataset[0]
for episode_data in dataset.iterate_episodes():
    observations = episode_data.observations
    actions = episode_data.actions
    rewards = episode_data.rewards
    terminations = episode_data.terminations
    truncations = episode_data.truncations
    infos = episode_data.infos

print(first)