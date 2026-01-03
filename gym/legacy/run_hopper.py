import gymnasium as gym
env = gym.make("Hopper-v5", render_mode="rgb_array", width=1280, height=720)
print(env)