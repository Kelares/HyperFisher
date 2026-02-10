import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import matplotlib.pyplot as plt

env = gym.make("MiniGrid-MemoryS7-v0")
env = FlatObsWrapper(env)
obs, _ = env.reset()

print(f"Flattened shape: {obs.shape}") # Should be 2835 for S13, smaller for S7

# Reshape it back to a grid to see if it makes sense
# Note: FlatObsWrapper usually flattens (width * height * channels)
grid_size = 7 

for i in range(10):
    try:
        print
        image_part = obs[:grid_size*grid_size*3].reshape(grid_size, grid_size, 3)
        plt.imshow(image_part)
        plt.title("What the Model Sees (Reconstructed)")
        plt.show()
    except:
        print("Flattening logic doesn't match standard 7x7x3 structure.")
    x = env.step(env.action_space.sample())
    obs = x[0]

