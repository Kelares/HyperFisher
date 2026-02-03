import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from sb3_contrib import RecurrentPPO


# 1. Setup Environment
env_id = "MiniGrid-MemoryS13Random-v0"
env = gym.make(env_id, render_mode=None)
env = FlatObsWrapper(env)  # Flattens the 7x7x3 view into a vector

# 2. Define the Model
# We use a larger MLP because the 'Memory' task requires internalizing logic
model = RecurrentPPO(
    "MlpLstmPolicy", 
    env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,           # Batch size for updates
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,          # Important: Helps the agent keep exploring
    device="auto"
)

# 3. Train
# MemoryS17 is tricky; it might need 500k to 1M steps to "click"
print("Training started... This may take a while.")
model.learn(total_timesteps=1000000)

# 4. Save the expert
model.save("ppo_minigrid_expert")
print("Model saved!")