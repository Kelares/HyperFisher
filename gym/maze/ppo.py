import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper, OneHotPartialObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

# 1. Setup Environment
env_id = "MiniGrid-MemoryS7-v0"
env = gym.make(env_id, render_mode=None)
env = OneHotPartialObsWrapper(env)
env = FlatObsWrapper(env)

env = Monitor(env) # Tracks stats for success_rate

# 2. Define the Model with specific Recurrent Hyperparameters
model = RecurrentPPO(
    "MlpLstmPolicy", 
    env, 
    verbose=1,
    learning_rate=1e-4,     # Lowered for LSTM stability
    n_steps=512,            # Increased to capture full trajectories
    batch_size=128,         # Larger batch for more stable gradients
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.05,          # Balanced exploration
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], qf=[128, 128]), # Wider feature extraction
        lstm_hidden_size=256,                        # Larger memory capacity
        n_lstm_layers=2,                             # Deepened memory
    ),
    device="auto"
)

# 3. Train
# Memory tasks often have a "Step Function" learning curve: 
# It stays at 0.5 for a long time, then suddenly jumps to 0.95.
print("Training started... If success_rate stays at 0.5, let it run longer.")
model.learn(total_timesteps=1000000)

model.save("oracle_S7")


# Transitions to get from wrapper
# Soft Q Imitation Learning,
# Behavior cloning
# A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.
