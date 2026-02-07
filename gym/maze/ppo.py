import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

# 1. Setup Environment
env_id = "MiniGrid-MemoryS13Random-v0"
env = gym.make(env_id, render_mode=None)
env = FlatObsWrapper(env)
env = Monitor(env) # Tracks stats for success_rate

# 2. Define the Model with specific Recurrent Hyperparameters
model = RecurrentPPO(
    "MlpLstmPolicy", 
    env, 
    verbose=1,
    learning_rate=2e-4,     # Slightly lower LR often helps LSTMs
    n_steps=128,            # Smaller rollout window per 'actor' 
    batch_size=64,          # Standard batch size
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.1,           # HIGHER entropy coefficient (Crucial to break 50% plateau)
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], qf=[64, 64]), # Feature extractor layers
        lstm_hidden_size=128,                    # Larger memory capacity
        n_lstm_layers=1,
    ),
    tensorboard_log="./tb_logs/",
    device="auto"
)

# 3. Train
# Memory tasks often have a "Step Function" learning curve: 
# It stays at 0.5 for a long time, then suddenly jumps to 0.95.
print("Training started... If success_rate stays at 0.5, let it run longer.")
model.learn(total_timesteps=1000000)

model.save("ppo_minigrid_expert_v2")
