import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    # This connects the CNN output to the LSTM
    lstm_hidden_size=256,
)

# 1. Setup Environment
env_id = "MiniGrid-MemoryS7-v0"
env = gym.make(env_id, render_mode=None)
env = ImgObsWrapper(env) # Convert dict to (7, 7, 3) 

model = RecurrentPPO(
    "CnnLstmPolicy", # Use the CNN-compatible Recurrent Policy
    env, 
    policy_kwargs=policy_kwargs, 
    verbose=1,
    n_steps=512,      # Long enough to see the hint and the goal [cite: 1]
    ent_coef=0.01
)


# 3. Train
# Memory tasks often have a "Step Function" learning curve: 
# It stays at 0.5 for a long time, then suddenly jumps to 0.95.
print("Training started... If success_rate stays at 0.5, let it run longer.")
model.learn(total_timesteps=200000)

model.save("oracle_S7_ext")


# Transitions to get from wrapper
# Soft Q Imitation Learning,
# Behavior cloning
# A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.
