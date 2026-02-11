import torch
import torch.nn as nn
from mamba_ssm import Mamba

# --- CONSTANTS FOR MINIGRID-MEMORY ---
STATE_DIM = 2835
ACT_DIM = 7     # Discrete actions 0-6
HIDDEN_SIZE = 128

class FlexibleMiniGridEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        
        # KEY CHANGE: This ensures that no matter if the input is 84x84 or 100x100,
        # the output of the CNN is always 4x4 spatially.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.flat = nn.Flatten()
        # 64 channels * 4 * 4 grid = 1024
        self.linear = nn.Linear(64 * 4 * 4, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = self.flat(x)
        return self.linear(x)
    
class DecisionMamba(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size):
        super().__init__()
        # Use the flexible CNN to process pixel observations
        self.state_encoder = FlexibleMiniGridEncoder(output_dim=hidden_size)
        
        # Embed discrete actions (e.g., 0-6 in MiniGrid)
        self.action_embed = nn.Embedding(act_dim, hidden_size)
        
        # Embed continuous RTG values
        self.rtg_embed = nn.Linear(1, hidden_size)

        self.hidden_size = hidden_size

        # Mamba Backbone
        self.backbone = Mamba(
            d_model=hidden_size,
            d_state=64, # High d_state helps with long-term memory tasks like S9
            d_conv=4,
            expand=2,
        )
        
        # Action Head: Predicts logits for the next action
        self.predict_action = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, rtgs):
        # states: (batch, seq_len, 3, 84, 84)
        batch_size, seq_len, c, h, w = states.shape
        
        # 1. Process states through CNN
        # Flatten batch and seq_len for the CNN
        flat_states = states.reshape(-1, c, h, w) 
        state_embeddings = self.state_encoder(flat_states) # (batch*seq_len, hidden_size)
        state_embeddings = state_embeddings.reshape(batch_size, seq_len, -1)
        
        # 2. Embed others
        action_embeddings = self.action_embed(actions)
        rtg_embeddings = self.rtg_embed(rtgs)
        
        # 3. Interleave (R1, S1, A1, R2, S2, A2...)
        # Decision Transformer/Mamba usually stacks these into a single sequence
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_size)
        
        # 4. Pass through Mamba
        output = self.backbone(stacked_inputs)
        
        # 5. Predict action from the state tokens
        # Typically we extract every 3rd token (the ones corresponding to state embeddings)
        logits = self.predict_action(output[:, 1::3, :])
        return logits


def create_actor(device):
    model = DecisionMamba(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        hidden_size=HIDDEN_SIZE,
    )
    return model.to(device)