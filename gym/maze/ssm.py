import torch
import torch.nn as nn
from mamba_ssm import Mamba

# --- CONSTANTS FOR MINIGRID-MEMORY ---
STATE_DIM = 2835
ACT_DIM = 7     # Discrete actions 0-6
HIDDEN_SIZE = 128

class DecisionMamba(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Embeddings
        self.embed_s = nn.Linear(state_dim, hidden_size)
        # Use nn.Embedding for discrete actions
        self.embed_a = nn.Embedding(act_dim, hidden_size)
        self.embed_R = nn.Linear(1, hidden_size)

        # 2. Backbone: Mamba
        self.backbone = Mamba(
            d_model=hidden_size,
            d_state=64, # High d_state is good for memory tasks
            d_conv=4,
            expand=2,
        )
        
        # 3. Action Head (Classification)
        # We remove Tanh and output logits for CrossEntropyLoss
        self.predict_action = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, returns):
        B, T, _ = states.shape 

        # --- A. Embed Inputs ---
        s_emb = self.embed_s(states)
        # actions is now (B, T) long integers
        a_emb = self.embed_a(actions) 
        r_emb = self.embed_R(returns)
        
        # --- B. Stack Sequence ---
        # Format: [R1, s1, a1, R2, s2, a2 ...]
        stacked_inputs = torch.stack((r_emb, s_emb, a_emb), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)

        # --- C. Pass through Mamba ---
        hidden_states = self.backbone(stacked_inputs)

        # --- D. Extract Action Predictions ---
        # Reshape to (B, T, 3, H)
        hidden_states = hidden_states.view(B, T, 3, self.hidden_size)
        
        # Predict Action from the State token's hidden state (index 1)
        state_hidden = hidden_states[:, :, 1, :] 
        logits = self.predict_action(state_hidden)

        return logits # Returns (B, T, 7)


def create_actor(device):
    model = DecisionMamba(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        hidden_size=HIDDEN_SIZE,
    )
    return model.to(device)