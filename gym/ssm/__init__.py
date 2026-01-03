import torch
import torch.nn as nn

# Try to import the official Mamba library
# If this fails, you will need to install it: pip install mamba-ssm
try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ Error: mamba_ssm library not found.")
    print("   Run: pip install mamba-ssm")
    # Fallback to a dummy class to prevent immediate crash during import
    Mamba = None

# --- CONSTANTS FOR HOPPER ---
# Hardcoded to match your env. 
# ideally these would pass through create_actor, but your train.py signature is fixed.
STATE_DIM = 11
ACT_DIM = 3
HIDDEN_SIZE = 128
MAX_LENGTH = 20  # Matches CONTEXT_LEN in your other files

class DecisionMamba(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, max_length):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Embeddings
        # We project everything to size 128 (hidden_size)
        self.embed_s = nn.Linear(state_dim, hidden_size)
        self.embed_a = nn.Linear(act_dim, hidden_size)
        self.embed_R = nn.Linear(1, hidden_size)

        # 2. Backbone: Mamba
        # This replaces the GPT/Transformer block
        if Mamba is None:
            raise ImportError("Please install mamba-ssm to train this model.")
            
        self.backbone = Mamba(
            d_model=hidden_size, # Model dimension
            d_state=64,          # SSM state dimension (N)
            d_conv=4,            # Local convolution width
            expand=2,            # Block expansion factor
        )
        
        # 3. Action Head
        # Projects the hidden state back to an Action
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() # Hopper actions are usually -1 to 1
        )

    def forward(self, states, actions, returns):
        B, T, _ = states.shape # Batch, Time, Dim

        # --- A. Embed Inputs ---
        s_emb = self.embed_s(states)
        a_emb = self.embed_a(actions)
        r_emb = self.embed_R(returns)
        
        # --- B. Stack Sequence ---
        # We want: [R1, s1, a1, R2, s2, a2 ...]
        # Stack dim=1 makes it (B, 3, T, H). 
        # Permute/Reshape makes it (B, 3*T, H).
        stacked_inputs = torch.stack((r_emb, s_emb, a_emb), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)

        # --- C. Pass through Mamba ---
        # Mamba processes the whole sequence in parallel (O(L) or O(log L))
        hidden_states = self.backbone(stacked_inputs)

        # --- D. Extract Action Predictions ---
        # The sequence is:
        # Index 0: R1
        # Index 1: s1  --> Predicts a1 (This is what we want!)
        # Index 2: a1
        # Index 3: R2
        # Index 4: s2  --> Predicts a2
        # ...
        # We slice indices starting at 1, with step 3.
        
        # Reshape back to (B, T, 3, H) to easily pluck the 2nd element
        hidden_states = hidden_states.view(B, T, 3, self.hidden_size)
        
        # Take the "State" token's output (index 1)
        state_hidden = hidden_states[:, :, 1, :] 

        # Predict Action
        action_preds = self.predict_action(state_hidden)

        return action_preds

# --- FACTORY FUNCTION ---
# This matches the signature called in train.py: "model.create_actor(DEVICE)"
def create_actor(device):
    model = DecisionMamba(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        hidden_size=HIDDEN_SIZE,
        max_length=MAX_LENGTH
    )
    return model.to(device)