from torchrl.modules import DTActor, DecisionTransformer
import torch

def create_actor(DEVICE):
    dt_config = DecisionTransformer.default_config()
    dt_config.n_embd = 128
    dt_config.n_layer = 3
    dt_config.n_head = 1
    dt_config.n_positions = 1000 # Max episode steps

    actor = DTActor(
        state_dim=11, 
        action_dim=3, 
        transformer_config=dt_config, 
        device=DEVICE
    ).to(DEVICE)

    return actor