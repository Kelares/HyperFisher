import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, target_network_template, device):
        super().__init__()
        self.device = device
        self.layers = target_network_template.to(self.device)
        
    def spawn(self, task_id):
        pass

    def forward(self, x):
        return self.layers(x)

