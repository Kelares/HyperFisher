import copy
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, target_network_template, device):
        super().__init__()
        self.device = device
        
        # 1. Create a full, independent clone of the template architecture
        self.layers = copy.deepcopy(target_network_template).to(self.device)
        
        # 2. Reinitialize the weights so it starts from scratch
        self._reinitialize_weights()
        
    def _reinitialize_weights(self):
        """Recursively resets parameters for all standard PyTorch layers."""
        for module in self.layers.modules():
            # Standard layers like nn.Linear, nn.Conv2d, etc., have this method
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def spawn(self, task_id):
        self.task_id = task_id
        pass

    def forward(self, x):
        return self.layers(x, self.task_id)