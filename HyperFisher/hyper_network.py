import torch
import torch.nn as nn
from torch.func import functional_call
from math import ceil


class HyperNetwork(nn.Module):
    def __init__(self, config, target_network_template: nn.Module, device: torch.device):
        super().__init__()
        self.device = device
        
        # 1. The Modular Target Network
        # We accept ANY architecture (MLP, CNN, etc.) and freeze it
        self.target_network = target_network_template.to(self.device)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        self.num_target_params = sum(p.numel() for p in self.target_network.parameters())
        print(self.num_target_params)

        # CHUNKING
        self.chunk_size = 1000
        self.num_of_chunks = ceil( self.num_target_params / self.chunk_size )

        self.chunk_emb = nn.Embedding(
            num_embeddings=self.num_of_chunks, 
            embedding_dim=config.embedding_dim
        ).to(self.device)
        ##########

        # 2. Task Embeddings (No shared context)
        self.task_emb = nn.Embedding(
            num_embeddings=config.num_tasks, 
            embedding_dim=config.embedding_dim
        ).to(self.device)
        


        # 3. Modular Hypernetwork Generator
        # config.hyper_hidden_dim defines the bottleneck (e.g., 16)
        bottleneck_dim = getattr(config, 'hyper_hidden_dim', 16)
        

        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, bottleneck_dim), # * 2 because we concat 2 embedding layers
            nn.ReLU(),
            nn.Linear(bottleneck_dim, self.chunk_size)
        ).to(self.device)

        # 4. Prevent variance explosion on the massive output layer
        with torch.no_grad():
            torch.nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.01)
            torch.nn.init.normal_(self.layers[-1].bias, mean=0.0, std=0.1)

        self.target_params = None
        
    def spawn(self, task_id):
        t_vec = self.task_emb(task_id).to(self.device)

        # COLLECT CHUNKS #
        chunks = []
        for chunk_id in range(self.num_of_chunks):
            chunk_id_tensor = torch.tensor([chunk_id], dtype=torch.long, device=device)

            c_vec = self.chunk_emb(chunk_id_tensor).to(self.device)
            x = torch.concat(t_vec, c_vec)
            chunks.append(self.layers(x).squeeze().to(self.device))
        self.target_params = self.get_params_dict(chunks)
        ##################

    def forward(self, x):
        return functional_call(self.target_network, self.target_params, x)

    def get_params_dict(self, flat_params):
        param_dict = {}
        pointer = 0
        for name, param in self.target_network.named_parameters():
            num_param = param.numel()
            param_dict[name] = flat_params[pointer:pointer + num_param].view_as(param)
            pointer += num_param
        return param_dict


