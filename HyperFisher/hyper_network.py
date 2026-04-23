import torch
import torch.nn as nn
from torch.func import functional_call
from math import ceil
from typing import List

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
        print("Target_network_param_n: ", self.num_target_params)

        # CHUNKING
        self.chunk_size = config.chunk_size
        if self.chunk_size:
            self.num_of_chunks = ceil( self.num_target_params / self.chunk_size )
            self.chunk_emb = nn.Embedding(
                num_embeddings=self.num_of_chunks, 
                embedding_dim=config.chunk_embedding_dim
            ).to(self.device)
        ##########

        # 2. Task Embeddings 
        self.task_emb = nn.Embedding(
            num_embeddings=config.num_tasks, 
            embedding_dim=config.task_embedding_dim
        ).to(self.device)


        # 3. Modular Hypernetwork Generator
        # config.hyper_hidden_dim defines the bottleneck (e.g., 16)
        bottleneck_dim = getattr(config, 'hyper_hidden_dim', 16)
        
        output_dim = self.chunk_size if self.chunk_size else self.num_target_params
        self.layers = nn.Sequential(
            nn.Linear(config.task_embedding_dim + config.chunk_embedding_dim, bottleneck_dim),
            nn.LeakyReLU(0.1),

            nn.Linear(bottleneck_dim, bottleneck_dim*2),
            nn.LeakyReLU(0.1), 

            nn.Linear(bottleneck_dim*2, output_dim)
        ).to(self.device)
        
        self.num_hyper_params = sum(p.numel() for p in self.layers.parameters())
        print("hyper_network_param_n: ", self.num_hyper_params)
        
        # 4. Prevent variance explosion on the massive output layer
        with torch.no_grad():
            torch.nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.05)
            torch.nn.init.normal_(self.layers[-1].bias, mean=0.0, std=0.01)

        self.target_params = None
        
    def spawn(self, task_id):
        # 1. Get the embedding and force it to be 1D (embedding_dim,)
        # This handles both task_id=0 and task_id=torch.tensor([0])
        t_emb = self.task_emb(task_id).view(-1) 

        # 1b. Repeat it into a 2D matrix (num_chunks, embedding_dim)
        t_vec = t_emb.repeat(self.num_of_chunks, 1)

        # 2. Get all chunk embeddings at once
        chunk_ids = torch.arange(self.num_of_chunks, device=self.device)
        c_vec = self.chunk_emb(chunk_ids) # [num_chunks, emb_dim]

        # 3. Concatenate and pass through MLP in one batch
        x = torch.cat([t_vec, c_vec], dim=1)
        flat_w = self.layers(x).view(-1) # Flatten all chunks into one long vector
        
        self.w = flat_w[:self.num_target_params] # Trim padding
        self.target_params = self.get_params_dict(self.w)

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

    # ── FIX 2: Only shared parameters should be projected ────────────────────
    # task_emb rows are task-specific — row t cannot affect task t'≠t, so
    # including them wastes projection budget on parameters that cannot cause
    # cross-task interference.
    @staticmethod
    def _shared_params(model: nn.Module) -> List[nn.Parameter]:
        """Return only the parameters shared across ALL tasks.

        Excludes task_emb because each task owns an independent embedding row
        and updates to that row can never affect any other task's output.
        Including it in the projection subspace would:
          (a) distort the Fisher diagonal with gradients that carry no
              cross-task interference signal, and
          (b) waste columns in G on directions that do not need protecting.
        """
        return list(model.layers.parameters()) + [model.chunk_emb.weight]