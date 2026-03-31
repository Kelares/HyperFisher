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
        torch.nn.init.constant_(self.task_emb.weight, 0.0)


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
        
        print("hyper_network_param_n: ", sum(p.numel() for p in self.layers.parameters()))
        
        # 4. Prevent variance explosion on the massive output layer
        with torch.no_grad():
            torch.nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.05)
            torch.nn.init.normal_(self.layers[-1].bias, mean=0.0, std=0.01)

        self.target_params = None
        
    def spawn(self, task_id):
        t_vec = self.task_emb(task_id).to(self.device)
        
        # Extract the integer value safely whether it's a tensor or int
        task_idx = task_id.item() if isinstance(task_id, torch.Tensor) else task_id
        
        # 1. Calculate active target params for this task
        # Shared params + parameters for head_{task_idx}
        active_param_names = [n for n, _ in self.target_network.named_parameters() 
                            if "heads" not in n or f"heads.head_{task_idx}" in n]
        
        task_target_n = sum(p.numel() for n, p in self.target_network.named_parameters() 
                            if n in active_param_names)

        # 2. Collect chunks as usual
        num_chunks = ceil(task_target_n / self.chunk_size)
        chunks = []
        for chunk_id in range(num_chunks):
            chunk_id_tensor = torch.tensor([chunk_id], device=self.device)
            c_vec = self.chunk_emb(chunk_id_tensor)
            x = torch.concat([t_vec, c_vec], dim=1)
            # Squeeze removes batch dim to make 1D vector
            chunks.append(self.layers(x).squeeze())
        
        self.w = torch.concat(chunks)[:task_target_n]
        
        # 3. Map to dict, but only for active parameters
        self.target_params = self.get_params_dict(self.w, active_param_names)

    def forward(self, x, task_id): 
        # Pass task_id as a keyword argument to the target network's forward
        return functional_call(
            self.target_network, 
            self.target_params, 
            args=(x,), 
            kwargs={'task_id': task_id}
        )

    def get_params_dict(self, flat_params, active_names):
        param_dict = {}
        pointer = 0
        for name, param in self.target_network.named_parameters():
            if name in active_names:
                num_p = param.numel()
                param_dict[name] = flat_params[pointer : pointer+num_p].view_as(param)
                pointer += num_p
        return param_dict

    def get_active_indices(self, task_id):
        """
        Returns a 1D tensor of global indices that correspond to the active
        parameters (shared + current head) for this task_id.
        """
        task_idx = task_id.item() if isinstance(task_id, torch.Tensor) else task_id
        indices = []
        current_idx = 0
        
        for name, param in self.target_network.named_parameters():
            num_p = param.numel()
            # If it's a shared layer, or it's the specific head for this task
            if "heads" not in name or f"heads.head_{task_idx}" in name:
                indices.extend(range(current_idx, current_idx + num_p))
            current_idx += num_p
            
        return torch.tensor(indices, dtype=torch.long, device=self.device)