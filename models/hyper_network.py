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
        self.task_embedding_lr = 0.05
        self.task_emb = nn.Embedding(
            num_embeddings=config.num_tasks, 
            embedding_dim=config.task_embedding_dim
        ).to(self.device)
        # self.task_emb.weight.requires_grad = False # 🔑 


        # 3. Modular Hypernetwork Generator
        # config.hyper_hidden_dim defines the bottleneck (e.g., 8)
        self.hidden_dim = getattr(config, 'hyper_hidden_dim', 8)
        
        output_dim = self.chunk_size if self.chunk_size else self.num_target_params
        self.layers = nn.Sequential(
            nn.Linear(config.task_embedding_dim + config.chunk_embedding_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1), 

            nn.Linear(self.hidden_dim, output_dim)
        ).to(self.device)
        
        self.num_hyper_params = sum(p.numel() for p in self.layers.parameters())
        print("hyper_network_param_n: ", self.num_hyper_params)
        
        # 4. Prevent variance explosion on the massive output layer
        with torch.no_grad():
            torch.nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.05)
            torch.nn.init.normal_(self.layers[-1].bias, mean=0.0, std=0.01)

        self.target_params = None
        self.num_shared_params = sum(p.numel() for p in self._shared_params)
        
        print("Num of chunks: ", self.num_of_chunks)
        
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
    @property
    def _shared_params(self) -> List[nn.Parameter]:
        """Return only the parameters shared across ALL tasks.

        Excludes task_emb because each task owns an independent embedding row
        and updates to that row can never affect any other task's output.
        Including it in the projection subspace would:
          (a) distort the Fisher diagonal with gradients that carry no
              cross-task interference signal, and
          (b) waste columns in G on directions that do not need protecting.
        """
        return list(self.layers.parameters()) + [self.chunk_emb.weight]
    
    def shrink(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.div_(self.num_of_chunks)

class HyperRegulizer():
    def __init__(self, beta: float = 0.01):
        self.beta = beta       
        self.old_weights = {}  

    def loss(self, model: nn.Module, current_task_id) -> torch.Tensor:
        """
        Von Oswald hypernetwork regularizer.
 
        Penalises drift in the generated weights for all previously seen tasks:
 
            L_reg = (1 / N_old) * Σ_{t' < t} ‖spawn(θ, emb_{t'}) − w_stored_{t'}‖²
 
        WHY this works better than FOPNG alone:
            FOPNG projects gradient directions in θ-space, which is an indirect
            proxy for preventing forgetting. The regularizer directly measures
            "did the hypernetwork start generating different weights for old
            tasks?" — hitting the actual forgetting mechanism. Together they are
            complementary: FOPNG prevents harmful update directions; the
            regularizer applies a restoring force toward old task solutions.
 
        Returns a scalar tensor (zero if no tasks stored or reg_lambda == 0).
        """
        device = next(model.parameters()).device
        if self.beta == 0.0 or not self.old_weights:
            return 0
 
        current_t = current_task_id.item() if hasattr(current_task_id, 'item') else int(current_task_id)
        old_task_ids = [t for t in self.old_weights if t != current_t]
        if not old_task_ids:
            return 0
 
        total = torch.tensor(0.0)
        for t in old_task_ids:
            w_stored = self.old_weights[t]         # frozen snapshot
            old_tid  = torch.tensor([t], dtype=torch.long, device=device)
            model.spawn(old_tid)                                    # recompute w under current θ
            w_now    = model.w                                      # differentiable w.r.t. θ
            total    = total + (w_now - w_stored).pow(2).sum()    # MSE per task
 


        # r_loss = total / len(old_task_ids)

        # The regularizer MSE is calculated on the FULL weight vector (sum of chunks)
        # We must divide the resulting loss by chunks to nullify the accumulation in backward()
        r_loss = total / len(old_task_ids) 
        loss = self.beta * r_loss

        return loss