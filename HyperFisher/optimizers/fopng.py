from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import wandb
from utils import _flat_grad, _apply_flat_update, calc_bwt, evaluate_accuracy, plot_overlap
import matplotlib.pyplot as plt
import numpy as np
import gc #Garbage Collector
from math import inf

# ─────────────────────────────────────────────────────────────────────────────
# FOPNG Class
# ─────────────────────────────────────────────────────────────────────────────
class FOPNG:
    def __init__(
        self,
        lr: float = 1e-3,
        lam: float = 1e-3,
        alpha: float = 0.5,
        grads_per_task: int = 80,
        max_directions: int = 400,
        fisher_samples: int = 1024,
    ):
        self.lr            = lr
        self.lam           = lam
        self.alpha         = alpha
        self.grads_per_task = grads_per_task
        self.max_directions = max_directions
        self.fisher_samples = fisher_samples

        self.F_old: Optional[Tensor] = None
        self.G:     Optional[Tensor] = None
        self._F_new: Optional[Tensor] = None
        self._A_inv: Optional[Tensor] = None
        self._device: Optional[torch.device] = None
        self.debug = 1
        self.fisher_after_task = {}

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

    def compute_fisher(self, model: nn.Module, loader: DataLoader, criterion: Callable) -> Tensor:
        return self.compute_fisher_diag(model, loader, criterion, self._device, self.fisher_samples)

    def compute_fisher_diag(
        self,
        hyper_network: nn.Module,
        task_id,
        loader: DataLoader,
        criterion: Callable,
        device: torch.device,
        max_samples: int = 1024,
    ) -> Tensor:
        """
        Compute diagonal Fisher in θ-space over shared parameters only.

        FIX 4 — why θ-space instead of w-space:
            G and the Fisher must live in the same space as the parameter
            updates that step() applies. step() updates θ directly, so the
            Fisher used in _fopng_update must also be in θ-space. If F were
            in w-space and the update in θ-space, the natural-gradient
            metric would be geometrically inconsistent.

        FIX 2 — why shared params only:
            task_emb is excluded via _shared_params(). See that docstring.

        Implementation:
            g_w  = ∂L/∂w   (one autograd.grad call in w-space)
            g_θ  = Jᵀ g_w  (one backward through spawn onto θ)
            F_θ += g_θ²     (diagonal approximation in θ-space)

        The double-spawn pattern (spawn → grad w.r.t. w, then spawn →
        backward onto θ) creates two independent computation graphs so
        no retain_graph=True is needed.
        """
        hyper_network.eval()
        shared = self._shared_params(hyper_network)
        D_theta = sum(p.numel() for p in shared)
        fisher = torch.zeros(D_theta, device=device)
        n_seen = 0

        with torch.enable_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                # ── Step 1: get g_w in w-space ──────────────────────────────
                hyper_network.zero_grad()
                hyper_network.spawn(task_id)
                w = hyper_network.w
                output = hyper_network(x)
                loss = criterion(output, y)
                (g_w,) = torch.autograd.grad(loss, w)   # graph freed here

                # ── Step 2: translate g_w → g_θ via Jᵀ ────────────────────
                # Fresh spawn so we get a new graph rooted at θ.
                hyper_network.zero_grad()
                hyper_network.spawn(task_id)
                hyper_network.w.backward(g_w.detach())  # θ.grad = Jᵀ g_w

                g_theta = torch.cat([
                    p.grad.view(-1) for p in shared if p.grad is not None
                ])
                fisher.add_(g_theta.detach().pow(2))    # diagonal F_θ

                n_seen += x.size(0)
                if n_seen >= max_samples:
                    break

        hyper_network.zero_grad()
        hyper_network.train()

        # Percentile-clipped normalization to fix Jᵀ amplification bias.
        #
        # WHY the old `fisher / fisher.max()` was broken:
        #   In θ-space, MLP parameters accumulate Jᵀ gradient contributions
        #   from every chunk (~206×), while chunk_emb[k] only accumulates
        #   from one chunk. This creates a ~206× dynamic range BEFORE
        #   normalization. Dividing by max then makes chunk_emb weights
        #   effectively zero (≈ 1/206 of max), collapsing the Fisher to a
        #   near-delta-function on a handful of MLP parameters.
        #   With mean ≈ 0.0002 and max = 1.0, the projection formula
        #   F_old * g ≈ 0 everywhere, so correction ≈ 0 and rho ≈ 1.
        #
        # FIX — clip the 99th-percentile outliers before normalizing:
        #   This neutralises the chunk-accumulation amplification while
        #   preserving the relative importance ordering within each
        #   parameter group. The resulting Fisher has mean ≈ 0.1–0.3
        #   instead of 0.0002, making the projection actually meaningful.
        fisher_nonzero = fisher[fisher > 0]
        if len(fisher_nonzero) > 0:
            p99 = torch.quantile(fisher_nonzero, 0.99)
            fisher = fisher.clamp(max=p99.item())
        if fisher.max() > 0:
            fisher = fisher / fisher.max()

            # Add a small floor (e.g., 1e-4) so the projection doesn't ignore 98% of the MLP
        fisher = fisher + 1e-4 
        if fisher.max() > 0:
            fisher = fisher / fisher.max()
        return fisher

    def prepare_epoch(self, F_new: Tensor) -> None:
        assert self.F_old is not None, "Call after_task() after task 1 before training task 2."
        self._F_new = F_new.detach().cpu()
        self._A_inv = self._build_A_inv(self.G, self.F_old, self._F_new, self.lam)

    def step(self, model: nn.Module, task_id, g_w: Tensor) -> float:
        """
        g_w : [D_w] gradient of loss w.r.t. generated weights w,
              computed by the training loop via autograd.grad(loss, w).

        FIX 4 — everything now lives in θ-space:
            Before: project in w-space → translate to θ via Jᵀ → rescale norm
            After:  translate g_w to θ-space first → project in θ-space →
                    apply directly to shared θ params

            Why this is more consistent:
            G and F_old/F_new are built in θ-space (see compute_fisher_diag
            and _collect_gradients). Running _fopng_update with a g that is
            in a *different* space than G would make the inner products
            Gᵀ F g geometrically meaningless. All three tensors must live in
            the same space.

        FIX 3 — the norm rescaling is removed:
            The old rescaling (scale = ‖v_star_w‖ / ‖Jᵀv_star_w‖) was needed
            because projection was in w-space but the update landed in θ-space,
            causing a magnitude mismatch. Moving everything to θ-space removes
            that mismatch entirely. lr inside _fopng_update already controls
            the step size.

        FIX 2 — only shared params are updated:
            task_emb is excluded. Its rows are task-specific so they cannot
            cause cross-task forgetting and need no protection.
        """
        assert self._A_inv is not None, "Call prepare_epoch(F_new) before step()."
        shared = self._shared_params(model)

        # ── 1. Translate g_w → g_θ via Jᵀ ───────────────────────────────
        # model.w.backward() populates .grad for EVERY parameter in spawn()'s
        # computation graph: layers, chunk_emb, AND task_emb.
        model.zero_grad()
        model.spawn(task_id)
        model.w.backward(g_w.detach())          # θ.grad = Jᵀ g_w

        g_theta = torch.cat([
            p.grad.view(-1) for p in shared if p.grad is not None
        ]).to(g_w.device)

        # ── 2. Update task_emb NOW, before zero_grad clears it ────────────
        # WHY task_emb needs a gradient step here:
        #   _shared_params() correctly excludes task_emb from the FOPNG
        #   projection because row t of task_emb only affects task t's
        #   spawn() output and can never cause cross-task forgetting.
        #   BUT that exclusion also means step() previously never updated
        #   task_emb at all for tasks 2+. The embedding for the current task
        #   stayed frozen at its random initialisation, forcing the shared
        #   MLP layers to compensate entirely — causing both slow convergence
        #   (task 2 stalls at ~0.47 loss vs task 1's ~0.19) and extra
        #   forgetting (shared layers pushed away from old-task solutions).
        #
        # WHY plain SGD (not projected):
        #   Projection is unnecessary because task_emb[t] cannot interfere
        #   with any other task. A simple lr-scaled gradient step is correct.
        with torch.no_grad():
            te_grad = model.task_emb.weight.grad
            if te_grad is not None:
                model.task_emb.weight.data.add_(-self.lr * te_grad)

        model.zero_grad()

        # ── 3. Project g_θ and compute natural-gradient step in θ-space ──
        v_star_theta, weighted_rho, correction_norm, raw_rho = self._fopng_update(
            g=g_theta, G=self.G, F_old=self.F_old, F_new=self._F_new,
            A_inv=self._A_inv, lr=self.lr, lam=self.lam,
        )

        # ── 4. Apply FOPNG update to shared θ only ────────────────────────
        pointer = 0
        with torch.no_grad():
            for p in shared:
                n = p.numel()
                p.data.add_(v_star_theta[pointer : pointer + n].view_as(p))
                pointer += n

        # weighted_rho: correct metric — retention in Fisher-important subspace
        # correction_norm: absolute gradient mass removed
        # raw_rho: kept for reference but misleading (≈1 by design in FOPNG)
        return weighted_rho, correction_norm, raw_rho

    def after_task(self, hyper_network: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        device = next(hyper_network.parameters()).device
        self._device = device

        F_new = self.compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
        self.fisher_after_task[task_id.item()] = F_new 
        # 1. CALCULATE OVERLAP BEFORE UPDATING F_OLD
        # At task 0, F_old is None, so we log 1.0 (perfect correlation with itself) or 0.0
        if self.F_old is not None:
            cosine_sim = self._cosine_similarity(self.F_old, F_new)
            pearson_corr = self._pearson_correlation(self.F_old, F_new)
            topk_iou = self._calculate_topk_iou(self.F_old, F_new)
        else:
            cosine_sim = 1.0
            pearson_corr = 1.0
            topk_iou = 1.0

        if self.F_old is None:
            self.F_old = F_new.detach().cpu()
        else:
            # Weighted average
            self.F_old = (1.0 - self.alpha) * self.F_old + self.alpha * F_new.detach().cpu()
            
            # NEW: Re-normalize so the most important parameter always has a weight of 1.0
            # Same percentile-clipped normalization as compute_fisher_diag.
            # The weighted average can accumulate the same outlier bias,
            # so we clip before normalizing here too.
            f_nonzero = self.F_old[self.F_old > 0]
            if len(f_nonzero) > 0:
                p99 = torch.quantile(f_nonzero, 0.99)
                self.F_old = self.F_old.clamp(max=p99.item())
            f_max = self.F_old.max()
            if f_max > 0:
                self.F_old = self.F_old / f_max

        new_cols = self._collect_gradients(hyper_network, task_id, loader, criterion)
        self.G   = new_cols if self.G is None else torch.cat([self.G, new_cols], dim=1)

        if self.G.shape[1] > self.max_directions:
            if self.debug:
                print("MAX N OF G REACHED: ", self.G.shape[1], "\n ##########################  \n", self.G)

                    
            # Uniformly sample indices across the entire chronological history. THE COLUMNS
            # This ensures every task gets an equal slice of the max_directions budget. Because the order is chronological
            indices = torch.linspace(
                0, self.G.shape[1] - 1, 
                steps=self.max_directions, 
                dtype=torch.long, 
                device=self.G.device
            )
            self.G = self.G[:, indices]

        f_nonzero = self.F_old[self.F_old > 0]
        logs = {
            "fopng/fisher/min": self.F_old.min().item(),
            "fopng/fisher/max": self.F_old.max().item(),
            "fopng/fisher/mean": self.F_old.mean().item(),
            # mean over non-zero entries — should be 0.1–0.4 with healthy Fisher
            # (was 0.0002 with /max normalization, indicating near-delta distribution)
            "fopng/fisher/mean_nonzero": f_nonzero.mean().item() if len(f_nonzero) > 0 else 0.0,
            "fopng/fisher/frac_nonzero": (len(f_nonzero) / len(self.F_old.view(-1))),
            "fopng/memory/G_cols": self.G.shape[1],
            "fopng/fisher_overlap/cosine": cosine_sim,
            "fopng/fisher_overlap/pearson": pearson_corr,
            "fopng/fisher_overlap/topk_iou": topk_iou,
            "task_completed": task_id.item() + 1
        }
        print(logs)

        wandb.log(logs)
        torch.cuda.empty_cache()
        gc.collect()


    def _collect_gradients(self, hyper_network: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> Tensor:
        """
        Collect loss-gradient directions in θ-space for the projection subspace G.

        WHY LOSS GRADIENTS (not random Jacobian projections):
            A previous version used random output directions v to sample the
            row space of ∂f/∂w. This is theoretically motivated (OGD theory
            calls for the subspace "induced by the Jacobian"), but it is
            catastrophically impractical at this dimensionality.

            In 65K-dimensional θ-space with 300 stored directions, a random
            vector Jᵀv and the actual loss gradient Jᵀg_w are nearly
            orthogonal by the curse of dimensionality. The inner product
            ⟨G_col, F_old·g⟩ ≈ 0 for all stored columns, making the FOPNG
            correction term ≈ 0, rho → 1.0, and the projection useless.

            Farajtabar et al. (2019) and every practical OGD implementation
            store LOSS gradients, not random Jacobian samples. Loss gradients
            are the task-relevant directions: they point exactly where a
            parameter update would increase task loss, which is precisely what
            we want to avoid. A budget of 300 loss-gradient directions will
            cover the critical forgetting subspace far more efficiently than
            300 random Jacobian samples.

        WHY θ-SPACE (Jᵀ translation):
            G must live in the same space as F_old/F_new and the updates in
            step(). We get g_w = ∂L/∂w in w-space first (cheap, no graph
            retention needed), then translate via Jᵀ to get g_θ.

        WHY SHARED PARAMS ONLY:
            task_emb excluded via _shared_params(). See that docstring.

        Returns
        -------
        G : [D_theta_shared, grads_per_task]
        """
        grads: List[Tensor] = []
        hyper_network.eval()
        shared = self._shared_params(hyper_network)

        with torch.enable_grad():
            while len(grads) < self.grads_per_task:
                for x, y in loader:
                    if len(grads) >= self.grads_per_task:
                        break
                    x, y = x.to(self._device), y.to(self._device)

                    # ── Step 1: get g_w = ∂L/∂w in w-space ───────────────
                    hyper_network.zero_grad()
                    hyper_network.spawn(task_id)
                    w = hyper_network.w
                    output = hyper_network(x)
                    loss = criterion(output, y)
                    (g_w,) = torch.autograd.grad(loss, w)       # graph freed

                    # ── Step 2: translate g_w → g_θ = Jᵀg_w ──────────────
                    hyper_network.zero_grad()
                    hyper_network.spawn(task_id)                # fresh graph
                    hyper_network.w.backward(g_w.detach())      # θ.grad = Jᵀ g_w

                    g_theta = torch.cat([
                        p.grad.view(-1) for p in shared if p.grad is not None
                    ])
                    # ADD THIS LINE: Normalize the direction so Task 1 gradients aren't "smaller" than Task 2
                    g_theta = g_theta / (torch.norm(g_theta) + 1e-8) 

                    grads.append(g_theta.detach().cpu())
                    hyper_network.zero_grad()


        hyper_network.train()
        return torch.stack(grads, dim=1)    # [D_theta_shared, grads_per_task]

    def _fopng_update(self, g, G, F_old, F_new, A_inv, lr, lam, eps=1e-8):
        g_cpu = g.detach().cpu()
        
        # ── 1. Projection ────────────────────────────────────────────────────
        F_old_g    = F_old * g_cpu
        GtFg       = G.t() @ F_old_g
        coeff      = A_inv @ GtFg
        correction = G @ coeff 
        
        Pg = g_cpu - correction

        # ── 2. Metrics ───────────────────────────────────────────────────────
        F_sqrt         = F_old.clamp(min=0).sqrt()
        weighted_rho   = ((F_sqrt * Pg).norm() / ((F_sqrt * g_cpu).norm() + eps)).item()
        correction_norm = correction.norm().item()
        raw_rho        = (Pg.norm() / (g_cpu.norm() + eps)).item()

        # ── 3. STABLE Natural Gradient ───────────────────────────────────────
        # Use a floor (e.g., 0.2) to prevent the 1000x amplification
        F_new_inv = 1.0 / (torch.sqrt(F_new + 0.2)) 
        v_raw = F_new_inv * Pg
        
        # ── 4. MANDATORY CLIPPING ────────────────────────────────────────────
        # This prevents the "corr" from destroying Task 1
        max_norm = 0.5 
        v_norm = torch.norm(v_raw)
        if v_norm > max_norm:
            v_raw = v_raw * (max_norm / v_norm)

        update_direction = -lr * v_raw
        return update_direction.to(g.device), weighted_rho, correction_norm, raw_rho


    def _build_A_inv(self, G, F_old, F_new, lam):
        # Use F_old as the primary metric. Scale G by F_old once.
        scale = F_old  # Dropped the square power for better signal-to-noise
        scaled_G = scale.unsqueeze(1) * G
        A = G.t() @ scaled_G
        A = A + lam * torch.eye(A.shape[0], device=A.device)
        return torch.linalg.pinv(A)          
    
    def _cosine_similarity(self, F_a, F_b):
        # Even though Fisher Matrix would have a different norm form if I used a full matrix,
        #  a diagonal one has the default euclidian form as it is just a vector.
        #   F_a o F_B    /
        #||F_a||||F_b||
        F_a_flat = F_a.detach().cpu().view(-1)
        F_b_flat = F_b.detach().cpu().view(-1)
        
        dot_product = torch.dot(F_a_flat, F_b_flat)
        norm_a = torch.norm(F_a_flat, p=2)
        norm_b = torch.norm(F_b_flat, p=2)
        
        return (dot_product / (norm_a * norm_b)).item()
    

    def _pearson_correlation(self, F_a, F_b):
        """
        Calculates the Pearson correlation coefficient between two tensors on the GPU.
        Pure PyTorch implementation
        """
        # 1. Flatten the tensors
        F_a_flat = F_a.detach().cpu().view(-1)
        F_b_flat = F_b.detach().cpu().view(-1)
        
        # 2. Calculate the means
        mean_a = torch.mean(F_a_flat)
        mean_b = torch.mean(F_b_flat)
        
        # 3. Mean-center the tensors
        A_centered = F_a_flat - mean_a
        B_centered = F_b_flat - mean_b
        
        # 4. Calculate covariance (numerator) and variances (denominator components)
        covariance = torch.sum(A_centered * B_centered)
        var_a = torch.sum(A_centered ** 2)
        var_b = torch.sum(B_centered ** 2)
        
        # 5. Calculate final coefficient (adding 1e-8 to avoid division by zero)
        pearson_r = covariance / (torch.sqrt(var_a * var_b) + 1e-8)
        
        # Return as a standard Python float
        return pearson_r.item()

    def _calculate_topk_iou(self, F_a, F_b, k_fraction=0.10):
        """
        Calculates the IoU of the top K important parameters between two Fisher matrices.
        
        Args:
            F_a (torch.Tensor): 1D tensor of diagonal Fisher values for Task A.
            F_b (torch.Tensor): 1D tensor of diagonal Fisher values for Task B.
            k_fraction (float): The percentage of total parameters to consider as "Top K".
                                Default is 0.10 (Top 10%).
                                
        Returns:
            float: The Intersection over Union (IoU) score between 0.0 and 1.0.
        """
        # 1. Flatten tensors to 1D (assuming they are diagonal approximations)
        F_a_flat = F_a.detach().cpu().view(-1)
        F_b_flat = F_b.detach().cpu().view(-1)
        
        assert F_a_flat.shape == F_b_flat.shape, "Fisher vectors must have the same size."
        
        # 2. Determine K based on the total number of parameters
        total_params = F_a_flat.numel()
        k = int(total_params * k_fraction)
        
        if k == 0:
            return 0.0
        
        # 3. Get the indices of the Top K values for both tasks
        # torch.topk returns a tuple of (values, indices). We only need the indices.
        _, indices_a = torch.topk(F_a_flat, k)
        _, indices_b = torch.topk(F_b_flat, k)
        
        # 4. Calculate Intersection using pure PyTorch (Fast on GPU)
        # Concatenate the two index tensors
        combined_indices = torch.cat((indices_a, indices_b))
        
        # Count how many times each index appears
        # An index appearing 2 times means it exists in both Top-K sets (Intersection)
        _, counts = combined_indices.unique(return_counts=True)
        intersection_size = (counts > 1).sum().item()
        
        # 5. Calculate Union and IoU
        union_size = (2 * k) - intersection_size
        
        iou = intersection_size / union_size
        
        return iou

    def frechet(self, F_1, F_2): # TODO FINISH FRECHET
        # Normalize to unit trace
        F_1_norm = F_1 / (F_1.sum() + 1e-8)
        F_2_norm = F_2 / (F_2.sum() + 1e-8)

        # Frechet distance (squared) for diagonal matrices
        # d^2 = 0.5 * sum( (sqrt(F1) - sqrt(F2))^2 )
        d_squared = 0.5 * torch.sum((torch.sqrt(F_1_norm) - torch.sqrt(F_2_norm))**2)

        fisher_overlap = 1.0 - d_squared.item()
        return fisher_overlap
    
    def compute_overlap_matrix(self):
        keys = list(self.fisher_after_task.keys())
        n = len(keys)
        # Initialize a symmetric matrix with 1s on the diagonal
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = self.frechet(
                    self.fisher_after_task[keys[i]], 
                    self.fisher_after_task[keys[j]]
                )
                matrix[i, j] = overlap
                matrix[j, i] = overlap  # Symmetry
                
        return matrix, keys

def train_fopng(
    hyper_network: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    lam: float = 1e-3,
    alpha: float = 0.5,
    grads_per_task: int = 80,
    max_directions: int = 400,
    fisher_samples: int = 1024,
    epochs: int = 5,
    max_epochs: int = None,
    first_task_optimizer_cls=torch.optim.Adam,
    task_classes: Optional[list] = None,
    verbose: bool = True,
) -> FOPNG:
    device = next(hyper_network.parameters()).device
    fopng = FOPNG(
        lr=lr, lam=lam, alpha=alpha,
        grads_per_task=grads_per_task,
        max_directions=max_directions,
        fisher_samples=fisher_samples,
    )
    results = {}
    global_epoch = 0

    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)
        
        if t == 0:
            if verbose: print(f"[FOPNG] Task 1 – {first_task_optimizer_cls.__name__}")
            opt = first_task_optimizer_cls(hyper_network.parameters(), lr=lr)
            for epoch in range(epochs):
                total_loss = 0.0
                hyper_network.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    hyper_network.spawn(task_id)
                    output = hyper_network(x)
                    loss = criterion(output, y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                wandb.log({"fopng/train/loss": avg_loss, "fopng/global_epoch": global_epoch, "task": t+1})
                global_epoch += 1
                if verbose: print(f"  epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")
            fopng.after_task(hyper_network, task_id, loader, criterion)

        else:
            if verbose: print(f"\n[FOPNG] Task {t+1}")

            best_loss = inf
            loss_repeat = 0
            max_epochs = max_epochs if max_epochs else epochs
            epoch = 0
            while best_loss >= 0.25 and loss_repeat < 5 and epoch < max_epochs:
                F_new = fopng.compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
                fopng.prepare_epoch(F_new)
                total_loss = 0.0
                total_weighted_rho = 0.0
                total_correction_norm = 0.0
                total_raw_rho = 0.0
                hyper_network.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    hyper_network.zero_grad()

                    hyper_network.spawn(task_id)
                    w = hyper_network.w
                    output = hyper_network(x)
                    loss = criterion(output, y)
                    total_loss += loss.item()

                    (g_w,) = torch.autograd.grad(loss, w)

                    weighted_rho, correction_norm, raw_rho = fopng.step(hyper_network, task_id, g_w.detach())
                    total_weighted_rho    += weighted_rho
                    total_correction_norm += correction_norm
                    total_raw_rho         += raw_rho

                n_batches = len(loader)
                avg_loss           = total_loss           / n_batches
                avg_weighted_rho   = total_weighted_rho   / n_batches
                avg_correction_norm= total_correction_norm/ n_batches
                avg_raw_rho        = total_raw_rho        / n_batches

                if best_loss < avg_loss:
                    loss_repeat += 1
                else:
                    loss_repeat = 0
                    best_loss = avg_loss

                wandb.log({
                    "fopng/train/loss":             avg_loss,
                    # weighted_rho: projection quality within Fisher-important subspace
                    # (the correct metric for FOPNG — want this LOW, close to 0)
                    "fopng/train/weighted_rho":     avg_weighted_rho,
                    # correction_norm: absolute gradient mass removed per step
                    # (want this non-trivially large relative to g_norm)
                    "fopng/train/correction_norm":  avg_correction_norm,
                    # raw_rho: ‖Pg‖/‖g‖ — kept for reference but ≈1 by design in FOPNG
                    "fopng/train/raw_rho":          avg_raw_rho,
                    "fopng/global_epoch":           global_epoch,
                    "task":                         t + 1,
                })
                global_epoch += 1

                if verbose: print(f"  epoch {epoch+1}/{max_epochs} loss={avg_loss:.4f} w_rho={avg_weighted_rho:.4f} corr={avg_correction_norm:.4e} raw_rho={avg_raw_rho:.4f}")
                epoch += 1

            fopng.after_task(hyper_network, task_id, loader, criterion)
                
        # ── Evaluate on ALL tasks using TEST loaders ───────────────────
        results[t+1] = []
        eval_metrics = {"task_completed": t+1}
        
        # CHANGED: Iterate over every single task, seen or unseen!
        for i in range(len(test_loaders)): 
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            acc = evaluate_accuracy(hyper_network, test_loaders[i], eval_task_id, task_classes=tc)
            results[t+1].append(acc)
            eval_metrics[f"fopng/eval/acc_task_{i+1}"] = acc
            if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")
            
        if t != 0:
            bwt = calc_bwt(results, task_id=t+1)
            eval_metrics["fopng/eval/bwt"] = bwt
            if verbose: print(f"BWT at task {t+1}: {bwt:.4f}")
                    # Normalize to unit trace


        wandb.log(eval_metrics)

    tasks_completed = sorted(list(results.keys())) # [1, 2, 3]
    num_eval_tasks = len(test_loaders)

    matrix, keys = fopng.compute_overlap_matrix()
    heat_map = plot_overlap(matrix, keys)
    wandb.log({"FOPNG FRECHET CORR MATRIX": wandb.Image(heat_map)})
    # 1. Log the overlapping FOPNG chart
    plt.figure(figsize=(10, 6))
    
    # Define a clean, distinct color palette
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        # Force solid line (linestyle='-') and cycle through colors
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title("FOPNG Hypernetwork: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    # Log the cleanly colored plot directly to W&B
    wandb.log({"FOPNG Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results

# ─────────────────────────────────────────────────────────────────────────────
# FOPNG+ Class
# ─────────────────────────────────────────────────────────────────────────────
class FOPNGPlus(FOPNG):
    """
    FOPNG+ — FOPNG with Jacobian-refreshed gradient memory.

    The core problem with FOPNG (and OGD) is G staleness:
        - At the end of task t, we collect gradient directions g_θ = Jᵀ(θ_t) g_w
          and store them in G.
        - By task t+2, the shared MLP has moved to θ_{t+2}, so the Jacobian
          J(θ_{t+2}) ≠ J(θ_t).
        - Stored directions no longer describe which θ movements harm old tasks,
          so the projection is geometrically stale and protection degrades.

    FOPNG+ fix (Bennani et al. 2020, OGD+):
        After completing each task, rebuild G from scratch by re-collecting
        gradient directions for ALL previously seen tasks under the CURRENT θ.
        This costs O(num_tasks × grads_per_task) extra gradient evaluations
        once per task boundary — cheap relative to training.

    Cost analysis (5 tasks, 300 grads/task):
        FOPNG:   300 grad evals at each task boundary  (total: 1500)
        FOPNG+:  300, 600, 900, 1200, 1500 at boundaries (total: 4500)
        The 3× overhead per run is well worth the forgetting reduction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Stores (task_id_tensor, loader) for every completed task so we can
        # re-collect gradients under the updated θ at each task boundary.
        self._task_history: List[tuple] = []

    def after_task(
        self,
        hyper_network: nn.Module,
        task_id,
        loader: DataLoader,
        criterion: Callable,
    ) -> None:
        device = next(hyper_network.parameters()).device
        self._device = device

        # ── 1. Fisher update (identical to FOPNG) ────────────────────────
        F_new = self.compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
        self.fisher_after_task[task_id.item()] = F_new

        if self.F_old is not None:
            cosine_sim  = self._cosine_similarity(self.F_old, F_new)
            pearson_corr = self._pearson_correlation(self.F_old, F_new)
            topk_iou    = self._calculate_topk_iou(self.F_old, F_new)
        else:
            cosine_sim = pearson_corr = topk_iou = 1.0

        if self.F_old is None:
            self.F_old = F_new.detach().cpu()
        else:
            self.F_old = (1.0 - self.alpha) * self.F_old + self.alpha * F_new.detach().cpu()
            f_nonzero = self.F_old[self.F_old > 0]
            if len(f_nonzero) > 0:
                p99 = torch.quantile(f_nonzero, 0.99)
                self.F_old = self.F_old.clamp(max=p99.item())
            f_max = self.F_old.max()
            if f_max > 0:
                self.F_old = self.F_old / f_max

        # ── 2. Store this task for future Jacobian refreshes ─────────────
        self._task_history.append((task_id.clone(), loader))

        # ── 3. FOPNG+: Rebuild G from scratch under current θ ────────────
        # Re-collect grads_per_task directions for EVERY previously seen task
        # using the current model weights. This replaces stale J(θ_old) vectors
        # with fresh J(θ_current) vectors, keeping the projection subspace
        # geometrically aligned with the actual parameter space.
        print(f"  [FOPNG+] Refreshing G for {len(self._task_history)} task(s) under current θ...")
        all_cols = []
        for prev_task_id, prev_loader in self._task_history:
            cols = self._collect_gradients(hyper_network, prev_task_id, prev_loader, criterion)
            all_cols.append(cols)

        G_fresh = torch.cat(all_cols, dim=1)   # [D_theta_shared, total_cols]

        # Apply max_directions budget — spread evenly across tasks
        if G_fresh.shape[1] > self.max_directions:
            indices = torch.linspace(
                0, G_fresh.shape[1] - 1,
                steps=self.max_directions,
                dtype=torch.long,
                device=G_fresh.device,
            )
            G_fresh = G_fresh[:, indices]

        self.G = G_fresh

        f_nonzero2 = self.F_old[self.F_old > 0]
        logs = {
            "fopng_plus/fisher/min":             self.F_old.min().item(),
            "fopng_plus/fisher/max":             self.F_old.max().item(),
            "fopng_plus/fisher/mean":            self.F_old.mean().item(),
            "fopng_plus/fisher/mean_nonzero":    f_nonzero2.mean().item() if len(f_nonzero2) > 0 else 0.0,
            "fopng_plus/fisher/frac_nonzero":    (len(f_nonzero2) / len(self.F_old.view(-1))),
            "fopng_plus/memory/G_cols":          self.G.shape[1],
            "fopng_plus/memory/tasks_in_history": len(self._task_history),
            "fopng_plus/fisher_overlap/cosine":  cosine_sim,
            "fopng_plus/fisher_overlap/pearson": pearson_corr,
            "fopng_plus/fisher_overlap/topk_iou": topk_iou,
            "task_completed": task_id.item() + 1,
        }
        print(logs)
        wandb.log(logs)
        torch.cuda.empty_cache()
        gc.collect()


def train_fopng_plus(
    hyper_network: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    lam: float = 1e-3,
    alpha: float = 0.5,
    grads_per_task: int = 80,
    max_directions: int = 400,
    fisher_samples: int = 1024,
    epochs: int = 5,
    max_epochs: int = None,
    first_task_optimizer_cls=torch.optim.Adam,
    task_classes: Optional[list] = None,
    verbose: bool = True,
) -> dict:
    """
    Training loop for FOPNG+.

    Identical to train_fopng except it uses FOPNGPlus, which rebuilds G
    from scratch under the current θ at every task boundary. The training
    loop itself (forward pass, step, eval) is unchanged.
    """
    device = next(hyper_network.parameters()).device
    fopng_plus = FOPNGPlus(
        lr=lr, lam=lam, alpha=alpha,
        grads_per_task=grads_per_task,
        max_directions=max_directions,
        fisher_samples=fisher_samples,
    )
    results = {}
    global_epoch = 0

    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)

        if t == 0:
            if verbose: print(f"[FOPNG+] Task 1 – {first_task_optimizer_cls.__name__}")
            opt = first_task_optimizer_cls(hyper_network.parameters(), lr=lr)
            for epoch in range(epochs):
                total_loss = 0.0
                hyper_network.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    hyper_network.spawn(task_id)
                    output = hyper_network(x)
                    loss = criterion(output, y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(loader)
                wandb.log({"fopng_plus/train/loss": avg_loss, "fopng_plus/global_epoch": global_epoch, "task": t + 1})
                global_epoch += 1
                if verbose: print(f"  epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")
            fopng_plus.after_task(hyper_network, task_id, loader, criterion)

        else:
            if verbose: print(f"\n[FOPNG+] Task {t+1}")
            best_loss = inf
            loss_repeat = 0
            _max_epochs = max_epochs if max_epochs else epochs
            epoch = 0

            while best_loss >= 0.25 and loss_repeat < 5 and epoch < _max_epochs:
                F_new = fopng_plus.compute_fisher_diag(hyper_network, task_id, loader, criterion, device)
                fopng_plus.prepare_epoch(F_new)
                total_loss = 0.0
                total_weighted_rho = 0.0
                total_correction_norm = 0.0
                total_raw_rho = 0.0
                hyper_network.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    hyper_network.zero_grad()
                    hyper_network.spawn(task_id)
                    w = hyper_network.w
                    output = hyper_network(x)
                    loss = criterion(output, y)
                    total_loss += loss.item()
                    (g_w,) = torch.autograd.grad(loss, w)
                    weighted_rho, correction_norm, raw_rho = fopng_plus.step(
                        hyper_network, task_id, g_w.detach()
                    )
                    total_weighted_rho    += weighted_rho
                    total_correction_norm += correction_norm
                    total_raw_rho         += raw_rho

                n_batches = len(loader)
                avg_loss            = total_loss            / n_batches
                avg_weighted_rho    = total_weighted_rho    / n_batches
                avg_correction_norm = total_correction_norm / n_batches
                avg_raw_rho         = total_raw_rho         / n_batches

                if best_loss < avg_loss:
                    loss_repeat += 1
                else:
                    loss_repeat = 0
                    best_loss = avg_loss

                wandb.log({
                    "fopng_plus/train/loss":            avg_loss,
                    "fopng_plus/train/weighted_rho":    avg_weighted_rho,
                    "fopng_plus/train/correction_norm": avg_correction_norm,
                    "fopng_plus/train/raw_rho":         avg_raw_rho,
                    "fopng_plus/global_epoch":          global_epoch,
                    "task":                             t + 1,
                })
                global_epoch += 1
                if verbose:
                    print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f} "
                          f"w_rho={avg_weighted_rho:.4f} corr={avg_correction_norm:.4e}")
                epoch += 1

            fopng_plus.after_task(hyper_network, task_id, loader, criterion)

        # ── Evaluate on ALL tasks ─────────────────────────────────────────
        results[t + 1] = []
        eval_metrics = {"task_completed": t + 1}
        for i in range(len(test_loaders)):
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            acc = evaluate_accuracy(hyper_network, test_loaders[i], eval_task_id, task_classes=tc)
            results[t + 1].append(acc)
            eval_metrics[f"fopng_plus/eval/acc_task_{i+1}"] = acc
            if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")

        if t != 0:
            bwt = calc_bwt(results, task_id=t + 1)
            eval_metrics["fopng_plus/eval/bwt"] = bwt
            if verbose: print(f"BWT for task {t+1}: {bwt:.4f}")

        wandb.log(eval_metrics)

    # ── Final plots ───────────────────────────────────────────────────────
    tasks_completed = sorted(results.keys())
    num_eval_tasks  = len(test_loaders)

    matrix, keys = fopng_plus.compute_overlap_matrix()
    heat_map = plot_overlap(matrix, keys)
    wandb.log({"FOPNG+ FRECHET CORR MATRIX": wandb.Image(heat_map)})

    plt.figure(figsize=(10, 6))
    cmap   = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5,
                 color=colors[i % len(colors)], label=f"{i+1}")
    plt.title("FOPNG+ Hypernetwork: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    wandb.log({"FOPNG+ Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results