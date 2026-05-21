from __future__ import annotations

from typing import Callable, List, Optional, Literal
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import wandb
from utils import get_grad_vector, calc_bwt, evaluate_accuracy, plot_overlap
import matplotlib.pyplot as plt
import numpy as np
import gc #Garbage Collector
from math import inf
from fisher import DiagonalFisherEstimator
from gradient import AVECollector, GradientMemory, GTLCollector, BoundaryCollector
from models.hyper_network import HyperRegulizer


# ─────────────────────────────────────────────────────────────────────────────
# Base Class
# ─────────────────────────────────────────────────────────────────────────────
class OP(ABC):
    def __init__(
            self,
            num_tasks,
            lr: float = 1e-3,
            lam: float = 1e-3,
            alpha: float = 0.5,
            grads_per_task: int = 80,
            max_directions: int = 400,
            fisher_samples: int = 1024,
            device_mode: Literal["cpu", "gpu", "hybrid"] = "hybrid",
            fisher_clipping: bool = False,
            normalize: bool = False,
        ):
        self.lr            = lr
        self.lam           = lam
        self.alpha         = alpha
        self.grads_per_task = grads_per_task

        # ── Device mode ───────────────────────────────────────────────────
        # "cpu"    — all FOPNG state and computation on CPU.
        # "gpu"    — all FOPNG state and computation on the model's GPU.
        # "hybrid" — large persistent matrices (Fisher, G, A_inv) live on CPU;
        #            per-step momentum buffer and parameter update run on GPU.
        self.calc_device = torch.device("cuda") if device_mode == "gpu" else torch.device("cpu") 

        self.num_tasks = num_tasks
        
        self.F_old: Optional[Tensor] = None
        self.G:     Optional[Tensor] = None
        self.F_new: Optional[Tensor] = None
        self.A_inv: Optional[Tensor] = None
        self.debug = True
        self.fisher_after_task = {}
        self.momentum_buffer = None
        self.task_momentum = None
        self.max_directions = max_directions
        self.quantile = 0.95
        self.damping = 0

        self.normalize = normalize
        self.FisherEstimator = DiagonalFisherEstimator(
            use_vmap = False, 
            fisher_samples=fisher_samples,
            clipping = fisher_clipping,
            normalization = normalize
        )
        self.gradient_memory = GradientMemory(mode="raw", max_directions=max_directions, normalization=normalize)
        self.GradientCollector = GTLCollector(grads_per_task)

    def prepare_epoch(self, F_new: Tensor) -> None:
        assert self.F_old is not None, "Call after_task() after task 1 before training task 2."
        self.F_new = F_new
        self.build_A_inv(self.gradient_memory.matrix, self.F_old, self.F_new)

    def step(self, model: nn.Module, task_id, g_theta: Tensor) -> float:

        assert self.A_inv is not None, "Call prepare_epoch(F_new) before step()."
     
        # 2. In FOPNG.step, update the task_emb logic:
        with torch.no_grad():
            if hasattr(model, "task_emb"):
                te_grad = model.task_emb.weight.grad
                if te_grad is not None:
                    model.task_emb.weight.data.add_(-self.lr * te_grad)

      
        v_star_theta, weighted_rho, correction_norm, raw_rho = self.update(
            g=g_theta, G=self.gradient_memory.matrix, F_old=self.F_old, F_new=self.F_new
        )

        pointer = 0
        with torch.no_grad():
            for p in model._shared_params:
                n = p.numel()
                p.data.add_(v_star_theta[pointer : pointer + n].view_as(p))
                pointer += n
        return v_star_theta.norm(), weighted_rho, correction_norm, raw_rho

    def after_task(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        F_new = self.FisherEstimator.estimate(model, task_id, loader, criterion, self.calc_device)
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
            self.F_old = F_new.detach().to(self.calc_device)
        else:
            # Arithmetic Mean: All tasks have exactly 1/N weight BIG CHANGE
            # n = task_id.item()+1
            # self.F_old = ((n - 1) / n) * self.F_old + (1.0 / n) * F_new.detach().to(self.calc_device)
            self.F_old = torch.max(self.F_old, F_new.detach())
            # self.F_old = (1 - self.alpha) * self.F_old + self.alpha * F_new.detach().to(self.calc_device)
            # CULPRIT FIX: Re-normalize so the combined history peak is always 1.0
            # self.F_old += F_new.detach()
            
        # if self.FisherEstimator.normalization:
        #     self.F_old /= model.num_of_chunks
            # if self.F_old.max() > 0:
            #     self.F_old = self.F_old / self.F_old.max()
            
        # 1. Collect the raw gradients for the current task
        self.GradientCollector.collect(
            memory = self.gradient_memory, 
            model = model, 
            dataloader = loader,
            device = self.calc_device, 
            task_id = task_id,
        )

        if self.debug:
            print(f"[{self.__name__}] Current G memory size: {len(self.gradient_memory)} / {self.max_directions}")

        logs = {
            f"{self.__name__}/fisher/min": self.F_old.min().item(),
            f"{self.__name__}/fisher/max": self.F_old.max().item(),
            f"{self.__name__}/fisher/mean": self.F_old.mean().item(),
            # mean over non-zero entries — should be 0.1–0.4 with healthy Fisher
            # (was 0.0002 with /max normalization, indicating near-delta distribution)
            f"{self.__name__}/memory/G_cols": len(self.gradient_memory),
            f"{self.__name__}/fisher_overlap/cosine": cosine_sim,
            f"{self.__name__}/fisher_overlap/pearson": pearson_corr,
            f"{self.__name__}/fisher_overlap/topk_iou": topk_iou,
            "task_completed": task_id.item() + 1
        }
        print(logs)

        wandb.log(logs)
        torch.cuda.empty_cache()
        gc.collect()


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


def get_magnitude_decay_lr(current_lr: float) -> float:
    """
    Decays LR perfectly through magnitudes:
    1e-2 -> 5e-3 -> 1e-3 -> 5e-4 -> 1e-4 ...
    """
    # Format to strict scientific notation to avoid float precision drift
    sci_str = f"{current_lr:.1e}"  # e.g., '1.0e-02' or '5.0e-03'
    mantissa, exp = sci_str.split('e')
    mantissa = float(mantissa)
    exp = int(exp)
    
    if mantissa >= 4.9: # If current LR starts with 5 (e.g., 0.05) -> drop to 0.01
        return 1.0 * (10 ** exp)
    else:               # If current LR starts with 1 (e.g., 0.01) -> drop to 0.005
        return 5.0 * (10 ** (exp - 1))

class PreFOPNG(OP):
    __name__ = "PreFOPNG"

    def update(self, g, G, F_old, F_new, eps=1e-8):
        """
        Compute FOPNG-PF update step with pre-multiplied Fisher gradients.
        
        KEY SIMPLIFICATION:
        - Original: F_old_g = F_old * gradient, then G_T_F_old_g = G.T @ F_old_g
        - FOPNG-PF: G_T_g = G.T @ gradient (G already has Fisher baked in)
        """
        F_new_inv_diag = 1.0 / (F_new + self.lam)
        
        G_T_g = G.T @ g
        
        A_inv_G_T_g = self.A_inv @ G_T_g
        correction = (G @ A_inv_G_T_g).view(-1)
        P_g = g - correction
    

        # Natural gradient step
        F_new_inv_P_g = P_g * F_new_inv_diag
        denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
        v_star = -self.lr * F_new_inv_P_g / (denom + 1e-8)

        # Metrics
        F_sqrt = F_old.clamp(min=0).sqrt()
        weighted_rho = ((F_sqrt * P_g).norm() / ((F_sqrt * g).norm() + eps)).item()
        return v_star, weighted_rho, correction.norm().item(), (P_g.norm() / (g.norm() + eps)).item()
           
    def build_A_inv(
            self,
            G: Tensor,
            F_old: Tensor,
            F_new: Tensor, # Included for signature compatibility
        ) -> Tensor:
        # Since G is already F*g, we just use it directly
        # A = G.T @ G (no F_old multiplication needed - already baked in G)
        F_new_inv_diag = 1.0 / (F_new + self.lam)
        A = G.T @ (F_new_inv_diag.view(-1, 1) * G) + self.lam * torch.eye(G.size(1), device=G.device)
        self.A_inv = torch.pinverse(A)
        print(self.A_inv)
        self.A = A
            
    def after_task(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        F_new = self.FisherEstimator.estimate(model, task_id, loader, criterion, self.calc_device)
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
            self.F_old = F_new.detach().to(self.calc_device)
        else:
            self.F_old = torch.max(self.F_old, F_new.detach())       

        # 1. Collect the raw gradients for the current task
        self.GradientCollector.collect_empirical_fisher_preconditioned(
            memory = self.gradient_memory, 
            model = model, 
            num_directions=self.grads_per_task,
            dataloader = loader,
            device = self.calc_device, 
            task_id = task_id,
            normalize = self.normalize
        )

        model.spawn(task_id)

        if self.debug:
            print(f"[{self.__name__}] Current G memory size: {len(self.gradient_memory)} / {self.max_directions}")

        logs = {
            f"{self.__name__}/fisher/min": self.F_old.min().item(),
            f"{self.__name__}/fisher/max": self.F_old.max().item(),
            f"{self.__name__}/fisher/mean": self.F_old.mean().item(),
            # mean over non-zero entries — should be 0.1–0.4 with healthy Fisher
            # (was 0.0002 with /max normalization, indicating near-delta distribution)
            f"{self.__name__}/memory/G_cols": len(self.gradient_memory),
            f"{self.__name__}/fisher_overlap/cosine": cosine_sim,
            f"{self.__name__}/fisher_overlap/pearson": pearson_corr,
            f"{self.__name__}/fisher_overlap/topk_iou": topk_iou,
            "task_completed": task_id.item() + 1
        }
        print(logs)

        wandb.log(logs)
        torch.cuda.empty_cache()
        gc.collect()


class FOPNG(OP):
    __name__ = "FOPNG"

    def update(self, g, G, F_old, F_new, eps=1e-8):
        if self.normalize:
            scale_factor = F_new.max().clamp(min=1.0)
        else:
            scale_factor = 1
        
        F_old_s = F_old / scale_factor
        F_new_s = F_new / scale_factor

        F_new_inv_diag = 1.0 / (F_new_s + self.lam)
            
        # Original projection logic
        F_old_g = F_old_s * g
        G_T_F_old_g = G.T @ F_old_g
        A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g
        correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old_s.squeeze()
        P_g = g - correction
        
         
        F_new_inv_P_g = P_g * F_new_inv_diag
        denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
        v_star = -self.lr * F_new_inv_P_g / (denom + 1e-8)
        
        F_old_s_sqrt = F_old_s.clamp(min=0).sqrt()

        # Metrics
        w_rho = ((F_old_s_sqrt * P_g).norm() / ((F_old_s_sqrt * g).norm() + eps)).item()
        return v_star, w_rho, correction.norm().item(), (P_g.norm() / (g.norm() + eps)).item()

    def build_A_inv(self, G, F_old, F_new) -> None:
        if self.normalize:
            scale_factor = F_new.max().clamp(min=1.0)
        else:
            scale_factor = 1

        F_new_s = F_new / scale_factor
        F_old_s = F_old / scale_factor

        scaled_lam = self.lam / scale_factor

        F_new_inv = 1.0 / (F_new_s + scaled_lam)
        F_old_diag = F_old_s.view(-1, 1)
        F_old_G = F_old_diag * G
        weighted_G = F_old_diag * (F_new_inv.view(-1, 1) * F_old_G)
        

        A = G.T @ weighted_G + scaled_lam * torch.eye(G.size(1), device=G.device)
        self.A_inv = torch.pinverse(A)
        self.A = A
               
        print("A: ", self.A.min().item(), self.A.mean().item(), self.A.max().item())
        print("A_inv: ", self.A_inv.min().item(), self.A_inv.mean().item(), self.A_inv.max().item())




class eFOPNG(OP):
    __name__ = "eFOPNG"
    '''Geometric interpretation of new kernel.'''
    '''The first term bounds the KL divergence on the new task; the second bounds the Fisher distance in the old task geometry. So a single trust region simultaneously constrains both — this is the precise sense in which the method is elastic, and it gives you the clean contrast with EWC: EWC achieves elasticity through an explicit penalty; eFOPNG achieves it by embedding the old-task geometry directly into the ambient metric.'''
    def update(self, g, G, F_old, F_new, eps=1e-8):
        # 1. THE INERTIA INVERSE (The fix for dead parameters)
        # We combine current and past importance so old important weights 
        # stay "heavy" and don't take huge jumps.
        if self.normalize:
            F_combined = F_new + F_old
            scale_factor = F_combined.max().clamp(min=1.0)
        else:
            scale_factor = 1

        # Create LOCAL scaled variables. No in-place operations!
        F_old_s = F_old / scale_factor
        F_new_s = F_new / scale_factor
        
        F_combined_s = F_new_s + F_old_s

        # 2. Scale the damping term to match the FIM scaling
        scaled_lam = self.lam / scale_factor
        F_c_inv = 1.0 / (F_combined_s + scaled_lam)
        
        # F_old_scaled = F_old / 18449
        # 2. Projection Logic
        F_old_g = F_old_s * g                                                 
        G_T_F_old_g = G.T @ F_old_g                                         
        A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g                        
        correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old_s.squeeze()     
        # Dynamically scale it if it overpowers the original gradient
        P_g = g - correction                                                

        # 3. Calculate Final Step using the Inertia Geometry
        F_inv_P_g = P_g * F_c_inv               
        
        # Trust Region Denominator
        denom = torch.sqrt((P_g * F_inv_P_g).sum() + 1e-8)    
        v_star = -self.lr * F_inv_P_g / (denom + 1e-8)

        # 5. DIAGNOSTIC: Calculate Rho on the actual update 'v_star'
        # This tells you if the final movement is actually safe.
        # We check the overlap of the physical update v_star with the past
        F_c_sqrt = F_combined_s.clamp(min=0).sqrt()

        w_rho = ((F_c_sqrt * P_g).norm() / ((F_c_sqrt * g).norm() + eps)).item()
        return v_star, w_rho, correction.norm().item(), (P_g.norm() / (g.norm() + eps)).item()

    def build_A_inv(self, G, F_old, F_new) -> None:
        # Use the same Inertia Geometry for the matrix A
        F_combined = F_new + F_old
        if self.normalize:
            scale_factor = F_combined.max().clamp(min=1.0)
        else:
            scale_factor = 1

        
        # Create LOCAL scaled variables. No in-place operations!
        F_old_s = F_old / scale_factor
        F_new_s = F_new / scale_factor
        F_combined_s = F_new_s + F_old_s

        # 2. Scale the damping term to match the FIM scaling
        scaled_lam = self.lam / scale_factor
        F_c_inv = 1.0 / (F_combined_s + scaled_lam)

        F_old_diag = F_old_s.view(-1, 1)
        F_old_G = F_old_diag * G
        weighted_G = F_old_diag * (F_c_inv.view(-1, 1) * F_old_G)

        A = G.T @ weighted_G + scaled_lam * torch.eye(G.size(1), device=G.device)
        self.A_inv = torch.pinverse(A)
        self.A = A
         
        print("A: ", self.A.min().item(), self.A.mean().item(), self.A.max().item())
        print("A_inv: ", self.A_inv.min().item(), self.A_inv.mean().item(), self.A_inv.max().item())

class preEFOPNG(OP):
    __name__ = "preEFOPNG"

    def build_A_inv(self, G, F_old, F_new) -> None:
        """
        G is already G̃ = F_task * g_raw (PreFisher gradients).
        For eFOPNG-PreFisher: A = G̃ᵀ Fc⁻¹ G̃,  Fc = F_new + F_old
        """
        F_combined = F_new + F_old
        scale_factor = F_combined.max().clamp(min=1.0)
        
        F_c_s = F_combined / scale_factor
        scaled_lam = self.lam / scale_factor
        F_c_inv_s = 1.0 / (F_c_s + scaled_lam)   # [D]

        # A = G̃ᵀ Fc⁻¹ G̃  (G̃ already has F_old baked in)
        weighted_G = F_c_inv_s.view(-1, 1) * G    # [D, K]
        A = G.T @ weighted_G                       # [K, K]
        A = A + scaled_lam * torch.eye(G.size(1), device=G.device)

        self.A_inv = torch.linalg.pinv(A)
        self.A = A
        self._scale_factor = scale_factor          # cache for update()

        print("A: ", A.min().item(), A.mean().item(), A.max().item())
        print("A_inv: ", self.A_inv.min().item(),
              self.A_inv.mean().item(), self.A_inv.max().item())

    def update(self, g, G, F_old, F_new, eps=1e-8):
        """
        G is G̃ (pre-Fisher gradients).
        P = I - G̃ (G̃ᵀ Fc⁻¹ G̃)⁻¹ G̃ᵀ
        v* = -lr * Fc⁻¹ Pg / ||Pg||_Fc
        """
        # Use cached scale from build_A_inv for consistency
        scale_factor = getattr(self, '_scale_factor',
                               (F_new + F_old).max().clamp(min=1.0))

        F_c_s      = (F_new + F_old) / scale_factor
        scaled_lam = self.lam / scale_factor
        F_c_inv_s  = 1.0 / (F_c_s + scaled_lam)   # [D]

        # --- Projection (never forms D×D) ---
        Gt_g           = G.T @ g                   # [K]
        A_inv_Gt_g     = self.A_inv @ Gt_g         # [K]
        correction     = G @ A_inv_Gt_g            # [D]
        P_g            = g - correction            # [D]

        # --- Natural gradient step under Fc metric ---
        F_c_inv_Pg  = P_g * F_c_inv_s             # [D]
        denom       = torch.sqrt((P_g * F_c_inv_Pg).sum() + 1e-8)
        v_star      = -self.lr * F_c_inv_Pg / (denom + 1e-8)

        # --- Diagnostics ---
        F_c_sqrt = F_c_s.clamp(min=0).sqrt()
        w_rho = ((F_c_sqrt * P_g).norm() /
                 ((F_c_sqrt * g).norm() + eps)).item()

        return (v_star,
                w_rho,
                correction.norm().item(),
                (P_g.norm() / (g.norm() + eps)).item())
    
    def after_task(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        F_new = self.FisherEstimator.estimate(model, task_id, loader, criterion, self.calc_device)
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
            self.F_old = F_new.detach().to(self.calc_device)
        else:
            self.F_old = torch.max(self.F_old, F_new.detach())       

        # 1. Collect the raw gradients for the current task
        self.GradientCollector.collect_empirical_fisher_preconditioned(
            memory = self.gradient_memory, 
            model = model, 
            num_directions=self.grads_per_task,
            dataloader = loader,
            device = self.calc_device, 
            task_id = task_id,
            normalize = self.normalize
        )

        model.spawn(task_id)

        if self.debug:
            print(f"[{self.__name__}] Current G memory size: {len(self.gradient_memory)} / {self.max_directions}")

        logs = {
            f"{self.__name__}/fisher/min": self.F_old.min().item(),
            f"{self.__name__}/fisher/max": self.F_old.max().item(),
            f"{self.__name__}/fisher/mean": self.F_old.mean().item(),
            # mean over non-zero entries — should be 0.1–0.4 with healthy Fisher
            # (was 0.0002 with /max normalization, indicating near-delta distribution)
            f"{self.__name__}/memory/G_cols": len(self.gradient_memory),
            f"{self.__name__}/fisher_overlap/cosine": cosine_sim,
            f"{self.__name__}/fisher_overlap/pearson": pearson_corr,
            f"{self.__name__}/fisher_overlap/topk_iou": topk_iou,
            "task_completed": task_id.item() + 1
        }
        print(logs)

        wandb.log(logs)
        torch.cuda.empty_cache()
        gc.collect()

class FNG(OP):
    """
    Fisher Natural Gradient.
    Uses the current task's Fisher information (F_new) to define a Riemannian 
    metric for the update step, ensuring the step size is consistent 
    within the parameter space.
    """
    __name__ = "FNG"

    def _fng_update(self, g, F_new, eps=1e-8):
        """
        Computes the natural gradient step: v* = -lr * F_inv * g / sqrt(g^T * F_inv * g)
        """
        if self.normalize:
            scale_factor = F_new.max().clamp(min=1.0)
        else:
            scale_factor = 1
        
        F_new_s = F_new / scale_factor
        # Pre-compute Fisher inverse diagonal with damping (lam)
        F_inv_diag = 1.0 / (F_new_s + self.lam)
            
        # Natural gradient pre-conditioning
        nat_grad = F_inv_diag * g
        
        # Compute the Fisher norm for normalization (denominator)
        # sqrt(g^T * F_inv * g)
        denom = torch.sqrt((g * nat_grad).sum() + eps)
        
        # Final update direction
        v_star = -self.lr * nat_grad / (denom + eps)
        
        # FNG does not project against G, so we return 1.0 for rho (no reduction)
        # and 0 for correction norm.
        return v_star, 1.0, 0.0, 1.0

    def build_A_inv(self, G, F_old, F_new) -> None:
        """
        FNG does not use the G-subspace projection (A_inv).
        We implement this as a dummy to satisfy the OP interface.
        """
        self.A_inv = torch.tensor([1.0]) # Dummy for assertion check
        self.A = None

    def step(self, model: nn.Module, task_id, g_theta: Tensor) -> float:
        """
        Overrides OP.step to use the FNG update logic.
        """
        # Ensure Fisher is computed for the current task
        assert self.F_new is not None, "F_new must be computed via prepare_epoch."
     
        # Update task specific embedding if it exists
        with torch.no_grad():
            if hasattr(model, "task_emb"):
                te_grad = model.task_emb.weight.grad
                if te_grad is not None:
                    model.task_emb.weight.data.add_(-self.lr * te_grad)

        # Compute the FNG specific update
        v_star_theta, weighted_rho, correction_norm, raw_rho = self._fng_update(
            g=g_theta, F_new=self.F_new
        )

        # Apply update to shared parameters
        pointer = 0
        with torch.no_grad():
            for p in model._shared_params:
                n = p.numel()
                p.data.add_(v_star_theta[pointer : pointer + n].view_as(p))
                pointer += n
                
        return v_star_theta.norm(), weighted_rho, correction_norm, raw_rho
    

class ONG(OP):
    __name__ = "ONG" # THIS SHOULD BE CALLED ONG

    def build_A_inv(self, G: Tensor, F_old: Tensor, F_new: Tensor) -> None:
        """
        In standard OGD, A is simply the correlation matrix of the memory.
        A = GᵀG + λI.
        Since G is orthonormalized via SVD in GradientMemory, A is nearly Identity.
        """
        # G is [D, m]. A is [m, m].
        self.A = G.T @ G + self.lam * torch.eye(G.size(1), device=G.device)
        
        # Use an adaptive damping scaled to the basis energy
        print(self.A)
        self.A_inv = torch.linalg.pinv(self.A)
        
    def update(self, g, G, F_old, F_new, eps=1e-8):
        """
        OGD on Natural Gradient:
        1. Compute the Natural Gradient direction.
        2. Project that direction to be orthogonal to history G.
        """
        if self.normalize:
            scale_factor = F_new.max().clamp(min=1.0)
        else:
            scale_factor = 1
        
        F_new_s = F_new / scale_factor
        # 1. Compute the Natural Gradient (v_nat = F_new⁻¹ g)
        # Using a sqrt(F + eps) floor for stability
        F_inv_diag = 1.0 / (F_new_s + self.lam)
        v_nat = F_inv_diag * g
        
        # 2. Euclidean OGD Projection
        # coeff = (Gᵀ G)⁻¹ Gᵀ v_nat
        coeff = self.A_inv @ (G.T @ v_nat)
        
        # correction = G (Gᵀ G)⁻¹ Gᵀ v_nat
        correction = G @ coeff
        
        # Projected direction v* = v_nat - correction
        v_star_unscaled = v_nat - correction

        # 3. Unit-step Natural Scaling (Optional but recommended for HyperNets)
        # Ensures the update doesn't explode if the Fisher landscape is flat
        denom = torch.sqrt((v_star_unscaled * g).sum() + eps)
        v_star = -self.lr * v_star_unscaled / (denom + eps)

        # Metrics for logging
        F_sqrt = F_new_s.clamp(min=0).sqrt()
        weighted_rho = ((F_sqrt * v_star_unscaled).norm() / ((F_sqrt * v_nat).norm() + eps)).item()
        
        return v_star, weighted_rho, correction.norm().item(), (v_star_unscaled.norm() / (v_nat.norm() + eps)).item()



class OGD(FOPNG):
    __name__ = "OGD"
    def prepare_epoch(self, F_new: Tensor) -> None:
        self.build_A_inv(self.gradient_memory.matrix, self.F_old, self.F_new)

    def build_A_inv(self, G: Tensor, F_old: Tensor, F_new: Tensor) -> None:
        """
        Normal OGD projection matrix: A = GᵀG + λI.
        Only cares about the Euclidean correlation of stored gradients.
        """
        # A is the [m x m] correlation matrix of the memory directions
        self.A = G.T @ G
        
        # Scale damping to the average basis energy for numerical stability
        avg_signal = torch.diag(self.A).mean()
        adaptive_lam = self.lam * avg_signal
        
        # Invert the Euclidean correlation matrix
        self.A_inv = torch.linalg.pinv(self.A + adaptive_lam * torch.eye(G.size(1), device=G.device))

    def update(self, g, G, F_old, F_new, eps=1e-8):
        """
        Standard OGD Update:
        1. Project the raw gradient 'g' to be orthogonal to history 'G'.
        2. Apply a standard SGD step (no Natural Gradient scaling).
        """
        
        # 1. Compute Euclidean Projection
        # coeff = (Gᵀ G + λI)⁻¹ Gᵀ g
        coeff = self.A_inv @ (G.T @ g)
        
        # correction = G · coeff
        correction = G @ coeff
        
        # Projected Gradient: Pg = g - correction
        Pg = g - correction
        
        # 2. Standard SGD step in the projected direction
        v_star = -self.lr * Pg

        # 3. Metrics (using standard Euclidean norms)
        rho = (Pg.norm() / (g.norm() + eps)).item()
        
        # Return signature matches FOPNG.step
        return v_star.to(g.device), rho, correction.norm().item(), rho
   
    def after_task(self, model: nn.Module, task_id, loader: DataLoader, criterion: Callable) -> None:
        self.F_old = None
        
        # 1. Collect the raw gradients for the current task
        self.GradientCollector.collect(
            memory = self.gradient_memory, 
            model = model, 
            dataloader = loader,
            device = self.calc_device, 
            task_id = task_id,
        )

        if self.debug:
            print(f"[{self.__name__}] Current G memory size: {len(self.gradient_memory)} / {self.max_directions}")


        torch.cuda.empty_cache()
        gc.collect()

# Map method names to their respective classes
METHOD_MAP = {
    "fopng": FOPNG,
    "efopng": eFOPNG,
    "ong": ONG,
    "ogd": OGD,
    "fng": FNG,
    "fopng_prefisher": PreFOPNG,
    "efopng_prefisher": preEFOPNG,
}    

def run_continual_method(
    method: str,
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    config,
    first_task_optimizer_cls
):
    """
    Unified entry point for all OP-based methods.
    """
    method_key = method.lower()
    if method_key not in METHOD_MAP:
        raise ValueError(f"Method {method} not found in METHOD_MAP")

    # 1. Handle Regularizer instance
    # We create it here so it can be shared between the Optimizer and the Train loop
    regulizer_instance = HyperRegulizer() if config.get("regulizer", True) else None
    
    # 2. Instantiate the specific OP class
    # We filter/pass relevant config args to the constructor
    optimizer = METHOD_MAP[method_key](
        num_tasks=len(train_loaders),
        lr=config.get("lr", 1e-3),
        lam=config.get("lam", 1e-3),
        alpha=config.get("alpha", 0.5),
        grads_per_task=config.get("grads_per_task", 80),
        max_directions=config.get("max_directions", 400),
        fisher_samples=config.get("fisher_samples", 1024),
        device_mode=config.get("device_mode", "hybrid"),
        fisher_clipping=config.get("fisher_clipping", False),
        normalize=config.get("normalize", False),
        
    )

    # 3. Call the generic engine
    return train(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        criterion=criterion,
        regulizer=regulizer_instance, # Pass the instance, not the bool
        optimizer=optimizer,
        lr=config.get("lr", 1e-3),
        first_task_lr=config.get("first_task_lr", 1e-3),
        first_task_optimizer_cls=first_task_optimizer_cls,
        epochs=config.get("epochs", 5),
        max_epochs=config.get("max_epochs"),
        task_classes=config.get("task_classes"),
        verbose=config.get("verbose", True),
        saved=config.get("saved", False),
        warmup=config.get("warmup", False),
    )

    
def train(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        criterion: Callable,
        regulizer: HyperRegulizer | None,
        optimizer: FOPNG | OGD | ONG,
        *,
        lr: float = 1e-3,
        first_task_lr: float = 1e-3,
        epochs: int = 5,
        max_epochs: int = None,
        first_task_optimizer_cls=torch.optim.AdamW,
        task_classes: Optional[list] = None,
        verbose: bool = True,
        saved: bool = False,
        warmup: bool = False,
    ):
    device = next(model.parameters()).device

    results = {"acc" : {}}
    global_epoch = 0
    loss_to_achieve = 0.1
    _max_epochs = max_epochs if max_epochs else epochs
    base_lr = lr
    save_label = "weights/first_run_weights"
    if not first_task_lr:
        first_task_lr = base_lr

    # save_path = "first_run_weights_0.14970794413238764_16.pt"
    # save_path = "first_run_weights_0.14454140998423098_16.pt"
    save_path = "weights/first_run_weights_0.08506506784658903_32.pt"
    for t, loader in enumerate(train_loaders):
        task_id = torch.tensor([t], dtype=torch.long, device=device)
        best_loss = inf
        loss_repeat = 0
        lr_patience_counter = 0
        best_parameters = None
        optimizer.lr = base_lr
        epoch = 0
        if t == 0:
            if not saved:
                if verbose: print(f"\n[{optimizer.__name__}] Task 1 – {first_task_optimizer_cls.__name__}")
                opt = first_task_optimizer_cls(model.parameters(), lr=first_task_lr, weight_decay=1e-4)  # This adds the L2 penalty)
                while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
                    total_loss = 0.0
                    model.train()

                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        model.zero_grad()
                        model.spawn(task_id)
                        output = model(x)
                        loss = criterion(output, y)
                        loss.backward()
                        opt.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(loader)
                    wandb.log({f"{optimizer.__name__}/train/loss": avg_loss, f"{optimizer.__name__}/global_epoch": global_epoch, "task": t+1})
                    global_epoch += 1
                    if verbose: print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f}")

                    # Intelligent Tracking & Scheduler
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        lr_patience_counter = 0
                        loss_repeat = 0
                        best_parameters = model.state_dict()
                    else:
                        lr_patience_counter += 1
                        loss_repeat += 1

                    if lr_patience_counter == 3:
                        for g in opt.param_groups:
                            g['lr'] = get_magnitude_decay_lr(g['lr'])
                        lr_patience_counter = 0
                        if verbose: print(f"    [Scheduler] Lowering LR to {opt.param_groups[0]['lr']}")


                    epoch += 1

                model.load_state_dict(best_parameters) # Load the best loss for the task and use it from now on.

                reason = f"best_loss: {best_loss}" if best_loss < loss_to_achieve else f"loss_repeat: {loss_repeat}" if loss_repeat < 10 else f"epoch: {epoch}"
                print(f"Task 1 Finished: {reason}")
                # save_path = f"{save_label}_{best_loss}_{model.hidden_dim}.pt"
                # torch.save(model.state_dict(), save_path)
            else:
                model.load_state_dict(torch.load(save_path, weights_only=True))
        else:
            if verbose: print(f"\n[{optimizer.__name__}] Task {t+1}")
            if warmup: # FROM EXPERIMENTS IT SEEMS RATHER DESTRUCTIVE.
                # FREEZING SHARED_PARAMS SO THE TASK EMBEDDING GETS AN EARLY START #
                for param in model._shared_params:
                    param.requires_grad = False
                ####################################################################
                active_params = filter(lambda p: p.requires_grad, model.parameters())
                opt = first_task_optimizer_cls(active_params, lr=0.1, weight_decay=1e-4)
                warmup_n = 15
                for i in range(warmup_n):
                    total_loss = 0.0
                    total_reg = 0.0

                    model.train()

                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        model.spawn(task_id)
                        output = model(x)
                        loss = criterion(output, y)

                        if regulizer:
                            w_penalty = regulizer.loss(model, task_id)
                            total_reg += w_penalty.item()
                            loss += w_penalty

                        loss.backward()
                        opt.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(loader)
                    avg_reg = total_reg / len(loader)

                    if verbose: print(f"embedding layer warm up {i+1}/{warmup_n} loss={avg_loss:.4f} avg_reg={avg_reg:.4f}")

                # UNFREEZING SHARED_PARAMS #
                for param in model._shared_params:
                    param.requires_grad = True
                model.zero_grad()  
                ############################

            while best_loss >= loss_to_achieve and loss_repeat < 10 and epoch < _max_epochs:
                F_new = optimizer.FisherEstimator.estimate(model, task_id, loader, criterion, device)
                optimizer.prepare_epoch(F_new)
                total_loss = 0.0
                total_reg = 0.0
                total_v_star_norm = 0.0
                total_weighted_rho = 0.0
                total_correction_norm = 0.0
                total_raw_rho = 0.0
                model.train()

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    model.zero_grad()
                    model.spawn(task_id)
                    output = model(x)
                    loss = criterion(output, y)

                    if regulizer:
                        w_penalty = regulizer.loss(model, task_id)
                        total_reg += w_penalty.item()
                        loss += w_penalty
                    total_loss += loss.item()
                    
                    loss.backward()

                    g_theta = get_grad_vector(model)
                    v_star_norm, weighted_rho, correction_norm, raw_rho = optimizer.step(model, task_id, g_theta.detach())
                    total_v_star_norm += v_star_norm
                    total_weighted_rho    += weighted_rho
                    total_correction_norm += correction_norm
                    total_raw_rho         += raw_rho

                n_batches = len(loader)
                avg_loss  = total_loss / n_batches
                avg_reg   = total_reg / n_batches


                avg_v_star_norm = total_v_star_norm / n_batches
                avg_weighted_rho   = total_weighted_rho   / n_batches
                avg_correction_norm= total_correction_norm/ n_batches
                avg_raw_rho        = total_raw_rho        / n_batches

                # 2. INTELLIGENT LOSS TRACKING
                # Require a meaningful improvement (e.g., 0.0001) to reset patience
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    lr_patience_counter = 0
                    loss_repeat = 0
                    best_parameters = model.state_dict()

                else:
                    lr_patience_counter += 1
                    loss_repeat += 1

                # 3. REDUCE LR ON PLATEAU
                # If loss hasn't improved for 5 epochs, cut speed in half
                if lr_patience_counter == 3:
                    optimizer.lr = get_magnitude_decay_lr(optimizer.lr)
                    lr_patience_counter = 0 # Reset so we don't decay again immediately
                    if verbose: print(f"    [Scheduler] Loss stalled. Lowering LR to {optimizer.lr}")
    

                wandb.log({
                    f"{optimizer.__name__}/train/loss":             avg_loss,
                    f"{optimizer.__name__}/train/reg_penalty":       avg_reg,

                    # weighted_rho: projection quality within Fisher-important subspace
                    # (the correct metric for FOPNG — want this LOW, close to 0)
                    f"{optimizer.__name__}/train/weighted_rho":     avg_weighted_rho,
                    # correction_norm: absolute gradient mass removed per step
                    # (want this non-trivially large relative to g_norm)
                    f"{optimizer.__name__}/train/correction_norm":  avg_correction_norm,
                    # raw_rho: ‖Pg‖/‖g‖ — kept for reference but ≈1 by design in FOPNG
                    f"{optimizer.__name__}/train/raw_rho":          avg_raw_rho,
                    f"{optimizer.__name__}/global_epoch":           global_epoch,
                    "task":                         t + 1,
                })
                global_epoch += 1

                if verbose: print(f"  epoch {epoch+1}/{_max_epochs} loss={avg_loss:.4f} v_star_norm={avg_v_star_norm:.4f} w_rho={avg_weighted_rho:.4f} corr={avg_correction_norm:.4e} raw_rho={avg_raw_rho:.4f}, lr={optimizer.lr}")
                if regulizer: print(f" reg_loss={avg_reg:.4f}")
                epoch += 1

            model.load_state_dict(best_parameters) #Load the best loss for the task and use it from now on.

        optimizer.after_task(model, task_id, loader, criterion)
        if regulizer:
            model.spawn(task_id)
            regulizer.old_weights[task_id.item()] = model.w.detach()

        # ── Evaluate on ALL tasks using TEST loaders ───────────────────
        results["acc"][t+1] = []
        eval_metrics = {"task_completed": t+1}
        
        # CHANGED: Iterate over every single task, seen or unseen!
        for i in range(len(test_loaders)): 
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc = task_classes[i] if task_classes is not None else None
            acc = evaluate_accuracy(model, test_loaders[i], eval_task_id, task_classes=tc)
            results["acc"][t+1].append(acc)
            eval_metrics[f"{optimizer.__name__}/eval/acc_task_{i+1}"] = acc
            if verbose: print(f"  Task {i+1} Acc: {acc*100:.1f}%")
            
        if t != 0:
            bwt = calc_bwt(results["acc"], task_id=t+1)
            eval_metrics[f"{optimizer.__name__}/eval/bwt"] = bwt
            if verbose: print(f"BWT at task {t+1}: {bwt:.4f}")
            results["bwt"] = bwt

        wandb.log(eval_metrics)

    tasks_completed = sorted(list(results["acc"].keys())) # [1, 2, 3]
    num_eval_tasks = len(test_loaders)
    if type(optimizer) is not OGD:
        matrix, keys = optimizer.compute_overlap_matrix()
        heat_map = plot_overlap(matrix, keys)
        wandb.log({f"{optimizer.__name__} FRECHET CORR MATRIX": wandb.Image(heat_map)})

    # 1. Log the overlapping FOPNG chart
    plt.figure(figsize=(10, 6))
    
    # Define a clean, distinct color palette
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_eval_tasks)]
    
    for i in range(num_eval_tasks):
        accs = [results["acc"][t][i] for t in tasks_completed]
        # Force solid line (linestyle='-') and cycle through colors
        plt.plot(tasks_completed, accs, marker='o', linestyle='-', linewidth=2.5, 
                 color=colors[i % len(colors)], label=f"{i+1}")

    plt.title(f"{optimizer.__name__}  Hypernetwork: All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    
    # Log the cleanly colored plot directly to W&B
    wandb.log({f"{optimizer.__name__}  Overlapping Accuracies (Colored)": wandb.Image(plt)})
    plt.close()

    return results