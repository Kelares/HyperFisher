"""
fopng.py — Fisher-Orthogonal Projected Natural Gradient (FOPNG / FOPNG+)

Overview
────────
FOPNG prevents catastrophic forgetting in continual learning by:
  1. Maintaining a gradient memory subspace G ⊆ ℝ^{D_θ} that spans the
     loss-gradient directions of all previously seen tasks.
  2. Projecting each new gradient *out of* G before applying it, ensuring
     parameter updates cannot increase old-task losses.
  3. Pre-conditioning the projected gradient by the inverse diagonal Fisher
     (a natural-gradient step) to respect parameter curvature.

All operations (G, F_old, A_inv) live in θ-space (the space of shared
parameters), keeping gradient projection, Fisher computation, and parameter
updates geometrically consistent.

Architecture generalisation
────────────────────────────
The algorithm is decoupled from the model via a ModelInterface ABC.  Two
concrete implementations are provided:

  HyperNetworkInterface — wraps a hypernetwork that uses spawn(task_id) to
    generate weights w.  Gradients require a two-pass Jacobian transpose
    (g_w → g_θ = Jᵀ g_w) to move from w-space into θ-space.

  DirectModelInterface — wraps any standard network (MLP, CNN, …).  θ-space
    gradients are obtained directly from a single loss.backward() call.

Usage
─────
# Hypernetwork (existing API — unchanged):
results = train_fopng(hyper_net, train_loaders, test_loaders, criterion)

# Plain MLP (new):
results = train_fopng(mlp, train_loaders, test_loaders, criterion,
                      use_hypernetwork=False)

# Full control:
interface  = HyperNetworkInterface(hyper_net)   # or DirectModelInterface(mlp)
fopng      = FOPNG(lr=1e-3, ...)               # or FOPNGPlus(...)
results    = train_continual(interface, fopng, ...)
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from math import inf
from typing import Callable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader

from utils import calc_bwt, evaluate_accuracy, plot_overlap


# =============================================================================
# MODEL INTERFACE
# =============================================================================

class ModelInterface(ABC):
    """
    Abstract interface that decouples the FOPNG algorithm from the underlying
    model architecture.

    All FOPNG core methods accept a ModelInterface instance, making them
    agnostic to whether the model is a hypernetwork or a direct network.
    """

    # ── Model state ───────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def shared_params(self) -> List[nn.Parameter]:
        """
        Parameters that FOPNG protects against forgetting.

        For a hypernetwork: the shared MLP parameters only (task_emb excluded,
        as its rows are task-specific and cannot cause cross-task forgetting).
        For a direct model: all model parameters.
        """
        ...

    @abstractmethod
    def all_params(self) -> List[nn.Parameter]:
        """All trainable parameters (used for task-1 plain-SGD optimiser)."""
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear gradient fields on all model parameters."""
        ...

    @abstractmethod
    def train_mode(self) -> None:
        """Switch the underlying model to training mode."""
        ...

    @abstractmethod
    def eval_mode(self) -> None:
        """Switch the underlying model to evaluation mode."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Return a full model checkpoint for saving/restoring."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore model weights from a checkpoint."""
        ...

    # ── Gradient computation ──────────────────────────────────────────────────

    @abstractmethod
    def compute_loss(
        self,
        x: Tensor,
        y: Tensor,
        criterion: Callable,
        task_id: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass → scalar loss tensor.
        Used in the task-1 plain-SGD loop where the optimiser owns backward().
        """
        ...

    @abstractmethod
    def grad_theta(
        self,
        x: Tensor,
        y: Tensor,
        criterion: Callable,
        task_id: Optional[Tensor] = None,
    ) -> Tuple[float, Tensor]:
        """
        Compute the θ-space gradient of the loss.

        Returns
        ───────
        loss_val : float   — scalar loss (for logging)
        g_theta  : [D_shared] 1-D gradient tensor in θ-space

        After this method returns, .grad fields on shared_params are still
        populated.  Callers that need them (e.g. update_task_specific_params)
        MUST use them before the next zero_grad() call.

        Implementation details differ by architecture:
          Hypernetwork → two-pass Jᵀ translation (see HyperNetworkInterface).
          Direct model → single backward pass (see DirectModelInterface).
        """
        ...

    @abstractmethod
    def update_task_specific_params(self, lr: float) -> None:
        """
        Raw SGD step on any parameters excluded from the FOPNG projection
        (e.g. the task embedding in a hypernetwork).

        Must be called AFTER grad_theta() while .grad fields are still live.
        For direct models this is a no-op.
        """
        ...

    # ── Evaluation ────────────────────────────────────────────────────────────

    @abstractmethod
    def evaluate(
        self,
        loader: DataLoader,
        task_id: Optional[Tensor] = None,
        task_classes: Optional[list] = None,
    ) -> float:
        """
        Accuracy on a given data loader.
        task_id is passed to hypernetworks to select generated weights;
        it is ignored by direct models.
        """
        ...

    # ── Optional: task-embedding warm-up ──────────────────────────────────────

    @property
    def has_task_embeddings(self) -> bool:
        """
        True if this model has task-specific parameters that need a warm-up
        phase before FOPNG projection begins.  Default: False.
        """
        return False

    def freeze_shared(self) -> None:
        """Freeze shared parameters (during task-embedding warm-up)."""
        for p in self.shared_params:
            p.requires_grad_(False)

    def unfreeze_shared(self) -> None:
        """Re-enable gradient computation for shared parameters."""
        for p in self.shared_params:
            p.requires_grad_(True)

    def warmup_task_embeddings(
        self,
        loader: DataLoader,
        criterion: Callable,
        n_epochs: int,
        lr: float,
        device: torch.device,
        task_id: Optional[Tensor] = None,
        optimizer_cls=torch.optim.SGD,
        verbose: bool = True,
    ) -> None:
        """
        Warm up task-specific parameters while shared parameters are frozen.
        Default implementation is a no-op; override in hypernetwork subclasses.
        """
        pass


# ─────────────────────────────────────────────────────────────────────────────

class HyperNetworkInterface(ModelInterface):
    """
    ModelInterface for hypernetworks following the spawn(task_id) protocol.

    Required model attributes / methods
    ─────────────────────────────────────
    model.spawn(task_id)    — populate model.w from task_emb[task_id]
    model.w                 — the generated flat weight tensor
    model(x)                — run the target network with current model.w
    model._shared_params    — list of shared MLP parameters (excludes task_emb)
    model.task_emb          — nn.Embedding (one row per task)
    model.task_embedding_lr — float: LR for the raw task_emb SGD step
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @property
    def shared_params(self) -> List[nn.Parameter]:
        return self.model._shared_params

    def all_params(self) -> List[nn.Parameter]:
        return list(self.model.parameters())

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)

    def compute_loss(
        self,
        x: Tensor,
        y: Tensor,
        criterion: Callable,
        task_id: Optional[Tensor] = None,
    ) -> Tensor:
        """Standard forward pass via spawn."""
        self.model.spawn(task_id)
        return criterion(self.model(x), y)

    def grad_theta(
        self,
        x: Tensor,
        y: Tensor,
        criterion: Callable,
        task_id: Optional[Tensor] = None,
    ) -> Tuple[float, Tensor]:
        """
        Two-pass Jacobian-transpose gradient computation in θ-space.

        WHY two passes are needed
        ──────────────────────────
        The hypernetwork maps θ → w → output.  A standard loss.backward()
        accumulates gradients in w-space.  FOPNG's projection matrix G and
        Fisher F both live in θ-space, so we must translate:
            g_θ = Jᵀ g_w  (where J = ∂w/∂θ)

        Pass 1 — g_w = ∂L/∂w:
            spawn(task_id) builds a computation graph rooted at w.
            autograd.grad(loss, w) yields g_w and immediately frees the graph,
            so no retain_graph=True is required.

        Pass 2 — g_θ = Jᵀ g_w:
            A second spawn(task_id) creates an independent graph rooted at θ.
            model.w.backward(g_w) backpropagates g_w through the generator,
            populating θ.grad = Jᵀ g_w for every shared parameter.
        """
        # ── Pass 1: g_w in generated-weight space ─────────────────────────────
        self.model.zero_grad()
        self.model.spawn(task_id)
        w      = self.model.w
        output = self.model(x)
        loss   = criterion(output, y)
        (g_w,) = torch.autograd.grad(loss, w)   # frees graph; no retain needed

        # ── Pass 2: g_θ = Jᵀ g_w ─────────────────────────────────────────────
        self.model.zero_grad()
        self.model.spawn(task_id)               # fresh graph rooted at θ
        self.model.w.backward(g_w.detach())     # θ.grad ← Jᵀ g_w

        g_theta = torch.cat([
            p.grad.view(-1) for p in self.shared_params if p.grad is not None
        ])
        return loss.item(), g_theta

    def update_task_specific_params(self, lr: float) -> None:
        """
        Plain SGD step on the current task's task_emb row.

        WHY plain SGD (not projected):
            task_emb[t] only affects task t's spawn() output; it cannot
            interfere with any other task.  Projecting it would be
            unnecessarily restrictive.

        WHY called separately from FOPNG.step():
            .grad fields from grad_theta() are still live here.  They will be
            cleared on the next grad_theta() call's opening zero_grad().
        """
        with torch.no_grad():
            te_grad = self.model.task_emb.weight.grad
            if te_grad is not None:
                self.model.task_emb.weight.data.add_(-lr * te_grad)

    def evaluate(
        self,
        loader: DataLoader,
        task_id: Optional[Tensor] = None,
        task_classes: Optional[list] = None,
    ) -> float:
        return evaluate_accuracy(self.model, loader, task_id, task_classes=task_classes)

    @property
    def has_task_embeddings(self) -> bool:
        return True

    def warmup_task_embeddings(
        self,
        loader: DataLoader,
        criterion: Callable,
        n_epochs: int,
        lr: float,
        device: torch.device,
        task_id: Optional[Tensor] = None,
        optimizer_cls=torch.optim.SGD,
        verbose: bool = True,
    ) -> None:
        """
        Warm-up phase for tasks 2+: train only task_emb while shared params
        are frozen.

        WHY warm up task embeddings first:
            At task t, task_emb[t] is randomly initialised.  Without a head
            start, the shared MLP must fully compensate for the new task,
            pushing its weights away from old-task solutions.  A short
            embedding-only phase gives task_emb[t] a meaningful starting point
            before FOPNG begins protecting the shared parameters.
        """
        self.freeze_shared()
        active_params = [p for p in self.model.parameters() if p.requires_grad]
        opt = optimizer_cls(active_params, lr=lr, weight_decay=1e-4)

        self.train_mode()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = self.compute_loss(x, y, criterion, task_id)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if verbose:
                print(f"  task_emb warmup  epoch {epoch+1}/{n_epochs}  "
                      f"loss={total_loss / len(loader):.4f}")

        self.unfreeze_shared()


# ─────────────────────────────────────────────────────────────────────────────

class DirectModelInterface(ModelInterface):
    """
    ModelInterface for standard feed-forward networks (MLP, CNN, etc.).

    All parameters are treated as shared — FOPNG protects all of them.
    task_id arguments are accepted for interface compatibility but ignored.
    No task-embedding warm-up phase is needed.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        # Snapshot once so the parameter list stays consistent across calls.
        self._shared: List[nn.Parameter] = list(model.parameters())

    @property
    def shared_params(self) -> List[nn.Parameter]:
        return self._shared

    def all_params(self) -> List[nn.Parameter]:
        return list(self.model.parameters())

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)

    def compute_loss(
        self,
        x: Tensor,
        y: Tensor,
        criterion: Callable,
        task_id: Optional[Tensor] = None,   # ignored
    ) -> Tensor:
        return criterion(self.model(x), y)

    def grad_theta(
        self,
        x: Tensor,
        y: Tensor,
        criterion: Callable,
        task_id: Optional[Tensor] = None,   # ignored
    ) -> Tuple[float, Tensor]:
        """
        Single-pass gradient: θ-space gradients are read directly from .grad
        fields after a standard backward() call.
        """
        self.model.zero_grad()
        loss = self.compute_loss(x, y, criterion)
        loss.backward()

        g_theta = torch.cat([
            p.grad.view(-1) for p in self.shared_params if p.grad is not None
        ])
        return loss.item(), g_theta

    def update_task_specific_params(self, lr: float) -> None:
        """No-op: all parameters are shared; nothing to update separately."""
        pass

    def evaluate(
        self,
        loader: DataLoader,
        task_id: Optional[Tensor] = None,   # ignored
        task_classes: Optional[list] = None,
    ) -> float:
        return evaluate_accuracy(self.model, loader, task_id, task_classes=task_classes)


# =============================================================================
# UTILITY
# =============================================================================

def get_magnitude_decay_lr(current_lr: float) -> float:
    """
    Decay the learning rate through clean magnitude steps:
        1e-2 → 5e-3 → 1e-3 → 5e-4 → 1e-4 → …

    Uses string parsing to avoid floating-point drift when repeatedly
    dividing by powers of 10.
    """
    sci_str        = f"{current_lr:.1e}"       # e.g. '1.0e-03'
    mantissa_str, exp_str = sci_str.split('e')
    mantissa       = float(mantissa_str)
    exp            = int(exp_str)

    if mantissa >= 4.9:
        # 5.0e-N → 1.0e-N (step mantissa down, keep exponent)
        return 1.0 * (10 ** exp)
    else:
        # 1.0e-N → 5.0e-(N+1)
        return 5.0 * (10 ** (exp - 1))


# =============================================================================
# FOPNG
# =============================================================================

class FOPNG:
    """
    Fisher-Orthogonal Projected Natural Gradient.

    Core idea
    ─────────
    At each gradient step for task t+1, FOPNG:
      1. Projects g_θ out of the gradient memory subspace G (which spans
         the loss-sensitive directions of all previous tasks).
      2. Scales the projected gradient by the inverse diagonal Fisher F_new⁻¹
         to obtain a natural-gradient step in the orthogonal complement.

    This guarantees that updates to shared θ cannot increase old-task losses
    (within the span of G), while still making progress on the current task.

    Notation
    ─────────
    D_θ      — number of shared parameters
    K        — number of columns currently in G  (≤ max_directions)
    G        — [D_θ, K]  gradient memory matrix (θ-space)
    F_old    — [D_θ]     running mean diagonal Fisher of seen tasks
    F_new    — [D_θ]     current-task diagonal Fisher (refreshed each epoch)
    A        — [K, K]    Gram matrix  A = Gᵀ diag(F_new⁻¹) G + λI
    A_inv    — [K, K]    pseudo-inverse of A (cached by prepare_epoch)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        lam: float = 1e-3,
        alpha: float = 0.5,
        grads_per_task: int = 80,
        max_directions: int = 400,
        fisher_samples: int = 1024,
        damping: float = 0.2,
        device_mode: Literal["cpu", "gpu", "hybrid"] = "hybrid",
    ) -> None:
        # ── Optimiser hyper-parameters ────────────────────────────────────────
        self.lr       = lr        # natural-gradient step size
        self.lam      = lam       # Tikhonov regularisation for A⁻¹
        self.alpha    = alpha     # reserved (future EWC blend)
        self.damping  = damping   # diagonal damping in Fisher inversion

        # ── Memory budget ─────────────────────────────────────────────────────
        self.grads_per_task  = grads_per_task   # gradient columns collected per task
        self.max_directions  = max_directions   # max columns in G before SVD compression
        self.fisher_samples  = fisher_samples   # data points for Fisher estimation

        # ── Fisher normalisation ──────────────────────────────────────────────
        # Clips outlier Fisher values before normalisation.
        # Needed because shared MLP params accumulate Jacobian contributions
        # from every chunk (~206×), while chunk_emb params only accumulate one,
        # creating a large dynamic range that collapses /max normalisation.
        self.quantile = 0.95

        # ── Device placement ──────────────────────────────────────────────────
        # "cpu"    — everything on CPU.
        # "gpu"    — everything on the model's GPU.
        # "hybrid" — large matrices (G, F_old, A_inv) on CPU; per-step
        #            momentum and parameter application on GPU.
        self.device_mode = device_mode

        # ── State (built incrementally as tasks are completed) ────────────────
        self.F_old:  Optional[Tensor] = None   # running mean diagonal Fisher
        self.G:      Optional[Tensor] = None   # gradient memory [D_θ, K]
        self._F_new: Optional[Tensor] = None   # current-task Fisher (prepare_epoch)
        self._A_inv: Optional[Tensor] = None   # cached projection matrix
        self._device: Optional[torch.device] = None  # inferred at after_task

        # ── Per-task Fisher archive for overlap analysis ───────────────────────
        self.fisher_after_task: dict = {}

        # ── Verbosity ─────────────────────────────────────────────────────────
        self.debug = True

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def prepare_epoch(self, F_new: Tensor) -> None:
        """
        Cache the current-epoch Fisher and pre-compute A⁻¹.
        Must be called once per epoch, before any step() calls.

        F_new : [D_θ] diagonal Fisher for the current task, obtained via
                compute_fisher_diag() evaluated on the current model state.
        """
        assert self.F_old is not None, (
            "prepare_epoch() requires F_old. "
            "Call after_task() at the end of task 1 first."
        )
        self._F_new = F_new
        self._A_inv = self._build_A_inv(self.G, self.F_old, F_new)

    def step(
        self,
        interface: ModelInterface,
        g_theta: Tensor,
    ) -> Tuple[float, float, float]:
        """
        Apply one FOPNG parameter update given a pre-computed θ-space gradient.

        The caller is responsible for obtaining g_theta via
        interface.grad_theta() and (for hypernetworks) calling
        interface.update_task_specific_params() while .grad is still live.

        Parameters
        ──────────
        interface : ModelInterface wrapping the model being trained.
        g_theta   : [D_shared] gradient in θ-space (output of grad_theta()).

        Returns
        ───────
        (weighted_rho, correction_norm, raw_rho) — projection quality metrics.
          weighted_rho    — ‖√F_old · Pg‖ / ‖√F_old · g‖
                            Retention fraction in the Fisher-important subspace.
                            Want LOW (close to 0) — confirms old-task components
                            are being removed.
          correction_norm — ‖correction‖
                            Absolute gradient mass removed per step.
          raw_rho         — ‖Pg‖ / ‖g‖
                            Overall retention ratio (≈1 by design; not the key
                            metric).
        """
        assert self._A_inv is not None, "Call prepare_epoch(F_new) before step()."

        model_device = g_theta.device
        target_dev   = self._get_target_device(model_device)  # where matrix math runs
        compute_dev  = self._get_compute_device(model_device)  # where update is applied

        # ── 1. Project and natural-gradient scale ─────────────────────────────
        # g_theta is moved to target_dev to match G / F_old / F_new / A_inv.
        v_star, weighted_rho, correction_norm, raw_rho = self._fopng_update(
            g=g_theta.to(target_dev),
            G=self.G,
            F_old=self.F_old,
            F_new=self._F_new,
            A_inv=self._A_inv,
        )

        # ── 2. Route the update to the parameter device ───────────────────────
        # cpu    → stays on CPU   (compute_dev == CPU)
        # gpu    → no-op          (target_dev == compute_dev == GPU)
        # hybrid → CPU → GPU      (target_dev = CPU, compute_dev = GPU)
        v_star = v_star.to(compute_dev)

        # ── 3. Apply to shared parameters ────────────────────────────────────
        pointer = 0
        with torch.no_grad():
            for p in interface.shared_params:
                n = p.numel()
                p.data.add_(v_star[pointer : pointer + n].view_as(p))
                pointer += n

        return weighted_rho, correction_norm, raw_rho

    def after_task(
        self,
        interface: ModelInterface,
        task_id: Tensor,
        loader: DataLoader,
        criterion: Callable,
    ) -> None:
        """
        End-of-task bookkeeping.  Must be called once after every task.

        Steps performed:
          1. Compute and archive the diagonal Fisher for this task.
          2. Log Fisher overlap metrics between this task and F_old.
          3. Update the running mean F_old (arithmetic mean over seen tasks).
          4. Collect loss-gradient directions and append them to G.
          5. SVD-compress G if it exceeds max_directions.
          6. Reset per-task state (momentum buffer).
          7. Log Fisher health and overlap metrics to W&B.
        """
        device     = next(iter(interface.shared_params)).device
        self._device = device
        target_dev = self._get_target_device(device)

        # ── 1. Compute and archive this task's Fisher ─────────────────────────
        F_new = self.compute_fisher_diag(interface, task_id, loader, criterion, device)
        self.fisher_after_task[task_id.item()] = F_new

        # ── 2. Fisher overlap statistics (computed BEFORE updating F_old) ─────
        if self.F_old is not None:
            cosine_sim   = self._cosine_similarity(self.F_old, F_new)
            pearson_corr = self._pearson_correlation(self.F_old, F_new)
            topk_iou     = self._calculate_topk_iou(self.F_old, F_new)
        else:
            # Task 1: no prior Fisher to compare against; baseline = 1.0
            cosine_sim = pearson_corr = topk_iou = 1.0

        # ── 3. Update running mean Fisher ─────────────────────────────────────
        # Arithmetic mean: each task contributes exactly 1/N weight.
        if self.F_old is None:
            self.F_old = F_new.detach().to(target_dev)
        else:
            n = task_id.item() + 1   # total tasks seen (task_id is 0-indexed)
            self.F_old = (
                ((n - 1) / n) * self.F_old
                + (1.0 / n) * F_new.detach().to(target_dev)
            )

        # ── 4. Collect gradient directions and append to G ────────────────────
        new_cols = self._collect_gradients(interface, task_id, loader, criterion)
        self.G   = new_cols if self.G is None else torch.cat([self.G, new_cols], dim=1)

        # ── 5. SVD-compress G if it exceeds the memory budget ─────────────────
        if self.G.shape[1] > self.max_directions:
            if self.debug:
                print(f"[FOPNG] G exceeded budget ({self.G.shape[1]} cols). "
                      f"Compressing to {self.max_directions} via SVD.")
            U, _, _ = torch.linalg.svd(self.G, full_matrices=False)
            self.G  = U[:, : self.max_directions]

        if self.debug:
            print(f"[FOPNG] G memory: {self.G.shape[1]} / {self.max_directions} cols")

        # ── 6. Reset per-task state ───────────────────────────────────────────
        print("  [FOPNG] Momentum buffer reset for next task.")

        # ── 7. Log Fisher health and overlap metrics ──────────────────────────
        f_nonzero = self.F_old[self.F_old > 0]
        logs = {
            "fopng/fisher/min":             self.F_old.min().item(),
            "fopng/fisher/max":             self.F_old.max().item(),
            "fopng/fisher/mean":            self.F_old.mean().item(),
            # mean_nonzero should be 0.1–0.4 with healthy percentile clipping
            # (was ~0.0002 with /max normalisation, indicating a near-delta distribution)
            "fopng/fisher/mean_nonzero":    f_nonzero.mean().item() if len(f_nonzero) > 0 else 0.0,
            "fopng/fisher/frac_nonzero":    len(f_nonzero) / self.F_old.numel(),
            "fopng/memory/G_cols":          self.G.shape[1],
            "fopng/fisher_overlap/cosine":  cosine_sim,
            "fopng/fisher_overlap/pearson": pearson_corr,
            "fopng/fisher_overlap/topk_iou": topk_iou,
            "task_completed":               task_id.item() + 1,
        }
        print(logs)
        wandb.log(logs)

        torch.cuda.empty_cache()
        gc.collect()

    def compute_fisher_diag(
        self,
        interface: ModelInterface,
        task_id: Optional[Tensor],
        loader: DataLoader,
        criterion: Callable,
        device: torch.device,
        max_samples: Optional[int] = None,
    ) -> Tensor:
        """
        Estimate the diagonal Fisher Information Matrix in θ-space via the
        squared-gradient approximation:  F_θ ≈ E[g_θ g_θᵀ]_diag.

        WHY θ-space (not w-space):
            G and parameter updates in step() all live in θ-space.  Computing
            F in w-space would make the natural-gradient metric inconsistent
            with the projection geometry.

        WHY percentile-clipped normalisation:
            In a hypernetwork's θ-space, shared MLP parameters accumulate Jᵀ
            contributions from every chunk (~206×), while chunk_emb params
            accumulate only one chunk's contribution.  This creates a large
            dynamic range.  Dividing by the max then collapses chunk_emb
            importance to ~1/206, making the projection near-useless.
            Clipping at the 95th percentile removes these outliers while
            preserving the relative importance ordering within each group.
            For a direct model the dynamic range is much smaller, but the
            same normalisation procedure is applied for consistency.

        Returns
        ───────
        fisher : [D_θ_shared] normalised diagonal Fisher, on target_dev.
        """
        max_samples = max_samples or self.fisher_samples
        target_dev  = self._get_target_device(device)

        interface.eval_mode()
        D_theta = sum(p.numel() for p in interface.shared_params)
        fisher  = torch.zeros(D_theta, device=target_dev)
        n_seen  = 0

        with torch.enable_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                # grad_theta() populates .grad; call zero_grad() after extracting.
                _, g_theta = interface.grad_theta(x, y, criterion, task_id)
                interface.zero_grad()

                # Skip numerically unstable batches
                if torch.isnan(g_theta).any() or torch.isinf(g_theta).any():
                    continue

                # Accumulate squared gradient (diagonal Fisher approximation)
                fisher.add_(g_theta.detach().to(target_dev).pow(2))

                n_seen += x.size(0)
                if n_seen >= max_samples:
                    break

        interface.zero_grad()
        interface.train_mode()

        # ── Percentile-clipped normalisation ─────────────────────────────────
        fisher_nonzero = fisher[fisher > 0]
        if len(fisher_nonzero) > 0:
            clip_val = torch.quantile(fisher_nonzero, self.quantile)
            fisher   = fisher.clamp(max=clip_val.item())

        if fisher.max() > 0:
            fisher = fisher / fisher.max()

        # Add a small floor so low-importance parameters are not entirely
        # ignored by the projection (avoids near-zero Fisher collapsing).
        fisher = fisher + 1e-4
        if fisher.max() > 0:
            fisher = fisher / fisher.max()

        return fisher

    # =========================================================================
    # OVERLAP ANALYSIS
    # =========================================================================

    def compute_overlap_matrix(self) -> Tuple[np.ndarray, list]:
        """
        Build a symmetric N×N matrix of pairwise Fréchet overlaps between the
        per-task Fishers archived during training.
        """
        keys   = list(self.fisher_after_task.keys())
        n      = len(keys)
        matrix = np.eye(n)   # diagonal = 1.0 (perfect self-overlap)

        for i in range(n):
            for j in range(i + 1, n):
                ov           = self._frechet_overlap(
                    self.fisher_after_task[keys[i]],
                    self.fisher_after_task[keys[j]],
                )
                matrix[i, j] = ov
                matrix[j, i] = ov

        return matrix, keys

    # =========================================================================
    # PRIVATE — PROJECTION CORE
    # =========================================================================

    def _fopng_update(
        self,
        g: Tensor,
        G: Tensor,
        F_old: Tensor,
        F_new: Tensor,
        A_inv: Tensor,
        eps: float = 1e-8,
    ) -> Tuple[Tensor, float, float, float]:
        """
        Core FOPNG projection and natural-gradient step.

        All input tensors must reside on the same device (G.device).

        Algorithm
        ─────────
        1. Aligned Riemannian projection — remove old-task components from g:

               correction = G · A⁻¹ · Gᵀ · F_new⁻¹ · g
               Pg         = g − correction

           The F_new⁻¹ weighting in the numerator aligns the projection metric
           with the A denominator, making the geometry consistent.

        2. Natural gradient precondition:
               v_raw = F_new⁻¹ · Pg

        3. Gradient norm clipping (max_norm = 0.5) for stability.

        Returns
        ───────
        v_star          : −lr · clipped update vector (ready to add to θ)
        weighted_rho    : Fisher-weighted retention ‖√F_old · Pg‖ / ‖√F_old · g‖
        correction_norm : ‖correction‖ — how much gradient mass was removed
        raw_rho         : ‖Pg‖ / ‖g‖  — raw retention ratio
        """
        F_new_inv = 1.0 / (F_new + self.damping)   # elementwise inverse with damping

        # ── 1. Riemannian projection ─────────────────────────────────────────
        GtFg       = G.t() @ (F_new_inv * g)        # [K] — project g into G's span
        coeff      = A_inv @ GtFg                    # [K] — solve A · coeff = Gᵀ F_new⁻¹ g
        correction = G @ coeff                       # [D_θ] — reconstruct old-task component
        Pg         = g - correction                  # [D_θ] — projected gradient

        # ── 2. Projection quality metrics ────────────────────────────────────
        F_sqrt          = F_old.clamp(min=0).sqrt()
        weighted_rho    = ((F_sqrt * Pg).norm() / ((F_sqrt * g).norm() + eps)).item()
        correction_norm = correction.norm().item()
        raw_rho         = (Pg.norm() / (g.norm() + eps)).item()

        # ── 3. Natural gradient step ─────────────────────────────────────────
        v_raw = F_new_inv * Pg

        # ── 4. Gradient norm clipping ─────────────────────────────────────────
        max_norm = 0.5
        v_norm   = torch.norm(v_raw)
        v_star   = v_raw * (max_norm / v_norm) if v_norm > max_norm else v_raw

        return -(self.lr * v_star), weighted_rho, correction_norm, raw_rho

    def _build_A_inv(
        self,
        G: Tensor,
        F_old: Tensor,
        F_new: Tensor,
    ) -> Tensor:
        """
        Pre-compute A⁻¹ where  A = Gᵀ · diag(F_new⁻¹) · G + λI.

        Cached by prepare_epoch() so it is only computed once per epoch,
        not once per batch.  Uses pinv for numerical stability near singularity.
        """
        scale    = 1.0 / (F_new + self.damping)    # F_new⁻¹ diagonal
        scaled_G = scale.unsqueeze(1) * G           # [D_θ, K]
        A        = G.t() @ scaled_G                 # [K, K]
        A        = A + self.lam * torch.eye(A.shape[0], device=A.device)
        return torch.linalg.pinv(A)

    def _collect_gradients(
        self,
        interface: ModelInterface,
        task_id: Optional[Tensor],
        loader: DataLoader,
        criterion: Callable,
    ) -> Tensor:
        """
        Collect loss-gradient directions in θ-space to build / extend G.

        WHY loss gradients (not random Jacobian samples):
            In high-dimensional θ-space, a random vector v and the loss gradient
            g are nearly orthogonal (curse of dimensionality), so random Jacobian
            samples Jᵀv yield ⟨G_col, F g⟩ ≈ 0 — making the projection useless.
            Loss gradients point exactly where updates would increase task loss,
            providing dense coverage of the critical forgetting subspace per
            stored direction.
            (cf. Farajtabar et al. 2019, OGD paper.)

        Returns
        ───────
        G_cols : [D_θ_shared, grads_per_task] — on target_dev.
        """
        target_dev = self._get_target_device(self._device)
        grads: List[Tensor] = []

        interface.eval_mode()

        with torch.enable_grad():
            # Loop over batches repeatedly until we have enough directions.
            while len(grads) < self.grads_per_task:
                for x, y in loader:
                    if len(grads) >= self.grads_per_task:
                        break

                    x, y = x.to(self._device), y.to(self._device)
                    _, g_theta = interface.grad_theta(x, y, criterion, task_id)
                    interface.zero_grad()

                    # Skip numerically unstable batches
                    if torch.isnan(g_theta).any() or torch.isinf(g_theta).any():
                        continue

                    grads.append(g_theta.detach().to(target_dev))

        interface.train_mode()
        return torch.stack(grads, dim=1)    # [D_θ_shared, grads_per_task]

    # =========================================================================
    # PRIVATE — FISHER OVERLAP METRICS
    # =========================================================================

    def _cosine_similarity(self, F_a: Tensor, F_b: Tensor) -> float:
        """
        Cosine similarity between two diagonal Fisher vectors.
        (Diagonal Fisher is a vector, so its metric is standard Euclidean.)
        """
        a = F_a.detach().cpu().view(-1)
        b = F_b.detach().cpu().view(-1)
        return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)).item()

    def _pearson_correlation(self, F_a: Tensor, F_b: Tensor) -> float:
        """Pearson correlation coefficient between two Fisher vectors."""
        a    = F_a.detach().cpu().view(-1).float()
        b    = F_b.detach().cpu().view(-1).float()
        a_c  = a - a.mean()
        b_c  = b - b.mean()
        denom = (torch.sum(a_c**2) * torch.sum(b_c**2)).sqrt() + 1e-8
        return (torch.sum(a_c * b_c) / denom).item()

    def _calculate_topk_iou(
        self,
        F_a: Tensor,
        F_b: Tensor,
        k_fraction: float = 0.10,
    ) -> float:
        """
        Intersection-over-Union of the top-K important parameter indices
        between two tasks, where K = k_fraction × D_θ.

        A high IoU means both tasks rely on the same parameters — potential
        for interference.  A low IoU means they use disjoint sub-networks.
        """
        a = F_a.detach().cpu().view(-1)
        b = F_b.detach().cpu().view(-1)
        assert a.shape == b.shape

        k = max(1, int(a.numel() * k_fraction))
        _, idx_a     = torch.topk(a, k)
        _, idx_b     = torch.topk(b, k)

        combined   = torch.cat([idx_a, idx_b])
        _, counts  = combined.unique(return_counts=True)
        inter      = (counts > 1).sum().item()
        union      = 2 * k - inter

        return inter / union

    def _frechet_overlap(self, F_1: Tensor, F_2: Tensor) -> float:
        """
        Fréchet-distance-based overlap between two diagonal Fisher distributions.
        Both are normalised to unit trace before computing the distance.

        Returns a score in [0, 1] where 1 = identical distributions.
        """
        F_1n = F_1 / (F_1.sum() + 1e-8)
        F_2n = F_2 / (F_2.sum() + 1e-8)
        d_sq = 0.5 * torch.sum((F_1n.sqrt() - F_2n.sqrt()) ** 2)
        return (1.0 - d_sq.item())

    # =========================================================================
    # PRIVATE — DEVICE ROUTING
    # =========================================================================

    def _get_target_device(self, model_device: torch.device) -> torch.device:
        """
        Device for large persistent matrices (G, F_old, A_inv) and heavy
        matrix-multiply operations.
        """
        if self.device_mode == "gpu":
            return model_device
        return torch.device("cpu")   # cpu and hybrid → CPU

    def _get_compute_device(self, model_device: torch.device) -> torch.device:
        """
        Device for per-step lightweight work (parameter update application).
        """
        if self.device_mode == "cpu":
            return torch.device("cpu")
        return model_device   # gpu and hybrid → GPU (model_device)


# =============================================================================
# FOPNG+
# =============================================================================

class FOPNGPlus(FOPNG):
    """
    FOPNG+ extends FOPNG by rebuilding G from scratch under the current θ at
    every task boundary, fixing the gradient-memory staleness problem.

    The staleness problem in FOPNG
    ───────────────────────────────
    Gradient directions collected at task t use J(θ_t).  By task t+2 the
    shared weights have moved to θ_{t+2}, so J(θ_{t+2}) ≠ J(θ_t).  Stored
    columns of G no longer faithfully describe which directions harm old tasks,
    and projection quality degrades over time.

    FOPNG+ fix (Bennani et al. 2020, OGD+)
    ────────────────────────────────────────
    After completing task t, re-collect gradient directions for ALL previously
    seen tasks evaluated under the CURRENT θ.  Cost: O(seen_tasks ×
    grads_per_task) gradient evaluations per task boundary — cheap relative to
    the full training loop.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Stores (task_id_tensor, DataLoader) for every completed task so we
        # can re-collect under a fresh θ at each boundary.
        self._task_history: List[tuple] = []

    def after_task(
        self,
        interface: ModelInterface,
        task_id: Tensor,
        loader: DataLoader,
        criterion: Callable,
    ) -> None:
        """
        End-of-task bookkeeping for FOPNG+.  Identical to FOPNG.after_task
        except step 4: G is rebuilt from scratch (not incrementally extended).
        """
        device     = next(iter(interface.shared_params)).device
        self._device = device
        target_dev = self._get_target_device(device)

        # ── 1. Compute and archive this task's Fisher ─────────────────────────
        F_new = self.compute_fisher_diag(interface, task_id, loader, criterion, device)
        self.fisher_after_task[task_id.item()] = F_new

        # ── 2. Fisher overlap statistics ──────────────────────────────────────
        if self.F_old is not None:
            cosine_sim   = self._cosine_similarity(self.F_old, F_new)
            pearson_corr = self._pearson_correlation(self.F_old, F_new)
            topk_iou     = self._calculate_topk_iou(self.F_old, F_new)
        else:
            cosine_sim = pearson_corr = topk_iou = 1.0

        # ── 3. Update running mean Fisher ─────────────────────────────────────
        if self.F_old is None:
            self.F_old = F_new.detach().to(target_dev)
        else:
            n = task_id.item() + 1
            self.F_old = (
                ((n - 1) / n) * self.F_old
                + (1.0 / n) * F_new.detach().to(target_dev)
            )

        # ── 4. FOPNG+: store this task and rebuild G under current θ ──────────
        self._task_history.append((task_id.clone(), loader))

        print(f"  [FOPNG+] Refreshing G for {len(self._task_history)} task(s) "
              f"under current θ …")
        all_cols = [
            self._collect_gradients(interface, prev_tid, prev_loader, criterion)
            for prev_tid, prev_loader in self._task_history
        ]
        G_fresh = torch.cat(all_cols, dim=1)    # [D_θ_shared, total_cols]

        # ── 5. SVD-compress G if it exceeds the memory budget ─────────────────
        if G_fresh.shape[1] > self.max_directions:
            if self.debug:
                print(f"[FOPNG+] Compressing G ({G_fresh.shape[1]} cols) "
                      f"to {self.max_directions} via SVD.")
            U, _, _ = torch.linalg.svd(G_fresh, full_matrices=False)
            G_fresh = U[:, : self.max_directions]

        self.G = G_fresh

        # ── 6. Reset per-task state ───────────────────────────────────────────
        print("  [FOPNG+] Momentum buffer reset for next task.")

        # ── 7. Log ────────────────────────────────────────────────────────────
        f_nonzero = self.F_old[self.F_old > 0]
        logs = {
            "fopng_plus/fisher/min":              self.F_old.min().item(),
            "fopng_plus/fisher/max":              self.F_old.max().item(),
            "fopng_plus/fisher/mean":             self.F_old.mean().item(),
            "fopng_plus/fisher/mean_nonzero":     f_nonzero.mean().item() if len(f_nonzero) > 0 else 0.0,
            "fopng_plus/fisher/frac_nonzero":     len(f_nonzero) / self.F_old.numel(),
            "fopng_plus/memory/G_cols":           self.G.shape[1],
            "fopng_plus/memory/tasks_in_history": len(self._task_history),
            "fopng_plus/fisher_overlap/cosine":   cosine_sim,
            "fopng_plus/fisher_overlap/pearson":  pearson_corr,
            "fopng_plus/fisher_overlap/topk_iou": topk_iou,
            "task_completed":                     task_id.item() + 1,
        }
        print(logs)
        wandb.log(logs)

        torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_continual(
    interface: ModelInterface,
    fopng: FOPNG,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    epochs: int = 5,
    max_epochs: Optional[int] = None,
    optimizer_cls=torch.optim.SGD,
    task_classes: Optional[list] = None,
    warmup_epochs: int = 5,
    warmup_lr: float = 0.1,
    loss_target_task1: float = 0.20,
    loss_target_task_n: float = 0.15,
    patience: int = 10,
    log_prefix: str = "fopng",
    verbose: bool = True,
) -> dict:
    """
    Unified continual-learning training loop for FOPNG and FOPNG+.

    Works with any (ModelInterface, FOPNG) combination:
        • (HyperNetworkInterface, FOPNG)      — original FOPNG for hypernetworks
        • (HyperNetworkInterface, FOPNGPlus)  — FOPNG+ for hypernetworks
        • (DirectModelInterface,  FOPNG)      — FOPNG for plain MLP / CNN
        • (DirectModelInterface,  FOPNGPlus)  — FOPNG+ for plain MLP / CNN

    Training protocol
    ─────────────────
    Task 1:
        Plain SGD with all parameters until loss < loss_target_task1, patience
        exhausted, or the epoch budget is reached.
        → fopng.after_task() builds G and F_old.

    Tasks 2+:
        [If interface.has_task_embeddings]
          Warm-up: freeze shared params, train task_emb for warmup_epochs.
        FOPNG loop: compute F_new, prepare_epoch(), then for each batch:
          1. grad_theta()                       — θ-space gradient
          2. update_task_specific_params()      — task_emb SGD step (no-op for MLP)
          3. fopng.step()                       — project, scale, apply
        → fopng.after_task() updates G and F_old.

    Parameters
    ──────────
    interface          : ModelInterface wrapping the model.
    fopng              : FOPNG or FOPNGPlus instance.
    train_loaders      : one DataLoader per task (training data).
    test_loaders       : one DataLoader per task (evaluation data).
    criterion          : loss function.
    lr                 : base learning rate (reset after each task).
    epochs             : default epoch budget per task.
    max_epochs         : override epoch budget (uses `epochs` if None).
    optimizer_cls      : optimiser class for task-1 and warm-up phases.
    task_classes       : optional per-task class label remapping.
    warmup_epochs      : task-embedding warm-up epochs for tasks 2+.
    warmup_lr          : learning rate during the warm-up phase.
    loss_target_task1  : stop task-1 when avg loss drops below this.
    loss_target_task_n : stop task-N FOPNG loop when avg loss drops below this.
    patience           : consecutive non-improving epochs before early stopping.
    log_prefix         : W&B key prefix ("fopng" or "fopng_plus").
    verbose            : print progress to stdout.

    Returns
    ───────
    results : dict[task_index → List[acc_per_task]] — accuracy matrix.
    """
    device      = next(iter(interface.shared_params)).device
    _max_epochs = max_epochs if max_epochs else epochs
    results     = {}
    global_epoch = 0

    for t, loader in enumerate(train_loaders):
        task_id           = torch.tensor([t], dtype=torch.long, device=device)
        best_loss         = inf
        loss_repeat       = 0
        lr_patience_count = 0
        best_state        = None
        epoch             = 0

        # ─────────────────────────────────────────────────────────────────────
        # TASK 1 — plain SGD, no projection
        # ─────────────────────────────────────────────────────────────────────
        if t == 0:
            if verbose:
                print(f"[{log_prefix.upper()}] Task 1  —  {optimizer_cls.__name__}")

            opt = optimizer_cls(interface.all_params(), lr=lr, weight_decay=1e-4)

            while (best_loss >= loss_target_task1
                   and loss_repeat < patience
                   and epoch < _max_epochs):

                interface.train_mode()
                total_loss = 0.0

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = interface.compute_loss(x, y, criterion, task_id)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                wandb.log({
                    f"{log_prefix}/train/loss": avg_loss,
                    f"{log_prefix}/global_epoch": global_epoch,
                    "task": t + 1,
                })
                global_epoch += 1

                if verbose:
                    print(f"  epoch {epoch+1}/{_max_epochs}  loss={avg_loss:.4f}")

                # LR plateau decay
                if lr_patience_count >= 5:
                    for g in opt.param_groups:
                        g['lr'] = get_magnitude_decay_lr(g['lr'])
                    lr_patience_count = 0
                    if verbose:
                        print(f"    [LR scheduler] → {opt.param_groups[0]['lr']:.2e}")

                if avg_loss < best_loss:
                    best_loss         = avg_loss
                    loss_repeat       = 0
                    lr_patience_count = 0
                    best_state        = interface.state_dict()
                else:
                    loss_repeat      += 1
                    lr_patience_count += 1

                epoch += 1

            reason = (f"loss={best_loss:.4f}" if best_loss < loss_target_task1
                      else f"patience={loss_repeat}")
            print(f"Task 1 finished: {reason}")

        # ─────────────────────────────────────────────────────────────────────
        # TASKS 2+ — optional warm-up, then FOPNG-projected loop
        # ─────────────────────────────────────────────────────────────────────
        else:
            if verbose:
                print(f"\n[{log_prefix.upper()}] Task {t+1}")

            # ── Warm-up: train task-specific parameters while shared are frozen ─
            # HyperNetworkInterface handles this; DirectModelInterface is a no-op.
            if interface.has_task_embeddings:
                interface.warmup_task_embeddings(
                    loader=loader,
                    criterion=criterion,
                    n_epochs=warmup_epochs,
                    lr=warmup_lr,
                    device=device,
                    task_id=task_id,
                    optimizer_cls=optimizer_cls,
                    verbose=verbose,
                )

            # LR for the plain-SGD task_emb update inside each FOPNG step
            task_emb_lr = (
                interface.model.task_embedding_lr
                if interface.has_task_embeddings else 0.0
            )

            # ── FOPNG-projected natural gradient loop ──────────────────────────
            while (best_loss >= loss_target_task_n
                   and loss_repeat < patience
                   and epoch < _max_epochs):

                # Recompute F_new and A_inv once per epoch (θ has moved since last
                # epoch, so the Fisher estimate needs to be refreshed).
                F_new = fopng.compute_fisher_diag(
                    interface, task_id, loader, criterion, device
                )
                fopng.prepare_epoch(F_new)

                interface.train_mode()
                total_loss = total_w_rho = total_corr = total_raw_rho = 0.0

                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    # 1. Obtain θ-space gradient (two-pass for hypernetwork,
                    #    single-pass for direct model).
                    loss_val, g_theta = interface.grad_theta(x, y, criterion, task_id)
                    total_loss += loss_val

                    # 2. Update task-specific parameters (task_emb) while .grad
                    #    is still populated from grad_theta().
                    interface.update_task_specific_params(task_emb_lr)

                    # 3. Project g_theta out of old-task subspace, apply natural
                    #    gradient step to shared parameters.
                    w_rho, corr, raw_rho = fopng.step(interface, g_theta)
                    total_w_rho   += w_rho
                    total_corr    += corr
                    total_raw_rho += raw_rho

                n_batches   = len(loader)
                avg_loss    = total_loss    / n_batches
                avg_w_rho   = total_w_rho   / n_batches
                avg_corr    = total_corr    / n_batches
                avg_raw_rho = total_raw_rho / n_batches

                # LR plateau decay for FOPNG's internal lr
                if lr_patience_count >= 5:
                    fopng.lr = get_magnitude_decay_lr(fopng.lr)
                    lr_patience_count = 0
                    if verbose:
                        print(f"    [LR scheduler] FOPNG lr → {fopng.lr:.2e}")

                if avg_loss < best_loss:
                    best_loss         = avg_loss
                    loss_repeat       = 0
                    lr_patience_count = 0
                    best_state        = interface.state_dict()
                else:
                    loss_repeat      += 1
                    lr_patience_count += 1

                wandb.log({
                    f"{log_prefix}/train/loss":            avg_loss,
                    f"{log_prefix}/train/weighted_rho":    avg_w_rho,
                    f"{log_prefix}/train/correction_norm": avg_corr,
                    f"{log_prefix}/train/raw_rho":         avg_raw_rho,
                    f"{log_prefix}/global_epoch":          global_epoch,
                    "task": t + 1,
                })
                global_epoch += 1

                if verbose:
                    print(f"  epoch {epoch+1}/{_max_epochs}  "
                          f"loss={avg_loss:.4f}  "
                          f"w_rho={avg_w_rho:.4f}  "
                          f"corr={avg_corr:.4e}  "
                          f"raw_rho={avg_raw_rho:.4f}  "
                          f"lr={fopng.lr:.2e}")
                epoch += 1

            # Restore best checkpoint and reset LR for the next task.
            if best_state is not None:
                interface.load_state_dict(best_state)
            fopng.lr = lr

        # ── End-of-task bookkeeping ────────────────────────────────────────────
        fopng.after_task(interface, task_id, loader, criterion)

        # ─────────────────────────────────────────────────────────────────────
        # EVALUATION — all tasks after completing task t
        # ─────────────────────────────────────────────────────────────────────
        results[t + 1] = []
        eval_metrics   = {"task_completed": t + 1}

        for i, test_loader in enumerate(test_loaders):
            eval_task_id = torch.tensor([i], dtype=torch.long, device=device)
            tc           = task_classes[i] if task_classes else None
            acc          = interface.evaluate(test_loader, eval_task_id, tc)
            results[t + 1].append(acc)
            eval_metrics[f"{log_prefix}/eval/acc_task_{i+1}"] = acc
            if verbose:
                print(f"  Task {i+1} Acc: {acc * 100:.1f}%")

        if t != 0:
            bwt = calc_bwt(results, task_id=t + 1)
            eval_metrics[f"{log_prefix}/eval/bwt"] = bwt
            if verbose:
                print(f"  BWT at task {t+1}: {bwt:.4f}")

        wandb.log(eval_metrics)

    # ── Final plots ────────────────────────────────────────────────────────────
    tasks_completed = sorted(results.keys())
    num_eval_tasks  = len(test_loaders)

    matrix, keys = fopng.compute_overlap_matrix()
    wandb.log({f"{log_prefix.upper()} FRECHET CORR MATRIX":
               wandb.Image(plot_overlap(matrix, keys))})

    plt.figure(figsize=(10, 6))
    cmap   = plt.get_cmap('gist_rainbow')
    colors = [cmap(v) for v in np.linspace(0, 1, num_eval_tasks)]

    for i in range(num_eval_tasks):
        accs = [results[t][i] for t in tasks_completed]
        plt.plot(tasks_completed, accs,
                 marker='o', linestyle='-', linewidth=2.5,
                 color=colors[i], label=str(i + 1))

    plt.title(f"{log_prefix.upper()} — All Tasks", fontsize=14, fontweight='bold')
    plt.xlabel("Tasks Completed", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(tasks_completed)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Evaluated Task", loc="lower left")
    wandb.log({f"{log_prefix.upper()} Overlapping Accuracies": wandb.Image(plt)})
    plt.close()

    return results


# =============================================================================
# CONVENIENCE WRAPPERS
# Drop-in replacements for the original train_fopng / train_fopng_plus API.
# =============================================================================

def train_fopng(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    lam: float = 1e-3,
    damping: float = 0.2,
    alpha: float = 0.5,
    grads_per_task: int = 80,
    max_directions: int = 400,
    fisher_samples: int = 1024,
    epochs: int = 5,
    max_epochs: Optional[int] = None,
    first_task_optimizer_cls=torch.optim.SGD,
    task_classes: Optional[list] = None,
    warmup_epochs: int = 15,
    verbose: bool = True,
    device_mode: Literal["cpu", "gpu", "hybrid"] = "hybrid",
    use_hypernetwork: bool = True,
) -> dict:
    """
    FOPNG training loop (no G refresh between tasks).

    Parameters
    ──────────
    use_hypernetwork : True  → wraps model in HyperNetworkInterface
                       False → wraps model in DirectModelInterface (plain MLP)

    All other parameters mirror the original API.
    """
    interface = (HyperNetworkInterface(model) if use_hypernetwork
                 else DirectModelInterface(model))
    fopng = FOPNG(
        lr=lr, lam=lam, damping=damping, alpha=alpha,
        grads_per_task=grads_per_task,
        max_directions=max_directions,
        fisher_samples=fisher_samples,
        device_mode=device_mode,
    )
    return train_continual(
        interface=interface, fopng=fopng,
        train_loaders=train_loaders, test_loaders=test_loaders,
        criterion=criterion, lr=lr, epochs=epochs, max_epochs=max_epochs,
        optimizer_cls=first_task_optimizer_cls, task_classes=task_classes,
        warmup_epochs=warmup_epochs, log_prefix="fopng", verbose=verbose,
    )


def train_fopng_plus(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    criterion: Callable,
    *,
    lr: float = 1e-3,
    lam: float = 1e-3,
    damping: float = 0.02,
    alpha: float = 0.5,
    grads_per_task: int = 80,
    max_directions: int = 400,
    fisher_samples: int = 1024,
    epochs: int = 5,
    max_epochs: Optional[int] = None,
    first_task_optimizer_cls=torch.optim.SGD,
    task_classes: Optional[list] = None,
    warmup_epochs: int = 5,
    verbose: bool = True,
    device_mode: Literal["cpu", "gpu", "hybrid"] = "hybrid",
    use_hypernetwork: bool = True,
) -> dict:
    """
    FOPNG+ training loop (G rebuilt from scratch under current θ each task).

    Parameters
    ──────────
    use_hypernetwork : True  → wraps model in HyperNetworkInterface
                       False → wraps model in DirectModelInterface (plain MLP)

    All other parameters mirror the original API.
    """
    interface = (HyperNetworkInterface(model) if use_hypernetwork
                 else DirectModelInterface(model))
    fopng_plus = FOPNGPlus(
        lr=lr, lam=lam, alpha=alpha, damping=damping,
        grads_per_task=grads_per_task,
        max_directions=max_directions,
        fisher_samples=fisher_samples,
        device_mode=device_mode,
    )
    return train_continual(
        interface=interface, fopng=fopng_plus,
        train_loaders=train_loaders, test_loaders=test_loaders,
        criterion=criterion, lr=lr, epochs=epochs, max_epochs=max_epochs,
        optimizer_cls=first_task_optimizer_cls, task_classes=task_classes,
        warmup_epochs=warmup_epochs, log_prefix="fopng_plus", verbose=verbose,
    )