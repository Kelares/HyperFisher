"""
FOPNG: Fisher-Orthogonal Projected Natural Gradient Descent
============================================================
Garg, Kolhe, Peng, Gopalam — UC Berkeley (ICML 2026)

All equations are taken directly from the paper.

─────────────────────────────────────────────────────────────
CORE UPDATE  (Theorem 1, eq. 5)
─────────────────────────────────────────────────────────────

	v* = -η · F_new⁻¹ P g / sqrt( gᵀ Pᵀ F_new⁻¹ P g )

where

	P  = I  −  F_old G  A⁻¹  Gᵀ F_old          (projection matrix)
	A  = Gᵀ F_old F_new⁻¹ F_old G  +  λ I       (m×m weighted Gram matrix)

	G        — [D × m]  columns are gradients stored from previous tasks
	F_old    — diagonal empirical Fisher, exponential average over old tasks
	F_new    — diagonal empirical Fisher on current task data
	λ        — ridge for numerical stability  (NOT an EWC penalty)
	η        — learning rate (trust-region radius in Fisher metric)

─────────────────────────────────────────────────────────────
GRADIENT STORAGE  (Section 4.3)
─────────────────────────────────────────────────────────────
After finishing task t, collect grads_per_task gradients and
append them as new columns of G.

─────────────────────────────────────────────────────────────
FISHER ESTIMATION  (Section 4.4, eq. 9)
─────────────────────────────────────────────────────────────
Diagonal empirical Fisher on a random batch B:

	F̂_diag(θ) = (1/|B|) Σ_{(x,y)∈B}  [∇_θ log p_θ(y|x)]²

F_old is maintained as a moving average after each task:

	F_old ← (1 − α) F_old + α F_new

─────────────────────────────────────────────────────────────
LIFECYCLE
─────────────────────────────────────────────────────────────

	# task 1: plain training
	train_task1(model, loader, optimizer)
	fopng.after_task(model, loader, criterion)   # seeds F_old and G

	# task t >= 2
	for epoch in range(E):
		F_new = fopng.compute_fisher(model, loader, criterion)
		fopng.prepare_epoch(F_new)               # builds A_inv once
		for x, y in loader:
			loss = criterion(model(x), y)
			loss.backward()
			fopng.step(model)                    # project + nat-grad update
	fopng.after_task(model, loader, criterion)
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Low-level utilities
# ─────────────────────────────────────────────────────────────────────────────

def _flat_grad(model: nn.Module) -> Tensor:
	"""Flatten all parameter .grad fields into a single vector [D]."""
	parts = []
	for p in model.parameters():
		if p.grad is not None:
			parts.append(p.grad.detach().view(-1))
		else:
			parts.append(p.data.new_zeros(p.numel()))
	return torch.cat(parts)


def _apply_flat_update(model: nn.Module, update: Tensor) -> None:
	"""Add a flat update vector to model parameters in-place: θ ← θ + update."""
	offset = 0
	for p in model.parameters():
		n = p.numel()
		p.data.add_(update[offset: offset + n].view_as(p))
		offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Diagonal Fisher estimation  (Section 4.4, eq. 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fisher_diag(
	model: nn.Module,
	loader: DataLoader,
	criterion: Callable,
	device: torch.device,
	max_samples: int = 1024,
) -> Tensor:
	"""
	F̂_diag(θ) = (1/N) Σ_{(x,y)}  [∇_θ log p_θ(y|x)]²

	For cross-entropy,  ∇_θ log p = −∇_θ L,  so we use the loss gradient.
	Accumulates squared gradients over up to max_samples data points.
	Returns a vector of shape [D].
	"""
	model.eval()
	D = sum(p.numel() for p in model.parameters())
	fisher = torch.zeros(D, device=device)
	n_seen = 0

	with torch.enable_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			model.zero_grad()
			loss = criterion(model(x), y)
			loss.backward()
			g = _flat_grad(model)          # [D]
			fisher.add_(g.pow(2))
			n_seen += x.size(0)
			if n_seen >= max_samples:
				break

	model.zero_grad()
	model.train()
	return fisher / max(n_seen, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core math  (Theorem 1)
# ─────────────────────────────────────────────────────────────────────────────

def _build_A_inv(
	G: Tensor,      # [D, m]
	F_old: Tensor,  # [D]
	F_new: Tensor,  # [D]
	lam: float,
) -> Tensor:
	"""
	A  = Gᵀ F_old F_new⁻¹ F_old G  +  λ I      [m × m]

	With diagonal Fishers, row i of (F_old F_new⁻¹ F_old G) is:

		F_old[i]² / F_new[i]  ×  G[i, :]

	Precomputed once per epoch.  Returns A⁻¹  [m × m].
	"""
	F_new_inv = 1.0 / (F_new + lam)                 # [D]
	scale     = (F_old ** 2) * F_new_inv             # [D]   F_old² / F_new
	scaled_G  = scale.unsqueeze(1) * G               # [D, m]
	A         = G.t() @ scaled_G                     # [m, m]
	A         = A + lam * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
	return torch.linalg.pinv(A)                      # [m, m]


def _fopng_update(
	g: Tensor,      # [D]   current task gradient
	G: Tensor,      # [D, m] gradient memory
	F_old: Tensor,  # [D]
	F_new: Tensor,  # [D]
	A_inv: Tensor,  # [m, m]
	lr: float,
	lam: float,
	eps: float = 1e-8,
) -> Tensor:
	"""
	Compute v*  (Theorem 1, eq. 5).

	Step 1 — project g:
		Pg = g  −  F_old G A⁻¹ Gᵀ F_old g

	Step 2 — unit natural gradient *descent* step in F_new metric:
		v* = -η · F_new⁻¹ Pg / sqrt( Pgᵀ F_new⁻¹ Pg )

	The minus sign is required because g points uphill (gradient of the loss),
	so F_new⁻¹ Pg also points uphill.  We negate to descend.
	Applied as  θ ← θ + v*  (i.e. θ ← θ - η · normalised_natural_grad).
	"""
	# ── projection ────────────────────────────────────────────────────
	F_old_g  = F_old * g                          # [D]    F_old · g
	GtFg     = G.t() @ F_old_g                    # [m]    Gᵀ F_old g
	coeff    = A_inv @ GtFg                        # [m]    A⁻¹ Gᵀ F_old g
	Pg       = g - F_old * (G @ coeff)             # [D]    Pg = g − F_old G A⁻¹ Gᵀ F_old g

	# ── unit natural gradient ──────────────────────────────────────────
	F_new_inv    = 1.0 / (F_new + lam)            # [D]
	F_new_inv_Pg = F_new_inv * Pg                  # [D]    F_new⁻¹ Pg
	fisher_norm  = torch.sqrt((Pg * F_new_inv_Pg).sum() + eps)   # scalar

	return -lr * F_new_inv_Pg / fisher_norm       # [D]  negative = descent


# ─────────────────────────────────────────────────────────────────────────────
# FOPNG  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

class FOPNG:
	"""
	Fisher-Orthogonal Projected Natural Gradient continual-learning method.

	Parameters
	----------
	lr             : learning rate η (trust-region radius in Fisher metric)
	lam             : ridge λ for numerical stability  (NOT an EWC penalty)
					  Applied twice: once to invert F_new, once to invert A.
					  Paper sweeps lam ∈ {1e-4, 5e-4, 1e-3, 1e-2}.
	alpha           : F_old EMA weight.  F_old ← (1−α) F_old + α F_new.
					  Paper fixes α = 0.5 (insensitive, Section A.2).
	grads_per_task  : k, gradient vectors stored after each task.
					  Paper uses k = 80.
	max_directions  : hard cap on total columns of G (oldest dropped).
					  Paper uses 400 (800 for Split-CIFAR100).
	fisher_samples  : max data points for Fisher estimation.
	"""

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
		self.lam            = lam
		self.alpha          = alpha
		self.grads_per_task = grads_per_task
		self.max_directions = max_directions
		self.fisher_samples = fisher_samples

		# Persistent state
		self.F_old: Optional[Tensor] = None   # [D]
		self.G:     Optional[Tensor] = None   # [D, m]

		# Epoch-level cache set by prepare_epoch()
		self._F_new: Optional[Tensor] = None  # [D]
		self._A_inv: Optional[Tensor] = None  # [m, m]

		self._device: Optional[torch.device] = None

	# ── Public API ────────────────────────────────────────────────────────────

	def compute_fisher(
		self,
		model: nn.Module,
		loader: DataLoader,
		criterion: Callable,
	) -> Tensor:
		"""Estimate diagonal Fisher on current-task data.  Returns [D]."""
		return compute_fisher_diag(
			model, loader, criterion, self._device, self.fisher_samples
		)

	def prepare_epoch(self, F_new: Tensor) -> None:
		"""
		Call once per epoch before the batch loop.

		Caches F_new and precomputes A⁻¹ = (Gᵀ F_old F_new⁻¹ F_old G + λI)⁻¹
		so the [m×m] inversion is paid once per epoch, not once per batch.
		"""
		assert self.F_old is not None, \
			"Call after_task() after task 1 before training task 2."
		self._F_new = F_new
		self._A_inv = _build_A_inv(self.G, self.F_old, F_new, self.lam)

	def step(self, model: nn.Module) -> None:
		"""
		Apply the FOPNG update for one batch.

		Call after loss.backward().  Zeroes gradients after applying the update.
		Requires prepare_epoch() to have been called this epoch.
		"""
		assert self._A_inv is not None, \
			"Call prepare_epoch(F_new) before step()."

		g      = _flat_grad(model)
		v_star = _fopng_update(
			g=g, G=self.G, F_old=self.F_old, F_new=self._F_new,
			A_inv=self._A_inv, lr=self.lr, lam=self.lam,
		)
		_apply_flat_update(model, v_star)   # θ ← θ + v*
		model.zero_grad()

	def after_task(
		self,
		model: nn.Module,
		loader: DataLoader,
		criterion: Callable,
	) -> None:
		"""
		Call after finishing all epochs on a task (including task 1).

		1. Estimates F_new on this task's data.
		2. Updates F_old ← (1−α) F_old + α F_new.
		3. Collects grads_per_task gradients and appends them to G.
		4. Drops oldest columns if G exceeds max_directions.
		"""
		device = next(model.parameters()).device
		self._device = device

		# 1 & 2: Fisher update
		F_new = self.compute_fisher(
			model, loader, criterion
		)
		if self.F_old is None:
			self.F_old = F_new.clone()
		else:
			self.F_old = (1.0 - self.alpha) * self.F_old + self.alpha * F_new

		# 3: Collect gradients and append to G
		new_cols = self._collect_gradients(model, loader, criterion)  # [D, k]
		self.G   = new_cols if self.G is None else \
				   torch.cat([self.G, new_cols], dim=1)               # [D, m+k]

		# 4: Enforce cap
		if self.G.shape[1] > self.max_directions:
			self.G = self.G[:, -self.max_directions:]

	# ── Internal ──────────────────────────────────────────────────────────────

	def _collect_gradients(
		self,
		model: nn.Module,
		loader: DataLoader,
		criterion: Callable,
	) -> Tensor:
		"""
		Collect up to grads_per_task gradient vectors at final task parameters.
		Returns [D, k].
		"""
		grads: List[Tensor] = []
		model.eval()
		with torch.enable_grad():
			for x, y in loader:
				if len(grads) >= self.grads_per_task:
					break
				x, y = x.to(self._device), y.to(self._device)
				model.zero_grad()
				loss = criterion(model(x), y)
				loss.backward()
				grads.append(_flat_grad(model).clone())
		model.zero_grad()
		model.train()
		return torch.stack(grads, dim=1)   # [D, k]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_fopng(
	model: nn.Module,
	task_loaders: List[DataLoader],
	criterion: Callable,
	*,
	lr: float = 1e-3,
	lam: float = 1e-3,
	alpha: float = 0.5,
	grads_per_task: int = 80,
	max_directions: int = 400,
	fisher_samples: int = 1024,
	epochs: int = 5,
	first_task_optimizer_cls=torch.optim.Adam,
	verbose: bool = True,
) -> FOPNG:
	"""
	Full continual training loop matching Algorithm 1 of the paper.

	task_loaders : one DataLoader per task  [D_1, ..., D_T]
	Returns the fitted FOPNG instance.
	"""
	device = next(model.parameters()).device

	fopng = FOPNG(
		lr=lr, lam=lam, alpha=alpha,
		grads_per_task=grads_per_task,
		max_directions=max_directions,
		fisher_samples=fisher_samples,
	)

	for t, loader in enumerate(task_loaders):

		# ── Task 1: standard optimizer ────────────────────────────────
		if t == 0:
			if verbose:
				print(f"[FOPNG] Task 1 – {first_task_optimizer_cls.__name__}")
			opt = first_task_optimizer_cls(model.parameters(), lr=lr)
			for epoch in range(epochs):
				total = 0.0
				model.train()
				for x, y in loader:
					x, y = x.to(device), y.to(device)
					opt.zero_grad()
					loss = criterion(model(x), y)
					loss.backward()
					opt.step()
					total += loss.item()
				if verbose:
					print(f"  epoch {epoch+1}/{epochs}  "
						  f"loss={total/len(loader):.4f}")
			fopng.after_task(model, loader, criterion)
			if verbose:
				print(f"  G: {fopng.G.shape}  "
					  f"F_old: [{fopng.F_old.min():.2e}, "
					  f"{fopng.F_old.max():.2e}]")

		# ── Tasks 2+: FOPNG ───────────────────────────────────────────
		else:
			if verbose:
				print(f"\n[FOPNG] Task {t+1}")
			for epoch in range(epochs):
				F_new = fopng.compute_fisher(model, loader, criterion)
				fopng.prepare_epoch(F_new)           # A_inv computed here
				total = 0.0
				model.train()
				for x, y in loader:
					x, y = x.to(device), y.to(device)
					loss = criterion(model(x), y)
					loss.backward()
					total += loss.item()
					fopng.step(model)
				if verbose:
					print(f"  epoch {epoch+1}/{epochs}  "
						  f"loss={total/len(loader):.4f}")
			fopng.after_task(model, loader, criterion)
			if verbose:
				print(f"  G: {fopng.G.shape}  "
					  f"F_old: [{fopng.F_old.min():.2e}, "
					  f"{fopng.F_old.max():.2e}]")
				
		# ── Evaluate all tasks ─────────────────────────────────
		print("\n" + "=" * 60)
		print(f"Final accuracy on all tasks after training task {t+1}:")
		print("=" * 60)
		for t, loader in enumerate(loaders):
			acc = evaluate_accuracy(model, loader)
			print(f"  Task {t+1}: {acc*100:.1f}%")

		print(f"\n  G shape : {fopng.G.shape}")
		print(f"  F_old   : min={fopng.F_old.min():.2e}  "
				f"max={fopng.F_old.max():.2e}")

	return fopng


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
from torch.utils.data import TensorDataset

torch.manual_seed(0)
device = torch.device("cpu")

model = nn.Sequential(
	nn.Linear(20, 32), nn.ReLU(),
	nn.Linear(32, 5),
).to(device)

criterion = nn.CrossEntropyLoss()

NUM_CLASSES = 5
INPUT_DIM   = 20

def make_task(n=800, task_id=0, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    # Initialize all inputs to zero
    X = torch.zeros(n, INPUT_DIM)
    
    # Each task gets its own exclusive chunk of 6 features
    start_idx = task_id * 6
    end_idx = start_idx + 6
    
    # Only populate the active features for this task
    X[:, start_idx:end_idx] = torch.randn(n, 6, generator=rng)
    
    # Create weights ONLY for those active features
    W = torch.randn(NUM_CLASSES, 6, generator=rng)
    
    # The target depends solely on this task's specific features
    y = (X[:, start_idx:end_idx] @ W.t()).argmax(dim=1)
    
    return DataLoader(TensorDataset(X, y), batch_size=10, shuffle=True)

# Update the loader creation to pass the task_id
loaders = [make_task(n=800, task_id=t, seed=t) for t in range(3)]

def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
	model.eval()
	correct, total = 0, 0
	with torch.no_grad():
		for x, y in loader:
			preds = model(x).argmax(dim=1)
			correct += (preds == y).sum().item()
			total   += y.size(0)
	model.train()
	return correct / total


fopng = train_fopng(
	model, loaders, criterion,
	lr=1e-2, lam=1e-2, alpha=0.5,
	grads_per_task=80, max_directions=400,
	epochs=10, verbose=True,
)

print("\nSmoke-test passed.")
print(f"  G shape : {fopng.G.shape}")
print(f"  F_old   : min={fopng.F_old.min():.2e}  max={fopng.F_old.max():.2e}")

print("\n" + "=" * 60)
print("\n" + "=" * 60)

print("SGD COMPARISON")
#-----------SGD attempt ---------------#

model2 = nn.Sequential(
	nn.Linear(20, 32), nn.ReLU(),
	nn.Linear(32, 5),
).to(device)
sgd = torch.optim.SGD(model2.parameters(), lr=1e-2)
epochs = 10
for t, loader in enumerate(loaders):
	for epoch in range(epochs):
		total = 0.0
		model2.train()
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			sgd.zero_grad()
			loss = criterion(model2(x), y)
			loss.backward()
			sgd.step()
			total += loss.item()

		print(f"  epoch {epoch+1}/{epochs}  "
				f"loss={total/len(loader):.4f}")

	# ── Evaluate all tasks ─────────────────────────────────
	print("\n" + "=" * 60)
	print(f"Final accuracy on all tasks after training task {t+1}:")
	print("=" * 60)
	for t, loader in enumerate(loaders):
		acc = evaluate_accuracy(model2, loader)
		print(f"  Task {t+1}: {acc*100:.1f}%")

