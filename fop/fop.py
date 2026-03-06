import math
from collections import defaultdict
from matplotlib.pyplot import step
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math
import time
import pdb
import asdl
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
from torch import nn
from typing import List, Union, Any, Iterable
from collections import OrderedDict
from asdl import NaturalGradientMaker
from asdl import PreconditioningConfig, PreconditionedGradientMaker
from asdl.vector import ParamVector
from asdl.matrices import *
from asdl.symmatrix import *
from asdl.fisher import FISHER_MC, LOSS_CROSS_ENTROPY
_module_level_shapes = [
    SHAPE_LAYER_WISE,
    SHAPE_KRON,
    SHAPE_SWIFT_KRON,
    SHAPE_KFE,
    SHAPE_UNIT_WISE,
    SHAPE_DIAG,
]
class DistKFACGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig,
                 fisher_type: str = FISHER_MC, loss_type: str = LOSS_CROSS_ENTROPY,
                 scale: float = 1, grad_scale: float = 1,
                 n_mc_samples: int = 1, var: float = 1., seed: int = None, swift=False):
        fisher_shape = [SHAPE_SWIFT_KRON if swift else SHAPE_KRON,
                        (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                        (nn.BatchNorm2d, SHAPE_UNIT_WISE),
                        (nn.LayerNorm, SHAPE_UNIT_WISE)]
        super().__init__(model, config,
                         fisher_type=fisher_type,
                         fisher_shape=fisher_shape,
                         loss_type=loss_type,
                         scale=scale,
                         grad_scale=grad_scale,
                         n_mc_samples=n_mc_samples,
                         var=var,
                         seed=seed
                         )
    def multi_run(self,model, x_full,t_full, loss_fn,label_smoothing=0.1):
        dummy_full = self.setup_model_call(model, x_full)
        self.setup_loss_call(loss_fn, dummy_full, t_full, label_smoothing=label_smoothing)
        print("Running DistKFACGradientMaker with distributed settings.")
        step = self.state["step"]
        if step==0:
            print("This KFAC precondition with data_parallel instead of the original asdl implementation")
        self._startup()
        if self.do_forward_and_backward(step):
            self.forward()
            self.backward()
        if self.do_update_curvature(step):
            self.update_curvature()
        if self.do_update_preconditioner(step):
            self.update_preconditioner()
        self.precondition_data_parallel()
        self._teardown()
        self.state["step"] += 1
        return self._model_output, self._loss.mean()
    def run(self,model, x_full,t_full, loss_fn,label_smoothing=0.1):
        dummy_full = self.setup_model_call(model, x_full)
        self.setup_loss_call(loss_fn, dummy_full, t_full, label_smoothing=label_smoothing)
        return super().forward_and_backward()
    def precondition_data_parallel(self, vectors: ParamVector = None, grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.grad_scale
        if self.world_size > 1:
            self._reduce_scatter_gradients_by_partition()
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                if self.world_rank == self.partitions[enum_shape][enum_module]:
                    if not self.is_module_for_inv_and_precondition(module):
                        continue
                    self._precondition_module_parallel(module, shape, vectors, grad_scale=grad_scale, use_inv=use_inv)
        params = [p for p in self.parameters_for(SHAPE_FULL)]
        if len(params) > 0:
            fisher = self._get_full_fisher()
            if fisher is None:
                raise ValueError(f'Fisher of shape {SHAPE_FULL} has not been calculated.')
            if vectors is None:
                vectors = ParamVector(params, [p.grad for p in params])
            if vectors is None:
                raise ValueError('gradient has not been calculated.')
            if grad_scale != 1:
                vectors.mul_(grad_scale)
            fisher.mvp(vectors=vectors, use_inv=use_inv, inplace=True)
        if self.world_size > 1:
            self._all_gather_preconditioned_gradients_by_partition()
    def _reduce_scatter_gradients_by_partition(self):
        if not dist.is_initialized():
            return
        rank_modules = {rank: [] for rank in range(self.world_size)}
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                owning_rank = self.partitions[enum_shape][enum_module]
                if self.is_module_for_inv_and_precondition(module):
                    rank_modules[owning_rank].append(module)
        for rank, modules in rank_modules.items():
            if len(modules) == 0:
                continue
            grads_for_rank = []
            for module in modules:
                if hasattr(module, 'weight') and module.weight.requires_grad and module.weight.grad is not None:
                    grads_for_rank.append(module.weight.grad)
                if self._bias_requires_grad(module) and module.bias.grad is not None:
                    grads_for_rank.append(module.bias.grad)
            if len(grads_for_rank) == 0:
                continue
            packed_grad = parameters_to_vector(grads_for_rank)
            dist.reduce(packed_grad, dst=rank, op=dist.ReduceOp.AVG)
            if self.world_rank == rank:
                vector_to_parameters(packed_grad, grads_for_rank)
    def _all_gather_preconditioned_gradients_by_partition(self):
        if not dist.is_initialized():
            return
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                owning_rank = self.partitions[enum_shape][enum_module]
                if not self.is_module_for_inv_and_precondition(module):
                    continue
                grads_to_broadcast = []
                if hasattr(module, 'weight') and module.weight.requires_grad and module.weight.grad is not None:
                    grads_to_broadcast.append(module.weight.grad)
                if self._bias_requires_grad(module) and module.bias.grad is not None:
                    grads_to_broadcast.append(module.bias.grad)
                if len(grads_to_broadcast) == 0:
                    continue
                packed_grad = parameters_to_vector(grads_to_broadcast)
                dist.broadcast(packed_grad, src=owning_rank)
                vector_to_parameters(packed_grad, grads_to_broadcast)
    def _bias_requires_grad(self,module):
            return hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad
    def _precondition_module_parallel(self, module, shape=None, vectors: ParamVector = None,
                            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None,
                            grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.grad_scale
        if shape is None:
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
        if vectors is not None:
            vec_weight = vectors.get_vector_by_param(module.weight, None)
            vec_bias = vectors.get_vector_by_param(module.bias, None)
        if shape is None:
            raise ValueError(f'No shape is assigned to module: {module}.')
        matrix = self._get_module_symmatrix(module, shape)
        if matrix is None:
            raise ValueError(f'Matrix of shape {shape} for module {module} has not been calculated.')
        if vec_weight is None and module.weight.requires_grad:
            vec_weight = module.weight.grad
        if vec_weight is None:
            raise ValueError(f'weight gradient for module {module} has not been calculated.')
        if self._bias_requires_grad(module):
            if vec_bias is None:
                vec_bias = module.bias.grad
            if vec_bias is None:
                raise ValueError(f'bias gradient for module {module} has not been calculated.')
        if grad_scale != 1:
            vec_weight.data.mul_(grad_scale)
            if vec_bias is not None:
                vec_bias.data.mul_(grad_scale)
        if not use_inv or matrix.has_inv:
            kwargs = dict(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=use_inv, inplace=True)
            if shape == SHAPE_KFE:
                kwargs['eps'] = self.config.damping
            matrix.mvp(**kwargs)
class FOPGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig,
                 fisher_type: str = FISHER_MC, loss_type: str = LOSS_CROSS_ENTROPY,
                 scale: float = 1, grad_scale: float = 1, beta: float = 0., beta_adaptive: bool = True,
                 eta_adaptive: bool = True,
                 n_mc_samples: int = 1, var: float = 1., seed: int = None, swift=False):
        fisher_shape = [SHAPE_SWIFT_KRON if swift else SHAPE_KRON,
                        (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                        (nn.BatchNorm2d, SHAPE_UNIT_WISE),
                        (nn.LayerNorm, SHAPE_UNIT_WISE)]
        super().__init__(
            model,
            config,
            fisher_type=fisher_type,
            fisher_shape=fisher_shape,
            loss_type=loss_type,
            scale=scale,
            grad_scale=grad_scale,
            n_mc_samples=n_mc_samples,
            var=var,
            seed=seed,
        )
        self.beta_adaptive = beta_adaptive
        self.eta_adaptive = eta_adaptive
        self.beta = beta
        print(
            f"The status of eta_adaptive: {self.eta_adaptive}, beta_adaptive: {self.beta_adaptive}"
        )
        self.eps = 1e-12
        self.beta = beta
        print(f"FOPGradientMaker initialized with beta={self.beta}, seed={seed}")
        self._model_params_with_grad = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        _zero_grads = [
            torch.zeros_like(p.data, device=p.device)
            for p in self._model_params_with_grad
        ]
        self._g1_vec = ParamVector(
            self._model_params_with_grad, [z.clone() for z in _zero_grads]
        )
        self._g2_vec = ParamVector(
            self._model_params_with_grad, [z.clone() for z in _zero_grads]
        )
        self.primary_grads ={}
        self.secondary_grads = {}
        proj_params_set = set()
        proj_params_ordered = []
        for module in self.modules_for(SHAPE_KRON):
            proj_params_set.add(module.weight)
            proj_params_ordered.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                proj_params_set.add(module.bias)
                proj_params_ordered.append(module.bias)
        self._proj_params_set = proj_params_set
        self._proj_params_ordered = proj_params_ordered
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.rank == 0:
                print("Initializing FOPGradientMaker with distributed settings.")
            if self.world_size % 2 != 0:
                raise ValueError("This algorithm requires an even number of GPUs.")
            self.is_primary_rank = (self.rank % 2 == 0)
            primary_ranks = [i for i in range(self.world_size) if i % 2 == 0]
            secondary_ranks = [i for i in range(self.world_size) if i % 2 != 0]
            self.primary_group = dist.new_group(ranks=primary_ranks)
            self.secondary_group = dist.new_group(ranks=secondary_ranks)
            if self.rank == 0:
                print(
                    f"FOPGradientMaker initialized with world_size={self.world_size}, "
                    f"rank={self.rank}, primary_group={primary_ranks}, secondary_group={secondary_ranks}"
                )
        else:
            self.world_size = 1
            self.rank = 0
            self.is_primary_rank = True
            self.primary_group = None
            self.secondary_group = None
            print("FOPGradientMaker initialized without distributed settings.")
    def run(self,model, x_full,t_full, loss_fn,label_smoothing=0.1):
        step = self.state["step"]
        self._startup()
        x1,x2 =torch.chunk(x_full,2,dim=0)
        t1,t2 =torch.chunk(t_full,2,dim=0)
        dummy_full = self.setup_model_call(model, x_full)
        self.setup_loss_call(loss_fn, dummy_full, t_full, label_smoothing=label_smoothing)
        output_1 = model(x1)
        self._loss_1 = loss_fn(output_1, t1, label_smoothing=label_smoothing)
        loss_1 = self._loss_1.item()
        g1_tensors_tuple = torch.autograd.grad(self._loss_1, self._model_params_with_grad, retain_graph=True, create_graph=False)
        self.model.zero_grad()
        output_2 = model(x2)
        self._loss_2 = loss_fn(output_2, t2, label_smoothing=label_smoothing)
        loss_2 = self._loss_2.item()
        g2_tensors_tuple = torch.autograd.grad(self._loss_2, self._model_params_with_grad, retain_graph=False, create_graph=False)
        g1_proj_tensors = []
        g2_proj_tensors = []
        for i, p in enumerate(self._model_params_with_grad):
            grad_val_1 = g1_tensors_tuple[i]
            grad_val_2 = g2_tensors_tuple[i]
            p.grad = (grad_val_1 + grad_val_2) / 2.0
            if p in self._proj_params_set:
                g1_proj_tensors.append(grad_val_1.clone())
                g2_proj_tensors.append(grad_val_2.clone())
        self.set_g1_values(g1_proj_tensors)
        self.set_g2_values(g2_proj_tensors)
        if self.do_update_curvature(step):
            self.update_curvature()
        if self.do_update_preconditioner(step):
            self.update_preconditioner()
        del g1_tensors_tuple, g2_tensors_tuple
        self.precondition_new()
        self._teardown()
        self.state["step"] += 1
        final_output = torch.cat((output_1, output_2), dim=0)
        return final_output, (loss_1+loss_2)/2
    def forward_and_backward(self):
        step = self.state["step"]
        self._startup()
        if self.do_forward_and_backward(step):
            self.forward()
            self.backward()
        if self.do_update_curvature(step):
            self.update_curvature()
        if self.do_update_preconditioner(step):
            self.update_preconditioner()
        self.precondition()
        self._teardown()
        self.state["step"] += 1
        return self._model_output, self._loss.mean()
    def set_g1_values(self, g1_proj_tensors: list[torch.Tensor]):
        self._g1_vec = ParamVector(self._proj_params_ordered, g1_proj_tensors)
    def set_g2_values(self, g2_proj_tensors: list[torch.Tensor]):
        self._g2_vec = ParamVector(self._proj_params_ordered, g2_proj_tensors)
    def precondition_new(self, vectors: ParamVector = None, grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.grad_scale
        param_to_g_1 =OrderedDict((p, v_avg) for p, v_avg in zip(self._g1_vec.params(), self._g1_vec.values()))
        param_to_g_2 =OrderedDict((p, v_diff) for p, v_diff in zip(self._g2_vec.params(), self._g2_vec.values()))
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                if self.world_rank == self.partitions[enum_shape][enum_module]:
                    if not self.is_module_for_inv_and_precondition(module):
                        continue
                    g_1_w_module, g_1_b_module, g_2_w_module, g_2_b_module = None, None, None, None
                    has_weight = hasattr(module, 'weight') and module.weight.requires_grad
                    has_bias = hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad
                    if has_weight:
                        g_1_w_module = param_to_g_1.get(module.weight)
                        g_2_w_module = param_to_g_2.get(module.weight)
                    if has_bias:
                        g_1_b_module = param_to_g_1.get(module.bias)
                        g_2_b_module = param_to_g_2.get(module.bias)
                    additional_modules=(g_1_w_module, g_1_b_module, g_2_w_module, g_2_b_module)
                    self._precondition_module_new(module,additional_modules,shape, vectors, grad_scale=grad_scale, use_inv=use_inv)
        params = [p for p in self.parameters_for(SHAPE_FULL)]
        if len(params) > 0:
            fisher = self._get_full_fisher()
            if fisher is None:
                raise ValueError(f'Fisher of shape {SHAPE_FULL} has not been calculated.')
            if vectors is None:
                vectors = ParamVector(params, [p.grad for p in params])
            if vectors is None:
                raise ValueError('gradient has not been calculated.')
            if grad_scale != 1:
                vectors.mul_(grad_scale)
            fisher.mvp(vectors=vectors, use_inv=use_inv, inplace=True)
        if self.world_size > 1:
            if self.do_accumulate:
                self.all_gather_or_reduce_grad()
            else:
                self.all_reduce_all_grad(async_op=False)
    def _precondition_module_new(
        self,
        module,
        additional_modules=None,
        shape=None,
        vectors: ParamVector = None,
        vec_weight: torch.Tensor = None,
        vec_bias: torch.Tensor = None,
        grad_scale=None,
        use_inv=True,
    ):
        has_weight = hasattr(module, 'weight') and module.weight.requires_grad
        has_bias = hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad
        current_grad_scale = grad_scale if grad_scale is not None else self.grad_scale
        g_1_w_module, g_1_b_module, g_2_w_module, g_2_b_module = additional_modules
        device = g_1_w_module.device
        g_avg_w_module = (g_1_w_module + g_2_w_module)/2
        if g_2_b_module is not None:
            g_avg_b_module = (g_1_b_module + g_2_b_module)/2
        else:
            g_avg_b_module = None
        g_diff_w_module = g_1_w_module - g_2_w_module
        if g_2_b_module is not None:
            g_diff_b_module = g_1_b_module - g_2_b_module
        if shape is None:
            for s_enum, s_val in enumerate(
                _module_level_shapes
            ):
                if module in self.modules_for(s_val):
                    shape = s_val
                    break
            if shape is None:
                super()._precondition_module(
                    module,
                    shape,
                    vectors,
                    vec_weight,
                    vec_bias,
                    current_grad_scale,
                    use_inv,
                )
                return
        if self._g1_vec is None or self._g2_vec is None:
            raise RuntimeError(
                "g1 and/or g2 were not properly set before _precondition_module."
            )
        if vectors is not None:
            vec_weight = vectors.get_vector_by_param(module.weight, None)
            vec_bias = vectors.get_vector_by_param(module.bias, None)
        if shape is None:
            raise ValueError(f'No shape is assigned to module: {module}.')
        F_module = self._get_module_symmatrix(module, shape)
        EPS =self.eps
        g_avg_w_module =  (g_1_w_module +  g_2_w_module) / 2
        g_diff_w_module = g_1_w_module - g_2_w_module
        g_avg_b_module, g_diff_b_module = None, None
        if has_bias:
            g_avg_b_module = (g_1_b_module + g_2_b_module) / 2
            g_diff_b_module = g_1_b_module - g_2_b_module
        eta_l_star_module = torch.tensor(1.0, device=device)
        if F_module is not None and ((has_weight and g_avg_w_module is not None) or
                               (has_bias and g_avg_b_module is not None)):
            mvp_F_g_avg_result = F_module.mvp(vec_weight=g_avg_w_module, vec_bias=g_avg_b_module, use_inv=False, inplace=False)
            F_g_avg_w, F_g_avg_b = self._unpack_mvp_result(F_module, mvp_F_g_avg_result, has_weight, has_bias, g_avg_w_module, g_avg_b_module)
            dot_g_diff_F_g_avg = torch.tensor(0., device=device)
            dot_g_avg_F_g_avg = torch.tensor(0., device=device)
            if g_avg_w_module is not None and F_g_avg_w is not None and g_diff_w_module is not None:
                dot_g_diff_F_g_avg += torch.sum(g_diff_w_module * F_g_avg_w)
                dot_g_avg_F_g_avg += torch.sum(g_avg_w_module * F_g_avg_w)
            if g_avg_b_module is not None and F_g_avg_b is not None and g_diff_b_module is not None:
                dot_g_diff_F_g_avg += torch.sum(g_diff_b_module * F_g_avg_b)
                dot_g_avg_F_g_avg += torch.sum(g_avg_b_module * F_g_avg_b)
            projection_coeff = torch.tensor(0., device=dot_g_avg_F_g_avg.device)
            if torch.abs(dot_g_avg_F_g_avg) > EPS:
                 projection_coeff = dot_g_diff_F_g_avg / dot_g_avg_F_g_avg
            g_diff_perp_w_module, g_diff_perp_b_module = None, None
            if has_weight and g_avg_w_module is not None and g_diff_w_module is not None:
                g_diff_perp_w_module = g_diff_w_module - projection_coeff * g_avg_w_module
            elif has_weight and g_diff_w_module is not None :
                g_diff_perp_w_module = g_diff_w_module.clone()
            if has_bias and g_avg_b_module is not None and g_diff_b_module is not None:
                g_diff_perp_b_module = g_diff_b_module - projection_coeff * g_avg_b_module
            elif has_bias and g_diff_b_module is not None:
                g_diff_perp_b_module = g_diff_b_module.clone()
            beta_star_module = torch.tensor(self.beta, device=device)
            if self.beta_adaptive  and F_module.has_inv and ((has_weight and g_diff_perp_w_module is not None) or
                                    (has_bias and g_diff_perp_b_module is not None)):
                mvp_Finv_g_diff_perp_result = F_module.mvp(vec_weight=g_diff_perp_w_module, vec_bias=g_diff_perp_b_module, use_inv=True, inplace=False)
                Finv_g_diff_perp_w, Finv_g_diff_perp_b = self._unpack_mvp_result(F_module, mvp_Finv_g_diff_perp_result, has_weight, has_bias, g_diff_perp_w_module, g_diff_perp_b_module)
                D_module = torch.tensor(0., device=device)
                if g_avg_w_module is not None and Finv_g_diff_perp_w is not None:
                    D_module += torch.sum(g_avg_w_module * Finv_g_diff_perp_w)
                if g_avg_b_module is not None and Finv_g_diff_perp_b is not None:
                    D_module += torch.sum(g_avg_b_module * Finv_g_diff_perp_b)
                E_module = torch.tensor(0., device=device)
                if g_diff_perp_w_module is not None and Finv_g_diff_perp_w is not None:
                    E_module += torch.sum(g_diff_perp_w_module * Finv_g_diff_perp_w)
                if g_diff_perp_b_module is not None and Finv_g_diff_perp_b is not None:
                    E_module += torch.sum(g_diff_perp_b_module * Finv_g_diff_perp_b)
                if torch.abs(E_module) > EPS:
                    beta_star_module = D_module / (E_module + EPS)
                beta_star_module = torch.clamp(beta_star_module, min=-0.5, max=0.5)
            g_comb_w_module, g_comb_b_module = None, None
            if has_weight and g_avg_w_module is not None:
                g_comb_w_module = g_avg_w_module.clone()
                if g_diff_perp_w_module is not None:
                    g_comb_w_module += beta_star_module * g_diff_perp_w_module
            if has_bias and g_avg_b_module is not None:
                g_comb_b_module = g_avg_b_module.clone()
                if g_diff_perp_b_module is not None:
                    g_comb_b_module += beta_star_module * g_diff_perp_b_module
            g_A1_w_final = g_comb_w_module
            g_A1_b_final = g_comb_b_module
            if self.eta_adaptive and  F_module.has_inv and ((has_weight and g_comb_w_module is not None) or
                                   (has_bias and g_comb_b_module is not None)):
                mvp_Finv_g_comb_result = F_module.mvp(vec_weight=g_comb_w_module, vec_bias=g_comb_b_module, use_inv=True, inplace=False)
                Finv_g_comb_w, Finv_g_comb_b = self._unpack_mvp_result(F_module, mvp_Finv_g_comb_result, has_weight, has_bias, g_comb_w_module, g_comb_b_module)
                Numerator_eta = torch.tensor(0., device=device)
                if g_avg_w_module is not None and Finv_g_comb_w is not None:
                    Numerator_eta += 2.0 * torch.sum(g_avg_w_module * Finv_g_comb_w)
                if g_avg_b_module is not None and Finv_g_comb_b is not None:
                    Numerator_eta += 2.0 * torch.sum(g_avg_b_module * Finv_g_comb_b)
                Denominator_eta = torch.tensor(0., device=device)
                if g_comb_w_module is not None and Finv_g_comb_w is not None:
                    Denominator_eta += torch.sum(g_comb_w_module * Finv_g_comb_w)
                if g_comb_b_module is not None and Finv_g_comb_b is not None:
                    Denominator_eta += torch.sum(g_comb_b_module * Finv_g_comb_b)
                if torch.abs(Denominator_eta) > EPS:
                    eta_l_star_module = Numerator_eta / (Denominator_eta + EPS)
                eta_l_star_module = torch.clamp(eta_l_star_module, min=0.0, max=2.0)
        else:
            if has_weight and g_avg_w_module is not None:
                g_A1_w_final = g_avg_w_module.clone()
            if has_bias and g_avg_b_module is not None:
                g_A1_b_final = g_avg_b_module.clone()
        effective_scale = current_grad_scale * eta_l_star_module.item()
        final_grad_w_to_set, final_grad_b_to_set = None, None
        if has_weight:
            if g_A1_w_final is not None:
                final_grad_w_to_set = g_A1_w_final.clone()
                if effective_scale != 1.:
                    final_grad_w_to_set.mul_(effective_scale)
                module.weight.grad = final_grad_w_to_set
            elif module.weight.grad is not None: module.weight.grad.zero_()
            else: module.weight.grad = torch.zeros_like(module.weight.data)
        if has_bias:
            if g_A1_b_final is not None:
                final_grad_b_to_set = g_A1_b_final.clone()
                if effective_scale != 1.:
                    final_grad_b_to_set.mul_(effective_scale)
                module.bias.grad = final_grad_b_to_set
            elif module.bias.grad is not None: module.bias.grad.zero_()
            else: module.bias.grad = torch.zeros_like(module.bias.data)
        if F_module is not None and use_inv and F_module.has_inv:
            kwargs_final_mvp = dict(
                vec_weight=module.weight.grad if has_weight else None,
                vec_bias=module.bias.grad if has_bias else None,
                use_inv=True,
                inplace=True
            )
            if shape == SHAPE_KFE: kwargs_final_mvp['eps'] = self.config.damping
            if (has_weight and module.weight.grad is not None and torch.any(module.weight.grad != 0)) or (has_bias and module.bias.grad is not None and torch.any(module.bias.grad != 0)):
                F_module.mvp(**kwargs_final_mvp)
    def _bias_requires_grad(self,module):
            return hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad
    def _unpack_mvp_result(
        self, F_m_obj, mvp_result_val, has_w, has_b, vec_w_provided, vec_b_provided
    ):
        res_w, res_b = None, None
        if isinstance(F_m_obj, Kron) or isinstance(F_m_obj, KFE):
            if vec_b_provided is not None and has_b:
                if isinstance(mvp_result_val, tuple) and len(mvp_result_val) == 2:
                    res_w, res_b = mvp_result_val
                else:
                    raise TypeError(
                        f"{type(F_m_obj)}.mvp w bias ret unexp: {type(mvp_result_val)}"
                    )
            elif vec_w_provided is not None and has_w:
                if torch.is_tensor(mvp_result_val):
                    res_w = mvp_result_val
                else:
                    raise TypeError(
                        f"{type(F_m_obj)}.mvp w/o bias ret unexp: {type(mvp_result_val)}"
                    )
        elif isinstance(F_m_obj, Diag):
            if not isinstance(mvp_result_val, list):
                raise TypeError(f"Diag.mvp non-list: {type(mvp_result_val)}")
            idx = 0
            if vec_w_provided is not None and has_w:
                res_w = mvp_result_val[idx]
                idx += 1
            if vec_b_provided is not None and has_b:
                res_b = mvp_result_val[idx]
                idx += 1
            if idx != len(mvp_result_val):
                raise ValueError(f"Diag.mvp list len err.")
        elif isinstance(F_m_obj, UnitWise):
            if isinstance(mvp_result_val, tuple) and len(mvp_result_val) == 2:
                res_w, res_b = mvp_result_val
            elif vec_w_provided is None and vec_b_provided is None:
                pass
            else:
                raise TypeError(f"UnitWise.mvp ret unexp: {type(mvp_result_val)}")
        elif isinstance(F_m_obj, SymMatrix):
            if (
                vec_w_provided is not None
                and vec_b_provided is not None
                and has_w
                and has_b
            ):
                if isinstance(mvp_result_val, tuple) and len(mvp_result_val) == 2:
                    res_w, res_b = mvp_result_val
                else:
                    raise TypeError(f"SymM.mvp (w&b) ret unexp: {type(mvp_result_val)}")
            elif vec_w_provided is not None and has_w:
                if isinstance(mvp_result_val, list) and len(mvp_result_val) == 1:
                    res_w = mvp_result_val[0]
                elif torch.is_tensor(mvp_result_val):
                    res_w = mvp_result_val
                else:
                    raise TypeError(f"SymM.mvp (w) ret unexp: {type(mvp_result_val)}")
            elif vec_b_provided is not None and has_b:
                if isinstance(mvp_result_val, list) and len(mvp_result_val) == 1:
                    res_b = mvp_result_val[0]
                elif torch.is_tensor(mvp_result_val):
                    res_b = mvp_result_val
                else:
                    raise TypeError(f"SymM.mvp (b) ret unexp: {type(mvp_result_val)}")
        else:
            raise TypeError(f"Unknown F_m type for unpacking: {type(F_m_obj)}")
        return res_w, res_b
    def calculate_global_g1_g2(self):
        if not dist.is_initialized():
            return
        local_grad_vec = parameters_to_vector([p.grad for p in self.model.parameters() if p.grad is not None])
        if self.is_primary_rank:
            dist.all_reduce(local_grad_vec, op=dist.ReduceOp.AVG, group=self.primary_group)
        else:
            dist.all_reduce(local_grad_vec, op=dist.ReduceOp.AVG, group=self.secondary_group)
        g1_global_vec = torch.zeros_like(local_grad_vec)
        g2_global_vec = torch.zeros_like(local_grad_vec)
        if self.is_primary_rank:
            g1_global_vec = local_grad_vec
            dist.broadcast(g1_global_vec, src=0, group=dist.group.WORLD)
            dist.broadcast(g2_global_vec, src=1, group=dist.group.WORLD)
        else:
            g2_global_vec = local_grad_vec
            dist.broadcast(g1_global_vec, src=0, group=dist.group.WORLD)
            dist.broadcast(g2_global_vec, src=1, group=dist.group.WORLD)
        params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
        param_slices = {}
        current_pos = 0
        for p in params_with_grad:
            num_params = p.numel()
            param_slices[p] = (current_pos, current_pos + num_params)
            current_pos += num_params
        g1_tensors = []
        g2_tensors = []
        for p in self._proj_params_ordered:
            sl = param_slices.get(p)
            if sl is None:
                g1_tensors.append(torch.zeros_like(p, device=g1_global_vec.device))
                g2_tensors.append(torch.zeros_like(p, device=g2_global_vec.device))
                continue
            start, end = sl
            g1_tensors.append(g1_global_vec[start:end].view_as(p))
            g2_tensors.append(g2_global_vec[start:end].view_as(p))
        self.set_g1_values(g1_tensors)
        self.set_g2_values(g2_tensors)
    def multi_run(self,model, x_full,t_full, loss_fn,label_smoothing=0.1):
        dummy_full = self.setup_model_call(model, x_full)
        self.setup_loss_call(loss_fn, dummy_full, t_full, label_smoothing=label_smoothing)
        step = self.state["step"]
        self._startup()
        self.forward()
        self.backward()
        if self.do_update_curvature(step):
            self.update_curvature()
        if self.do_update_preconditioner(step):
            self.update_preconditioner()
        self.calculate_global_g1_g2()
        self.precondition_new()
        self._teardown()
        self.state["step"] += 1
        return self._model_output, self._loss.mean()
