'''
Adapted and extended from: https://github.com/RAIVNLab/supsup/blob/master/mnist.ipynb
'''
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import copy

# Subnetwork forward from hidden networks
# Mask derived using Piggyback method (threshold by
# a constant a)
# paper: https://arxiv.org/abs/1801.06519
class GetSubnetDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        # A strict threshold gives an exact zero a well-defined inactive state,
        # which is required by ternary stored masks. Unquantized continuous
        # scores equal zero with probability zero, so their behavior is unchanged.
        return (scores > a).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

class GetSubnetContinuous(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float() * scores

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

# may not work as well as the alternative above (which is the default choice)
class GetSubnetContinuousV2:
    @staticmethod
    def apply(scores, a=0):
        return (scores >= a).float() * scores

def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

def _linear_comb_beta_weights(module, beta_indices):
    raw = module.betas[module.task, beta_indices]
    if getattr(module, 'use_uniform_betas', False):
        return torch.ones_like(raw) / raw.numel()
    return torch.softmax(raw, dim=-1)

NEW_MASK_RANDOM = 'random'
NEW_MASK_LINEAR_COMB = 'linear_comb'
MASK_SCORE_NORMALIZATION_NONE = 'none'
MASK_SCORE_NORMALIZATION_L2 = 'l2'
MASK_SCORE_NORMALIZATION_L2_STE = 'l2_ste'
MASK_SCORE_NORMALIZATION_RMS_STE = 'rms_ste'
MASK_SCORE_NORMALIZATION_MATCH_CURRENT_L2 = 'match_current_l2'
MASK_SCORE_NORMALIZATION_SOFTMAX_ACROSS_MASKS = 'softmax_across_masks'
MASK_SCORE_NORMALIZATION_STD = 'std'
MASK_SCORE_NORMALIZATION_ZSCORE = 'zscore'
MASK_SCORE_NORMALIZATION_SIGN = 'sign'
MASK_SCORE_NORMALIZATION_MODES = {
    MASK_SCORE_NORMALIZATION_NONE,
    MASK_SCORE_NORMALIZATION_L2,
    MASK_SCORE_NORMALIZATION_L2_STE,
    MASK_SCORE_NORMALIZATION_RMS_STE,
    MASK_SCORE_NORMALIZATION_MATCH_CURRENT_L2,
    MASK_SCORE_NORMALIZATION_SOFTMAX_ACROSS_MASKS,
    MASK_SCORE_NORMALIZATION_STD,
    MASK_SCORE_NORMALIZATION_ZSCORE,
    MASK_SCORE_NORMALIZATION_SIGN,
}

MASK_STORAGE_QUANTIZATION_NONE = 'none'
MASK_STORAGE_QUANTIZATION_BINARY = 'binary'
MASK_STORAGE_QUANTIZATION_TERNARY = 'ternary'
MASK_STORAGE_QUANTIZATION_MODES = {
    MASK_STORAGE_QUANTIZATION_NONE,
    MASK_STORAGE_QUANTIZATION_BINARY,
    MASK_STORAGE_QUANTIZATION_TERNARY,
}


class MaskScoreNormalizationMixin:
    def _init_mask_score_normalization(self):
        self.mask_score_normalization = MASK_SCORE_NORMALIZATION_NONE
        self.mask_score_normalize_current = False
        self.mask_score_skip_current_normalization_on_abstain = False
        self.mask_score_norm_eps = 1e-8
        self.mask_score_softmax_temperature = 1.0
        self.learned_transfer_gate_enabled = False
        self.learned_transfer_gate = None
        self.learned_transfer_gate_features = None
        self.learned_transfer_gate_prior_weights = {}
        self.use_sparsemax_fixed_composition = False
        self.use_raw_cosine_fixed_composition = False
        self.selected_task_weights = {}
        self.composition_support_size = None

    def set_selected_task_weights(self, prior_weights):
        self.selected_task_weights = {
            int(index): float(weight)
            for index, weight in prior_weights.items()
        }

    def set_composition_support_size(self, support_size):
        self.composition_support_size = (
            None if support_size is None else max(int(support_size), 0)
        )

    def _sparsemax_fixed_composition_weights(self, selected, reference):
        """Allocate one equal component to the current mask.

        Sparsemax determines the relative distribution over the ``m`` selected
        priors. The current mask receives ``1 / (m + 1)`` and the normalized
        prior distribution receives the remaining ``m / (m + 1)`` mass.
        """
        if not selected:
            return reference.new_ones(1)

        prior_weights = reference.new_tensor(
            [
                max(self.selected_task_weights.get(int(index), 0.0), 0.0)
                for index in selected
            ]
        )
        weight_sum = prior_weights.sum()
        if float(weight_sum.detach().cpu()) > 1e-8:
            prior_weights = prior_weights / weight_sum
        else:
            prior_weights = torch.ones_like(prior_weights) / len(selected)

        allocation_support_size = self.composition_support_size
        if allocation_support_size is None:
            allocation_support_size = len(selected)
        allocation_support_size = max(allocation_support_size, len(selected))
        component_count = allocation_support_size + 1
        current_weight = 1.0 / component_count
        prior_mass = allocation_support_size / component_count
        return torch.cat(
            [
                prior_mass * prior_weights,
                reference.new_tensor([current_weight]),
            ],
            dim=0,
        )

    def _raw_cosine_fixed_composition_weights(self, selected, reference):
        """Use absolute cosine evidence as fixed prior composition mass.

        ``selected_task_weights`` contains ``w_i * max(s_i, 0)``, where
        sparsemax weight ``w_i`` determines relative relevance and raw cosine
        similarity ``s_i`` preserves absolute evidence. Any unallocated mass
        is assigned to the current task mask.
        """
        if not selected:
            return reference.new_ones(1)

        prior_weights = reference.new_tensor(
            [
                max(self.selected_task_weights.get(int(index), 0.0), 0.0)
                for index in selected
            ]
        )
        prior_mass = prior_weights.sum()
        prior_mass_value = float(prior_mass.detach().cpu())
        if prior_mass_value > 1.0:
            prior_weights = prior_weights / prior_mass
            current_weight = 0.0
        else:
            current_weight = max(1.0 - prior_mass_value, 0.0)
        return torch.cat(
            [prior_weights, reference.new_tensor([current_weight])],
            dim=0,
        )

    def configure_learned_transfer_gate(self, enabled=True, initial_probability=0.5):
        self.learned_transfer_gate_enabled = bool(enabled)
        if not self.learned_transfer_gate_enabled:
            if getattr(self, 'betas', None) is not None:
                self.betas.requires_grad_(True)
            return []

        probability = float(initial_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError('initial_probability must be strictly between 0 and 1')
        if self.learned_transfer_gate is None:
            self.learned_transfer_gate = nn.Linear(4, 1)
            reference = self.betas if self.betas is not None else self.scores[0]
            self.learned_transfer_gate.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            nn.init.zeros_(self.learned_transfer_gate.weight)
            nn.init.constant_(
                self.learned_transfer_gate.bias,
                math.log(probability / (1.0 - probability)),
            )
        if getattr(self, 'betas', None) is not None:
            # Learned-gate mode replaces layerwise beta softmax with the
            # retrieval-conditioned source/current mixture below.
            self.betas.requires_grad_(False)
        return list(self.learned_transfer_gate.parameters())

    def set_learned_transfer_gate_context(
        self,
        prior_weights,
        features,
    ):
        self.learned_transfer_gate_prior_weights = {
            int(index): float(weight)
            for index, weight in prior_weights.items()
        }
        self.learned_transfer_gate_features = torch.as_tensor(
            features,
            dtype=torch.float32,
        ).reshape(4)

    def _learned_transfer_gate_weights(self, selected, reference):
        if not selected:
            return reference.new_ones(1)
        if self.learned_transfer_gate is None:
            raise RuntimeError('learned transfer gate is enabled but not configured')

        if self.learned_transfer_gate_features is None:
            features = reference.new_zeros(4)
        else:
            features = self.learned_transfer_gate_features.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        gate = torch.sigmoid(self.learned_transfer_gate(features).reshape(()))
        conditional = reference.new_tensor(
            [
                self.learned_transfer_gate_prior_weights.get(int(index), 0.0)
                for index in selected
            ]
        )
        conditional_sum = conditional.sum()
        if float(conditional_sum.detach().cpu()) > 1e-8:
            conditional = conditional / conditional_sum
        else:
            conditional = torch.zeros_like(conditional)
        return torch.cat(
            [gate * conditional, (1.0 - gate).reshape(1)],
            dim=0,
        )

    def _selected_composition_weights(self, selected, reference):
        if getattr(self, 'learned_transfer_gate_enabled', False):
            return self._learned_transfer_gate_weights(selected, reference)
        if getattr(self, 'use_raw_cosine_fixed_composition', False):
            return self._raw_cosine_fixed_composition_weights(
                selected,
                reference,
            )
        if getattr(self, 'use_sparsemax_fixed_composition', False):
            return self._sparsemax_fixed_composition_weights(selected, reference)
        return _linear_comb_beta_weights(self, selected + [self.task])

    @torch.no_grad()
    def learned_transfer_gate_value(self):
        if (
            not getattr(self, 'learned_transfer_gate_enabled', False)
            or self.learned_transfer_gate is None
        ):
            return None
        reference = next(self.learned_transfer_gate.parameters())
        features = self.learned_transfer_gate_features
        if features is None:
            features = reference.new_zeros(4)
        else:
            features = features.to(device=reference.device, dtype=reference.dtype)
        return float(
            torch.sigmoid(self.learned_transfer_gate(features).reshape(())).item()
        )

    def _normalize_mask_score(self, score):
        mode = getattr(
            self,
            'mask_score_normalization',
            MASK_SCORE_NORMALIZATION_NONE,
        )
        eps = getattr(self, 'mask_score_norm_eps', 1e-8)
        if mode == MASK_SCORE_NORMALIZATION_NONE:
            return score
        if mode == MASK_SCORE_NORMALIZATION_L2:
            return score / score.norm(p=2).clamp_min(eps)
        if mode == MASK_SCORE_NORMALIZATION_L2_STE:
            # Use unit-L2 scores in the forward pass while preserving the
            # identity gradient with respect to the trainable raw score.
            norm = score.detach().norm(p=2).clamp_min(eps)
            normalized = score / norm
            return score + (normalized - score).detach()
        if mode == MASK_SCORE_NORMALIZATION_RMS_STE:
            # Canonicalize score scale in the forward pass without changing
            # the current mask's raw-score gradient parameterization.
            rms = score.detach().square().mean().add(eps).sqrt()
            normalized = score / rms
            return score + (normalized - score).detach()
        if mode == MASK_SCORE_NORMALIZATION_MATCH_CURRENT_L2:
            # This mode needs the current score as its scale reference and is
            # therefore handled jointly by _composition_scores.
            return score
        if mode == MASK_SCORE_NORMALIZATION_STD:
            return score / score.std(unbiased=False).clamp_min(eps)
        if mode == MASK_SCORE_NORMALIZATION_ZSCORE:
            std = score.std(unbiased=False).clamp_min(eps)
            return (score - score.mean()) / std
        if mode == MASK_SCORE_NORMALIZATION_SIGN:
            return score.sign()
        raise ValueError('Unknown mask_score_normalization: {0}'.format(mode))

    def _composition_scores(self, prior_scores, current_score):
        """Prepare prior/current scores for one mask composition.

        ``match_current_l2`` leaves the trainable current score unchanged and
        rescales each frozen prior to its detached L2 norm. This controls
        cross-mask scale without changing the current mask's gradient
        parameterization when retrieval starts or stops.
        """
        mode = getattr(
            self,
            'mask_score_normalization',
            MASK_SCORE_NORMALIZATION_NONE,
        )
        if mode == MASK_SCORE_NORMALIZATION_SOFTMAX_ACROSS_MASKS:
            if not prior_scores:
                return [], current_score
            eps = getattr(self, 'mask_score_norm_eps', 1e-8)
            temperature = max(
                float(getattr(self, 'mask_score_softmax_temperature', 1.0)),
                eps,
            )
            stacked = torch.stack(prior_scores + [current_score], dim=0)
            attention = torch.softmax(stacked / temperature, dim=0)
            attended_scores = attention * stacked
            return list(attended_scores[:-1].unbind(dim=0)), attended_scores[-1]
        if mode == MASK_SCORE_NORMALIZATION_MATCH_CURRENT_L2:
            eps = getattr(self, 'mask_score_norm_eps', 1e-8)
            current_norm = current_score.detach().norm(p=2).clamp_min(eps)
            prepared_priors = [
                score * (current_norm / score.detach().norm(p=2).clamp_min(eps))
                for score in prior_scores
            ]
            return prepared_priors, current_score

        prepared_priors = [
            self._composition_score(score, is_current=False)
            for score in prior_scores
        ]
        prepared_current = self._composition_score(
            current_score,
            is_current=True,
            has_active_priors=bool(prior_scores),
        )
        return prepared_priors, prepared_current

    def _composition_score(
        self,
        score,
        is_current=False,
        has_active_priors=True,
    ):
        mode = getattr(
            self,
            'mask_score_normalization',
            MASK_SCORE_NORMALIZATION_NONE,
        )
        if mode == MASK_SCORE_NORMALIZATION_NONE:
            return score
        if is_current and not getattr(
            self,
            'mask_score_normalize_current',
            False,
        ):
            return score
        if (
            is_current
            and not has_active_priors
            and getattr(
                self,
                'mask_score_skip_current_normalization_on_abstain',
                False,
            )
        ):
            return score
        return self._normalize_mask_score(score)


class MultitaskMaskLinear(MaskScoreNormalizationMixin, nn.Linear):
    def __init__(self, *args, discrete=True, num_tasks=1, new_mask_type=NEW_MASK_RANDOM, \
        bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self._init_mask_score_normalization()
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

        self.task = -1
        self.num_tasks_learned = 0
        self.use_uniform_betas = False
        self.new_mask_type = new_mask_type
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
            self.selected_task_indices = None
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetDiscrete if discrete else GetSubnetContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j])
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
        else:
            # Subnet forward pass (given task info in self.task)
            #subnet = self._subnet_class.apply(self.scores[self.task])
            subnet = self._forward_mask()
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task])

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]
        if self.task < self.num_tasks_learned:
            # this is a task that has been seen before (with established/trained mask).
            # fetch mask and use (either for eval or to continue training).
            return self._subnet_class.apply(_subnet)

        # otherwise, this is a new task. check if the first task
        if self.task == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet)

        if self.selected_task_indices is not None:
            # combine only the selected prior masks plus the current task mask
            selected = [idx for idx in self.selected_task_indices if idx < self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in selected],
                _subnet,
            )
            _subnets.append(_current_subnet)

            _betas = self._selected_composition_weights(selected, _subnet)

            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)
        else:
            # otherwise, a new task and it is not the first task. combine task mask with
            # masks from previous tasks.
            # note: should not update scores/masks from previous tasks. only update their coeffs/betas
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in range(self.task)],
                _subnet,
            )
            assert len(_subnets) > 0, 'an error occured'
            _betas = _linear_comb_beta_weights(self, slice(0, self.task + 1))
            _subnets.append(_current_subnet)
            assert len(_betas) == len(_subnets), 'an error ocurred'
            _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
            # element wise sum of various masks (weighted sum)
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)

    @torch.no_grad()
    def consolidate_mask(self):
        if self.new_mask_type == NEW_MASK_RANDOM:
            return
        if self.task <= 0:
            return
        if self.task < self.num_tasks_learned:
            # re-visiting a task that has been previously learnt
            # (no need to consolidate)
            return

        if self.selected_task_indices is not None:
            _subnet = self.scores[self.task]
            # combine only the selected prior masks plus the current task mask
            selected = [idx for idx in self.selected_task_indices if idx < self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in selected],
                _subnet,
            )
            _subnets.append(_current_subnet)

            _betas = self._selected_composition_weights(selected, _subnet)

            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            self.scores[self.task].data = _subnet_linear_comb.data
            return
        else:
            _subnet = self.scores[self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in range(self.task)],
                _subnet,
            )
            assert len(_subnets) > 0, 'an error occured'
            _betas = _linear_comb_beta_weights(self, slice(0, self.task + 1))
            _subnets.append(_current_subnet)
            assert len(_betas) == len(_subnets), 'an error ocurred'
            _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
            # element wise sum of various masks (weighted sum)
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            self.scores[self.task].data = _subnet_linear_comb.data
            return

    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks()
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                k = task + 1
                # MASK-LC
                self.betas.data[task, 0:k] = 1. / k
                #print(f'BETAS INIT: {self.betas}')

                # MASK-BLC
                # set the coeff for the new task to a fixed starting value of 0.25.
                # this is set to 0.25 for now. The remaining prob are shared equally
                # by previous task masks.
                #threshold_ = 0.25
                #remain_prob = 1. - threshold_
                #self.betas.data[task, k-1] = threshold_
                #self.betas.data[task, 0:k-1] = remain_prob / (k-1)

                #self.betas.data[task, 0:k] = torch.log(self.betas.data[task, 0:k])


class MultitaskLoRALinear(MaskScoreNormalizationMixin, nn.Linear):
    """Frozen linear layer with one fixed-rank LoRA adapter per task.

    In compositional mode, retrieved source deltas and the current delta use
    AMSC's fixed sparsemax coefficients. At the task boundary their weighted
    sum is projected back to ``rank`` with truncated SVD, preventing rank and
    storage growth across the curriculum.
    """

    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        num_tasks=1,
        compositional=False,
        bias=False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        self._init_mask_score_normalization()
        self.rank = int(rank)
        self.num_tasks = int(num_tasks)
        self.compositional = bool(compositional)
        self.task = -1
        self.num_tasks_learned = 0
        self.selected_task_indices = None

        self.weight.requires_grad_(False)
        signed_constant(self)
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for _ in range(self.num_tasks):
            a = nn.Parameter(torch.empty(self.rank, in_features))
            b = nn.Parameter(torch.zeros(out_features, self.rank))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_A.append(a)
            self.lora_B.append(b)

    def _delta(self, task):
        return self.lora_B[task] @ self.lora_A[task]

    def _effective_delta(self):
        current = self._delta(self.task)
        if (
            not self.compositional
            or self.task == 0
            or self.task < self.num_tasks_learned
        ):
            return current

        selected = [
            int(index)
            for index in (self.selected_task_indices or [])
            if 0 <= int(index) < self.task
        ]
        if not selected:
            return current
        weights = self._sparsemax_fixed_composition_weights(selected, current)
        deltas = [self._delta(index).detach() for index in selected] + [current]
        return torch.stack(
            [weight * delta for weight, delta in zip(weights, deltas)],
            dim=0,
        ).sum(dim=0)

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
        return F.linear(x, self.weight + self._effective_delta(), self.bias)

    @torch.no_grad()
    def consolidate_mask(self):
        if (
            not self.compositional
            or self.task <= 0
            or self.task < self.num_tasks_learned
        ):
            return
        delta = self._effective_delta()
        u, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
        kept = min(self.rank, singular_values.numel())
        root = singular_values[:kept].clamp_min(0).sqrt()
        b = u[:, :kept] * root.unsqueeze(0)
        a = root.unsqueeze(1) * vh[:kept, :]
        self.lora_A[self.task].zero_()
        self.lora_B[self.task].zero_()
        self.lora_A[self.task][:kept].copy_(a)
        self.lora_B[self.task][:, :kept].copy_(b)

    def cache_masks(self):
        return

    def set_task(self, task, new_task=False):
        self.task = int(task)

# Subnetwork forward from hidden networks
# Sparse mask (using edge-pop algorithm)
class GetSubnetSparseDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class GetSubnetSparseContinuous(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# may not work as well as the alternative above (which is the default choice)
class GetSubnetSparseContinuousV2:
    @staticmethod
    def apply(scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0

        return out

class MultitaskMaskLinearSparse(MaskScoreNormalizationMixin, nn.Linear):
    def __init__(self, *args, discrete=True, num_tasks=1, sparsity=0.5, \
        new_mask_type=NEW_MASK_RANDOM, bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self._init_mask_score_normalization()
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

        # sparsity for top k%, edge pop up algorithm
        self.sparsity = sparsity

        self.task = -1
        self.num_tasks_learned = 0
        self.use_uniform_betas = False
        self.new_mask_type = new_mask_type
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
            self.selected_task_indices = None
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetSparseDiscrete if discrete else GetSubnetSparseContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j], self.sparsity)
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
        else:
            # Subnet forward pass (given task info in self.task)
            #subnet = self._subnet_class.apply(self.scores[self.task], self.sparsity)
            subnet = self._forward_mask()
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task], self.sparsity)

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]
        if self.task < self.num_tasks_learned:
            # this is a task that has been seen before (with established/trained mask).
            # fetch mask and use (either for eval or to continue training).
            return self._subnet_class.apply(_subnet, self.sparsity)

        # otherwise, this is a new task. check if the first task
        if self.task == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet, self.sparsity)

        if getattr(self, 'selected_task_indices', None) is not None:
            selected = [idx for idx in self.selected_task_indices if idx < self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in selected],
                _subnet,
            )
            _subnets.append(_current_subnet)

            _betas = self._selected_composition_weights(selected, _subnet)

            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb, self.sparsity)

        # otherwise, a new task and it is not the first task. combine task mask with
        # masks from previous tasks.
        # note: should not update scores/masks from previous tasks. only update their coeffs/betas
        _subnets, _current_subnet = self._composition_scores(
            [self.scores[idx].detach() for idx in range(self.task)],
            _subnet,
        )
        assert len(_subnets) > 0, 'an error occured'
        _betas = _linear_comb_beta_weights(self, slice(0, self.task + 1))
        _subnets.append(_current_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return self._subnet_class.apply(_subnet_linear_comb, self.sparsity)

    @torch.no_grad()
    def consolidate_mask(self):
        if self.new_mask_type == NEW_MASK_RANDOM:
            return
        if self.task <= 0:
            return
        if self.task < self.num_tasks_learned:
            # re-visiting a task that has been previously learnt
            # (no need to consolidate)
            return
        _subnet = self.scores[self.task]
        if getattr(self, 'selected_task_indices', None) is not None:
            selected = [idx for idx in self.selected_task_indices if idx < self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in selected],
                _subnet,
            )
            _subnets.append(_current_subnet)

            _betas = self._selected_composition_weights(selected, _subnet)

            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            self.scores[self.task].data = _subnet_linear_comb.data
            return

        _subnets, _current_subnet = self._composition_scores(
            [self.scores[idx].detach() for idx in range(self.task)],
            _subnet,
        )
        assert len(_subnets) > 0, 'an error occured'
        _betas = _linear_comb_beta_weights(self, slice(0, self.task + 1))
        _subnets.append(_current_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        self.scores[self.task].data = _subnet_linear_comb.data
        return

    def __repr__(self):
        return f"MultitaskMaskLinearSparse({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks()
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                k = task + 1
                self.betas.data[task, 0:k] = 1. / k

# Utility functions
def set_model_task(model, task, verbose=False, new_task=False):
    for n, m in model.named_modules():
        if isinstance(m, (MultitaskMaskLinear, MultitaskMaskLinearSparse,
                          MultitaskMaskConv2D, MultitaskLoRALinear)):
            if verbose:
                print(f"=> Set task of {n} to {task}")
            m.set_task(task, new_task)

def cache_masks(model, verbose=False):
    for n, m in model.named_modules():
        if isinstance(m, (MultitaskMaskLinear, MultitaskMaskLinearSparse,
                          MultitaskMaskConv2D, MultitaskLoRALinear)):
            if verbose:
                print(f"=> Caching mask state for {n}")
            m.cache_masks()

def set_num_tasks_learned(model, num_tasks_learned, verbose=True):
    for n, m in model.named_modules():
        if isinstance(m, (MultitaskMaskLinear, MultitaskMaskLinearSparse,
                          MultitaskMaskConv2D, MultitaskLoRALinear)):
            if verbose:
                print(f"=> Setting learned tasks of {n} to {num_tasks_learned}")
            m.num_tasks_learned = num_tasks_learned

def set_mask_score_normalization(
    model,
    mode=MASK_SCORE_NORMALIZATION_NONE,
    normalize_current=False,
    skip_current_normalization_on_abstain=False,
    eps=1e-8,
    softmax_temperature=1.0,
    verbose=False,
):
    if mode not in MASK_SCORE_NORMALIZATION_MODES:
        raise ValueError(
            'Unknown mask score normalization mode {0}. Expected one of {1}'.format(
                mode,
                sorted(MASK_SCORE_NORMALIZATION_MODES),
            )
        )
    if softmax_temperature <= 0.0:
        raise ValueError('softmax_temperature must be positive')
    for n, m in model.named_modules():
        if isinstance(
            m,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse, MultitaskMaskConv2D),
        ):
            if verbose:
                print(
                    f"=> Setting mask score normalization of {n} to {mode} "
                    f"(normalize_current={normalize_current}, "
                    f"skip_current_normalization_on_abstain="
                    f"{skip_current_normalization_on_abstain}, eps={eps}, "
                    f"softmax_temperature={softmax_temperature})"
                )
            m.mask_score_normalization = mode
            m.mask_score_normalize_current = normalize_current
            m.mask_score_skip_current_normalization_on_abstain = (
                skip_current_normalization_on_abstain
            )
            m.mask_score_norm_eps = eps
            m.mask_score_softmax_temperature = softmax_temperature

def get_mask(model, task, raw_score=True):
    mask = {}
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or isinstance(m, MultitaskMaskConv2D):
            mask[n] = m.get_mask(task, raw_score)
    return mask

def set_mask(model, mask, task):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or isinstance(m, MultitaskMaskConv2D):
            m.set_mask(mask[n], task)

def consolidate_mask(model):
    for n, m in model.named_modules():
        if isinstance(m, (MultitaskMaskLinear, MultitaskMaskLinearSparse,
                          MultitaskMaskConv2D, MultitaskLoRALinear)):
            m.consolidate_mask()


@torch.no_grad()
def quantize_task_mask_scores(
    model,
    task,
    mode=MASK_STORAGE_QUANTIZATION_NONE,
    threshold_rms=0.1,
):
    """Quantize one consolidated task mask before adding it to the library.

    The stored levels are {-1, +1} for binary masks and {-1, 0, +1} for
    ternary masks. Ternary zeros are scores within ``threshold_rms`` times the
    layer-task tensor RMS. L2-STE composition makes the absolute level scale
    immaterial while retaining the sign and dead-zone structure.
    """
    if mode not in MASK_STORAGE_QUANTIZATION_MODES:
        raise ValueError(
            'Unknown mask storage quantization mode {0}. Expected one of {1}'.format(
                mode, sorted(MASK_STORAGE_QUANTIZATION_MODES)
            )
        )
    if threshold_rms < 0.0:
        raise ValueError('mask storage quantization threshold must be non-negative')
    if mode == MASK_STORAGE_QUANTIZATION_NONE:
        return []

    rows = []
    mask_types = (
        MultitaskMaskLinear,
        MultitaskMaskLinearSparse,
        MultitaskMaskConv2D,
    )
    for name, module in model.named_modules():
        if not isinstance(module, mask_types):
            continue
        scores = module.scores[int(task)].data
        rms = scores.square().mean().sqrt().clamp_min(1e-12)
        if mode == MASK_STORAGE_QUANTIZATION_BINARY:
            quantized = torch.where(
                scores > 0,
                torch.ones_like(scores),
                -torch.ones_like(scores),
            )
        else:
            threshold = float(threshold_rms) * rms
            quantized = torch.zeros_like(scores)
            quantized[scores > threshold] = 1.0
            quantized[scores < -threshold] = -1.0
        scores.copy_(quantized)
        rows.append(
            {
                'layer': name,
                'task': int(task),
                'mode': mode,
                'threshold_rms': float(threshold_rms),
                'negative_fraction': float((quantized < 0).float().mean()),
                'zero_fraction': float((quantized == 0).float().mean()),
                'positive_fraction': float((quantized > 0).float().mean()),
            }
        )
    return rows

def set_selected_task_indices(model, indices):
    for n, m in model.named_modules():
        if isinstance(m, (MultitaskMaskLinear, MultitaskMaskLinearSparse,
                          MultitaskMaskConv2D, MultitaskLoRALinear)):
            m.selected_task_indices = indices

def set_selected_task_weights(model, prior_weights):
    for _, module in model.named_modules():
        if isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse,
             MultitaskMaskConv2D, MultitaskLoRALinear),
        ):
            module.set_selected_task_weights(prior_weights)

def set_composition_support_size(model, support_size):
    for _, module in model.named_modules():
        if isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse,
             MultitaskMaskConv2D, MultitaskLoRALinear),
        ):
            module.set_composition_support_size(support_size)

def configure_learned_transfer_gate(
    model,
    enabled=True,
    initial_probability=0.5,
):
    """Enable a differentiable retrieval/current gate on every mask layer."""
    parameters = []
    for _, module in model.named_modules():
        if isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse,
             MultitaskMaskConv2D, MultitaskLoRALinear),
        ):
            parameters.extend(
                module.configure_learned_transfer_gate(
                    enabled=enabled,
                    initial_probability=initial_probability,
                )
            )
    return parameters

def set_learned_transfer_gate_context(model, prior_weights, features):
    for _, module in model.named_modules():
        if isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse, MultitaskMaskConv2D),
        ):
            module.set_learned_transfer_gate_context(prior_weights, features)

def get_learned_transfer_gate_values(model):
    values = {}
    for name, module in model.named_modules():
        if isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse, MultitaskMaskConv2D),
        ):
            value = module.learned_transfer_gate_value()
            if value is not None:
                values[name] = value
    return values

@torch.no_grad()
def initialize_retrieval_beta_logits(
    model,
    task,
    prior_weights,
    current_weight,
    prior_indices=None,
    initialize_current=True,
    eps=1e-8,
):
    """Initialize newly admitted beta logits from retrieval confidence masses.

    The first accepted support is initialized directly from the requested
    masses. On later support expansions, retained logits are left untouched and
    new logits are placed relative to their log-sum-exp, preserving the learned
    relative weighting of retained priors and the current mask. Returns
    parameter/index triples so callers can clear optimizer moments only for
    entries whose values changed.
    """
    task = int(task)
    prior_indices = (
        list(prior_weights)
        if prior_indices is None
        else [int(idx) for idx in prior_indices]
    )
    changed = []
    for _, module in model.named_modules():
        if not isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse, MultitaskMaskConv2D),
        ):
            continue
        betas = getattr(module, 'betas', None)
        if betas is None or task < 0 or task >= betas.shape[0]:
            continue

        valid_new = [
            prior_idx
            for prior_idx in prior_indices
            if 0 <= prior_idx < task and prior_idx < betas.shape[1]
        ]
        changed_indices = list(valid_new)
        if initialize_current:
            for prior_idx in valid_new:
                weight = max(float(prior_weights.get(prior_idx, 0.0)), eps)
                betas.data[task, prior_idx] = math.log(weight)
            if task < betas.shape[1]:
                betas.data[task, task] = math.log(
                    max(float(current_weight), eps)
                )
                changed_indices.append(task)
        elif valid_new:
            selected = [
                int(idx)
                for idx in (getattr(module, 'selected_task_indices', None) or [])
                if 0 <= int(idx) < task
            ]
            retained = [idx for idx in selected if idx not in valid_new]
            old_active = retained + [task]
            old_logits = betas.data[task, old_active]
            old_logsumexp = torch.logsumexp(old_logits, dim=0).item()
            new_mass = min(
                sum(
                    max(float(prior_weights.get(prior_idx, 0.0)), 0.0)
                    for prior_idx in valid_new
                ),
                1.0 - eps,
            )
            retained_mass = max(1.0 - new_mass, eps)
            for prior_idx in valid_new:
                weight = max(float(prior_weights.get(prior_idx, 0.0)), eps)
                betas.data[task, prior_idx] = (
                    old_logsumexp + math.log(weight / retained_mass)
                )

        if changed_indices:
            changed.append((betas, task, sorted(set(changed_indices))))
    return changed

def set_uniform_beta_composition(model, enabled=True):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or isinstance(m, MultitaskMaskConv2D):
            m.use_uniform_betas = bool(enabled)
            if enabled:
                m.use_sparsemax_fixed_composition = False
                m.use_raw_cosine_fixed_composition = False
            if getattr(m, 'betas', None) is not None:
                m.betas.requires_grad_(not bool(enabled))

def set_sparsemax_fixed_composition(model, enabled=True):
    for _, module in model.named_modules():
        if isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse,
             MultitaskMaskConv2D, MultitaskLoRALinear),
        ):
            module.use_sparsemax_fixed_composition = bool(enabled)
            if enabled:
                module.use_uniform_betas = False
                module.use_raw_cosine_fixed_composition = False
            if getattr(module, 'betas', None) is not None:
                module.betas.requires_grad_(not bool(enabled))

def set_raw_cosine_fixed_composition(model, enabled=True):
    for _, module in model.named_modules():
        if isinstance(
            module,
            (MultitaskMaskLinear, MultitaskMaskLinearSparse, MultitaskMaskConv2D),
        ):
            module.use_raw_cosine_fixed_composition = bool(enabled)
            if enabled:
                module.use_uniform_betas = False
                module.use_sparsemax_fixed_composition = False
            if getattr(module, 'betas', None) is not None:
                module.betas.requires_grad_(not bool(enabled))

# Multitask Model, a simple fully connected model in this case
class SampleMaskModel(nn.Module):
    def __init__(self, hidden_size, num_tasks):
        super().__init__()
        self.model = nn.Sequential(
            MultitaskMaskLinear(
                50,
                hidden_size,
                num_tasks=num_tasks,
                bias=False
            ),
            nn.ReLU(),
            MultitaskMaskLinear(
                hidden_size,
                hidden_size,
                num_tasks=num_tasks,
                bias=False
            ),
            nn.ReLU(),
            MultitaskMaskLinear(
                hidden_size,
                10,
                num_tasks=num_tasks,
                bias=False
            )
        )

    def forward(self, x):
        return self.model(x)

class MultitaskMaskConv2D(MaskScoreNormalizationMixin, nn.Conv2d):
    def __init__(self, *args, discrete=True, num_tasks=1, new_mask_type=NEW_MASK_RANDOM, \
        bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self._init_mask_score_normalization()
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

        self.task = -1
        self.num_tasks_learned = 0
        self.use_uniform_betas = False
        self.new_mask_type = new_mask_type
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
            self.selected_task_indices = None
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetDiscrete if discrete else GetSubnetContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j])
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
        else:
            # Subnet forward pass (given task info in self.task)
            #subnet = self._subnet_class.apply(self.scores[self.task])
            subnet = self._forward_mask()
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
        else:
            #subnet = GetSubnet.apply(self.scores[self.task])
            subnet = self._forward_mask()

        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task])

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]
        if self.task < self.num_tasks_learned:
            # this is a task that has been seen before (with established/trained mask).
            # fetch mask and use (either for eval or to continue training).
            return self._subnet_class.apply(_subnet)

        # otherwise, this is a new task. check if the first task
        if self.task == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet)

        if self.selected_task_indices is not None:
            # combine only the selected prior masks plus the current task mask
            selected = [idx for idx in self.selected_task_indices if idx < self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in selected],
                _subnet,
            )
            _subnets.append(_current_subnet)

            _betas = self._selected_composition_weights(selected, _subnet)

            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)
        else:
            # otherwise, a new task and it is not the first task. combine task mask with
            # masks from previous tasks.
            # note: should not update scores/masks from previous tasks. only update their coeffs/betas
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in range(self.task)],
                _subnet,
            )
            assert len(_subnets) > 0, 'an error occured'
            _betas = _linear_comb_beta_weights(self, slice(0, self.task + 1))
            _subnets.append(_current_subnet)
            assert len(_betas) == len(_subnets), 'an error ocurred'
            _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
            # element wise sum of various masks (weighted sum)
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)

    @torch.no_grad()
    def consolidate_mask(self):
        if self.new_mask_type == NEW_MASK_RANDOM:
            return
        if self.task <= 0:
            return
        if self.task < self.num_tasks_learned:
            # re-visiting a task that has been previously learnt
            # (no need to consolidate)
            return

        if self.selected_task_indices is not None:
            _subnet = self.scores[self.task]
            # combine only the selected prior masks plus the current task mask
            selected = [idx for idx in self.selected_task_indices if idx < self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in selected],
                _subnet,
            )
            _subnets.append(_current_subnet)

            _betas = self._selected_composition_weights(selected, _subnet)

            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            self.scores[self.task].data = _subnet_linear_comb.data
            return
        else:
            _subnet = self.scores[self.task]
            _subnets, _current_subnet = self._composition_scores(
                [self.scores[idx].detach() for idx in range(self.task)],
                _subnet,
            )
            assert len(_subnets) > 0, 'an error occured'
            _betas = _linear_comb_beta_weights(self, slice(0, self.task + 1))
            _subnets.append(_current_subnet)
            assert len(_betas) == len(_subnets), 'an error ocurred'
            _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
            # element wise sum of various masks (weighted sum)
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            self.scores[self.task].data = _subnet_linear_comb.data
            return

    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks()
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                k = task + 1
                self.betas.data[task, 0:k] = 1. / k
                #print(f'BETAS INIT: {self.betas}')
