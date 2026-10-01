import numpy as np
import pickle
import os
import json
import time
import datetime
import torch
from .torch_utils import *
from ..mask_modules import (
    get_learned_transfer_gate_values,
    initialize_retrieval_beta_logits,
    set_learned_transfer_gate_context,
    set_composition_support_size,
    set_selected_task_indices,
    set_selected_task_weights,
)

def _csv_float(value):
    if value is None:
        return 'nan'
    try:
        if not np.isfinite(value):
            return 'nan'
    except TypeError:
        return 'nan'
    return f'{float(value):.6f}'

def _should_log_parameter_histograms(config, iteration):
    if not getattr(config, 'log_parameter_histograms', False):
        return False
    interval = getattr(config, 'histogram_log_interval', None)
    if interval is None:
        interval = getattr(config, 'iteration_log_interval', 1)
    interval = int(interval)
    return interval > 0 and iteration % interval == 0

def _should_save_iteration_snapshots(config, iteration):
    if not getattr(config, 'save_iteration_snapshots', False):
        return False
    interval = getattr(config, 'iteration_snapshot_interval', None)
    if interval is None:
        interval = getattr(config, 'iteration_log_interval', 1)
    interval = int(interval)
    return interval > 0 and iteration % interval == 0


def _composition_networks(agent, include_target=False):
    yield agent.network
    target_network = getattr(agent, 'target_network', None)
    if (
        include_target
        and target_network is not None
        and target_network is not agent.network
    ):
        yield target_network


def _set_agent_selected_task_indices(agent, indices):
    include_target = bool(
        getattr(agent.config, 'selection_learned_gate', False)
    )
    for network in _composition_networks(agent, include_target=include_target):
        set_selected_task_indices(network, indices)


def _set_agent_selected_task_weights(agent, prior_weights):
    for network in _composition_networks(agent, include_target=True):
        set_selected_task_weights(network, prior_weights)


def _set_agent_composition_support_size(agent, support_size):
    for network in _composition_networks(agent, include_target=True):
        set_composition_support_size(network, support_size)


def _set_agent_learned_gate_context(agent, prior_weights, features):
    for network in _composition_networks(agent, include_target=True):
        set_learned_transfer_gate_context(network, prior_weights, features)


@torch.no_grad()
def _log_task_beta_composition(
    network,
    file_handle,
    learn_block_idx,
    task_idx,
):
    """Record the active composition weights at task completion."""
    lines = []
    for layer_name, module in network.named_modules():
        betas = getattr(module, 'betas', None)
        if betas is None or getattr(betas, 'ndim', 0) != 2:
            continue
        if task_idx >= betas.shape[0]:
            continue

        selected = getattr(module, 'selected_task_indices', None)
        if selected is None:
            selected = list(range(task_idx))
        else:
            selected = sorted({
                int(idx) for idx in selected if 0 <= int(idx) < task_idx
            })
        active_indices = selected + [task_idx]
        logits = betas[task_idx, active_indices]
        learned_gate_enabled = bool(
            getattr(module, 'learned_transfer_gate_enabled', False)
        )
        if learned_gate_enabled:
            weights = module._selected_composition_weights(
                selected,
                betas[task_idx],
            )
            logits = torch.log(weights.clamp_min(1e-8))
            composition_mode = 'learned_gate'
            transfer_gate = (
                module.learned_transfer_gate_value() if selected else np.nan
            )
        elif getattr(module, 'use_sparsemax_fixed_composition', False):
            weights = module._selected_composition_weights(
                selected,
                betas[task_idx],
            )
            logits = torch.log(weights.clamp_min(1e-8))
            composition_mode = 'sparsemax_fixed'
            transfer_gate = np.nan
        elif getattr(module, 'use_raw_cosine_fixed_composition', False):
            weights = module._selected_composition_weights(
                selected,
                betas[task_idx],
            )
            logits = torch.log(weights.clamp_min(1e-8))
            composition_mode = 'raw_cosine_fixed'
            transfer_gate = np.nan
        elif getattr(module, 'use_uniform_betas', False):
            weights = torch.ones_like(logits) / logits.numel()
            composition_mode = 'uniform_beta'
            transfer_gate = np.nan
        else:
            weights = torch.softmax(logits, dim=-1)
            composition_mode = 'beta'
            transfer_gate = np.nan
        effective_n = float(
            1.0 / torch.sum(weights.detach().float().pow(2)).item()
        )
        allocation_support_size = getattr(
            module,
            'composition_support_size',
            None,
        )
        if allocation_support_size is None:
            allocation_support_size = len(selected)
        if (
            composition_mode == 'sparsemax_fixed'
            and allocation_support_size > len(selected)
        ):
            composition_mode = 'sparsemax_fixed_top1_matched_mass'
        for component_idx, logit, weight in zip(
            active_indices,
            logits.detach().cpu().tolist(),
            weights.detach().cpu().tolist(),
        ):
            role = 'current' if component_idx == task_idx else 'prior'
            lines.append(
                f"{learn_block_idx},{task_idx},{layer_name},{component_idx},"
                f"{role},{float(logit):.8f},{float(weight):.8f},"
                f"{len(selected)},{effective_n:.8f},{composition_mode},"
                f"{_csv_float(transfer_gate)},{allocation_support_size}\n"
            )
    file_handle.writelines(lines)

def _sparsemax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    finite = np.isfinite(logits)
    if not finite.any():
        return np.zeros_like(logits, dtype=np.float64)

    z = np.full_like(logits, -np.inf, dtype=np.float64)
    z[finite] = logits[finite]
    z_finite = z[finite]
    z_sorted = np.sort(z_finite)[::-1]
    cssv = np.cumsum(z_sorted)
    ks = np.arange(1, z_sorted.size + 1)
    support = 1 + ks * z_sorted > cssv
    if not support.any():
        return np.zeros_like(logits, dtype=np.float64)

    k = ks[support][-1]
    tau = (cssv[k - 1] - 1.0) / k
    weights = np.zeros_like(logits, dtype=np.float64)
    weights[finite] = np.maximum(z_finite - tau, 0.0)
    return weights

def _standardize_np(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    standardized = np.full_like(values, -np.inf, dtype=np.float64)
    if not finite.any():
        return standardized, np.nan, np.nan
    mean = float(values[finite].mean())
    scale = max(float(values[finite].std()), 1e-3)
    standardized[finite] = (values[finite] - mean) / scale
    return standardized, mean, scale

def _l2_normalize_np(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    normalized = np.full_like(values, -np.inf, dtype=np.float64)
    if not finite.any():
        return normalized, 0.0, np.nan
    norm = float(np.linalg.norm(values[finite]))
    if norm <= eps:
        normalized[finite] = 0.0
        return normalized, 0.0, norm
    normalized[finite] = values[finite] / norm
    return normalized, 0.0, norm

def _normalize_similarity_for_selection(cfg, values):
    mode = getattr(cfg, 'selection_similarity_normalization', None)
    if mode is None:
        mode = 'zscore' if getattr(cfg, 'selection_normalize_similarities', True) else 'none'
    if mode == 'zscore':
        normalized, center, scale = _standardize_np(values)
    elif mode == 'l2':
        normalized, center, scale = _l2_normalize_np(values)
    elif mode == 'none':
        normalized = np.asarray(values, dtype=np.float64).copy()
        center = 0.0
        scale = 1.0
    else:
        raise ValueError(
            'selection_similarity_normalization must be one of: zscore, l2, none'
        )
    return normalized, center, scale, mode

def _normalize_competence_for_selection(cfg, value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value):
        return np.nan

    mode = getattr(cfg, 'selection_competence_normalization', 'clip01')
    if mode == 'none':
        return value
    if mode == 'clip01':
        return float(np.clip(value, 0.0, 1.0))
    raise ValueError(
        'selection_competence_normalization must be one of: clip01, none'
    )

def _selection_strategy(cfg):
    strategy = getattr(cfg, 'select_strategy', None)
    if strategy is None:
        strategy = getattr(cfg, 'selection_strategy', 'amsc')
    return str(strategy)


_ORACLE_TRANSFER_CONFIG_CACHE = {}


def _load_oracle_transfer_config(cfg):
    path = getattr(cfg, 'oracle_transfer_config', None)
    if not path:
        raise ValueError(
            "select_strategy='oracle_transfer' requires "
            'config.oracle_transfer_config'
        )
    path = os.path.abspath(os.path.expanduser(str(path)))
    if path not in _ORACLE_TRANSFER_CONFIG_CACHE:
        with open(path, 'r') as file_handle:
            payload = json.load(file_handle)
        targets = payload.get('targets')
        if not isinstance(targets, dict):
            raise ValueError(
                'oracle transfer config must contain a targets object: {0}'.format(
                    path
                )
            )
        _ORACLE_TRANSFER_CONFIG_CACHE[path] = payload
    return _ORACLE_TRANSFER_CONFIG_CACHE[path]


def _oracle_transfer_entry(cfg, task_idx):
    payload = _load_oracle_transfer_config(cfg)
    entry = payload['targets'].get(str(int(task_idx)), {})
    sources = [int(index) for index in entry.get('sources', [])]
    weights = [float(value) for value in entry.get('weights', [])]
    scores = [float(value) for value in entry.get('scores', weights)]
    if len(weights) != len(sources) or len(scores) != len(sources):
        raise ValueError(
            'oracle transfer entry {0} has inconsistent source, weight, or '
            'score lengths'.format(task_idx)
        )
    if len(set(sources)) != len(sources):
        raise ValueError(
            'oracle transfer entry {0} contains duplicate sources'.format(task_idx)
        )
    if any(index < 0 or index >= task_idx for index in sources):
        raise ValueError(
            'oracle transfer entry {0} must contain only causal source indices '
            'smaller than the target index'.format(task_idx)
        )
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError(
            'oracle transfer entry {0} contains invalid weights'.format(task_idx)
        )
    weight_sum = float(np.sum(weights))
    if sources and weight_sum <= 0.0:
        raise ValueError(
            'oracle transfer entry {0} has no positive weight'.format(task_idx)
        )
    normalized_weights = (
        [value / weight_sum for value in weights] if sources else []
    )
    return sources, normalized_weights, scores

def _reset_beta_optimizer_moments(agent, changed_entries):
    """Clear optimizer moments for beta entries changed by retrieval initialization."""
    optimizers = []
    for attr in ('opt', 'actor_opt', 'critic_opt'):
        optimizer = getattr(agent, attr, None)
        if optimizer is not None and optimizer not in optimizers:
            optimizers.append(optimizer)

    for parameter, row, columns in changed_entries:
        for optimizer in optimizers:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            for value in state.values():
                if not torch.is_tensor(value) or value.shape != parameter.shape:
                    continue
                value[row, columns] = 0

def _oracle_prior_indices(cfg, task_idx):
    strategy = _selection_strategy(cfg)
    if strategy == 'amsc':
        return None
    if strategy == 'oracle_transfer':
        sources, _weights, _scores = _oracle_transfer_entry(cfg, task_idx)
        return sources
    if strategy not in ['oracle_all', 'oracle_depth_prefix', 'oracle_parent']:
        raise ValueError(
            'select_strategy must be one of: amsc, oracle_all, '
            'oracle_depth_prefix, oracle_parent, oracle_transfer'
        )

    family_stride = getattr(cfg, 'family_stride', None)
    if family_stride is None:
        raise ValueError(
            "select_strategy='{0}' requires config.family_stride".format(strategy)
        )
    family_stride = int(family_stride)
    if family_stride <= 0:
        raise ValueError('config.family_stride must be a positive integer')

    same_family = [
        int(idx)
        for idx in range(task_idx)
        if idx % family_stride == task_idx % family_stride
    ]
    if strategy == 'oracle_parent':
        return same_family[-1:] if same_family else []
    return same_family

def _oracle_selection_details(
    cfg,
    task_idx,
    sims,
    emb_indices,
    selected,
    prior_competence=None,
):
    candidate_selected = [int(idx) for idx in selected]
    weighting = str(getattr(cfg, 'oracle_weighting', 'uniform'))
    if weighting not in ['uniform', 'similarity', 'transfer']:
        raise ValueError(
            'oracle_weighting must be one of: uniform, similarity, transfer'
        )

    sim_lookup = {}
    if sims is not None:
        sims_cpu = sims.detach().float().cpu().numpy()
        sim_lookup = {
            int(prev_idx): float(sims_cpu[row_idx])
            for row_idx, prev_idx in enumerate(emb_indices)
            if int(prev_idx) < task_idx
        }
    candidates = [
        (idx, sim_lookup.get(idx, np.nan))
        for idx in range(task_idx)
    ]

    weight_by_index = {}
    score_by_index = {}
    similarity_mean = np.nan
    similarity_scale = np.nan
    normalization_enabled = False
    normalization_name = '{0}:{1}'.format(_selection_strategy(cfg), weighting)

    if weighting == 'uniform':
        selected = candidate_selected
        selected_weight = 1.0 / len(selected) if selected else 0.0
        weight_by_index = {idx: selected_weight for idx in selected}
        score_by_index = {
            idx: sim_lookup.get(idx, np.nan) for idx in selected
        }
    elif weighting == 'similarity':
        available = [
            (idx, sim_lookup.get(idx, np.nan))
            for idx in candidate_selected
            if np.isfinite(sim_lookup.get(idx, np.nan))
        ]
        if available:
            indices = [idx for idx, _similarity in available]
            similarities = np.asarray(
                [similarity for _idx, similarity in available],
                dtype=np.float64,
            )
            normalized, similarity_mean, similarity_scale, mode = (
                _normalize_similarity_for_selection(cfg, similarities)
            )
            temperature = max(
                float(getattr(cfg, 'selection_soft_temperature', 1.0)),
                1e-6,
            )
            sparse_weights = _sparsemax_np(normalized / temperature)
            selected = [
                idx for idx, weight in zip(indices, sparse_weights)
                if weight > 0.0
            ]
            weight_by_index = {
                idx: float(weight)
                for idx, weight in zip(indices, sparse_weights)
                if weight > 0.0
            }
            score_by_index = {
                idx: float(score)
                for idx, score in zip(indices, normalized / temperature)
            }
            normalization_enabled = mode != 'none'
            normalization_name = '{0}:similarity_{1}'.format(
                _selection_strategy(cfg), mode
            )
        else:
            selected = []
    else:
        if _selection_strategy(cfg) != 'oracle_transfer':
            raise ValueError(
                "oracle_weighting='transfer' requires "
                "select_strategy='oracle_transfer'"
            )
        transfer_sources, transfer_weights, transfer_scores = (
            _oracle_transfer_entry(cfg, task_idx)
        )
        selected = transfer_sources
        weight_by_index = dict(zip(transfer_sources, transfer_weights))
        score_by_index = dict(zip(transfer_sources, transfer_scores))

    selected_set = set(selected)

    details = {}
    for prev_idx, sim_val in candidates:
        prior_perf = np.nan
        if prior_competence is not None and prev_idx < len(prior_competence):
            prior_perf = _normalize_competence_for_selection(
                cfg,
                prior_competence[prev_idx],
            )
        weight = weight_by_index.get(prev_idx, 0.0)
        score = score_by_index.get(prev_idx, sim_val)
        details[int(prev_idx)] = {
            'weight': float(weight),
            'pre_shuffle_weight': float(weight),
            'score': float(score) if np.isfinite(score) else np.nan,
            'selection_similarity': float(sim_val) if np.isfinite(sim_val) else np.nan,
            'similarity_mean': similarity_mean,
            'similarity_scale': similarity_scale,
            'normalization_enabled': normalization_enabled,
            'similarity_normalization': normalization_name,
            'shuffled_support': False,
            'prior_competence': float(prior_perf) if np.isfinite(prior_perf) else np.nan,
            'competence_eligible': True,
            'competence_score': float(prior_perf) if np.isfinite(prior_perf) else np.nan,
            'competence_weight': 1.0,
            'competence_selected': True,
            'competence_mean': np.nan,
            'competence_scale': np.nan,
            'competence_floor': float(getattr(cfg, 'selection_competence_floor', 0.0)),
            'competence_gate_disabled': True,
            'conditional_weight': float(weight),
        }
    return selected, details

def _sparsemax_similarity_selection(
    cfg,
    task_idx,
    sims,
    emb_indices,
    rng=None,
    prior_competence=None,
):
    """Select prior masks from sparsemax support over task similarities.

    The returned sparsemax weights define relative relevance over prior masks.
    When the optional hard gate is enabled, their similarity-weighted sum q
    decides whether the entire prior support is admitted. The current task
    mask is not part of this gate; it is always appended inside the
    linear-combination mask layers.

    By default, per-event standardization makes temperature operate on relative
    similarity gaps instead of the raw cosine scale. L2 and raw-cosine modes
    are available as ablations. The competence gate is enabled when
    prior_competence is provided unless --selection_disable_competence_gate is
    set. It is a floor-based admissibility check over normalized own-task
    performance, not a second sparsemax selector. The Shuffled ablation
    preserves the similarity sparsemax support size and weight multiset but
    randomly assigns them to competence-eligible prior tasks.
    """
    if sims is None:
        return [], {}

    sims_cpu = sims.detach().float().cpu().numpy()
    candidates = []
    for row_idx, prev_idx in enumerate(emb_indices):
        if prev_idx >= task_idx:
            continue
        sim_val = float(sims_cpu[row_idx])
        if np.isfinite(sim_val):
            candidates.append((prev_idx, sim_val))

    if not candidates:
        return [], {}

    temperature = max(float(getattr(cfg, 'selection_soft_temperature', 1.0)), 1e-6)
    disable_competence_gate = bool(
        getattr(cfg, 'selection_disable_competence_gate', False)
    ) or prior_competence is None
    competence_floor = float(getattr(cfg, 'selection_competence_floor', 0.0))
    shuffle_support = bool(getattr(cfg, 'selection_shuffle_support', False))
    hard_gate_enabled = bool(getattr(cfg, 'selection_hard_gate', False))
    hard_gate_threshold = float(
        getattr(cfg, 'selection_hard_gate_threshold', 2.0 ** -0.5)
    )
    beta_init_gamma = float(getattr(cfg, 'selection_beta_init_gamma', 2.0))
    if not 0.0 <= hard_gate_threshold <= 1.0:
        raise ValueError('selection_hard_gate_threshold must be in [0, 1]')
    if beta_init_gamma <= 0.0:
        raise ValueError('selection_beta_init_gamma must be positive')
    prior_indices = [idx for idx, _sim in candidates]
    sim_values = np.asarray([sim for _idx, sim in candidates], dtype=np.float64)
    gate_similarity_max = float(np.max(sim_values))
    gate_similarity_mean = float(np.mean(sim_values))
    gate_similarity_std = float(np.std(sim_values))

    selection_similarities, similarity_mean, similarity_scale, normalization_mode = (
        _normalize_similarity_for_selection(cfg, sim_values)
    )
    normalize_similarities = normalization_mode != 'none'

    competence_values = np.asarray(
        [
            _normalize_competence_for_selection(
                cfg,
                prior_competence[idx] if prior_competence is not None else np.nan,
            )
            for idx in prior_indices
        ],
        dtype=np.float64,
    )
    competence_eligible = np.ones_like(competence_values, dtype=bool)
    if not disable_competence_gate:
        competence_eligible = (
            np.isfinite(competence_values)
            & (competence_values > competence_floor)
        )

    sim_logits = np.full_like(selection_similarities, -np.inf, dtype=np.float64)
    sim_logits[competence_eligible] = (
        selection_similarities[competence_eligible] / temperature
    )
    pre_shuffle_weights = _sparsemax_np(sim_logits)
    sparse_weights = pre_shuffle_weights.copy()
    if shuffle_support and sparse_weights.size > 1:
        if rng is None:
            raise ValueError(
                'selection_shuffle_support requires a dedicated random generator'
            )
        shuffled_values = sparse_weights[competence_eligible]
        shuffled_values = shuffled_values[rng.permutation(shuffled_values.size)]
        sparse_weights = np.zeros_like(sparse_weights, dtype=np.float64)
        sparse_weights[competence_eligible] = shuffled_values

    clipped_similarities = np.clip(sim_values, 0.0, 1.0)
    raw_absolute_weights = sparse_weights * clipped_similarities
    absolute_confidence_q = float(
        np.clip(raw_absolute_weights.sum(), 0.0, 1.0)
    )
    if absolute_confidence_q > 1e-8:
        conditional_weights = raw_absolute_weights / absolute_confidence_q
    else:
        conditional_weights = np.zeros_like(raw_absolute_weights)

    hard_gate_reuse = (
        (not hard_gate_enabled)
        or absolute_confidence_q > hard_gate_threshold
    )
    beta_init_prior_mass = float(absolute_confidence_q ** beta_init_gamma)
    beta_init_current_mass = 1.0 - beta_init_prior_mass
    beta_init_weights = beta_init_prior_mass * conditional_weights
    if not hard_gate_reuse:
        sparse_weights = np.zeros_like(sparse_weights)
        beta_init_weights = np.zeros_like(beta_init_weights)
        beta_init_prior_mass = 0.0
        beta_init_current_mass = 1.0

    competence_weights = np.where(competence_eligible, 1.0, 0.0)
    competence_scores = competence_values.copy()

    selected = [
        int(idx)
        for idx, sim_weight, eligible in zip(
            prior_indices,
            sparse_weights,
            competence_eligible,
        )
        if sim_weight > 0.0 and eligible
    ]
    details = {
        int(idx): {
            'weight': float(weight if eligible else 0.0),
            'pre_shuffle_weight': float(pre_shuffle_weight),
            'score': float(score),
            'selection_similarity': float(selection_similarity),
            'similarity_mean': similarity_mean,
            'similarity_scale': similarity_scale,
            'normalization_enabled': normalize_similarities,
            'similarity_normalization': normalization_mode,
            'shuffled_support': shuffle_support,
            'prior_competence': float(competence),
            'competence_eligible': bool(eligible),
            'competence_score': float(competence_score),
            'competence_weight': float(comp_weight),
            'competence_selected': bool(eligible),
            'competence_mean': np.nan,
            'competence_scale': np.nan,
            'competence_floor': competence_floor,
            'competence_gate_disabled': disable_competence_gate,
            'absolute_confidence_q': absolute_confidence_q,
            'absolute_weight': float(absolute_weight),
            'conditional_weight': float(conditional_weight),
            'gate_similarity_max': gate_similarity_max,
            'gate_similarity_mean': gate_similarity_mean,
            'gate_similarity_std': gate_similarity_std,
            'hard_gate_enabled': hard_gate_enabled,
            'hard_gate_threshold': hard_gate_threshold,
            'hard_gate_reuse': hard_gate_reuse,
            'beta_init_weight': float(beta_init_weight),
            'beta_init_prior_mass': beta_init_prior_mass,
            'beta_init_current_mass': beta_init_current_mass,
            'beta_init_gamma': beta_init_gamma,
        }
        for (
            idx,
            weight,
            pre_shuffle_weight,
            score,
            selection_similarity,
            competence,
            eligible,
            competence_score,
            comp_weight,
            absolute_weight,
            conditional_weight,
            beta_init_weight,
        ) in zip(
            prior_indices,
            sparse_weights,
            pre_shuffle_weights,
            sim_logits,
            selection_similarities,
            competence_values,
            competence_eligible,
            competence_scores,
            competence_weights,
            raw_absolute_weights,
            conditional_weights,
            beta_init_weights,
        )
    }
    return selected, details


def _top1_with_matched_prior_mass(cfg, selected, details):
    """Retain the strongest source without changing prior/current mass.

    The returned support size records how many sources sparsemax originally
    selected. Fixed sparsemax composition uses that size to preserve the
    baseline allocation of m/(m+1) to priors and 1/(m+1) to the current mask.
    """
    retrieved_support_size = len(selected)
    if (
        not getattr(cfg, 'selection_top1_matched_prior_mass', False)
        or retrieved_support_size <= 1
    ):
        return list(selected), retrieved_support_size

    top_source = max(
        selected,
        key=lambda idx: (float(details[idx].get('weight', 0.0)), -int(idx)),
    )
    return [int(top_source)], retrieved_support_size

def _itr_log(logger, agent, iteration, dict_logs):
    logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
        iteration, agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards)
    ))
    logger.scalar_summary('last_episode_reward/avg', np.mean(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/std', np.std(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/max', np.max(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/min', np.min(agent.last_episode_rewards))
    logger.scalar_summary('iteration_reward/avg', np.mean(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/std', np.std(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/max', np.max(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/min', np.min(agent.iteration_rewards))

    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('debug/{0}_avg_norm'.format(tag), np.mean(value_norm))
            logger.scalar_summary('debug/{0}_avg'.format(tag), value.mean())
            logger.scalar_summary('debug/{0}_std'.format(tag), value.std())
            logger.scalar_summary('debug/{0}_max'.format(tag), value.max())
            logger.scalar_summary('debug/{0}_min'.format(tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('debug_extended/{0}_avg'.format(key), np.mean(value))
        logger.scalar_summary('debug_extended/{0}_std'.format(key), np.std(value))
        logger.scalar_summary('debug_extended/{0}_max'.format(key), np.max(value))
        logger.scalar_summary('debug_extended/{0}_min'.format(key), np.min(value))

    return

# metaworld/continualworld
def _itr_log_mw(logger, agent, iteration, dict_logs):
    logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f, ' \
        'mean/max/min success rate %f/%f/%f'%(
        iteration, agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards),
        np.mean(agent.iteration_success_rate),
        np.max(agent.iteration_success_rate),
        np.min(agent.iteration_success_rate)
    ))
    logger.scalar_summary('last_episode_reward/avg', np.mean(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/std', np.std(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/max', np.max(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/min', np.min(agent.last_episode_rewards))
    logger.scalar_summary('iteration_reward/avg', np.mean(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/std', np.std(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/max', np.max(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/min', np.min(agent.iteration_rewards))

    logger.scalar_summary('last_episode_success_rate/avg', np.mean(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/std', np.std(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/max', np.max(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/min', np.min(agent.last_episode_success_rate))
    logger.scalar_summary('iteration_success_rate/avg', np.mean(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/std', np.std(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/max', np.max(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/min', np.min(agent.iteration_success_rate))

    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('debug/{0}_avg_norm'.format(tag), np.mean(value_norm))
            logger.scalar_summary('debug/{0}_avg'.format(tag), value.mean())
            logger.scalar_summary('debug/{0}_std'.format(tag), value.std())
            logger.scalar_summary('debug/{0}_max'.format(tag), value.max())
            logger.scalar_summary('debug/{0}_min'.format(tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('debug_extended/{0}_avg'.format(key), np.mean(value))
        logger.scalar_summary('debug_extended/{0}_std'.format(key), np.std(value))
        logger.scalar_summary('debug_extended/{0}_max'.format(key), np.max(value))
        logger.scalar_summary('debug_extended/{0}_min'.format(key), np.min(value))

    return

# run iterations, lifelong learning
# used by either a baseline agent (with no task knowledge preservation) or
# an agent with knowledge preservation via supermask superposition (ss)
# modules on: PPO agent or PPO agent with supermask
# modules off: detect and resource manager
def run_iterations_w_oracle(agent, tasks_info):
    config = agent.config

    log_path_tstats = config.log_dir + '/task_stats'
    if not os.path.exists(log_path_tstats):
        os.makedirs(log_path_tstats)
    log_path_eval = config.log_dir + '/eval_stats'
    if not os.path.exists(log_path_eval):
        os.makedirs(log_path_eval)
    random_seed(config.seed)
    agent_name = agent.__class__.__name__

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    num_tasks = len(tasks_info)
    # track how many times each task selected each prior task (for post-run summaries)
    selection_counts = [[0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    selection_rng = np.random.RandomState(int(config.seed) + 104729)
    prior_competence = np.full(num_tasks, np.nan, dtype=np.float64)
    eval_data_fh = open(config.logger.log_dir + '/eval_metrics.csv', 'a', buffering=1)
    sims_csv_path = config.logger.log_dir + '/task_similarities.csv'
    sims_csv_fh = open(sims_csv_path, 'w', buffering=1)
    sims_csv_fh.write(
        'learn_block,task_idx,iteration,total_steps,prev_idx,similarity,'
        'pre_shuffle_selected,selected,pre_shuffle_weight,weight,score,'
        'temperature,normalization_enabled,similarity_normalization,shuffled_support,'
        'prior_competence,competence_eligible,competence_selected,'
        'competence_weight,competence_score,'
        'competence_floor,competence_gate_disabled,'
        'absolute_confidence_q,hard_gate_enabled,hard_gate_threshold,'
        'hard_gate_reuse,beta_init_weight,beta_init_prior_mass,'
        'beta_init_current_mass,beta_init_gamma,conditional_weight,'
        'gate_similarity_max,gate_similarity_mean,gate_similarity_std,'
        'learned_gate_enabled,learned_gate_mean\n'
    )
    beta_csv_path = config.logger.log_dir + '/beta_composition.csv'
    beta_csv_fh = open(beta_csv_path, 'w', buffering=1)
    beta_csv_fh.write(
        'learn_block,task_idx,layer,component_idx,component_role,beta_logit,'
        'beta_weight,selected_prior_count,effective_n,composition_mode,'
        'transfer_gate,allocation_support_size\n'
    )

    eval_data = []
    metric_icr = [] # icr => total cumulative reward

    if agent.task.name == config.ENV_METAWORLD or agent.task.name == config.ENV_CONTINUALWORLD:
        itr_log_fn = _itr_log_mw
    else:
        itr_log_fn = _itr_log

    for learn_block_idx in range(config.cl_num_learn_blocks):
        config.logger.info('********** start of learning block {0}'.format(learn_block_idx))
        eval_results = {task_idx:[] for task_idx in range(len(tasks_info))}

        for task_idx, task_info in enumerate(tasks_info):
            config.logger.info('*****start training on task {0}'.format(task_idx))
            config.logger.info('name: {0}'.format(task_info['name']))
            config.logger.info('task: {0}'.format(task_info['task']))
            config.logger.info('task_label: {0}'.format(task_info['task_label']))

            states = agent.task.reset_task(task_info)
            agent.states = config.state_normalizer(states)
            agent.data_buffer.clear()
            agent.task_train_start(task_info['task_label'])
            initial_oracle_selected = _oracle_prior_indices(config, task_idx)
            initial_weight_details = {}
            if initial_oracle_selected is None:
                initial_oracle_selected = []
            else:
                initial_oracle_selected, initial_weight_details = (
                    _oracle_selection_details(
                        config,
                        task_idx,
                        sims=None,
                        emb_indices=[],
                        selected=initial_oracle_selected,
                        prior_competence=prior_competence,
                    )
                )
            _set_agent_selected_task_indices(agent, initial_oracle_selected)
            _set_agent_composition_support_size(
                agent,
                len(initial_oracle_selected),
            )
            _set_agent_selected_task_weights(
                agent,
                {
                    idx: initial_weight_details[idx].get('weight', 0.0)
                    for idx in initial_oracle_selected
                },
            )
            if getattr(config, 'selection_learned_gate', False):
                _set_agent_learned_gate_context(
                    agent,
                    prior_weights={},
                    features=[0.0, 0.0, 0.0, 0.0],
                )
            retrieval_beta_initialized_priors = set()
            retrieval_beta_current_initialized = False
            retrieval_frozen = False

            while True:
                # ---- agent iteration ----
                dict_logs = agent.iteration()
                iteration += 1

                total_steps = agent.total_steps
                steps.append(total_steps)
                rewards.append(float(np.mean(agent.iteration_rewards)))

                # ---- locals to reduce attribute overhead
                cfg = agent.config
                detect_freq = getattr(cfg, "detect_frequency", 0) or 0
                select_freq = getattr(cfg, "select_frequency", 0) or 0

                # ---- detect / embedding update ----
                if hasattr(agent, "detect"):
                    if detect_freq and iteration % detect_freq == 0 and agent.data_buffer.size() >= (agent.detect.get_num_samples()):
                        # Extract the configured SAR embedding batch.
                        sar_data = agent.extract_sar(batch_size=config.detect_num_samples)

                        # Update
                        new_embedding = agent.compute_task_embedding(sar_data, agent.task.action_dim)
                        agent._update_embedding(
                            task_idx=task_idx,
                            new_emb=new_embedding,
                            ema=float(getattr(cfg, 'embedding_ema', 0.5)),
                        )

                # ---- selection step ----
                should_select = (
                    iteration
                    and select_freq
                    and (iteration % select_freq == 0)
                    and task_idx > 0
                    and not retrieval_frozen
                )
                if should_select:
                    oracle_selected = _oracle_prior_indices(cfg, task_idx)
                    _, sims = agent.select_similar(
                        task_idx=task_idx,
                        threshold=-float("inf"),
                        topk=None,
                    )
                    if oracle_selected is not None:
                        selected, weight_details = _oracle_selection_details(
                            cfg,
                            task_idx,
                            sims,
                            agent._emb_indices,
                            oracle_selected,
                            prior_competence=prior_competence,
                        )
                        retrieved_support_size = len(selected)
                        _set_agent_selected_task_indices(agent, selected)
                        _set_agent_composition_support_size(
                            agent,
                            retrieved_support_size,
                        )
                        _set_agent_selected_task_weights(
                            agent,
                            {
                                idx: weight_details[idx].get('weight', 0.0)
                                for idx in selected
                            },
                        )
                        for idx in selected:
                            selection_counts[task_idx][idx] += 1
                    elif sims is not None:
                        selected, weight_details = _sparsemax_similarity_selection(
                            cfg,
                            task_idx,
                            sims,
                            agent._emb_indices,
                            rng=selection_rng,
                            prior_competence=prior_competence,
                        )
                        selected, retrieved_support_size = (
                            _top1_with_matched_prior_mass(
                                cfg,
                                selected,
                                weight_details,
                            )
                        )
                        _set_agent_selected_task_indices(agent, selected)
                        _set_agent_composition_support_size(
                            agent,
                            retrieved_support_size,
                        )
                        _set_agent_selected_task_weights(
                            agent,
                            {
                                idx: weight_details[idx].get(
                                    'absolute_weight'
                                    if getattr(
                                        cfg,
                                        'selection_raw_cosine_fixed_composition',
                                        False,
                                    )
                                    else 'weight',
                                    0.0,
                                )
                                for idx in selected
                            },
                        )
                        if getattr(cfg, 'selection_learned_gate', False):
                            detail_values = list(weight_details.values())
                            if detail_values:
                                gate_detail = detail_values[0]
                                gate_prior_weights = {
                                    idx: weight_details[idx].get(
                                        'conditional_weight',
                                        0.0,
                                    )
                                    for idx in selected
                                }
                                gate_features = [
                                    gate_detail.get('absolute_confidence_q', 0.0),
                                    gate_detail.get('gate_similarity_max', 0.0),
                                    gate_detail.get('gate_similarity_mean', 0.0),
                                    gate_detail.get('gate_similarity_std', 0.0),
                                ]
                                _set_agent_learned_gate_context(
                                    agent,
                                    prior_weights=gate_prior_weights,
                                    features=gate_features,
                                )
                        if bool(
                            getattr(cfg, 'selection_retrieval_beta_init', False)
                        ) and selected:
                            new_priors = [
                                idx for idx in selected
                                if idx not in retrieval_beta_initialized_priors
                            ]
                            initialize_current = (
                                not retrieval_beta_current_initialized
                            )
                            if new_priors or initialize_current:
                                prior_weights = {
                                    idx: weight_details[idx].get(
                                        'beta_init_weight',
                                        0.0,
                                    )
                                    for idx in selected
                                }
                                current_weight = next(
                                    iter(weight_details.values())
                                ).get('beta_init_current_mass', 1.0)
                                changed = initialize_retrieval_beta_logits(
                                    agent.network,
                                    task_idx,
                                    prior_weights,
                                    current_weight,
                                    prior_indices=new_priors,
                                    initialize_current=initialize_current,
                                )
                                _reset_beta_optimizer_moments(agent, changed)
                                retrieval_beta_initialized_priors.update(
                                    new_priors
                                )
                                retrieval_beta_current_initialized = True
                        for idx in selected:
                            selection_counts[task_idx][idx] += 1
                    else:
                        continue

                    selected_set = set(selected)
                    sparsemax_list = []
                    if sims is None:
                        sims_cpu = [np.nan for _idx in range(task_idx)]
                        emb_indices = list(range(task_idx))
                    else:
                        sims_cpu = sims.detach().float().cpu().tolist()
                        emb_indices = agent._emb_indices
                    lines = []
                    temperature = float(getattr(cfg, 'selection_soft_temperature', 1.0))
                    normalization_enabled = bool(
                        getattr(cfg, 'selection_normalize_similarities', True)
                    )
                    shuffled_support = bool(
                        getattr(cfg, 'selection_shuffle_support', False)
                    )
                    similarity_normalization = getattr(
                        cfg,
                        'selection_similarity_normalization',
                        'zscore' if normalization_enabled else 'none',
                    )
                    learned_gate_enabled = bool(
                        getattr(cfg, 'selection_learned_gate', False)
                    )
                    learned_gate_values = get_learned_transfer_gate_values(
                        agent.network
                    )
                    learned_gate_mean = (
                        float(np.mean(list(learned_gate_values.values())))
                        if learned_gate_values else np.nan
                    )
                    normalization_mean = np.nan
                    normalization_scale = np.nan
                    for sim_val, prev_idx in zip(sims_cpu, emb_indices):
                        if prev_idx >= task_idx:
                            continue
                        detail = weight_details.get(
                            prev_idx,
                            {
                                'weight': 0.0,
                                'pre_shuffle_weight': 0.0,
                                'score': np.nan,
                                'similarity_mean': np.nan,
                                'similarity_scale': np.nan,
                                'normalization_enabled': normalization_enabled,
                                'similarity_normalization': similarity_normalization,
                                'shuffled_support': shuffled_support,
                                'prior_competence': np.nan,
                                'competence_eligible': False,
                                'competence_selected': False,
                                'competence_weight': 0.0,
                                'competence_score': np.nan,
                                'competence_floor': float(
                                    getattr(cfg, 'selection_competence_floor', 0.0)
                                ),
                                'competence_gate_disabled': bool(
                                    getattr(cfg, 'selection_disable_competence_gate', False)
                                ),
                                'absolute_confidence_q': np.nan,
                                'conditional_weight': 0.0,
                                'gate_similarity_max': np.nan,
                                'gate_similarity_mean': np.nan,
                                'gate_similarity_std': np.nan,
                                'hard_gate_enabled': bool(
                                    getattr(cfg, 'selection_hard_gate', False)
                                ),
                                'hard_gate_threshold': float(
                                    getattr(
                                        cfg,
                                        'selection_hard_gate_threshold',
                                        2.0 ** -0.5,
                                    )
                                ),
                                'hard_gate_reuse': False,
                                'beta_init_weight': 0.0,
                                'beta_init_prior_mass': 0.0,
                                'beta_init_current_mass': 1.0,
                                'beta_init_gamma': float(
                                    getattr(cfg, 'selection_beta_init_gamma', 2.0)
                                ),
                            },
                        )
                        weight = float(detail['weight'])
                        pre_shuffle_weight = float(detail['pre_shuffle_weight'])
                        score = float(detail['score'])
                        prior_perf = detail.get('prior_competence', np.nan)
                        competence_eligible = bool(
                            detail.get('competence_eligible', False)
                        )
                        competence_selected = bool(
                            detail.get('competence_selected', False)
                        )
                        competence_weight = detail.get('competence_weight', 0.0)
                        competence_score = detail.get('competence_score', np.nan)
                        competence_floor = float(
                            detail.get(
                                'competence_floor',
                                getattr(cfg, 'selection_competence_floor', 0.0),
                            )
                        )
                        competence_gate_disabled = bool(
                            detail.get(
                                'competence_gate_disabled',
                                getattr(cfg, 'selection_disable_competence_gate', False),
                            )
                        )
                        absolute_confidence_q = detail.get(
                            'absolute_confidence_q',
                            np.nan,
                        )
                        conditional_weight = detail.get(
                            'conditional_weight',
                            0.0,
                        )
                        gate_similarity_max = detail.get(
                            'gate_similarity_max',
                            np.nan,
                        )
                        gate_similarity_mean = detail.get(
                            'gate_similarity_mean',
                            np.nan,
                        )
                        gate_similarity_std = detail.get(
                            'gate_similarity_std',
                            np.nan,
                        )
                        hard_gate_enabled = bool(
                            detail.get('hard_gate_enabled', False)
                        )
                        hard_gate_threshold = detail.get(
                            'hard_gate_threshold',
                            2.0 ** -0.5,
                        )
                        hard_gate_reuse = bool(
                            detail.get('hard_gate_reuse', False)
                        )
                        beta_init_weight = detail.get('beta_init_weight', 0.0)
                        beta_init_prior_mass = detail.get(
                            'beta_init_prior_mass',
                            0.0,
                        )
                        beta_init_current_mass = detail.get(
                            'beta_init_current_mass',
                            1.0,
                        )
                        beta_init_gamma = detail.get('beta_init_gamma', 2.0)
                        row_normalization_enabled = bool(
                            detail.get('normalization_enabled', normalization_enabled)
                        )
                        normalization_mean = float(detail['similarity_mean'])
                        normalization_scale = float(detail['similarity_scale'])
                        similarity_normalization = detail.get(
                            'similarity_normalization',
                            similarity_normalization,
                        )
                        sparsemax_list.append((prev_idx, float(sim_val), weight, score))
                        lines.append(
                            (
                                f"{learn_block_idx},{task_idx},{iteration},{total_steps},"
                                f"{prev_idx},{sim_val:.6f},"
                                f"{int(pre_shuffle_weight > 0.0)},"
                                f"{int(prev_idx in selected_set)},"
                                f"{pre_shuffle_weight:.6f},{weight:.6f},"
                                f"{_csv_float(score)},{temperature:.6f},"
                                f"{int(row_normalization_enabled)},"
                                f"{similarity_normalization},"
                                f"{int(shuffled_support)},"
                                f"{_csv_float(prior_perf)},"
                                f"{int(competence_eligible)},"
                                f"{int(competence_selected)},"
                                f"{_csv_float(competence_weight)},"
                                f"{_csv_float(competence_score)},"
                                f"{competence_floor:.6f},"
                                f"{int(competence_gate_disabled)},"
                                f"{_csv_float(absolute_confidence_q)},"
                                f"{int(hard_gate_enabled)},"
                                f"{_csv_float(hard_gate_threshold)},"
                                f"{int(hard_gate_reuse)},"
                                f"{_csv_float(beta_init_weight)},"
                                f"{_csv_float(beta_init_prior_mass)},"
                                f"{_csv_float(beta_init_current_mass)},"
                                f"{_csv_float(beta_init_gamma)},"
                                f"{_csv_float(conditional_weight)},"
                                f"{_csv_float(gate_similarity_max)},"
                                f"{_csv_float(gate_similarity_mean)},"
                                f"{_csv_float(gate_similarity_std)},"
                                f"{int(learned_gate_enabled)},"
                                f"{_csv_float(learned_gate_mean)}\n"
                            )
                        )
                    sims_csv_fh.writelines(lines)
                    sparsemax_list.sort(key=lambda x: x[2], reverse=True)

                    cfg.logger.info(
                        "Selected priors ({0}): {1}\n"
                        "current mask: unconditional in linear combination\n"
                        "temperature: {2:.6f}\n"
                        "similarity normalization enabled: {3}\n"
                        "similarity normalization mode: {4}\n"
                        "similarity normalization center/scale: {5:.6f}/{6:.6f}\n"
                        "shuffled support: {7}\n"
                        "competence gate disabled: {8}\n"
                        "prior competence table: {9}".format(
                            _selection_strategy(cfg),
                            sparsemax_list,
                            temperature,
                            normalization_enabled,
                            similarity_normalization,
                            normalization_mean,
                            normalization_scale,
                            shuffled_support,
                            getattr(cfg, 'selection_disable_competence_gate', False),
                            prior_competence[:task_idx].tolist(),
                        )
                    )
                    if getattr(cfg, 'selection_freeze_after_first', False):
                        retrieval_frozen = True
                        cfg.logger.info(
                            'retrieval frozen after first valid estimate for '
                            'task {0} at iteration {1}'.format(
                                task_idx,
                                iteration,
                            )
                        )
                                
                # ---- logging iteration stats ----
                if iteration % config.iteration_log_interval == 0:
                    itr_log_fn(config.logger, agent, iteration, dict_logs)

                    if _should_save_iteration_snapshots(config, iteration):
                        with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                            (agent_name, config.tag, agent.task.name), 'wb') as f:
                            pickle.dump({'rewards': rewards, 'steps': steps}, f)
                        agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                            agent.task.name))
                    if _should_log_parameter_histograms(config, iteration):
                        for tag, value in agent.network.named_parameters():
                            tag = tag.replace('.', '/')
                            config.logger.histo_summary(tag, value.data.cpu().numpy())
                        if hasattr(agent, 'layers_output'):
                            for tag, value in agent.layers_output:
                                tag = 'layer_output/' + tag
                                config.logger.histo_summary(tag, value.data.cpu().numpy())

                # ---- evaluation block ----
                if (agent.config.eval_interval is not None and \
                    iteration % agent.config.eval_interval == 0):
                    config.logger.info('*****agent / evaluation block')
                    _tasks = tasks_info
                    _names = [eval_task_info['name'] for eval_task_info in _tasks]
                    config.logger.info('eval tasks: {0}'.format(', '.join(_names)))
                    eval_data.append(np.zeros(len(_tasks),))
                    for eval_task_idx, eval_task_info in enumerate(_tasks):
                        agent.task_eval_start(eval_task_info['task_label'])
                        eval_states = agent.evaluation_env.reset_task(eval_task_info)
                        agent.evaluation_states = eval_states
                        # performance (perf) can be success rate in (meta-)continualworld or
                        # rewards in other environments
                        perf, eps = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                        agent.task_eval_end()
                        mean_perf = float(np.mean(perf))
                        eval_data[-1][eval_task_idx] = mean_perf
                    _record = np.concatenate([eval_data[-1], np.array(time.time()).reshape(1,)])
                    np.savetxt(eval_data_fh, _record.reshape(1, -1), delimiter=',', fmt='%.4f')
                    del _record
                    icr = eval_data[-1].sum()
                    metric_icr.append(icr)
                    tpot = np.sum(metric_icr)
                    config.logger.info('*****cl evaluation:')
                    config.logger.info('cl eval ICR: {0}'.format(icr))
                    config.logger.info('cl eval TPOT: {0}'.format(tpot))
                    config.logger.scalar_summary('cl_eval/icr', icr)
                    config.logger.scalar_summary('cl_eval/tpot', np.sum(metric_icr))


                # check whether task training has been completed
                task_steps_limit = config.max_steps * (num_tasks * learn_block_idx + task_idx + 1)
                if config.max_steps and agent.total_steps >= task_steps_limit:
                    with open(log_path_tstats + '/%s-%s-online-stats-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)

                    if hasattr(agent, 'seen_tasks'):
                        config.logger.info('cacheing mask for current task')
                    _log_task_beta_composition(
                        agent.network,
                        beta_csv_fh,
                        learn_block_idx,
                        task_idx,
                    )
                    ret = agent.task_train_end()
                    if getattr(config, 'save_task_checkpoints', False):
                        agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (agent_name, \
                            config.tag, agent.task.name, learn_block_idx+1, task_idx+1))
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    task_start_idx = len(rewards)
                    break
            # end of while True. current task training
            # evaluate agent across task exposed to agent so far
            config.logger.info('evaluating agent across all tasks exposed so far to agent')
            for j in range(task_idx+1):
                _eval_task = tasks_info[j]
                agent.task_eval_start(_eval_task['task_label'])

                eval_states = agent.evaluation_env.reset_task(tasks_info[j])
                agent.evaluation_states = eval_states
                perf, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                eval_results[j] += perf
                if j == task_idx:
                    mean_perf = float(np.mean(perf))
                    prior_competence[task_idx] = _normalize_competence_for_selection(
                        config,
                        mean_perf,
                    )
                    config.logger.info(
                        'selection competence for task {0}: raw={1:.6f}, '
                        'normalized={2:.6f}'.format(
                            task_idx,
                            mean_perf,
                            prior_competence[task_idx],
                        )
                    )

                agent.task_eval_end()

                with open(log_path_eval+'/rewards-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(perf, f)
                with open(log_path_eval+'/episodes-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(episodes, f)
        # end for each task
        print('eval stats')
        with open(log_path_eval + '/eval_full_stats.bin', 'wb') as f: pickle.dump(eval_results, f)

        f = open(log_path_eval + '/eval_stats.csv', 'w')
        f.write('task_id,avg_reward\n')
        for k, v in eval_results.items():
            print('{0}: {1:.4f}'.format(k, np.mean(v)))
            f.write('{0},{1:.4f}\n'.format(k, np.mean(v)))
            config.logger.scalar_summary('zeval/task_{0}/avg_reward'.format(k), np.mean(v))
        f.close()
        config.logger.info('********** end of learning block {0}\n'.format(learn_block_idx))
    # end for learning block
    eval_data_fh.close()
    sims_csv_fh.close()
    beta_csv_fh.close()
    if hasattr(agent, 'detect'):
        config.logger.info('***** selection counts (current task -> prior task: count)')
        print('selection counts (current task -> prior task: count)')
        for curr_idx in range(num_tasks):
            if curr_idx == 0:
                summary = 'none'
            else:
                summary = ', '.join([f'{prior}:{selection_counts[curr_idx][prior]}' for prior in range(curr_idx)])
            config.logger.info(f'task {curr_idx}: {summary}')

    if len(eval_data) > 0:
        to_save = np.stack(eval_data, axis=0)
        with open(config.logger.log_dir + '/eval_metrics.npy', 'wb') as f:
            np.save(f, to_save)
    agent.close()
    return steps, rewards
