#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
lifelong (continual) learning experiments using supermask
superpostion algorithm in RL.
https://arxiv.org/abs/2006.14769
'''

import json
import copy
import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
import argparse


def apply_mask_score_normalization(agent, args):
    set_mask_score_normalization(
        agent.network,
        mode=args.mask_score_normalization,
        normalize_current=args.mask_score_normalize_current,
        skip_current_normalization_on_abstain=(
            args.mask_score_skip_current_normalization_on_abstain
        ),
        eps=args.mask_score_norm_eps,
        softmax_temperature=args.mask_score_softmax_temperature,
    )
    if hasattr(agent, 'target_network'):
        set_mask_score_normalization(
            agent.target_network,
            mode=args.mask_score_normalization,
            normalize_current=args.mask_score_normalize_current,
            skip_current_normalization_on_abstain=(
                args.mask_score_skip_current_normalization_on_abstain
            ),
            eps=args.mask_score_norm_eps,
            softmax_temperature=args.mask_score_softmax_temperature,
        )


##### (Meta)CT-graph environment
'''
ppo, baseline (no lifelong learning), task boundary (oracle) given
'''
def ppo_baseline_mctgraph(name, args):
    env_config_path = args.env_config_path
    task_label_input_disabled = args.disable_task_label_input

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'baseline'
    config.seed = args.seed
    random_seed(config.seed)
    exp_suffix = '-no_task_label' if task_label_input_disabled else ''
    exp_id = '-{0}-{1}{2}'.format(config.seed, args.exp_id, exp_suffix)
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4

    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = env_config_['num_tasks']
    del env_config_
    config.use_task_label_input = not task_label_input_disabled

    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_CL(
            state_dim,
            task_label_dim=None if task_label_input_disabled else label_dim,
            hidden_units=(200, 200, 200),
        ),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True
    config.reset_optimizer_on_task_change = args.reset_optimizer_on_task_change
    config.log_parameter_histograms = args.log_parameter_histograms
    config.histogram_log_interval = args.histogram_log_interval
    config.save_task_checkpoints = args.save_task_checkpoints
    config.save_iteration_snapshots = args.save_iteration_snapshots
    config.iteration_snapshot_interval = args.iteration_snapshot_interval
    config.mask_score_normalization = args.mask_score_normalization
    config.mask_score_normalize_current = args.mask_score_normalize_current
    config.mask_score_norm_eps = args.mask_score_norm_eps
    config.mask_score_softmax_temperature = args.mask_score_softmax_temperature

    config.eval_interval = 10
    config.task_ids = np.arange(num_tasks).tolist()

    agent = BaselineAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

'''
ppo, supermask lifelong learning, task boundary (oracle) given
'''
def ppo_ll_mctgraph(name, args):
    env_config_path = args.env_config_path
    task_label_input_disabled = args.disable_task_label_input
    use_lora = args.algo == 'll_lora'
    lora_compositional = use_lora and args.lora_composition == 'sparsemax'

    if args.select_strategy == 'oracle_transfer':
        if not args.oracle_transfer_config:
            raise ValueError(
                '--select_strategy oracle_transfer requires '
                '--oracle_transfer_config'
            )
        if args.oracle_weighting != 'transfer':
            raise ValueError(
                '--select_strategy oracle_transfer requires '
                '--oracle_weighting transfer'
            )
    elif args.oracle_weighting == 'transfer':
        raise ValueError(
            '--oracle_weighting transfer requires '
            '--select_strategy oracle_transfer'
        )

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'lora' if use_lora else 'supermask'
    config.lora_rank = args.lora_rank
    config.lora_composition = args.lora_composition
    config.seed = args.seed
    random_seed(config.seed)
    exp_suffix = '-no_task_label' if task_label_input_disabled else ''
    if args.select_strategy != 'amsc':
        exp_suffix += '-{0}-{1}'.format(
            args.select_strategy, args.oracle_weighting
        )
    if args.selection_no_normalization or args.selection_similarity_normalization == 'none':
        exp_suffix += '-no_norm'
    elif args.selection_similarity_normalization == 'l2':
        exp_suffix += '-sim_l2'
    if args.selection_shuffle_support:
        exp_suffix += '-shuffled'
    if args.selection_disable_competence_gate:
        exp_suffix += '-no_comp_gate'
    if args.selection_uniform_betas:
        exp_suffix += '-uniform_betas'
    if args.selection_sparsemax_fixed_composition:
        exp_suffix += '-sparsemax_fixed'
    if use_lora:
        exp_suffix += '-rank_{0}-{1}'.format(
            args.lora_rank, args.lora_composition
        )
        if lora_compositional and not args.selection_sparsemax_fixed_composition:
            raise ValueError(
                '--lora_composition sparsemax requires '
                '--selection_sparsemax_fixed_composition'
            )
    if args.mask_score_storage_quantization != 'none':
        exp_suffix += '-store_{0}'.format(args.mask_score_storage_quantization)
        if args.mask_score_storage_quantization == 'ternary':
            exp_suffix += '-thr_{0:g}'.format(
                args.mask_score_storage_threshold_rms
            ).replace('.', 'p')
    if args.selection_raw_cosine_fixed_composition:
        exp_suffix += '-raw_cosine_fixed'
    if args.selection_freeze_after_first:
        exp_suffix += '-freeze_first'
    if args.selection_top1_matched_prior_mass:
        exp_suffix += '-top1_matched_mass'
    if args.selection_hard_gate:
        exp_suffix += '-hard_gate'
    if args.mask_score_skip_current_normalization_on_abstain:
        exp_suffix += '-raw_current_on_abstain'
    if args.selection_retrieval_beta_init:
        exp_suffix += '-retrieval_beta_init'
    if args.selection_learned_gate:
        exp_suffix += '-learned_gate'
    if (
        args.selection_retrieval_beta_init
        and (
            args.selection_uniform_betas
            or args.selection_sparsemax_fixed_composition
            or args.selection_raw_cosine_fixed_composition
        )
    ):
        raise ValueError(
            '--selection_retrieval_beta_init cannot be combined with '
            'fixed composition weights'
        )
    if sum([
        args.selection_uniform_betas,
        args.selection_sparsemax_fixed_composition,
        args.selection_raw_cosine_fixed_composition,
    ]) > 1:
        raise ValueError(
            'fixed composition modes are mutually exclusive'
        )
    if args.mask_score_storage_threshold_rms < 0.0:
        raise ValueError(
            '--mask_score_storage_threshold_rms must be non-negative'
        )
    if args.selection_raw_cosine_fixed_composition:
        if args.new_task_mask != 'linear_comb':
            raise ValueError(
                '--selection_raw_cosine_fixed_composition requires '
                '--new_task_mask linear_comb'
            )
        if args.select_strategy != 'amsc':
            raise ValueError(
                '--selection_raw_cosine_fixed_composition requires '
                '--select_strategy amsc'
            )
    if args.selection_freeze_after_first and args.select_strategy != 'amsc':
        raise ValueError(
            '--selection_freeze_after_first requires --select_strategy amsc'
        )
    if args.selection_top1_matched_prior_mass:
        if args.select_strategy != 'amsc':
            raise ValueError(
                '--selection_top1_matched_prior_mass requires '
                '--select_strategy amsc'
            )
        if not args.selection_sparsemax_fixed_composition:
            raise ValueError(
                '--selection_top1_matched_prior_mass requires '
                '--selection_sparsemax_fixed_composition'
            )
    if not 0.0 <= args.selection_hard_gate_threshold <= 1.0:
        raise ValueError('--selection_hard_gate_threshold must be in [0, 1]')
    if args.selection_beta_init_gamma <= 0.0:
        raise ValueError('--selection_beta_init_gamma must be positive')
    if (
        args.mask_score_normalization == 'match_current_l2'
        and (
            args.mask_score_normalize_current
            or args.mask_score_skip_current_normalization_on_abstain
        )
    ):
        raise ValueError(
            '--mask_score_normalization match_current_l2 always leaves the '
            'current score raw; do not pass current-normalization flags'
        )
    if (
        args.mask_score_skip_current_normalization_on_abstain
        and not args.mask_score_normalize_current
    ):
        raise ValueError(
            '--mask_score_skip_current_normalization_on_abstain requires '
            '--mask_score_normalize_current'
        )
    if args.selection_learned_gate:
        if args.new_task_mask != 'linear_comb':
            raise ValueError(
                '--selection_learned_gate requires --new_task_mask linear_comb'
            )
        if args.select_strategy != 'amsc':
            raise ValueError(
                '--selection_learned_gate currently requires '
                '--select_strategy amsc'
            )
        incompatible = []
        if args.selection_uniform_betas:
            incompatible.append('--selection_uniform_betas')
        if args.selection_sparsemax_fixed_composition:
            incompatible.append('--selection_sparsemax_fixed_composition')
        if args.selection_raw_cosine_fixed_composition:
            incompatible.append('--selection_raw_cosine_fixed_composition')
        if args.selection_hard_gate:
            incompatible.append('--selection_hard_gate')
        if args.selection_retrieval_beta_init:
            incompatible.append('--selection_retrieval_beta_init')
        if incompatible:
            raise ValueError(
                '--selection_learned_gate cannot be combined with '
                + ', '.join(incompatible)
            )
        if not 0.0 < args.selection_learned_gate_initial_probability < 1.0:
            raise ValueError(
                '--selection_learned_gate_initial_probability must be in (0, 1)'
            )
    exp_id = '-{0}-mask-{1}-{2}{3}'.format(
        config.seed, args.new_task_mask, args.exp_id, exp_suffix)
    log_name = args.pathheader + '/' + name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4
    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = env_config_['num_tasks']
    del env_config_
    config.use_task_label_input = not task_label_input_disabled

    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    if use_lora:
        config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_LoRA(
            state_dim,
            action_dim,
            label_dim,
            phi_body=FCBody_LoRA(
                state_dim,
                task_label_dim=None if task_label_input_disabled else label_dim,
                hidden_units=(200, 200, 200),
                rank=args.lora_rank,
                num_tasks=num_tasks,
                compositional=lora_compositional,
            ),
            actor_body=DummyBody_CL(200),
            critic_body=DummyBody_CL(200),
            num_tasks=num_tasks,
            rank=args.lora_rank,
            compositional=lora_compositional,
        )
    else:
        config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(
            state_dim, action_dim, label_dim,
            phi_body=FCBody_SS(
                state_dim,
                task_label_dim=None if task_label_input_disabled else label_dim,
                hidden_units=(200, 200, 200),
                num_tasks=num_tasks,
                new_task_mask=args.new_task_mask,
            ),
            actor_body=DummyBody_CL(200),
            critic_body=DummyBody_CL(200),
            num_tasks=num_tasks,
            new_task_mask=args.new_task_mask)
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True
    config.reset_optimizer_on_task_change = args.reset_optimizer_on_task_change
    config.log_parameter_histograms = args.log_parameter_histograms
    config.histogram_log_interval = args.histogram_log_interval
    config.save_task_checkpoints = args.save_task_checkpoints
    config.save_iteration_snapshots = args.save_iteration_snapshots
    config.iteration_snapshot_interval = args.iteration_snapshot_interval
    config.mask_score_normalization = args.mask_score_normalization
    config.mask_score_normalize_current = args.mask_score_normalize_current
    config.mask_score_skip_current_normalization_on_abstain = (
        args.mask_score_skip_current_normalization_on_abstain
    )
    config.mask_score_norm_eps = args.mask_score_norm_eps
    config.mask_score_softmax_temperature = args.mask_score_softmax_temperature
    config.mask_score_storage_quantization = (
        args.mask_score_storage_quantization
    )
    config.mask_score_storage_threshold_rms = (
        args.mask_score_storage_threshold_rms
    )

    config.eval_interval = 10
    config.task_ids = np.arange(num_tasks).tolist()

    #=============================================================#
    #                   AMSC Hyperparameters
    #=============================================================#
    config.detect_reference_num = args.detect_reference_num
    config.detect_num_samples = args.detect_num_samples
    config.detect_frequency = args.detect_frequency
    config.embedding_ema = args.embedding_ema
    config.detect_reference_seed = args.detect_reference_seed
    config.legacy_wte_ema = args.legacy_wte_ema
    config.detect_embedding_method = args.detect_embedding_method
    config.swe_num_projections = args.swe_num_projections
    config.swe_num_quantiles = args.swe_num_quantiles
    config.swe_num_workers = args.swe_num_workers
    config.swe_seed = args.swe_seed
    config.swe_normalize_embedding = args.swe_normalize_embedding
    config.detect_fn = lambda input_dim, action_dim: Detect(
        config.detect_reference_num,
        input_dim, action_dim,
        config.detect_num_samples,
        device=Config.DEVICE,
        one_hot=True,
        normalized=True,
        embedding_method=args.detect_embedding_method,
        swe_num_projections=args.swe_num_projections,
        swe_num_quantiles=args.swe_num_quantiles,
        swe_num_workers=args.swe_num_workers,
        swe_seed=args.swe_seed,
        swe_normalize_embedding=args.swe_normalize_embedding,
        reference_seed=args.detect_reference_seed,
    )
    config.select_frequency = args.select_frequency
    config.select_strategy = args.select_strategy
    config.family_stride = args.family_stride
    config.oracle_weighting = args.oracle_weighting
    config.oracle_transfer_config = args.oracle_transfer_config
    config.selection_soft_temperature = args.selection_soft_temperature
    config.selection_similarity_normalization = (
        'none' if args.selection_no_normalization
        else args.selection_similarity_normalization
    )
    config.selection_normalize_similarities = (
        config.selection_similarity_normalization != 'none'
    )
    config.selection_shuffle_support = args.selection_shuffle_support
    config.selection_disable_competence_gate = args.selection_disable_competence_gate
    config.selection_competence_floor = args.selection_competence_floor
    config.selection_competence_normalization = args.selection_competence_normalization
    config.selection_uniform_betas = args.selection_uniform_betas
    config.selection_sparsemax_fixed_composition = (
        args.selection_sparsemax_fixed_composition
    )
    config.selection_raw_cosine_fixed_composition = (
        args.selection_raw_cosine_fixed_composition
    )
    config.selection_freeze_after_first = args.selection_freeze_after_first
    config.selection_top1_matched_prior_mass = (
        args.selection_top1_matched_prior_mass
    )
    config.selection_hard_gate = args.selection_hard_gate
    config.selection_hard_gate_threshold = args.selection_hard_gate_threshold
    config.selection_retrieval_beta_init = args.selection_retrieval_beta_init
    config.selection_beta_init_gamma = args.selection_beta_init_gamma
    config.selection_learned_gate = args.selection_learned_gate
    config.selection_learned_gate_initial_probability = (
        args.selection_learned_gate_initial_probability
    )
    #=============================================================#

    agent = DetectLLAgent(config)
    apply_mask_score_normalization(agent, args)
    if args.selection_learned_gate:
        configure_learned_transfer_gate(
            agent.network,
            enabled=True,
            initial_probability=args.selection_learned_gate_initial_probability,
        )
        agent._reset_optimizer()
    if args.selection_uniform_betas:
        set_uniform_beta_composition(agent.network, True)
    if args.selection_sparsemax_fixed_composition:
        set_sparsemax_fixed_composition(agent.network, True)
    if args.selection_raw_cosine_fixed_composition:
        set_raw_cosine_fixed_composition(agent.network, True)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', help='algorithm to run')
    parser.add_argument('--env_name', help='name of the evaluation environment. ' \
        'minigrid and ctgraph currently supported', default='ctgraph')
    parser.add_argument('--env_config_path', help='path to environment config', \
        default='./env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json')
        #./env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json
        #./env_configs/ct28/seed1/meta_ctgraph_ct28_random.json
        #./env_configs/ct28/seed1/meta_ctgraph_ct8_interleaved.json
        #./env_configs/ct28/seed1/meta_ctgraph_ct14_half_1.json
        #./env_configs/ct28/seed1/meta_ctgraph_ct14_half_2.json
        #./env_configs/ct28/seed1/meta_ctgraph_ct14_md.json
        #./env_configs/ct8.json
    parser.add_argument('--exp_id', help='experiment id', default='ct14_md', type=str)
    parser.add_argument('--max_steps', help='maximum number of training steps per task.', \
        default=51200*2, type=int)
    parser.add_argument('--new_task_mask', help='', \
        default='random', type=str)
    parser.add_argument(
        '--lora_rank',
        '--lora-rank',
        dest='lora_rank',
        type=int,
        default=4,
        help='rank of each task-local LoRA adapter',
    )
    parser.add_argument(
        '--lora_composition',
        '--lora-composition',
        dest='lora_composition',
        choices=['task_local', 'sparsemax'],
        default='task_local',
        help=(
            'task_local learns independent adapters; sparsemax composes '
            'retrieved adapters and recompresses the result at task boundaries'
        ),
    )
    parser.add_argument(
        '--legacy_wte_ema',
        '--legacy-wte-ema',
        dest='legacy_wte_ema',
        help=(
            'use the legacy WTE update, which averages the raw new embedding '
            'with the stored unit embedding before normalisation'
        ),
        action='store_true',
    )
    parser.add_argument(
        '--detect_reference_num',
        '--detect-reference-num',
        '--embedding-reference-size',
        dest='detect_reference_num',
        help='number of samples in the fixed LWE reference distribution',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--detect_num_samples',
        '--detect-num-samples',
        '--embedding-sample-count',
        dest='detect_num_samples',
        help='number of replay samples used for each online task embedding',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--detect_frequency',
        '--detect-frequency',
        '--embedding-update-frequency',
        '--f-emb',
        dest='detect_frequency',
        help='embedding update interval measured in PPO iterations',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--select_frequency',
        '--select-frequency',
        '--selection-frequency',
        '--f-sel',
        dest='select_frequency',
        help='prior-selection interval measured in PPO iterations',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--embedding_ema',
        '--embedding-ema',
        dest='embedding_ema',
        help='EMA coefficient for online task embeddings; 0 disables EMA',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--detect_reference_seed',
        '--detect-reference-seed',
        '--embedding-reference-seed',
        dest='detect_reference_seed',
        help='seed used only to sample the fixed LWE reference distribution',
        type=int,
        default=98,
    )
    parser.add_argument('--disable_task_label_input',
        help='do not concatenate the task label to the policy network input; task labels are still used for task switching/evaluation',
        action='store_true')
    parser.add_argument('--reset_optimizer_on_task_change',
        help='recreate the RMSprop optimizer at each task boundary',
        action='store_true')
    parser.add_argument('--log_parameter_histograms',
        help='enable TensorBoard parameter histograms; disabled by default because they create very large event files',
        action='store_true')
    parser.add_argument('--histogram_log_interval',
        help='iteration interval for parameter histograms when --log_parameter_histograms is enabled; defaults to iteration_log_interval',
        type=int,
        default=1)
    parser.add_argument('--save_task_checkpoints',
        help='save full per-task model checkpoints under task_stats; disabled by default because these files are very large',
        action='store_true')
    parser.add_argument('--save_iteration_snapshots',
        help='save latest model and online-stats snapshots during iteration logging; disabled by default because model snapshots are very large',
        action='store_true')
    parser.add_argument('--iteration_snapshot_interval',
        help='iteration interval for --save_iteration_snapshots; defaults to iteration_log_interval',
        type=int,
        default=None)
    parser.add_argument('--mask_score_normalization',
        help=(
            'optional normalization applied to mask score tensors before '
            'linear-combination composition; match_current_l2 leaves the '
            'current score raw and matches each prior score to its L2 norm; '
            'l2_ste and rms_ste use normalized forward values with identity '
            'score gradients; softmax_across_masks applies per-parameter '
            'attention over the active prior masks and current mask'
        ),
        choices=[
            'none', 'l2', 'l2_ste', 'rms_ste', 'match_current_l2',
            'softmax_across_masks', 'std', 'zscore', 'sign'
        ],
        default='none')
    parser.add_argument('--mask_score_normalize_current',
        help='also normalize the current task mask score during linear-combination composition',
        action='store_true')
    parser.add_argument(
        '--mask_score_skip_current_normalization_on_abstain',
        '--mask-score-skip-current-normalization-on-abstain',
        dest='mask_score_skip_current_normalization_on_abstain',
        help=(
            'leave the current mask score unnormalized when selective '
            'composition has no active prior masks'
        ),
        action='store_true',
    )
    parser.add_argument('--mask_score_norm_eps',
        help='epsilon used by mask score normalization modes',
        type=float,
        default=1e-8)
    parser.add_argument('--mask_score_softmax_temperature',
        help='temperature for softmax_across_masks parameter-level attention',
        type=float,
        default=1.0)
    parser.add_argument(
        '--mask_score_storage_quantization',
        '--mask-score-storage-quantization',
        dest='mask_score_storage_quantization',
        choices=['none', 'binary', 'ternary'],
        default='none',
        help=(
            'quantize each consolidated completed-task score tensor before it '
            'enters the reusable mask library'
        ),
    )
    parser.add_argument(
        '--mask_score_storage_threshold_rms',
        '--mask-score-storage-threshold-rms',
        dest='mask_score_storage_threshold_rms',
        type=float,
        default=0.1,
        help=(
            'ternary dead-zone magnitude as a multiple of each score tensor RMS'
        ),
    )
    parser.add_argument('--seed', help='seed for the experiment', default=8379, type=int)
    parser.add_argument('--pathheader', '--p', '-p', help='experiment header to log path for launcher.py', type=str, default='')
    parser.add_argument(
        '--select_strategy',
        '--select-strategy',
        dest='select_strategy',
        help='prior-mask selection strategy',
        choices=[
            'amsc', 'oracle_all', 'oracle_depth_prefix', 'oracle_parent',
            'oracle_transfer'
        ],
        default='amsc',
    )
    parser.add_argument(
        '--family_stride',
        '--family-stride',
        dest='family_stride',
        help='number of interleaved task families; required by oracle selection strategies',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--oracle_weighting',
        '--oracle-weighting',
        dest='oracle_weighting',
        help='fixed source weighting used inside an oracle-selected support',
        choices=['uniform', 'similarity', 'transfer'],
        default='uniform',
    )
    parser.add_argument(
        '--oracle_transfer_config',
        '--oracle-transfer-config',
        dest='oracle_transfer_config',
        help='JSON source/weight table required by oracle_transfer',
        default=None,
    )
    parser.add_argument(
        '--selection_soft_temperature',
        help='sparsemax temperature for similarity selection',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--selection_similarity_normalization',
        '--selection-similarity-normalization',
        dest='selection_similarity_normalization',
        help='normalization applied to similarity logits before sparsemax',
        choices=['zscore', 'l2', 'none'],
        default='zscore',
    )
    parser.add_argument(
        '--selection_no_normalization',
        '--selection-no-normalization',
        '--amsc_no_norm',
        dest='selection_no_normalization',
        help='NoNorm ablation: equivalent to --selection_similarity_normalization none',
        action='store_true',
    )
    parser.add_argument(
        '--selection_shuffle_support',
        '--selection-shuffle-support',
        '--amsc_shuffled',
        dest='selection_shuffle_support',
        help='Shuffled ablation: preserve sparsemax support size but randomly assign support to prior masks',
        action='store_true',
    )
    parser.add_argument(
        '--selection_disable_competence_gate',
        '--selection-disable-competence-gate',
        dest='selection_disable_competence_gate',
        help='disable the default prior competence gate and select using similarity only',
        action='store_true',
    )
    parser.add_argument(
        '--selection_competence_floor',
        '--selection-competence-floor',
        dest='selection_competence_floor',
        help='minimum normalized own-task performance required before a prior can pass the competence gate',
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '--selection_competence_normalization',
        '--selection-competence-normalization',
        dest='selection_competence_normalization',
        help='normalization applied to own-task performance before competence gating',
        choices=['clip01', 'none'],
        default='clip01',
    )
    parser.add_argument(
        '--selection_uniform_betas',
        '--selection-uniform-betas',
        '--amsc_uniform_betas',
        dest='selection_uniform_betas',
        help='ablation: use a uniform average over the selected support plus current mask instead of learned beta weights',
        action='store_true',
    )
    parser.add_argument(
        '--selection_sparsemax_fixed_composition',
        '--selection-sparsemax-fixed-composition',
        dest='selection_sparsemax_fixed_composition',
        help=(
            'ablation: freeze beta parameters, allocate 1/(m+1) to the '
            'current mask, and distribute m/(m+1) over selected priors '
            'using their normalized sparsemax retrieval weights'
        ),
        action='store_true',
    )
    parser.add_argument(
        '--selection_raw_cosine_fixed_composition',
        '--selection-raw-cosine-fixed-composition',
        dest='selection_raw_cosine_fixed_composition',
        help=(
            'freeze beta parameters, retain sparsemax-selected priors with '
            'weight w_i*max(cosine_i, 0), and allocate the remaining mass '
            'to the current mask'
        ),
        action='store_true',
    )
    parser.add_argument(
        '--selection_freeze_after_first',
        '--selection-freeze-after-first',
        dest='selection_freeze_after_first',
        help=(
            'freeze the support and composition weights after the first valid '
            'within-task retrieval estimate'
        ),
        action='store_true',
    )
    parser.add_argument(
        '--selection_top1_matched_prior_mass',
        '--selection-top1-matched-prior-mass',
        dest='selection_top1_matched_prior_mass',
        help=(
            'retain only the highest-weight sparsemax source while preserving '
            'the total prior/current allocation of the full retrieved support'
        ),
        action='store_true',
    )
    parser.add_argument(
        '--selection_hard_gate',
        '--selection-hard-gate',
        dest='selection_hard_gate',
        help='abstain from prior reuse unless q exceeds the absolute-confidence threshold',
        action='store_true',
    )
    parser.add_argument(
        '--selection_hard_gate_threshold',
        '--selection-hard-gate-threshold',
        dest='selection_hard_gate_threshold',
        help='minimum q required to admit the sparsemax-selected prior support',
        type=float,
        default=2.0 ** -0.5,
    )
    parser.add_argument(
        '--selection_retrieval_beta_init',
        '--selection-retrieval-beta-init',
        dest='selection_retrieval_beta_init',
        help='initialize newly admitted beta logits from retrieval confidence',
        action='store_true',
    )
    parser.add_argument(
        '--selection_beta_init_gamma',
        '--selection-beta-init-gamma',
        dest='selection_beta_init_gamma',
        help='exponent mapping q to total initial prior beta mass',
        type=float,
        default=2.0,
    )
    parser.add_argument(
        '--selection_learned_gate',
        '--selection-learned-gate',
        dest='selection_learned_gate',
        help=(
            'learn the total prior-versus-current transfer mass from retrieval '
            'statistics; sparsemax still defines the prior support'
        ),
        action='store_true',
    )
    parser.add_argument(
        '--selection_learned_gate_initial_probability',
        '--selection-learned-gate-initial-probability',
        dest='selection_learned_gate_initial_probability',
        help='initial sigmoid probability assigned to total prior reuse',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--detect_embedding_method',
        help='task embedding method used by Detect',
        choices=['lwe', 'swe'],
        default='lwe',
    )
    parser.add_argument(
        '--swe_num_projections',
        help='number of random projection directions for sliced Wasserstein embeddings',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--swe_num_quantiles',
        help='number of quantile support points per projection for sliced Wasserstein embeddings',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--swe_num_workers',
        help='number of parallel worker threads used to process SWE projection chunks',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--swe_seed',
        help='random projection seed for sliced Wasserstein embeddings',
        type=int,
        default=98,
    )
    parser.add_argument(
        '--swe_normalize_embedding',
        help='L2-normalize SWE embeddings before cosine selection',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.detect_reference_num <= 0:
        parser.error('--detect_reference_num must be positive')
    if args.detect_num_samples <= 0:
        parser.error('--detect_num_samples must be positive')
    if args.detect_frequency <= 0:
        parser.error('--detect_frequency must be positive')
    if args.select_frequency <= 0:
        parser.error('--select_frequency must be positive')
    if not 0.0 <= args.embedding_ema < 1.0:
        parser.error('--embedding_ema must be in [0, 1)')
    if args.selection_soft_temperature <= 0.0:
        parser.error('--selection_soft_temperature must be positive')
    if args.lora_rank <= 0:
        parser.error('--lora_rank must be positive')

    if args.env_name == 'ctgraph':
        name = Config.ENV_METACTGRAPH
        if args.algo == 'baseline':
            ppo_baseline_mctgraph(name, args)
        elif args.algo in ('ll_supermask', 'll_lora'):
            ppo_ll_mctgraph(name, args)
        else:
            raise ValueError('algo {0} not implemented'.format(args.algo))
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
