import subprocess
import shlex
import os
import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

commands_ctgraph_ppo = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py baseline --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py baseline --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py baseline --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py baseline --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py baseline --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py baseline --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py baseline --max_steps 51200 --seed 92 --disable_task_label_input']
]

# FOCCAL (CT-Graph)
commands_ctgraph_lc = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92']
]

commands_ctgraph_sc = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 51200 --seed 92 --disable_task_label_input']
]

commands_ctgraph_sc_oracle = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --selection_soft_temperature 1.0 --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --selection_soft_temperature 1.0 --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --selection_soft_temperature 1.0 --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --selection_soft_temperature 1.0 --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --selection_soft_temperature 1.0 --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --selection_soft_temperature 1.0 --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --selection_soft_temperature 1.0 --max_steps 51200 --seed 92 --disable_task_label_input']
]

commands_ctgraph_sc_d8_recovery = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --env_config_path ./env_configs/ct28/seed1/meta_ctgraph_ct29_d8_recovery_unique_labels.json --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --env_config_path ./env_configs/ct28/seed1/meta_ctgraph_ct29_d8_recovery_unique_labels.json --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --env_config_path ./env_configs/ct28/seed1/meta_ctgraph_ct29_d8_recovery_unique_labels.json --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --env_config_path ./env_configs/ct28/seed1/meta_ctgraph_ct29_d8_recovery_unique_labels.json --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --env_config_path ./env_configs/ct28/seed1/meta_ctgraph_ct29_d8_recovery_unique_labels.json --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --env_config_path ./env_configs/ct28/seed1/meta_ctgraph_ct29_d8_recovery_unique_labels.json --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --env_config_path ./env_configs/ct28/seed1/meta_ctgraph_ct29_d8_recovery_unique_labels.json --max_steps 51200 --seed 92 --disable_task_label_input']
]

commands_ctgraph_sc_swe = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 51200 --seed 92 --disable_task_label_input']
]

commands_ctgraph_sc_no_norm = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 51200 --seed 92 --disable_task_label_input']
]

commands_ctgraph_sc_no_comp = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 51200 --seed 92 --disable_task_label_input']
]

commands_ctgraph_sc_shuffled = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 51200 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 51200 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 51200 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 51200 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 51200 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 51200 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 51200 --seed 92 --disable_task_label_input']
]

commands_ctgraph_single_task_experts = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task1 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task1.json'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task6 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task6.json'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task17 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task17.json'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task39 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task39.json'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task82 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task82.json'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task169 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task169.json'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task342 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task342.json'],

    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4095 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4095.json'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4102 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4102.json'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4117 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4117.json'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4146 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4146.json'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4205 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4205.json'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4322 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4322.json'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task4557 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task4557.json'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8184 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8184.json'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8189 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8189.json'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8198 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8198.json'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8216 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8216.json'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8253 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8253.json'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8327 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8327.json'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task8474 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task8474.json'],

    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12278 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12278.json'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12284 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12284.json'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12297 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12297.json'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12323 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12323.json'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12375 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12375.json'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12478 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12478.json'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86 --exp_id task12684 --env_config_path ./env_configs/ct28/seed1_individual/meta_ctgraph_ct28_task12684.json']
]


commands_minigrid = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --selection_soft_temperature 1.0 --selection-disable-competence-gate --max_steps 512_000 --seed 92 --disable_task_label_input']
]

commands_minigrid_oracle = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed86.json --selection_soft_temperature 1.0 --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed87.json --selection_soft_temperature 1.0 --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed88.json --selection_soft_temperature 1.0 --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed89.json --selection_soft_temperature 1.0 --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed90.json --selection_soft_temperature 1.0 --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed91.json --selection_soft_temperature 1.0 --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --select_strategy oracle_depth_prefix --family_stride 4 --env_config_path ./env_configs/minigrid_object_remap_seed92.json --selection_soft_temperature 1.0 --max_steps 512_000 --seed 92 --disable_task_label_input']
]

commands_minigrid_no_comp = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --selection_soft_temperature 1.0 --selection_disable_competence_gate --max_steps 512_000 --seed 92 --disable_task_label_input']
]

commands_minigrid_no_norm = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --selection_soft_temperature 1.0 --selection_similarity_normalization none --max_steps 512_000 --seed 92 --disable_task_label_input']
]

commands_minigrid_shuffled = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --selection_soft_temperature 1.0 --selection_shuffle_support --max_steps 512_000 --seed 92 --disable_task_label_input']
]

commands_minigrid_swe = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --selection_soft_temperature 1.0 --detect_embedding_method swe --swe_num_projections 128 --swe_num_quantiles 128 --swe_num_workers 1 --max_steps 512_000 --seed 92 --disable_task_label_input']
]

commands_minigrid_single_task_experts = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_0.json --exp_id task0 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_1.json --exp_id task1 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_2.json --exp_id task2 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_3.json --exp_id task3 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_4.json --exp_id task4 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_5.json --exp_id task5 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_6.json --exp_id task6 --max_steps 512_000 --seed 92 --disable_task_label_input'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_7.json --exp_id task7 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_8.json --exp_id task8 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_9.json --exp_id task9 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_10.json --exp_id task10 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_11.json --exp_id task11 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_12.json --exp_id task12 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_13.json --exp_id task13 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_14.json --exp_id task14 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_15.json --exp_id task15 --max_steps 512_000 --seed 92 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/mg16_remap/minigrid_object_remap_seed841_16.json --exp_id task16 --max_steps 512_000 --seed 92 --disable_task_label_input'],
]


commands_continualworld = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_continualworld.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/continualworld_10.json --max_steps 2_000_000 --seed 92 --disable_task_label_input']
]


MIG_1_IDS = [
    'MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277',
    'MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa',
    'MIG-c432df19-0894-5232-ac1c-9a3440fc267e',
    'MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf',
    'MIG-35ecef79-db2e-590b-9e8c-2c07c787008e',
    'MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0',
    'MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e',
]

MIG_2_IDS = [
    'MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3',
    'MIG-4590f80d-be70-58e4-af75-eeb950255d4a',
    'MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2',
    'MIG-2593b912-5975-58e9-bc3d-495311cee807',
    'MIG-51069529-f343-59c6-bac7-a75648296e7b',
    'MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7',
    'MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1',
]

# Alternative host: two logical worker groups share seven 10 GB MIGs. Using
# groups 3 and 4 schedules two independent experiment queues on each MIG.
MIG_80GB_SHARED_IDS = [
    'MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3',
    'MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e',
    'MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6',
    'MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371',
    'MIG-62fabbf0-b8de-5040-b4db-93f62c477543',
    'MIG-fcda8ac4-e82d-5259-8495-c777d8c95d74',
    'MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e',
]

MIG_GROUPS = {
    0: MIG_1_IDS,
    1: MIG_2_IDS,
    3: MIG_80GB_SHARED_IDS,
    4: MIG_80GB_SHARED_IDS,
}

DEFAULT_SENSITIVITY_MIG_GROUPS = [0, 1]

def build_hard_gate_commands(
    environment,
    initialize_betas=False,
    skip_current_normalization_on_abstain=False,
    disable_competence_gate=False,
    mask_score_normalization='l2',
):
    if (
        mask_score_normalization == 'match_current_l2'
        and skip_current_normalization_on_abstain
    ):
        raise ValueError(
            'match_current_l2 already leaves the current score raw and cannot '
            'use skip_current_normalization_on_abstain'
        )
    commands = []
    for offset, seed in enumerate(range(86, 93)):
        common = (
            '--new_task_mask linear_comb '
            '--selection_soft_temperature 1.0 '
            '--selection_hard_gate '
            '--selection_hard_gate_threshold 0.70710678 '
            '--disable_task_label_input '
            f'--mask_score_normalization {mask_score_normalization}'
        )
        if mask_score_normalization != 'match_current_l2':
            common += ' --mask_score_normalize_current'
        if initialize_betas:
            common += (
                ' --selection_retrieval_beta_init'
                ' --selection_beta_init_gamma 2.0'
            )
        if skip_current_normalization_on_abstain:
            common += ' --mask_score_skip_current_normalization_on_abstain'
        if disable_competence_gate:
            common += ' --selection_disable_competence_gate'

        if environment == 'ctgraph':
            command = (
                f'python train_ctgraph.py ll_supermask {common} '
                f'--max_steps 51200 --seed {seed}'
            )
        elif environment == 'minigrid':
            command = (
                f'python train_minigrid.py ll_supermask {common} '
                f'--env_config_path '
                f'./env_configs/minigrid_object_remap_seed{seed}.json '
                f'--max_steps 512_000 --seed {seed}'
            )
        else:
            raise ValueError(f'unknown hard-gate environment: {environment}')
        commands.append([MIG_2_IDS[offset], command])
    return commands


def build_rms_ste_commands(
    environment,
    hard_gate=False,
    disable_competence_gate=True,
):
    """Run AMSC with forward-only unit-RMS mask-score normalization."""
    commands = []
    for offset, seed in enumerate(range(86, 93)):
        common = (
            '--new_task_mask linear_comb '
            '--selection_soft_temperature 1.0 '
            '--disable_task_label_input '
            '--mask_score_normalization rms_ste '
            '--mask_score_normalize_current'
        )
        if hard_gate:
            common += (
                ' --selection_hard_gate'
                ' --selection_hard_gate_threshold 0.70710678'
            )
        if disable_competence_gate:
            common += ' --selection_disable_competence_gate'

        if environment == 'ctgraph':
            command = (
                f'python train_ctgraph.py ll_supermask {common} '
                f'--max_steps 51200 --seed {seed}'
            )
        elif environment == 'minigrid':
            command = (
                f'python train_minigrid.py ll_supermask {common} '
                f'--env_config_path '
                f'./env_configs/minigrid_object_remap_seed{seed}.json '
                f'--max_steps 512_000 --seed {seed}'
            )
        else:
            raise ValueError(f'unknown RMS-STE environment: {environment}')
        commands.append([MIG_2_IDS[offset], command])
    return commands


def build_mask_score_composition_commands(
    environment,
    normalization,
    normalize_current=None,
    hard_gate=False,
    disable_competence_gate=True,
    softmax_temperature=1.0,
    fixed_composition=None,
    shuffle_support=False,
    select_strategy='amsc',
    family_stride=None,
    oracle_weighting='uniform',
    oracle_transfer_config=None,
    freeze_after_first=False,
    top1_matched_prior_mass=False,
    storage_quantization='none',
    storage_threshold_rms=0.1,
):
    """Run fixed-composition and mask-score normalization ablations."""
    if normalization not in {'none', 'l2_ste', 'softmax_across_masks'}:
        raise ValueError(
            'normalization must be none, l2_ste, or softmax_across_masks'
        )
    if fixed_composition not in {
        None, 'uniform', 'sparsemax', 'raw_cosine'
    }:
        raise ValueError(
            'fixed_composition must be None, uniform, sparsemax, or raw_cosine'
        )
    if storage_quantization not in {'none', 'binary', 'ternary'}:
        raise ValueError(
            'storage_quantization must be none, binary, or ternary'
        )
    if storage_threshold_rms < 0.0:
        raise ValueError('storage_threshold_rms must be non-negative')
    if normalize_current is None:
        normalize_current = normalization == 'l2_ste'
    normalize_current = bool(normalize_current)
    if normalize_current and normalization != 'l2_ste':
        raise ValueError(
            'normalize_current is only supported with l2_ste in this builder'
        )
    if select_strategy not in {
        'amsc', 'oracle_all', 'oracle_depth_prefix', 'oracle_parent',
        'oracle_transfer'
    }:
        raise ValueError('unknown selection strategy: {0}'.format(select_strategy))
    if (
        select_strategy in {'oracle_all', 'oracle_depth_prefix', 'oracle_parent'}
        and family_stride is None
    ):
        raise ValueError('oracle selection requires family_stride')
    if oracle_weighting not in {'uniform', 'similarity', 'transfer'}:
        raise ValueError(
            'oracle_weighting must be uniform, similarity, or transfer'
        )
    if select_strategy == 'oracle_transfer':
        if oracle_weighting != 'transfer' or not oracle_transfer_config:
            raise ValueError(
                'oracle_transfer requires transfer weighting and a config path'
            )
    elif oracle_weighting == 'transfer':
        raise ValueError('transfer weighting requires oracle_transfer')
    if freeze_after_first and select_strategy != 'amsc':
        raise ValueError('freeze_after_first requires AMSC selection')
    if top1_matched_prior_mass:
        if select_strategy != 'amsc':
            raise ValueError('top1_matched_prior_mass requires AMSC selection')
        if fixed_composition != 'sparsemax':
            raise ValueError(
                'top1_matched_prior_mass requires sparsemax fixed composition'
            )

    commands = []
    for offset, seed in enumerate(range(86, 93)):
        common = (
            '--new_task_mask linear_comb '
            '--selection_soft_temperature 1.0 '
            '--disable_task_label_input '
            f'--mask_score_normalization {normalization}'
        )
        if normalization == 'l2_ste' and normalize_current:
            common += ' --mask_score_normalize_current'
        elif normalization == 'softmax_across_masks':
            common += (
                f' --mask_score_softmax_temperature {softmax_temperature}'
            )
        if hard_gate:
            common += (
                ' --selection_hard_gate'
                ' --selection_hard_gate_threshold 0.70710678'
            )
        if disable_competence_gate:
            common += ' --selection_disable_competence_gate'
        if shuffle_support:
            common += ' --selection_shuffle_support'
        if freeze_after_first:
            common += ' --selection_freeze_after_first'
        if top1_matched_prior_mass:
            common += ' --selection_top1_matched_prior_mass'
        if select_strategy != 'amsc':
            common += (
                f' --select_strategy {select_strategy}'
                f' --oracle_weighting {oracle_weighting}'
            )
            if family_stride is not None:
                common += f' --family_stride {int(family_stride)}'
            if oracle_transfer_config is not None:
                common += (
                    f' --oracle_transfer_config {oracle_transfer_config}'
                )
        if fixed_composition == 'uniform':
            common += ' --selection_uniform_betas'
        elif fixed_composition == 'sparsemax':
            common += ' --selection_sparsemax_fixed_composition'
        elif fixed_composition == 'raw_cosine':
            common += ' --selection_raw_cosine_fixed_composition'
        if storage_quantization != 'none':
            common += (
                f' --mask_score_storage_quantization {storage_quantization}'
            )
            if storage_quantization == 'ternary':
                common += (
                    ' --mask_score_storage_threshold_rms '
                    f'{storage_threshold_rms}'
                )

        if environment == 'ctgraph':
            command = (
                f'python train_ctgraph.py ll_supermask {common} '
                f'--max_steps 51200 --seed {seed}'
            )
        elif environment == 'minigrid':
            command = (
                f'python train_minigrid.py ll_supermask {common} '
                f'--env_config_path '
                f'./env_configs/minigrid_object_remap_seed{seed}.json '
                f'--max_steps 512_000 --seed {seed}'
            )
        else:
            raise ValueError(
                f'unknown mask-score ablation environment: {environment}'
            )
        commands.append([MIG_2_IDS[offset], command])
    return commands


def build_ctgraph_lora_commands(rank, composition):
    """Build fixed-backbone LoRA controls for the CT28 mask study."""
    if rank <= 0:
        raise ValueError('LoRA rank must be positive')
    if composition not in {'task_local', 'sparsemax'}:
        raise ValueError('unknown LoRA composition: {0}'.format(composition))

    commands = []
    for offset, seed in enumerate(range(86, 93)):
        common = (
            f'--lora_rank {rank} '
            f'--lora_composition {composition} '
            '--disable_task_label_input '
            '--selection_disable_competence_gate'
        )
        if composition == 'sparsemax':
            common += (
                ' --new_task_mask linear_comb'
                ' --selection_sparsemax_fixed_composition'
                ' --selection_soft_temperature 1.0'
            )
        command = (
            f'python train_ctgraph.py ll_lora {common} '
            f'--max_steps 51200 --seed {seed}'
        )
        commands.append([MIG_2_IDS[offset], command])
    return commands


AMSC_SENSITIVITY_GRIDS = {
    'temperature': {
        'ctgraph': [0.25, 0.5, 1.0, 2.0, 4.0],
        'minigrid': [0.25, 0.5, 1.0, 2.0, 4.0],
    },
    'sample_count': {
        'ctgraph': [32, 64, 128, 256, 512],
        'minigrid': [128, 256, 512, 1024, 2048],
    },
    'reference_size': {
        'ctgraph': [10, 25, 50, 100, 200],
        'minigrid': [10, 25, 50, 100, 200],
    },
    'embedding_frequency': {
        'ctgraph': [1, 2, 5, 10],
        'minigrid': [1, 2, 5, 10],
    },
    'selection_frequency': {
        'ctgraph': [1, 2, 5, 10],
        'minigrid': [1, 2, 5, 10],
    },
    'ema': {
        'ctgraph': [0.0, 0.25, 0.5, 0.75, 0.9],
        'minigrid': [0.0, 0.25, 0.5, 0.75, 0.9],
    },
    'reference_seed': {
        'ctgraph': [11, 37, 71, 98, 131],
        'minigrid': [11, 37, 71, 98, 131],
    },
}

AMSC_SENSITIVITY_DEFAULTS = {
    'ctgraph': {
        'temperature': 1.0,
        'sample_count': 128,
        'reference_size': 50,
        'embedding_frequency': 1,
        'selection_frequency': 1,
        'ema': 0.5,
        'reference_seed': 98,
    },
    'minigrid': {
        'temperature': 1.0,
        'sample_count': 512,
        'reference_size': 50,
        'embedding_frequency': 1,
        'selection_frequency': 5,
        'ema': 0.5,
        'reference_seed': 98,
    },
}


def _sensitivity_value_label(value):
    return str(value).replace('-', 'm').replace('.', 'p')


def _build_amsc_sensitivity_command(environment, settings, seed):
    common = (
        '--new_task_mask linear_comb '
        '--disable_task_label_input '
        '--selection_disable_competence_gate '
        '--selection_sparsemax_fixed_composition '
        '--mask_score_normalization l2_ste '
        '--mask_score_normalize_current '
        f'--selection_soft_temperature {settings["temperature"]} '
        f'--detect_num_samples {settings["sample_count"]} '
        f'--detect_reference_num {settings["reference_size"]} '
        f'--detect_frequency {settings["embedding_frequency"]} '
        f'--select_frequency {settings["selection_frequency"]} '
        f'--embedding_ema {settings["ema"]} '
        f'--detect_reference_seed {settings["reference_seed"]}'
    )
    if environment == 'ctgraph':
        return (
            f'python train_ctgraph.py ll_supermask {common} '
            f'--max_steps 51200 --seed {seed}'
        )
    if environment == 'minigrid':
        return (
            f'python train_minigrid.py ll_supermask {common} '
            '--env_config_path '
            f'./env_configs/minigrid_object_remap_seed{seed}.json '
            f'--max_steps 512_000 --seed {seed}'
        )
    raise ValueError('unknown sensitivity environment: {0}'.format(environment))


def build_amsc_sensitivity_default_commands(environment):
    """Build the single canonical seven-seed sensitivity reference point."""
    if environment not in AMSC_SENSITIVITY_DEFAULTS:
        raise ValueError('unknown sensitivity environment: {0}'.format(environment))

    settings = dict(AMSC_SENSITIVITY_DEFAULTS[environment])
    return [
        [
            MIG_2_IDS[offset],
            _build_amsc_sensitivity_command(environment, settings, seed),
            'default',
        ]
        for offset, seed in enumerate(range(86, 93))
    ]


def build_amsc_primary_commands(environment, **overrides):
    """Build a fully specified seven-seed paper-primary AMSC configuration."""
    if environment not in AMSC_SENSITIVITY_DEFAULTS:
        raise ValueError('unknown primary AMSC environment: {0}'.format(environment))
    settings = dict(AMSC_SENSITIVITY_DEFAULTS[environment])
    unknown = set(overrides) - set(settings)
    if unknown:
        raise ValueError(
            'unknown primary AMSC setting(s): {0}'.format(sorted(unknown))
        )
    settings.update(overrides)
    return [
        [
            MIG_2_IDS[offset],
            _build_amsc_sensitivity_command(environment, settings, seed),
        ]
        for offset, seed in enumerate(range(86, 93))
    ]


def append_command_options(commands, options):
    """Append shared CLI options without discarding command metadata."""
    suffix = ' '.join(str(options).split())
    return [
        [command[0], f'{command[1]} {suffix}', *command[2:]]
        for command in commands
    ]


def interleave_labeled_command_sets(*command_sets):
    """Interleave equal-length command sets so variants launch concurrently."""
    if not command_sets:
        return []
    lengths = {len(commands) for _, commands in command_sets}
    if len(lengths) != 1:
        raise ValueError('interleaved command sets must have equal lengths')

    interleaved = []
    for index in range(lengths.pop()):
        for label, commands in command_sets:
            command = commands[index]
            interleaved.append([command[0], command[1], label])
    return interleaved


def build_amsc_sensitivity_commands(
    environment,
    parameter,
    include_default=False,
):
    """Build a seven-seed one-factor sweep without redundant defaults."""
    if environment not in AMSC_SENSITIVITY_DEFAULTS:
        raise ValueError('unknown sensitivity environment: {0}'.format(environment))
    if parameter not in AMSC_SENSITIVITY_GRIDS:
        raise ValueError('unknown sensitivity parameter: {0}'.format(parameter))

    defaults = AMSC_SENSITIVITY_DEFAULTS[environment]
    commands = []
    for value in AMSC_SENSITIVITY_GRIDS[parameter][environment]:
        if not include_default and value == defaults[parameter]:
            continue
        settings = dict(defaults)
        settings[parameter] = value
        value_label = _sensitivity_value_label(value)

        for offset, seed in enumerate(range(86, 93)):
            commands.append([
                MIG_2_IDS[offset],
                _build_amsc_sensitivity_command(environment, settings, seed),
                f'{parameter}/{parameter}_{value_label}',
            ])
    return commands


def build_soft_retrieval_commands(environment):
    """Use absolute confidence to initialize composition, not reject support."""
    commands = []
    for offset, seed in enumerate(range(86, 93)):
        common = (
            '--new_task_mask linear_comb '
            '--selection_soft_temperature 1.0 '
            '--selection_retrieval_beta_init '
            '--selection_beta_init_gamma 2.0 '
            '--disable_task_label_input '
            '--mask_score_normalization l2 '
            '--mask_score_normalize_current'
        )
        if environment == 'ctgraph':
            command = (
                f'python train_ctgraph.py ll_supermask {common} '
                f'--max_steps 51200 --seed {seed}'
            )
        elif environment == 'minigrid':
            command = (
                f'python train_minigrid.py ll_supermask {common} '
                f'--env_config_path '
                f'./env_configs/minigrid_object_remap_seed{seed}.json '
                f'--max_steps 512_000 --seed {seed}'
            )
        else:
            raise ValueError(f'unknown soft-retrieval environment: {environment}')
        commands.append([MIG_2_IDS[offset], command])
    return commands


def build_learned_gate_commands(environment):
    """Learn total prior reuse while retaining sparsemax source retrieval."""
    commands = []
    for offset, seed in enumerate(range(86, 93)):
        common = (
            '--new_task_mask linear_comb '
            '--selection_soft_temperature 1.0 '
            '--selection_learned_gate '
            '--selection_learned_gate_initial_probability 0.5 '
            '--disable_task_label_input'
        )
        if environment == 'ctgraph':
            command = (
                f'python train_ctgraph.py ll_supermask {common} '
                f'--max_steps 51200 --seed {seed}'
            )
        elif environment == 'minigrid':
            command = (
                f'python train_minigrid.py ll_supermask {common} '
                f'--env_config_path '
                f'./env_configs/minigrid_object_remap_seed{seed}.json '
                f'--max_steps 512_000 --seed {seed}'
            )
        elif environment == 'continualworld':
            command = (
                f'python train_continualworld.py ll_supermask {common} '
                f'--env_config_path ./env_configs/continualworld_10.json '
                f'--max_steps 2_000_000 --seed {seed}'
            )
        else:
            raise ValueError(f'unknown learned-gate environment: {environment}')
        commands.append([MIG_2_IDS[offset], command])
    return commands


commands_ctgraph_hard_gate = build_hard_gate_commands('ctgraph')
commands_ctgraph_hard_gate_beta_init = build_hard_gate_commands(
    'ctgraph',
    initialize_betas=True,
)
commands_minigrid_hard_gate = build_hard_gate_commands('minigrid')
commands_minigrid_hard_gate_beta_init = build_hard_gate_commands(
    'minigrid',
    initialize_betas=True,
)
commands_ctgraph_hard_gate_l2_raw_abstain = build_hard_gate_commands(
    'ctgraph',
    skip_current_normalization_on_abstain=True,
)
commands_ctgraph_hard_gate_l2_raw_abstain_no_comp = build_hard_gate_commands(
    'ctgraph',
    skip_current_normalization_on_abstain=True,
    disable_competence_gate=True,
)
commands_minigrid_hard_gate_l2_raw_abstain = build_hard_gate_commands(
    'minigrid',
    skip_current_normalization_on_abstain=True,
)
commands_minigrid_hard_gate_l2_raw_abstain_no_comp = build_hard_gate_commands(
    'minigrid',
    skip_current_normalization_on_abstain=True,
    disable_competence_gate=True,
)
commands_ctgraph_hard_gate_match_current_l2 = build_hard_gate_commands(
    'ctgraph',
    mask_score_normalization='match_current_l2',
)
commands_ctgraph_hard_gate_match_current_l2_no_comp = build_hard_gate_commands(
    'ctgraph',
    disable_competence_gate=True,
    mask_score_normalization='match_current_l2',
)
commands_minigrid_hard_gate_match_current_l2 = build_hard_gate_commands(
    'minigrid',
    mask_score_normalization='match_current_l2',
)
commands_minigrid_hard_gate_match_current_l2_no_comp = build_hard_gate_commands(
    'minigrid',
    disable_competence_gate=True,
    mask_score_normalization='match_current_l2',
)
commands_ctgraph_rms_ste_no_comp = build_rms_ste_commands('ctgraph')
commands_minigrid_rms_ste_no_comp = build_rms_ste_commands('minigrid')
commands_ctgraph_hard_gate_rms_ste_no_comp = build_rms_ste_commands(
    'ctgraph',
    hard_gate=True,
)
commands_minigrid_hard_gate_rms_ste_no_comp = build_rms_ste_commands(
    'minigrid',
    hard_gate=True,
)
commands_ctgraph_hard_gate_rms_ste = build_rms_ste_commands(
    'ctgraph',
    hard_gate=True,
    disable_competence_gate=False,
)
commands_minigrid_hard_gate_rms_ste = build_rms_ste_commands(
    'minigrid',
    hard_gate=True,
    disable_competence_gate=False,
)
commands_ctgraph_l2_ste_no_comp = build_mask_score_composition_commands(
    'ctgraph',
    'l2_ste',
)
commands_ctgraph_lora_rank4_task_local = build_ctgraph_lora_commands(
    4, 'task_local'
)
commands_ctgraph_lora_rank8_task_local = build_ctgraph_lora_commands(
    8, 'task_local'
)
commands_ctgraph_lora_rank4_sparsemax = build_ctgraph_lora_commands(
    4, 'sparsemax'
)
commands_ctgraph_lora_memory_baselines = (
    commands_ctgraph_lora_rank4_task_local
    + commands_ctgraph_lora_rank8_task_local
    + commands_ctgraph_lora_rank4_sparsemax
)
commands_minigrid_l2_ste_no_comp = build_mask_score_composition_commands(
    'minigrid',
    'l2_ste',
)
commands_ctgraph_l2_ste_uniform_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='uniform',
    )
)
commands_minigrid_l2_ste_uniform_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='uniform',
    )
)
commands_ctgraph_l2_ste_sparsemax_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
    )
)
commands_ctgraph_amsc_primary_rho0 = build_amsc_primary_commands(
    'ctgraph',
    ema=0.0,
)
commands_minigrid_amsc_primary_rho0_n1024_r10 = build_amsc_primary_commands(
    'minigrid',
    ema=0.0,
    sample_count=1024,
    reference_size=10,
)
commands_ctgraph_l2_ste_sparsemax_binary_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='binary',
    )
)
commands_ctgraph_l2_ste_sparsemax_ternary_0p1_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.1,
    )
)
commands_ctgraph_l2_ste_sparsemax_ternary_0p25_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.25,
    )
)
commands_ctgraph_l2_ste_sparsemax_ternary_0p5_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.5,
    )
)
commands_ctgraph_prior_l2_sparsemax_ternary_0p5_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        normalize_current=False,
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.5,
    )
)
commands_ctgraph_no_l2_sparsemax_ternary_0p5_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'none',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.5,
    )
)
commands_ctgraph_ternary_0p5_normalization_ablation_no_comp = (
    commands_ctgraph_prior_l2_sparsemax_ternary_0p5_storage_no_comp
    + commands_ctgraph_no_l2_sparsemax_ternary_0p5_storage_no_comp
)
commands_ctgraph_l2_ste_sparsemax_ternary_0p625_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.625,
    )
)
commands_ctgraph_l2_ste_sparsemax_ternary_0p75_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.75,
    )
)
commands_ctgraph_l2_ste_sparsemax_ternary_0p875_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=0.875,
    )
)
commands_ctgraph_l2_ste_sparsemax_ternary_1p0_storage_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        storage_quantization='ternary',
        storage_threshold_rms=1.0,
    )
)
commands_ctgraph_l2_ste_sparsemax_ternary_high_sweep_no_comp = (
    commands_ctgraph_l2_ste_sparsemax_ternary_0p625_storage_no_comp
    + commands_ctgraph_l2_ste_sparsemax_ternary_0p75_storage_no_comp
    + commands_ctgraph_l2_ste_sparsemax_ternary_0p875_storage_no_comp
    + commands_ctgraph_l2_ste_sparsemax_ternary_1p0_storage_no_comp
)
commands_ctgraph_l2_ste_sparsemax_quantization_no_comp = (
    commands_ctgraph_l2_ste_sparsemax_binary_storage_no_comp
    + commands_ctgraph_l2_ste_sparsemax_ternary_0p1_storage_no_comp
    + commands_ctgraph_l2_ste_sparsemax_ternary_0p25_storage_no_comp
    + commands_ctgraph_l2_ste_sparsemax_ternary_0p5_storage_no_comp
)
commands_ctgraph_l2_ste_sparsemax_freeze_first_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        freeze_after_first=True,
    )
)
commands_ctgraph_l2_ste_sparsemax_top1_matched_mass_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        top1_matched_prior_mass=True,
    )
)
commands_minigrid_l2_ste_sparsemax_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
    )
)
commands_minigrid_l2_ste_sparsemax_freeze_first_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
        freeze_after_first=True,
    )
)
commands_minigrid_l2_ste_sparsemax_top1_matched_mass_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
        top1_matched_prior_mass=True,
    )
)
commands_minigrid_l2_ste_retrieval_ablations_no_comp = (
    [
        [mig_id, command, 'freeze_first']
        for mig_id, command in
        commands_minigrid_l2_ste_sparsemax_freeze_first_no_comp
    ]
    + [
        [mig_id, command, 'top1_matched_mass']
        for mig_id, command in
        commands_minigrid_l2_ste_sparsemax_top1_matched_mass_no_comp
    ]
)

# Match the tuned MG16 paper configuration while changing only the ablated
# composition or retrieval component.
MG16_N1024_R50_OPTIONS = (
    '--detect_num_samples 1024 '
    '--detect_reference_num 50 '
    '--detect_frequency 1 '
    '--select_frequency 5 '
    '--embedding_ema 0.5 '
    '--detect_reference_seed 98'
)
commands_minigrid_n1024_learned_alpha_no_comp = append_command_options(
    commands_minigrid_l2_ste_no_comp,
    MG16_N1024_R50_OPTIONS,
)
commands_minigrid_n1024_uniform_no_comp = append_command_options(
    commands_minigrid_l2_ste_uniform_no_comp,
    MG16_N1024_R50_OPTIONS,
)
commands_minigrid_n1024_freeze_first_no_comp = append_command_options(
    commands_minigrid_l2_ste_sparsemax_freeze_first_no_comp,
    MG16_N1024_R50_OPTIONS,
)
commands_minigrid_n1024_priority_ablations_no_comp = (
    interleave_labeled_command_sets(
        ('learned_alpha', commands_minigrid_n1024_learned_alpha_no_comp),
        ('uniform', commands_minigrid_n1024_uniform_no_comp),
        ('freeze_first', commands_minigrid_n1024_freeze_first_no_comp),
    )
)
commands_ctgraph_l2_ste_raw_cosine_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='raw_cosine',
    )
)
commands_minigrid_l2_ste_raw_cosine_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='raw_cosine',
    )
)
commands_ctgraph_l2_ste_random_matched_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        shuffle_support=True,
    )
)
commands_minigrid_l2_ste_random_matched_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
        shuffle_support=True,
    )
)
commands_ctgraph_l2_ste_oracle_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_depth_prefix',
        family_stride=4,
    )
)
commands_minigrid_l2_ste_oracle_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_depth_prefix',
        family_stride=4,
    )
)
commands_ctgraph_l2_ste_oracle_similarity_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_depth_prefix',
        family_stride=4,
        oracle_weighting='similarity',
    )
)
commands_minigrid_l2_ste_oracle_similarity_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_depth_prefix',
        family_stride=4,
        oracle_weighting='similarity',
    )
)
commands_minigrid_l2_ste_transfer_oracle_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_transfer',
        oracle_weighting='transfer',
        oracle_transfer_config=(
            './env_configs/oracles/mg16_pairwise_best_prior_seed86.json'
        ),
    )
)
commands_ctgraph_l2_ste_transfer_oracle_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_transfer',
        oracle_weighting='transfer',
        oracle_transfer_config=(
            './env_configs/oracles/'
            'ct28_pairwise_composition_best_prior_seed92.json'
        ),
    )
)
commands_ctgraph_l2_ste_all_positive_composition_transfer_oracle_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_transfer',
        oracle_weighting='transfer',
        oracle_transfer_config=(
            './env_configs/oracles/'
            'ct28_pairwise_composition_all_positive_seed92.json'
        ),
    )
)
commands_ctgraph_l2_ste_all_positive_transfer_oracle_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_transfer',
        oracle_weighting='transfer',
        oracle_transfer_config=(
            './env_configs/oracles/'
            'ct28_pairwise_finetune_all_positive_seed92.json'
        ),
    )
)
commands_minigrid_l2_ste_all_positive_transfer_oracle_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        fixed_composition='sparsemax',
        select_strategy='oracle_transfer',
        oracle_weighting='transfer',
        oracle_transfer_config=(
            './env_configs/oracles/'
            'mg16_pairwise_all_positive_seed86.json'
        ),
    )
)
commands_ctgraph_sparsemax_fixed_no_l2_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'none',
        fixed_composition='sparsemax',
    )
)
commands_minigrid_sparsemax_fixed_no_l2_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'none',
        fixed_composition='sparsemax',
    )
)
commands_ctgraph_hard_gate_l2_ste_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'l2_ste',
        hard_gate=True,
    )
)
commands_minigrid_hard_gate_l2_ste_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'l2_ste',
        hard_gate=True,
    )
)
commands_ctgraph_softmax_masks_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'softmax_across_masks',
    )
)
commands_minigrid_softmax_masks_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'softmax_across_masks',
    )
)
commands_ctgraph_hard_gate_softmax_masks_no_comp = (
    build_mask_score_composition_commands(
        'ctgraph',
        'softmax_across_masks',
        hard_gate=True,
    )
)
commands_minigrid_hard_gate_softmax_masks_no_comp = (
    build_mask_score_composition_commands(
        'minigrid',
        'softmax_across_masks',
        hard_gate=True,
    )
)
commands_ctgraph_soft_retrieval = build_soft_retrieval_commands('ctgraph')
commands_minigrid_soft_retrieval = build_soft_retrieval_commands('minigrid')
commands_ctgraph_learned_gate = build_learned_gate_commands('ctgraph')
commands_minigrid_learned_gate = build_learned_gate_commands('minigrid')
commands_continualworld_learned_gate = build_learned_gate_commands(
    'continualworld'
)


COMMAND_SETS = {
    'ctgraph_lc': commands_ctgraph_lc,
    'minigrid': commands_minigrid,
    'minigrid_individual': commands_minigrid_single_task_experts,
    'ctgraph_ppo': commands_ctgraph_ppo,
    'ctgraph_sc': commands_ctgraph_sc,
    'ctgraph_individual': commands_ctgraph_single_task_experts,
    'continualworld': commands_continualworld,
    'ctgraph_sc_no_norm': commands_ctgraph_sc_no_norm,
    'ctgraph_sc_no_comp': commands_ctgraph_sc_no_comp,
    'ctgraph_sc_shuffled': commands_ctgraph_sc_shuffled,
    'ctgraph_sc_swe': commands_ctgraph_sc_swe,
    'minigrid_swe': commands_minigrid_swe,
    'ctgraph_sc_d8_recovery': commands_ctgraph_sc_d8_recovery,
    'minigrid_no_norm': commands_minigrid_no_norm,
    'minigrid_no_comp': commands_minigrid_no_comp,
    'minigrid_shuffled': commands_minigrid_shuffled,
    'ctgraph_sc_oracle': commands_ctgraph_sc_oracle,
    'minigrid_oracle': commands_minigrid_oracle,
    'ctgraph_hard_gate': commands_ctgraph_hard_gate,
    'ctgraph_hard_gate_beta_init': commands_ctgraph_hard_gate_beta_init,
    'minigrid_hard_gate': commands_minigrid_hard_gate,
    'minigrid_hard_gate_beta_init': commands_minigrid_hard_gate_beta_init,
    'ctgraph_hard_gate_l2_raw_abstain': (
        commands_ctgraph_hard_gate_l2_raw_abstain
    ),
    'ctgraph_hard_gate_l2_raw_abstain_no_comp': (
        commands_ctgraph_hard_gate_l2_raw_abstain_no_comp
    ),
    'minigrid_hard_gate_l2_raw_abstain': (
        commands_minigrid_hard_gate_l2_raw_abstain
    ),
    'minigrid_hard_gate_l2_raw_abstain_no_comp': (
        commands_minigrid_hard_gate_l2_raw_abstain_no_comp
    ),
    'ctgraph_hard_gate_match_current_l2': (
        commands_ctgraph_hard_gate_match_current_l2
    ),
    'ctgraph_hard_gate_match_current_l2_no_comp': (
        commands_ctgraph_hard_gate_match_current_l2_no_comp
    ),
    'minigrid_hard_gate_match_current_l2': (
        commands_minigrid_hard_gate_match_current_l2
    ),
    'minigrid_hard_gate_match_current_l2_no_comp': (
        commands_minigrid_hard_gate_match_current_l2_no_comp
    ),
    'ctgraph_rms_ste_no_comp': commands_ctgraph_rms_ste_no_comp,
    'minigrid_rms_ste_no_comp': commands_minigrid_rms_ste_no_comp,
    'ctgraph_hard_gate_rms_ste_no_comp': (
        commands_ctgraph_hard_gate_rms_ste_no_comp
    ),
    'minigrid_hard_gate_rms_ste_no_comp': (
        commands_minigrid_hard_gate_rms_ste_no_comp
    ),
    'ctgraph_hard_gate_rms_ste': commands_ctgraph_hard_gate_rms_ste,
    'minigrid_hard_gate_rms_ste': commands_minigrid_hard_gate_rms_ste,
    'ctgraph_l2_ste_no_comp': commands_ctgraph_l2_ste_no_comp,
    'ctgraph_lora_rank4_task_local': (
        commands_ctgraph_lora_rank4_task_local
    ),
    'ctgraph_lora_rank8_task_local': (
        commands_ctgraph_lora_rank8_task_local
    ),
    'ctgraph_lora_rank4_sparsemax': (
        commands_ctgraph_lora_rank4_sparsemax
    ),
    'ctgraph_lora_memory_baselines': (
        commands_ctgraph_lora_memory_baselines
    ),
    'minigrid_l2_ste_no_comp': commands_minigrid_l2_ste_no_comp,
    'ctgraph_l2_ste_uniform_no_comp': (
        commands_ctgraph_l2_ste_uniform_no_comp
    ),
    'minigrid_l2_ste_uniform_no_comp': (
        commands_minigrid_l2_ste_uniform_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_no_comp
    ),
    'ctgraph_amsc_primary_rho0': (
        commands_ctgraph_amsc_primary_rho0
    ),
    'minigrid_amsc_primary_rho0_n1024_r10': (
        commands_minigrid_amsc_primary_rho0_n1024_r10
    ),
    'ctgraph_l2_ste_sparsemax_binary_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_binary_storage_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_0p1_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_0p1_storage_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_0p25_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_0p25_storage_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_0p5_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_0p5_storage_no_comp
    ),
    'ctgraph_prior_l2_sparsemax_ternary_0p5_storage_no_comp': (
        commands_ctgraph_prior_l2_sparsemax_ternary_0p5_storage_no_comp
    ),
    'ctgraph_no_l2_sparsemax_ternary_0p5_storage_no_comp': (
        commands_ctgraph_no_l2_sparsemax_ternary_0p5_storage_no_comp
    ),
    'ctgraph_ternary_0p5_normalization_ablation_no_comp': (
        commands_ctgraph_ternary_0p5_normalization_ablation_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_0p625_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_0p625_storage_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_0p75_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_0p75_storage_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_0p875_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_0p875_storage_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_1p0_storage_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_1p0_storage_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_ternary_high_sweep_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_ternary_high_sweep_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_quantization_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_quantization_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_freeze_first_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_freeze_first_no_comp
    ),
    'ctgraph_l2_ste_sparsemax_top1_matched_mass_no_comp': (
        commands_ctgraph_l2_ste_sparsemax_top1_matched_mass_no_comp
    ),
    'minigrid_l2_ste_sparsemax_no_comp': (
        commands_minigrid_l2_ste_sparsemax_no_comp
    ),
    'minigrid_l2_ste_sparsemax_freeze_first_no_comp': (
        commands_minigrid_l2_ste_sparsemax_freeze_first_no_comp
    ),
    'minigrid_l2_ste_sparsemax_top1_matched_mass_no_comp': (
        commands_minigrid_l2_ste_sparsemax_top1_matched_mass_no_comp
    ),
    'minigrid_l2_ste_retrieval_ablations_no_comp': (
        commands_minigrid_l2_ste_retrieval_ablations_no_comp
    ),
    'minigrid_n1024_priority_ablations_no_comp': (
        commands_minigrid_n1024_priority_ablations_no_comp
    ),
    'ctgraph_l2_ste_raw_cosine_no_comp': (
        commands_ctgraph_l2_ste_raw_cosine_no_comp
    ),
    'minigrid_l2_ste_raw_cosine_no_comp': (
        commands_minigrid_l2_ste_raw_cosine_no_comp
    ),
    'ctgraph_l2_ste_random_matched_no_comp': (
        commands_ctgraph_l2_ste_random_matched_no_comp
    ),
    'minigrid_l2_ste_random_matched_no_comp': (
        commands_minigrid_l2_ste_random_matched_no_comp
    ),
    'ctgraph_l2_ste_oracle_no_comp': (
        commands_ctgraph_l2_ste_oracle_no_comp
    ),
    'minigrid_l2_ste_oracle_no_comp': (
        commands_minigrid_l2_ste_oracle_no_comp
    ),
    'ctgraph_l2_ste_oracle_similarity_no_comp': (
        commands_ctgraph_l2_ste_oracle_similarity_no_comp
    ),
    'minigrid_l2_ste_oracle_similarity_no_comp': (
        commands_minigrid_l2_ste_oracle_similarity_no_comp
    ),
    'minigrid_l2_ste_transfer_oracle_no_comp': (
        commands_minigrid_l2_ste_transfer_oracle_no_comp
    ),
    'ctgraph_l2_ste_transfer_oracle_no_comp': (
        commands_ctgraph_l2_ste_transfer_oracle_no_comp
    ),
    'ctgraph_l2_ste_all_positive_transfer_oracle_no_comp': (
        commands_ctgraph_l2_ste_all_positive_transfer_oracle_no_comp
    ),
    'ctgraph_l2_ste_all_positive_composition_transfer_oracle_no_comp': (
        commands_ctgraph_l2_ste_all_positive_composition_transfer_oracle_no_comp
    ),
    'minigrid_l2_ste_all_positive_transfer_oracle_no_comp': (
        commands_minigrid_l2_ste_all_positive_transfer_oracle_no_comp
    ),
    'ctgraph_sparsemax_fixed_no_l2_no_comp': (
        commands_ctgraph_sparsemax_fixed_no_l2_no_comp
    ),
    'minigrid_sparsemax_fixed_no_l2_no_comp': (
        commands_minigrid_sparsemax_fixed_no_l2_no_comp
    ),
    'ctgraph_hard_gate_l2_ste_no_comp': (
        commands_ctgraph_hard_gate_l2_ste_no_comp
    ),
    'minigrid_hard_gate_l2_ste_no_comp': (
        commands_minigrid_hard_gate_l2_ste_no_comp
    ),
    'ctgraph_softmax_masks_no_comp': commands_ctgraph_softmax_masks_no_comp,
    'minigrid_softmax_masks_no_comp': commands_minigrid_softmax_masks_no_comp,
    'ctgraph_hard_gate_softmax_masks_no_comp': (
        commands_ctgraph_hard_gate_softmax_masks_no_comp
    ),
    'minigrid_hard_gate_softmax_masks_no_comp': (
        commands_minigrid_hard_gate_softmax_masks_no_comp
    ),
    'ctgraph_soft_retrieval': commands_ctgraph_soft_retrieval,
    'minigrid_soft_retrieval': commands_minigrid_soft_retrieval,
    'ctgraph_learned_gate': commands_ctgraph_learned_gate,
    'minigrid_learned_gate': commands_minigrid_learned_gate,
    'continualworld_learned_gate': commands_continualworld_learned_gate,
}

for sensitivity_environment in ('ctgraph', 'minigrid'):
    default_commands = build_amsc_sensitivity_default_commands(
        sensitivity_environment
    )
    COMMAND_SETS['{0}_sensitivity_default'.format(sensitivity_environment)] = (
        default_commands
    )
    all_sensitivity_commands = list(default_commands)
    for sensitivity_parameter in AMSC_SENSITIVITY_GRIDS:
        command_set_name = '{0}_sensitivity_{1}'.format(
            sensitivity_environment,
            sensitivity_parameter,
        )
        parameter_commands = build_amsc_sensitivity_commands(
            sensitivity_environment,
            sensitivity_parameter,
        )
        COMMAND_SETS[command_set_name] = parameter_commands
        all_sensitivity_commands.extend(parameter_commands)
    COMMAND_SETS['{0}_sensitivity_all'.format(sensitivity_environment)] = (
        all_sensitivity_commands
    )

def parse_gpu_groups(raw_values):
    if raw_values is None:
        return None

    # Accept: --gpu 0 1, --gpu 0,1, --gpu "[0, 1]", or --gpu [0, 1].
    cleaned = ' '.join(raw_values).replace('[', ' ').replace(']', ' ').replace(',', ' ')
    values = cleaned.split()
    if not values:
        raise ValueError('--gpu requires at least one MIG group')

    groups = []
    for value in values:
        try:
            group = int(value)
        except ValueError as exc:
            raise ValueError(f'invalid MIG group {value!r}; available groups: {sorted(MIG_GROUPS)}') from exc
        if group not in MIG_GROUPS:
            raise ValueError(f'unknown MIG group {group}; available groups: {sorted(MIG_GROUPS)}')
        if group not in groups:
            groups.append(group)
    return groups


def build_mig_pool(group_ids):
    """Interleave selected groups so multi-group runs use every physical GPU."""
    pool = []
    max_group_size = max(len(MIG_GROUPS[group]) for group in group_ids)
    for slot in range(max_group_size):
        for group in group_ids:
            if slot < len(MIG_GROUPS[group]):
                pool.append(MIG_GROUPS[group][slot])
    return pool


def assign_mig_ids(commands, group_ids):
    if group_ids is None:
        return [command[0] for command in commands]

    pool = build_mig_pool(group_ids)
    return [pool[index % len(pool)] for index in range(len(commands))]


def assign_mig_queues(commands, group_ids):
    """Assign commands to logical queues and their CUDA-visible MIGs.

    Logical queue IDs keep repeated MIG UUIDs distinct. This permits groups 3
    and 4 to run two serial queues concurrently on each shared physical MIG.
    """
    mig_ids = assign_mig_ids(commands, group_ids)
    if group_ids is None:
        return [(mig_id, mig_id) for mig_id in mig_ids]

    pool_size = len(build_mig_pool(group_ids))
    return [
        ('slot-{0:02d}'.format(index % pool_size), mig_id)
        for index, mig_id in enumerate(mig_ids)
    ]


def _run_device_queue(queue_id, mig_id, command_texts, keep_going=False):
    """Run commands serially on one MIG while other MIG queues run in parallel."""
    failures = []
    for command_text in command_texts:
        queue_label = f'{queue_id} ({mig_id})'
        print(f'[start] {queue_label}: {command_text}', flush=True)
        process_env = dict(os.environ)
        process_env['CUDA_VISIBLE_DEVICES'] = mig_id
        result = subprocess.run(
            shlex.split(command_text),
            env=process_env,
            check=False,
        )
        if result.returncode != 0:
            failure = (
                f'command failed with exit code {result.returncode}: '
                f'{command_text}'
            )
            if not keep_going:
                raise RuntimeError(failure)
            failures.append(failure)
            print(f'[failed] {queue_label}: {failure}', flush=True)
            continue
        print(f'[done] {queue_label}: {command_text}', flush=True)
    return failures


def _command_seed(command_text):
    """Return the integer passed to --seed, if present."""
    tokens = shlex.split(command_text)
    for index, token in enumerate(tokens[:-1]):
        if token == '--seed':
            try:
                return int(tokens[index + 1])
            except ValueError:
                return None
    return None


def _command_option(command_text, option):
    """Return the value following a command-line option, if present."""
    tokens = shlex.split(command_text)
    for index, token in enumerate(tokens[:-1]):
        if token == option:
            return tokens[index + 1]
    return None


def _command_has_flag(command_text, flag):
    """Return whether a standalone command-line flag is present."""
    return flag in shlex.split(command_text)


def _completed_seed_exists(path_header, command_text):
    """Check for a completed config matching the seed and run-defining options."""
    seed = _command_seed(command_text)
    storage_mode = _command_option(
        command_text,
        '--mask_score_storage_quantization',
    )
    storage_threshold = _command_option(
        command_text,
        '--mask_score_storage_threshold_rms',
    )
    score_normalization = _command_option(
        command_text,
        '--mask_score_normalization',
    )
    normalize_current = _command_has_flag(
        command_text,
        '--mask_score_normalize_current',
    )
    lora_rank = _command_option(command_text, '--lora_rank')
    lora_composition = _command_option(command_text, '--lora_composition')
    embedding_ema = _command_option(command_text, '--embedding_ema')
    detect_num_samples = _command_option(command_text, '--detect_num_samples')
    detect_reference_num = _command_option(command_text, '--detect_reference_num')
    detect_frequency = _command_option(command_text, '--detect_frequency')
    select_frequency = _command_option(command_text, '--select_frequency')
    # Training scripts interpret -p as a header beneath their ./log directory.
    root = Path('log') / path_header
    if seed is None or not root.is_dir():
        return False

    for config_path in root.rglob('config.json'):
        try:
            with config_path.open('r') as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            continue
        try:
            completed_seed = int(config.get('seed'))
        except (TypeError, ValueError):
            continue
        if completed_seed != seed:
            continue

        if storage_mode is not None:
            if config.get('mask_score_storage_quantization', 'none') != storage_mode:
                continue
        if storage_threshold is not None:
            try:
                completed_threshold = float(
                    config.get('mask_score_storage_threshold_rms')
                )
                requested_threshold = float(storage_threshold)
            except (TypeError, ValueError):
                continue
            if not math.isclose(
                completed_threshold,
                requested_threshold,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                continue
        if score_normalization is not None:
            if config.get('mask_score_normalization', 'none') != score_normalization:
                continue
            if bool(
                config.get('mask_score_normalize_current', False)
            ) != normalize_current:
                continue
        if lora_rank is not None:
            try:
                if int(config.get('lora_rank')) != int(lora_rank):
                    continue
            except (TypeError, ValueError):
                continue
        if lora_composition is not None:
            if config.get('lora_composition') != lora_composition:
                continue
        numeric_options = (
            (embedding_ema, 'embedding_ema', float),
            (detect_num_samples, 'detect_num_samples', int),
            (detect_reference_num, 'detect_reference_num', int),
            (detect_frequency, 'detect_frequency', int),
            (select_frequency, 'select_frequency', int),
        )
        numeric_mismatch = False
        for requested, config_key, converter in numeric_options:
            if requested is None:
                continue
            try:
                observed = converter(config.get(config_key))
                expected = converter(requested)
            except (TypeError, ValueError):
                numeric_mismatch = True
                break
            if converter is float:
                if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12):
                    numeric_mismatch = True
                    break
            elif observed != expected:
                numeric_mismatch = True
                break
        if numeric_mismatch:
            continue
        return True
    return False



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        help='indicate which experiment is being run for command selection',
        type=str,
        default='ctgraph_sc',
        choices=sorted(COMMAND_SETS),
    )
    parser.add_argument('--exp', help='', type=str, default='')
    parser.add_argument(
        '--legacy_wte_ema',
        '--legacy-wte-ema',
        dest='legacy_wte_ema',
        help='append the legacy WTE EMA compatibility flag to every command',
        action='store_true',
    )
    parser.add_argument(
        '--gpu',
        nargs='+',
        metavar='GROUP',
        help=(
            'MIG group(s) used to schedule commands: 0 selects the first seven '
            'MIG instances and 1 selects the second seven on the original host. '
            'Groups 3 and 4 are two logical queues over the same seven 10 GB '
            'MIGs on the 80 GB host. Multiple groups are interleaved. Examples: '
            '--gpu 0, --gpu 0 1, --gpu 3 4, --gpu "[0, 1]". '
            'If omitted, each command keeps its hard-coded MIG UUID.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        help='print the resolved commands and GPU assignments without launching',
        action='store_true',
    )
    parser.add_argument(
        '--keep-going',
        help='continue the remaining commands in a logical MIG queue after a failure',
        action='store_true',
    )
    parser.add_argument(
        '--skip-completed',
        help=(
            'skip a seed when its output directory already contains the '
            'config.json completion marker; useful when restarting sweeps'
        ),
        action='store_true',
    )
    args = parser.parse_args()

    commands = COMMAND_SETS[args.env]
    gpu_groups = parse_gpu_groups(args.gpu)
    if gpu_groups is None and args.env.endswith('_sensitivity_all'):
        gpu_groups = list(DEFAULT_SENSITIVITY_MIG_GROUPS)
    assigned_queues = assign_mig_queues(commands, gpu_groups)

    path_header = args.exp if args.exp else args.env
    
    jobs_by_queue = defaultdict(list)
    queue_mig_ids = {}
    for command, (queue_id, mig_id) in zip(commands, assigned_queues):
        command_path_header = path_header
        if len(command) > 2 and command[2]:
            command_path_header = os.path.join(path_header, command[2])

        command_text = command[1] + f' -p {shlex.quote(command_path_header)}'
        if args.legacy_wte_ema:
            command_text += ' --legacy_wte_ema'
        if args.skip_completed and _completed_seed_exists(
            command_path_header,
            command_text,
        ):
            print(f'[skip completed] {command_path_header}: {command_text}')
            continue
        print(f'{queue_id} ({mig_id}), {command_text}')
        jobs_by_queue[queue_id].append(command_text)
        queue_mig_ids[queue_id] = mig_id

    if args.dry_run:
        return

    if not jobs_by_queue:
        print('No commands remain after filtering completed runs.')
        return

    failures = []
    with ThreadPoolExecutor(max_workers=len(jobs_by_queue)) as executor:
        futures = {
            executor.submit(
                _run_device_queue,
                queue_id,
                queue_mig_ids[queue_id],
                command_texts,
                args.keep_going,
            ): queue_id
            for queue_id, command_texts in jobs_by_queue.items()
        }
        for future in as_completed(futures):
            queue_id = futures[future]
            try:
                queue_failures = future.result()
                failures.extend(
                    (queue_id, failure) for failure in queue_failures
                )
            except Exception as exc:
                failures.append((queue_id, exc))

    if failures:
        details = '\n'.join(
            f'  {queue_id}: {exc}' for queue_id, exc in failures
        )
        raise SystemExit(f'{len(failures)} command(s) failed:\n{details}')


if __name__ == '__main__':
    main()
