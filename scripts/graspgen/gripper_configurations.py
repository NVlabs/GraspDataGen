# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gripper configurations for datagen.

This file contains predefined gripper configurations that can be used with the
--gripper_config argument. Each configuration can override any parameter from
gripper.py, grasp_guess.py, or grasp_sim.py.
"""

import sys
from graspgen_utils import print_blue

# Gripper configurations dictionary
GRIPPER_CONFIGS = {
    'intrinsic': {
        'gripper_file': 'bots/intrinsic_pinch_gripper_prismatic_flip.usd',
        'finger_colliders': ['finger_link', 'mount_link'],
        'base_frame': 'mount_link',
        'bite': 0.02,
    },
    'robotiq_2f_85': {
        'gripper_file': 'bots/robotiq_2f_85.usd',
        'finger_colliders': ['right_inner_finger', 'left_inner_finger'],
        'base_frame': 'base_link',
        'bite': 0.0185,  # half of 37mm
    },
    'onrobot_rg6': {
        'gripper_file': 'bots/onrobot_rg6.usd',
        'finger_colliders': ['right_inner_finger', 'left_inner_finger'],
        'base_frame': 'base_frame',
    },
    'franka_panda': {
        'gripper_file': 'bots/franka_panda.usd',
        'finger_colliders': ['panda_rightfinger', 'panda_leftfinger'],
        'base_frame': 'panda_hand',
    },
}


def get_gripper_config(gripper_type):
    """
    Get gripper configuration based on gripper type string.
    
    Args:
        gripper_type (str): One of the predefined gripper types
        
    Returns:
        dict: Dictionary containing parameter overrides for the specified
        gripper type
        
    Raises:
        ValueError: If gripper_type is not found in configurations
    """
    if gripper_type not in GRIPPER_CONFIGS:
        available = list(GRIPPER_CONFIGS.keys())
        raise ValueError(f"Unknown gripper type: {gripper_type}. "
                        f"Must be one of: {available}")
    
    return GRIPPER_CONFIGS[gripper_type]

def apply_gripper_config_to_args(args, gripper_config):
    """
    Apply gripper configuration overrides to the args object.
    Only overrides values that weren't explicitly provided on the command line.
    
    Args:
        args: The parsed arguments object
        gripper_config: Dictionary of parameter overrides
    """
    # Apply each override to the args object
    for param_name, value in gripper_config.items():
        if hasattr(args, param_name):
            # Check if this argument was explicitly provided on command line
            arg_flag = f"--{param_name}"
            if arg_flag in sys.argv:
                # Find the index of the flag in sys.argv
                try:
                    flag_index = sys.argv.index(arg_flag)
                    # Check if there's a value after the flag (not another flag)
                    if flag_index + 1 < len(sys.argv) and not sys.argv[flag_index + 1].startswith('-'):
                        print_blue(f"  Skipping {param_name}: {value} (explicitly provided on command line)")
                        continue
                except ValueError:
                    pass  # Flag not found, continue with override
            
            setattr(args, param_name, value)
            print_blue(f"  Overriding {param_name}: {value}")
        else:
            # Let argparse handle unknown parameters - this will raise an error
            raise ValueError(f"Unknown parameter '{param_name}' in gripper config "
                           f"'{args.gripper_config}'")


def list_available_grippers():
    """
    List all available gripper configurations.
    
    Returns:
        list: List of available gripper type names
    """
    return list(GRIPPER_CONFIGS.keys()) 