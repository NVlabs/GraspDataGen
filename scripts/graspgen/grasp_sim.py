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

#!/usr/bin/env python3
# flake8: noqa: E303, E111
import argparse
from graspgen_utils import add_arg_to_group, save_yaml, print_green, add_isaac_lab_args_if_needed, print_purple, print_red, print_yellow, print_blue, start_isaac_lab_if_needed, open_configuration_string_to_dict, str_to_bool
from gripper import add_gripper_args, collect_gripper_args, apply_gripper_configuration
from object import add_object_args, collect_object_args, ObjectConfig
import os

USE_ORIGIN_PLACEMENT = False

default_grasp_file = os.path.join(os.environ.get('GRASP_DATASET_DIR', ''), "grasp_guess_data/onrobot_rg6/mug.yaml")
 
default_max_num_envs = 1024
default_max_num_grasps = 0
default_env_spacing = 1.0

default_fps = 250.0
default_force_magnitude = 1.0
default_initial_grasp_duration = 1.0
default_tug_sequences = [[0.5, [0, 0, 1], 1.0], [0.5, [0, 2, 1], 1.0], [0.50, [0, -2, 1], 1.0], [0.50, [2, 0, 1], 1.0], [0.50, [-2, 0, 1], 1.0]]
default_start_with_pregrasp_cspace_position = True
default_open_limit = "" # TODO Move this to gripper? It's only needed in this file.
default_grasp_file_cspace_position = "{}"

default_disable_sim = False
default_record_pvd = False
default_debug_single_index = 0
default_output_failed_grasp_locations = False
default_flip_input_grasps = False
default_enable_ccd = True

def collect_grasp_sim_args(input_dict):
    desired_keys = [
        "default_grasp_file",
        "default_max_num_envs",
        "default_max_num_grasps",
        "default_env_spacing",
        "default_fps",
        "default_force_magnitude",
        "default_initial_grasp_duration",
        "default_tug_sequences",
        "default_start_with_pregrasp_cspace_position",
        "default_open_limit",
        "default_grasp_file_cspace_position",
        "default_disable_sim",
        "default_record_pvd",
        "default_debug_single_index",
        "default_output_failed_grasp_locations",
        "default_flip_input_grasps",
        "default_enable_ccd"]
    kwargs = {}
    for key in desired_keys:
        if key in input_dict:
            kwargs[key] = input_dict[key]
        else:
            # Use local default if not provided in input_dict
            kwargs[key] = globals()[key]
    return kwargs

def add_grasp_sim_args(parser, param_dict, default_grasp_file=default_grasp_file, 
                       default_max_num_envs=default_max_num_envs,
                       default_max_num_grasps=default_max_num_grasps,
                       default_env_spacing=default_env_spacing,
                       default_fps=default_fps,
                       default_force_magnitude=default_force_magnitude,
                       default_initial_grasp_duration=default_initial_grasp_duration,
                       default_tug_sequences=default_tug_sequences,
                       default_start_with_pregrasp_cspace_position=default_start_with_pregrasp_cspace_position,
                       default_open_limit=default_open_limit,
                       default_grasp_file_cspace_position=default_grasp_file_cspace_position,
                       default_disable_sim=default_disable_sim,
                       
                       
                       default_record_pvd=default_record_pvd,
                       default_debug_single_index=default_debug_single_index,
                       default_output_failed_grasp_locations=default_output_failed_grasp_locations,
                       default_flip_input_grasps=default_flip_input_grasps,
                       default_enable_ccd=default_enable_ccd):
    
    # Register argument groups since we'll be adding arguments to them
    from graspgen_utils import register_argument_group
    register_argument_group(parser, 'grasp_sim', 'grasp_sim', 'Grasp simulation options')
    
    add_gripper_args(parser, param_dict, **collect_gripper_args(param_dict))
    add_object_args(parser, param_dict, **collect_object_args(param_dict))
    
    # The grasp validation sim can take an input isaac grasp file, or a grasp guess buffer.
    # This allows us to run this stand alone or with user supplied grasps.
    # TODO: If the grasp file has only grasps and the gripper file, then get the gripper parameters, like open_limit, from the gripper file.
    add_arg_to_group('grasp_sim', parser, "--grasp_file", type=str, default=default_grasp_file, help="Path to grasp file.")

    # InteractiveSceneCfg args
    add_arg_to_group('grasp_sim', parser, "--max_num_envs", type=int, default=default_max_num_envs, help="Maximum number of environments to spawn.")
    add_arg_to_group('grasp_sim', parser, "--max_num_grasps", type=int, default=default_max_num_grasps, help="Maximum number of grasps to process (0 means use all grasps).")
    add_arg_to_group('grasp_sim', parser, "--env_spacing", type=float, default=default_env_spacing, help="Spacing between environments.")

    # Simulation args
    add_arg_to_group('grasp_sim', parser, "--fps", type=float, default=default_fps, help="Simulation FPS (physics timestep = 1/FPS)")
    add_arg_to_group('grasp_sim', parser, "--force_magnitude", type=float, default=default_force_magnitude, help="Magnitude of force in Newtons to apply to object")
    add_arg_to_group('grasp_sim', parser, "--initial_grasp_duration", type=float, default=default_initial_grasp_duration, help="Initial grasp duration in seconds.")
    add_arg_to_group('grasp_sim', parser, "--tug_sequences", type=str, # sequence_str: JSON string in format [[duration, [x,y,z], force_scale], ...]
        default=str(default_tug_sequences),
        help="JSON-formatted list of tug sequences to pull on the object. Format: [[duration, [x,y,z], force_scale], ...]")
    add_arg_to_group('grasp_sim', parser, "--start_with_pregrasp_cspace_position", type=str_to_bool, nargs='?', const=True, default=default_start_with_pregrasp_cspace_position,
        help="Start with the pregrasp cspace position instead of the cspace position from the grasp input.")
    # Scene args (TODO: Make a file loader for the scene args)
    add_arg_to_group('grasp_sim', parser, "--open_limit", type=str, default=default_open_limit,
                        choices=["lower", "upper"],
                        help="The open position of the gripper is at this limit (fallback if not in the grasp_file).")
    # Fallback cspace for grasp files that lack cspace_position / pregrasp_cspace_position
    add_arg_to_group('grasp_sim', parser, "--grasp_file_cspace_position", type=str, default=default_grasp_file_cspace_position,
                     help="JSON mapping joint name (str) -> position (float) to use when grasp file lacks cspace.")
    
    # debugging args
    add_arg_to_group('grasp_sim', parser, "--disable_sim", action="store_true", default=default_disable_sim,
        help="Disable simulation")
    add_arg_to_group('grasp_sim', parser, "--record_pvd", action="store_true", default=default_record_pvd,
        help="Record PVD files for debugging")
    # If we want to debug by looking at a single index from a file, then set this to the index... a value of 0 does nothing... 
    # if you want the first index in a file then set max_num_envs to 1 and this to 0, and use --force_headed so you can see the single grasp
    # before the simulation closes.  Tricky, I know, but it's for debugging, not normal use.
    add_arg_to_group('grasp_sim', parser, "--debug_single_index", type=int, default=default_debug_single_index,
        help="Load a single grasp index from a grasp file for debugging (0=no effect, 1=first grasp, etc.)")
    add_arg_to_group('grasp_sim', parser, "--output_failed_grasp_locations", action="store_true", default=default_output_failed_grasp_locations,
        help="Output the locations of failed grasps for debugging purposes.")
    add_arg_to_group('grasp_sim', parser, "--flip_input_grasps", action="store_true", default=default_flip_input_grasps,
        help="Flip (rotate 180 degrees around approach-axis) the input grasps for debugging purposes.")
    add_arg_to_group('grasp_sim', parser, "--enable_ccd", type=str_to_bool, nargs='?', const=True, default=default_enable_ccd,
        help="Enable continuous collision detection (CCD) in the physics simulation (prevents fast-moving objects from tunneling through each other).")

    add_isaac_lab_args_if_needed(parser)

import yaml
import numpy as np
import torch
import os
import sys
import math
import copy
import warp as wp
import json
from warp_kernels import (
    world_to_object_force_kernel, get_joint_pos_kernel, compute_relative_pos_and_rot_kernel, 
    get_joint_pos_kernel, set_is_success_kernel, get_body_pos_kernel, transform_inverse_kernel, get_cspace_positions_kernel, get_bite_points_kernel
)
import time
from datetime import datetime

def parse_tug_sequences(sequences):
    """Parse tug sequences from command line argument.
    
    Args:
        sequence_str: JSON string in format [[duration, [x,y,z], force_scale], ...]
        
    Returns:
        List of [duration, [x,y,z], force_scale] sequences
    """
    try:
        if isinstance(sequences, str):
            sequences = json.loads(sequences)
        # First validate the JSON structure
        if not isinstance(sequences, list):
            raise ValueError("Input must be a list of sequences")
        
        # Validate and normalize each sequence
        normalized_sequences = []
        for i, seq in enumerate(sequences):
            if not isinstance(seq, list) or len(seq) != 3:
                raise ValueError(f"Sequence {i} must be a list of [duration, [x,y,z], force_scale]")
            
            duration, direction, force_scale = seq
            
            # Validate duration
            if not isinstance(duration, (int, float)) or duration <= 0:
                raise ValueError(f"Duration in sequence {i} must be a positive number")
            
            # Validate direction
            if not isinstance(direction, list) or len(direction) != 3:
                raise ValueError(f"Direction in sequence {i} must be a list of 3 numbers [x,y,z]")
            if not all(isinstance(x, (int, float)) for x in direction):
                raise ValueError(f"Direction components in sequence {i} must be numbers")
            
            # Validate force scale
            if not isinstance(force_scale, (int, float)):
                raise ValueError(f"Force scale in sequence {i} must be a number")
            
            # Normalize direction and create sequence

            def normalize(v):
                """Normalize a vector to unit length.
                
                Args:
                    v: Vector to normalize as list of floats [x, y, z]
                    
                Returns:
                    Normalized vector as list of floats with unit length
                """
                norm = math.sqrt(sum(x*x for x in v))
                if norm == 0:
                    return v
                return [x/norm for x in v]
            
            normalized_sequences.append([float(duration), normalize(direction), float(force_scale)])
        
        return normalized_sequences
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Expected format: [[duration, [x,y,z], force_scale], ...]")
        sys.exit(1)
    except ValueError as e:
        print(f"Error in grasp sequences: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error parsing grasp sequences: {e}")
        sys.exit(1)

class GraspSimBuffer:
    def __init__(self, grasps, cspace_positions, bite_points, device):
        self.device = device
        self.num_grasps = len(grasps)
        self.transforms = wp.array(grasps, dtype=wp.transform, device=self.device)
        self.pregrasp_transforms = wp.clone(self.transforms, device=self.device)
        self.cspace_positions = wp.array(cspace_positions, dtype=wp.float32, device=self.device)
        self.pregrasp_cspace_positions = wp.clone(self.cspace_positions, device=self.device)
        self.is_success = wp.zeros(shape=self.num_grasps, dtype=wp.int32, device=self.device)
        self.bite_points = wp.array(bite_points, dtype=wp.vec3, device=self.device)
        self.pregrasp_bite_points = wp.clone(self.bite_points, device=self.device)

class GraspingSimulationConfig:
    def __init__(self, max_num_envs, env_spacing, fps, force_magnitude, initial_grasp_duration,
                 tug_sequences, start_with_pregrasp_cspace_position,
                 open_limit, disable_sim, record_pvd, debug_single_index,
                 output_failed_grasp_locations, flip_input_grasps, enable_ccd, device, max_num_grasps=0, grasp_file = None, grasp_guess_buffer = None, grasp_file_args = None):
        if grasp_guess_buffer is None and grasp_file is None:
            raise ValueError("GraspGuessBuffer or grasp_file must be provided")
        if grasp_guess_buffer is not None and grasp_file is not None:
            raise ValueError("Only one of GraspGuessBuffer or grasp_file must be provided")

        self.grasp_file = grasp_file
        self.grasp_file_args = grasp_file_args
        self.grasp_guess_buffer = grasp_guess_buffer
        self.max_num_envs = max_num_envs
        self.max_num_grasps = max_num_grasps
        self.env_spacing = env_spacing
        self.fps = fps
        self.force_magnitude = force_magnitude
        self.initial_grasp_duration = initial_grasp_duration
        self.tug_sequences = tug_sequences
        self.start_with_pregrasp_cspace_position = start_with_pregrasp_cspace_position
        self.open_limit = open_limit
        self.disable_sim = disable_sim
        
        self.record_pvd = record_pvd
        self.debug_single_index = debug_single_index
        self.output_failed_grasp_locations = output_failed_grasp_locations
        self.flip_input_grasps = flip_input_grasps
        self.enable_ccd = enable_ccd
        self.device = device

# There are parameters that go with the object creation.  Seems like I should create
# the sim based on teh gripper not hte object, and be able to run it on a per object 
# basis. Changing parameters for the simulation maybe, but not the gripper?  Simpler
# if setting can't be changed, but keeping the ones that can't in the config,
# and the others as paramters could work. 

class GraspingSimulation:
    def __init__(self, config, force_headed=False, wait_for_debugger_attach=False):
        self.config = config
        self.force_headed = force_headed
        self.wait_for_debugger_attach = wait_for_debugger_attach
        self._simulation_app = None
        self._kit_major_version = None
        self.validate_config()

    @classmethod
    def from_args(cls, args, grasp_guess_buffer=None):
        if grasp_guess_buffer is not None:
            grasp_file = None
            grasp_file_args = None
        else:
            grasp_file = args.grasp_file
            grasp_file_args = args
        grasp_sim_config = GraspingSimulationConfig(
            max_num_envs=args.max_num_envs, env_spacing=args.env_spacing, fps=args.fps, force_magnitude=args.force_magnitude, initial_grasp_duration=args.initial_grasp_duration, 
            tug_sequences=args.tug_sequences, start_with_pregrasp_cspace_position=args.start_with_pregrasp_cspace_position, 
            open_limit=args.open_limit, disable_sim=args.disable_sim, record_pvd=args.record_pvd, debug_single_index=args.debug_single_index, 
            output_failed_grasp_locations=args.output_failed_grasp_locations, flip_input_grasps=args.flip_input_grasps, enable_ccd=args.enable_ccd, device=args.device, max_num_grasps=args.max_num_grasps, grasp_file=grasp_file, grasp_file_args=grasp_file_args, grasp_guess_buffer=grasp_guess_buffer)
        return cls(grasp_sim_config, force_headed=args.force_headed, wait_for_debugger_attach=args.wait_for_debugger_attach)

    @property
    def simulation_app(self):
        """Get the simulation app instance."""
        if self._simulation_app is None:
            from graspgen_utils import get_simulation_app
            self._simulation_app = get_simulation_app(__file__, force_headed=self.force_headed, wait_for_debugger_attach=self.wait_for_debugger_attach)
        return self._simulation_app
    
    @property
    def kit_major_version(self):
        """Get the kit major version, initializing Isaac Lab if needed."""
        if self._kit_major_version is None:
            self._kit_major_version = self.simulation_app._app.get_build_version()[:3]
        return self._kit_major_version

    def validate_config(self):
        # Parse grasp sequences
        if isinstance(self.config.tug_sequences, str):
            self.tug_sequences = parse_tug_sequences(self.config.tug_sequences)
        else:
            self.tug_sequences = self.config.tug_sequences

        # get the grasps we need to simulate form the grasp file or the grasp guess buffer
        if self.config.grasp_file is not None:
            self.load_grasp_file()
        else:
            self.load_grasp_guess_buffer()
    
    def _setup_pvd_recording(self):
        """Setup PVD recording (requires Isaac Lab to be started)."""
        if self.config.record_pvd:            
            if not self.simulation_app.DEFAULT_LAUNCHER_CONFIG['headless']:
                print_red("PVD recording is only supported in headless mode, NOT recording PVD files")
            else:
                print_yellow("Recording PVD files")
                pvd_dir = "/tmp/pvdout2/"
                if not os.path.exists(pvd_dir):
                    os.makedirs(pvd_dir)
            
                from isaacsim.core.utils.extensions import enable_extension
                enable_extension("omni.physx.pvd")

                import carb
                settings_ = carb.settings.get_settings()
                settings_.set("/persistent/physics/omniPvdOvdRecordingDirectory", pvd_dir)
                settings_.set("/physics/omniPvdOutputEnabled", True)

    def load_grasp_file(self):
        # Load the grasp file
        with open(self.config.grasp_file, 'r') as f:
            self.original_grasp_yaml_data = yaml.safe_load(f)
        
        if not self.original_grasp_yaml_data or 'grasps' not in self.original_grasp_yaml_data:
            raise ValueError("Invalid grasp file format")
        
        # grasp_data['grasps'] may hold failed grasps, g,  (if g['confidence'] is 0,0)
        # we need to filter these out, and get a list of indices of valid grasps
        
        if self.config.debug_single_index:
            for i, g in enumerate(self.original_grasp_yaml_data['grasps'].values()):
                if i == self.config.debug_single_index:
                    valid_grasp_indices = [[i, g]]
                    break
        else:
            # Treat missing confidence as 1.0 (i.e., include all such grasps)
            valid_grasp_indices = [
                [i, g]
                for i, g in enumerate(self.original_grasp_yaml_data['grasps'].values())
                if ('confidence' not in g) or (g['confidence'] != 0.0)
            ]

        # Apply max_num_grasps limit if specified (0 means use all grasps)
        if self.config.max_num_grasps > 0:
            valid_grasp_indices = valid_grasp_indices[:self.config.max_num_grasps]

        # Pre-allocate numpy array for grasps
        num_grasps = len(valid_grasp_indices)
        grasps = np.zeros((num_grasps, 7))
        bite_points = np.zeros((num_grasps, 3))
        grasp_idx_map = np.zeros(num_grasps, dtype=int)

        grasp_names = list(self.original_grasp_yaml_data["grasps"].keys())
        first_grasp = self.original_grasp_yaml_data["grasps"][grasp_names[0]]
        
        # We need to get the initial cspace values to set the initial joint positions
        if self.config.start_with_pregrasp_cspace_position and 'pregrasp_cspace_position' in first_grasp:
            cspace_key = "pregrasp_cspace_position"
            bite_key = "pregrasp_bite_point"
        else:
            cspace_key = "cspace_position"
            bite_key = "bite_point"

        # Optional fallback: use --cspace_position JSON if the grasp file lacks cspace entries
        cspace_fallback = {}
        try:
            if hasattr(self.config, 'grasp_file_args') and self.config.grasp_file_args is not None:
                fallback_str = getattr(self.config.grasp_file_args, 'grasp_file_cspace_position', None)
                if isinstance(fallback_str, str) and fallback_str.strip() and fallback_str.strip() != '{}':
                    cspace_fallback = open_configuration_string_to_dict(fallback_str)
        except Exception:
            cspace_fallback = {}

        # Determine joint names either from the file or fallback
        if cspace_key in first_grasp and isinstance(first_grasp[cspace_key], dict) and len(first_grasp[cspace_key]) > 0:
            cspace_joint_names = list(first_grasp[cspace_key].keys())
        elif cspace_fallback:
            cspace_joint_names = list(cspace_fallback.keys())
        else:
            raise ValueError(f"Grasp file has no '{cspace_key}' and no fallback was provided. E.g. --grasp_file_cspace_position \"{{'finger_joint': 0.31}}\"")

        cspace_positions = [[0]*len(cspace_joint_names) for _ in range(num_grasps)]
        approach_axis = 2 if 'approach_axis' not in self.original_grasp_yaml_data else self.original_grasp_yaml_data['approach_axis']

        for i in range(num_grasps):
            grasp = valid_grasp_indices[i][1]
            grasp_idx_map[i] = valid_grasp_indices[i][0]
            # Extract position and orientation
            position = np.array(grasp['position'])
            orientation_xyz = np.array(grasp['orientation']['xyz'])
            orientation_w = grasp['orientation']['w']
            # wp.transforms are [x, y, z, qx, qy, qz, qw]
            if self.config.flip_input_grasps: #rotate the grasps for debugging
                zrot=wp.quat(0.0,0.0,0.0, 0.0)
                zrot[approach_axis] = 1.0
                orig=wp.quat(orientation_xyz[0], orientation_xyz[1], orientation_xyz[2], orientation_w)
                newquat=orig*zrot
                orientation_xyz = [newquat[0], newquat[1], newquat[2]]
                orientation_w = newquat[3]

            grasps[i] = np.concatenate([position, orientation_xyz, [orientation_w]])
            #print(grasps[i])
            # Robust bite point fallback: per-grasp bite_key first, then alternate, then top-level, then [0,0,0]
            bite_value = grasp.get(bite_key)
            if bite_value is None:
                alt_bite_key = "pregrasp_bite_point" if bite_key == "bite_point" else "bite_point"
                bite_value = grasp.get(alt_bite_key)
            if bite_value is None:
                bite_value = self.original_grasp_yaml_data.get("bite_point")
            if bite_value is None:
                bite_value = [0.0, 0.0, 0.0]
            bite_points[i] = np.array(bite_value)
            # Fill cspace from grasp if present, else fallback
            if cspace_key in grasp and isinstance(grasp[cspace_key], dict) and len(grasp[cspace_key]) > 0:
                for j, cspace_joint_name in enumerate(cspace_joint_names):
                    cspace_positions[i][j] = grasp[cspace_key][cspace_joint_name]
            elif cspace_fallback:
                for j, cspace_joint_name in enumerate(cspace_joint_names):
                    cspace_positions[i][j] = cspace_fallback[cspace_joint_name]
            else:
                raise ValueError(f"Grasp entry {i} missing '{cspace_key}' and no --cspace_position fallback provided")


        self.grasps = wp.array(grasps, dtype=wp.transform, device=self.config.device)
        self.bite_points = wp.array(bite_points, dtype=wp.vec3, device=self.config.device)
        self.grasp_idx_map = grasp_idx_map
        self.cspace_joint_names = cspace_joint_names
        self.cspace_positions = wp.array(cspace_positions, dtype=wp.float32, device=self.config.device)
        self.cspace_joint_indices = None

        # TODO: get rid of these
        def get_value_with_priority(key, grasp_file_value, command_line_value):
            """
            Get value with priority: command_line_value > grasp_file_value
            Priority order: grasp_file (lowest) < grasp_config < args (highest)
            """
            # If command line value is provided and different from default, use it
            if command_line_value is not None:
                # Check if this is a non-default value (indicating it was explicitly set)
                if key == "gripper_file" and command_line_value != "bots/onrobot_rg6.usd":
                    print_blue(f"Using command line {key}: {command_line_value} (overriding grasp file value: {grasp_file_value})")
                    return command_line_value
                elif key == "finger_colliders" and command_line_value != ["right_inner_finger", "left_inner_finger"]:
                    print_blue(f"Using command line {key}: {command_line_value} (overriding grasp file value: {grasp_file_value})")
                    return command_line_value
                elif key == "open_limit" and command_line_value != "":
                    print_blue(f"Using command line {key}: {command_line_value} (overriding grasp file value: {grasp_file_value})")
                    return command_line_value
            
            # Otherwise use grasp file value if it exists
            if grasp_file_value is not None:
                return grasp_file_value
            
            # Fallback to default values
            if key == "gripper_file":
                return "bots/onrobot_rg6.usd"
            elif key == "finger_colliders":
                return ["right_inner_finger", "left_inner_finger"]
            elif key == "open_limit":
                return ""
            elif key == "bite_point":
                return [0.0, 0.0, 0.0]
            elif key == "bite_body_idx":
                return 0
            else:
                print_red(f"{key} not found in grasp file or command line, using default fallback")
                # Return a safe default for unknown keys
                return None

        grasp_file_gripper = self.original_grasp_yaml_data.get("gripper_file")
        grasp_file_finger_colliders = self.original_grasp_yaml_data.get("finger_colliders")
        grasp_file_open_limit = self.original_grasp_yaml_data.get("open_limit")
        grasp_file_bite_point = self.original_grasp_yaml_data.get("bite_point")
        grasp_file_bite_body_idx = self.original_grasp_yaml_data.get("bite_body_idx")

        self.open_limit = get_value_with_priority("open_limit", grasp_file_open_limit, self.config.open_limit)
        self.gripper_file = get_value_with_priority("gripper_file", grasp_file_gripper, self.config.grasp_file_args.gripper_file)
        self.finger_colliders = get_value_with_priority("finger_colliders", grasp_file_finger_colliders, self.config.grasp_file_args.finger_colliders)
        self.object_config = ObjectConfig.from_isaac_grasp_dict(self.original_grasp_yaml_data, self.config.grasp_file_args)
        self.bite_point = get_value_with_priority("bite_point", grasp_file_bite_point, [0.0, 0.0, 0.0])
        self.bite_body_idx = get_value_with_priority("bite_body_idx", grasp_file_bite_body_idx, 0)

    def load_grasp_guess_buffer(self):
        ggb = self.config.grasp_guess_buffer
        gpr = ggb.gripper
        obj = ggb.object
        if self.config.flip_input_grasps: # TODO
            print_yellow("Flipping input grasps currently only works with grasp_file, not grasp_guess_buffer")

        if ggb.succ_buff is None:
            print_yellow("Grasp guess buffer has no successful grasps")
            self.grasps = wp.array(shape=0, dtype=wp.transform, device=self.config.device)
            return
        # Apply max_num_grasps limit if specified (0 means use all grasps)
        num_grasps = int(ggb.num_successes)
        if self.config.max_num_grasps > 0:
            num_grasps = min(num_grasps, self.config.max_num_grasps)
        
        # Clone the transforms with the limited number of grasps
        self.grasps = wp.array(shape=num_grasps, data=ggb.succ_buff.transforms, dtype=wp.transform, device=self.config.device)
        
        self.open_limit = gpr.open_limit
        self.cspace_joint_names = gpr.joint_names
        self.cspace_joint_indices = torch.arange(len(gpr.joint_names))
        self.driven_joints = gpr.driven_joints
        self.cspace_joint_indices = wp.array(self.cspace_joint_indices, dtype=wp.int32, device=self.config.device)  # type: ignore[arg-type]
        self.bite_body_idx = gpr.finger_indices[0]
        self.bite_point = gpr.bite_point
        offsets = ggb.succ_buff.pregrasp_offsets if self.config.start_with_pregrasp_cspace_position else ggb.succ_buff.offsets
        self.cspace_positions = wp.array(shape=(num_grasps, len(self.cspace_joint_names)), dtype=wp.float32, device=self.config.device)
        self.bite_points = wp.array(shape=num_grasps, dtype=wp.vec3, device=self.config.device)
        wp.launch(kernel=get_cspace_positions_kernel,
                  dim=(num_grasps, len(self.cspace_joint_names)),
                  inputs=[offsets, self.cspace_joint_indices, gpr.joint_cspace_pos, self.cspace_positions],
                  device=self.config.device)
        wp.launch(kernel=get_bite_points_kernel,
                  dim=num_grasps, 
                  inputs=[offsets, gpr.bite_points, self.bite_points],
                  device=self.config.device)
        self.object_config = obj.config
        self.gripper_file = gpr.config.gripper_file
        self.finger_colliders = gpr.config.finger_colliders

    def create_isaac_grasp_data(self, grasp_sim_buffer, save_successes = True, save_fails = False, only_driven_joints = True, save_to_folder = None, file_name_prefix = "", file_extension_prefix = ""):
        gsb = grasp_sim_buffer
        if self.config.grasp_file is not None:
            # Note only_driven_joints does not work with grasp_file, what comes in is what comes out as far as joints go.
            isaac_grasp_data = copy.deepcopy(self.original_grasp_yaml_data)
            if 'created_with' not in isaac_grasp_data or isaac_grasp_data['created_with'] != "grasp_guess":
                print_red("Grasp file was not created with grasp_guess")
            isaac_grasp_data["created_with"] = "grasp_sim"
            isaac_grasp_data["created_at"] = datetime.now().isoformat()
            # Update gripper_file to reflect the actual gripper file used in simulation
            isaac_grasp_data["gripper_file"] = self.gripper_file
            #is_suss_cpu = gsb.is_success.numpy().tolist()
            is_success = wp.to_torch(gsb.is_success)
            grasp_keys = list(isaac_grasp_data["grasps"].keys())
            num_successes = 0
            num_fails = 0
            if (save_successes and not save_fails) or (not save_successes and save_fails):
                env_ids = torch.where(is_success)[0] if save_successes else torch.where(~is_success)[0]
                old_grasps = isaac_grasp_data["grasps"]
                new_grasps = {}
                for env_id in env_ids:
                    env_id = int(env_id)
                    grasp_idx = self.grasp_idx_map[env_id]
                    new_grasps[grasp_keys[grasp_idx]] = old_grasps[grasp_keys[grasp_idx]]
                isaac_grasp_data["grasps"] = new_grasps
            else:
                env_ids = torch.arange(gsb.num_grasps)
            
            cspace_positions = gsb.cspace_positions.numpy()
            transforms = gsb.transforms.numpy()
            bite_points = gsb.bite_points.numpy()


            for env_id in env_ids:
                env_id = int(env_id)
                grasp_idx = self.grasp_idx_map[env_id]
                grasp = isaac_grasp_data["grasps"][grasp_keys[grasp_idx]]
                # Add joint positions
                # Ensure cspace_position dict exists; fill from computed positions (may originate from fallback)
                if "cspace_position" not in grasp or not isinstance(grasp["cspace_position"], dict):
                    grasp["cspace_position"] = {}
                for j, joint_name in enumerate(self.cspace_joint_names):
                    grasp["cspace_position"][str(joint_name)] = float(cspace_positions[env_id, j])
                
                # confidence
                confidence = float(is_success[env_id])
                if confidence:
                    num_successes += 1
                else:
                    num_fails += 1
                grasp["confidence"] = confidence
                grasp["pregrasp_position"] = copy.deepcopy(grasp["position"])
                grasp["pregrasp_orientation"] = copy.deepcopy(grasp["orientation"])
                # Use a safe fallback if original YAML lacked bite_point
                _pre_bite = grasp.get("bite_point", [0.0, 0.0, 0.0])
                grasp["pregrasp_bite_point"] = copy.deepcopy(_pre_bite)
                if not confidence and not self.config.output_failed_grasp_locations:
                    continue

                # Add position, orientation, and bite point
                grasp["position"] = transforms[env_id, :3].tolist()
                grasp["orientation"]["xyz"] = transforms[env_id, 3:6].tolist()
                grasp["orientation"]["w"] = float(transforms[env_id, 6])
                grasp["bite_point"] = bite_points[env_id].tolist()
            
            print_green(f"created {num_successes} successes and {num_fails} fails")
        elif self.config.grasp_guess_buffer is not None:
            ggb = self.config.grasp_guess_buffer
            isaac_grasp_data = {
                "format": "isaac_grasp",
                "format_version": "1.0",
                "created_with": "grasp_sim",
                "created_at": datetime.now().isoformat(),
                "object_file": self.object_config.object_file,
                "object_scale": self.object_config.object_scale,
                "gripper_file": self.gripper_file,
                "gripper_frame_link": ggb.gripper.config.base_frame,
                "open_limit": ggb.gripper.open_limit,
                "finger_colliders": ggb.gripper.config.finger_colliders,
                "base_length": ggb.gripper.base_length,
                "approach_axis": ggb.gripper.approach_axis,
                "bite_point": ggb.gripper.bite_point,
                "bite_body_idx": ggb.gripper.finger_indices[0],
                "grasps": {
                }
            }
            is_success = gsb.is_success.numpy()
            transforms = gsb.transforms.numpy()
            pregrasp_transforms = gsb.pregrasp_transforms.numpy()
            cspace_positions = gsb.cspace_positions.numpy()
            pregrasp_cspace_positions = gsb.pregrasp_cspace_positions.numpy()
            bite_points = gsb.bite_points.numpy()
            pregrasp_bite_points = gsb.pregrasp_bite_points.numpy()
            
            num_successes = 0
            num_fails = 0

            for i in range(gsb.num_grasps):
                confidence = float(is_success[i])
                if confidence:
                    num_successes += 1
                else:
                    num_fails += 1
                if not confidence and not save_fails:
                    continue
                if confidence and not save_successes:
                    continue
                if not confidence and not self.config.output_failed_grasp_locations:
                    continue
                grasp = {
                    "confidence": confidence,
                    "position": transforms[i, :3].tolist(),
                    "orientation": {
                        "w": float(transforms[i, 6]),
                        "xyz": transforms[i, 3:6].tolist()
                    },
                    "cspace_position": {
                    },
                    "bite_point": bite_points[i].tolist(),
                    "pregrasp_position": pregrasp_transforms[i, :3].tolist(),
                    "pregrasp_orientation": {
                        "w": float(pregrasp_transforms[i, 6]),
                        "xyz": pregrasp_transforms[i, 3:6].tolist()
                    },
                    "pregrasp_cspace_position": {
                    },
                    "pregrasp_bite_point": pregrasp_bite_points[i].tolist(),
                }
                for j, joint_name in enumerate(self.cspace_joint_names):
                    if only_driven_joints and j not in self.driven_joints:
                        continue
                    grasp[f"cspace_position"][str(joint_name)] = float(cspace_positions[i, j])
                    grasp[f"pregrasp_cspace_position"][str(joint_name)] = float(pregrasp_cspace_positions[i, j])
                isaac_grasp_data["grasps"][f"grasp_{i}"] = grasp
            print_green(f"created {num_successes} successes and {num_fails} fails")
        else:
            raise ValueError("No grasp guess creation only works for grasp_guess_buffer or grasp_file")

        #print(f"created {self.num_successes if save_successes else 0} successes and {self.num_fails if save_fails else 0} fails")
        output_file = None
        if save_to_folder is not None:
            from graspgen_utils import predict_grasp_data_filepath
            # Ensure gripper_file is a string for type safety
            gripper_file_str = str(self.gripper_file)
            gripper_name = os.path.splitext(os.path.basename(gripper_file_str))[0]
            output_file = predict_grasp_data_filepath(
                gripper_name,
                self.object_config.object_file,
                save_to_folder,
                file_name_prefix,
                file_extension_prefix
            )
            if output_file is not None:
                # Create the output directory
                output_folder = os.path.dirname(output_file)
                os.makedirs(output_folder, exist_ok=True)
                save_yaml(isaac_grasp_data, output_file=output_file)
        return isaac_grasp_data, output_file

    def get_usd_path(self, file_path):
        # Expand user path (handle ~ in file paths)
        file_path = os.path.expanduser(file_path)
        
        if file_path.lower().endswith(".usd") or file_path.lower().endswith(".usda") or file_path.lower().endswith(".usdz"):
            print_blue(f", with USD file: {file_path}")
            return file_path
        else:
            # when we create the usd file, we need to have the scale on there not only to 
            # tell when our own cached usd is valid, but also because kit's import obj feature
            # will use the usd file when "converting" if it object_foo.usd already exists for object_foo.obj
            usd_file = os.path.splitext(file_path)[0] + f".usd"
            if self.object_config.obj2usd_use_existing_usd and os.path.exists(usd_file):
                print_blue(f", and using existing USD file: {usd_file}")
                return usd_file
            
            print_blue(f", and creating USD file: {usd_file}")
            usd_file = self.create_usd(usd_file, file_path)
            return usd_file

    def create_usd(self, usd_file, file_path):
        from graspgen_utils import get_simulation_app
        # need to make sure isaac sim is started before importing isaaclab, and pxr modules
        _ = get_simulation_app(__file__, force_headed=self.force_headed, wait_for_debugger_attach=self.wait_for_debugger_attach)
        from mesh_utils import convert_mesh_to_usd
        from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
        
        physics_material = RigidBodyMaterialCfg(
            static_friction=self.object_config.obj2usd_friction,
            dynamic_friction=self.object_config.obj2usd_friction,
        )
        #physics_material = None
        usd_file = convert_mesh_to_usd(
            usd_file,
            file_path,
            overwrite=True,
            #vertex_scale=self.object_config.object_scale,
            mass=1.0,
            collision_approximation=self.object_config.obj2usd_collision_approximation,
            physics_material=physics_material,
        )
        return usd_file
    
    def build_grasp_sim_scene_cfg(self, num_envs):
        # Import Isaac Lab modules after ensuring Isaac Lab is started
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils import configclass
        from isaaclab.actuators import ImplicitActuatorCfg
        
        @configclass
        class GraspingSceneCfg(InteractiveSceneCfg):
            """Configuration for a grasping scene."""
            # ground plane
            #ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)))
            # lights
            dome_light = AssetBaseCfg(
                prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            )

            # gripper
            gripper = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                init_state=ArticulationCfg.InitialStateCfg(),#joint_pos={"finger_joint": -0.62},),
                actuators={
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=[".*",],
                        #effort_limit_sim=150.0,
                        #velocity_limit_sim=2.175,
                        #stiffness=400.0,
                        #damping=40.0,
                        stiffness=None,
                        damping=None,
                    ),
                },
            )

            # object
            # notice Z is 100.0, so the object is far away from the gripper when setting initial joint positions
            # we could avoid this if IsaacLab didn't solve collisions when setting initial joint positions
            object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 100.0*0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0))
            )

            # sensors to tell if the object is in grasp
            contact_forces0 = ContactSensorCfg(
                update_period=0.0, history_length=0, debug_vis=False
            )
            contact_forces1 = ContactSensorCfg(
                update_period=0.0, history_length=0, debug_vis=False
            )

        scene_cfg = GraspingSceneCfg(
                num_envs=num_envs, 
                env_spacing=self.config.env_spacing, 
                filter_collisions=True,
                replicate_physics=False,#
            )
        scene_cfg.gripper.spawn = sim_utils.UsdFileCfg(
            usd_path=self.gripper_file,
            activate_contact_sensors=True,
        )
        scene_cfg.object.spawn = sim_utils.UsdFileCfg(
            usd_path=self.usd_path,
            scale=(self.object_config.object_scale, self.object_config.object_scale, self.object_config.object_scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=not self.config.disable_sim,
            ),
            activate_contact_sensors=True,
        )
        # Ensure finger_colliders is a list for type safety
        finger_colliders_list = list(self.finger_colliders)
        scene_cfg.contact_forces0.prim_path="{ENV_REGEX_NS}/Robot/" + str(finger_colliders_list[0]) #+ ".*"
        scene_cfg.contact_forces1.prim_path="{ENV_REGEX_NS}/Robot/" + str(finger_colliders_list[1]) #+ ".*"
        return scene_cfg
    
    def get_initial_joint_pos(self, scene, joint_pos, num_envs, start_idx, buff):
        if self.cspace_joint_indices is None:
            self.cspace_joint_indices = [scene["gripper"].data.joint_names.index(name) for name in self.cspace_joint_names]
            self.cspace_joint_indices = wp.array(self.cspace_joint_indices, dtype=wp.int32, device=self.config.device)

        wp.launch(
            kernel=get_joint_pos_kernel,
            dim=(num_envs, len(self.cspace_joint_names)),
            inputs=[start_idx, 0, buff.pregrasp_cspace_positions, self.cspace_joint_indices],
            outputs=[joint_pos],
            device=self.config.device)
    
    def check_memory(self, prev_mem_list):
        print_purple(f"🔍 Checking memory usage...")
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss / 1024 / 1024  # Convert to MB
        prev_mem_list.append(rss)
        if len(prev_mem_list) > 1:
            mem_diff = rss - prev_mem_list[-2]
            print_purple(f"  Memory: {rss:.1f}MB (change: {mem_diff:+.1f}MB)")
        else:
            print_purple(f"  Memory: {rss:.1f}MB (baseline)")

    def clean_stage(self):
        # CRITICAL: Clear USD stage references (main leak source)
        
        # Track RSS memory at the beginning
        import psutil
        import os
        process = psutil.Process(os.getpid())
        rss_start = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print_blue(f"🧹 clean_stage() RSS Memory - Start: {rss_start:.1f} MB")
        
        try:
            # More aggressive USD cleanup
            import omni.usd
            from pxr import Usd, Sdf
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            # Clear all layers in the stage
            for layer in stage.GetLayerStack():
                if layer:
                    layer.Clear()
            
            # Clear root layer specifically
            root_layer = stage.GetRootLayer()
            if root_layer:
                root_layer.Clear()
            
            # Clear session layer
            session_layer = stage.GetSessionLayer()
            if session_layer:
                session_layer.Clear()
            
            stage = None
        
            # Get USD context and clear it
            if usd_context:
                usd_context.close_stage()
        except Exception as usd_e:
            print_red(f"⚠ USD cleanup failed: {usd_e}")
            import traceback
            traceback.print_exc()
        
        # Track RSS memory at the end
        rss_end = process.memory_info().rss / 1024 / 1024  # Convert to MB
        rss_change = rss_end - rss_start
        print_blue(f"🧹 clean_stage() RSS Memory - End: {rss_end:.1f} MB (Change: {rss_change:+.1f} MB)")
        
        # Force garbage collection after cleanup
        import gc
        gc.collect()

    def validate_grasps(self):
        # the max_num_envs may be smaller than the number of grasps we actually have, in that case,
        # we need to run more than one batch of simulations.
        if len(self.grasps) == 0:
            print_yellow("No grasps to validate")
            return None
        print_blue(f"  🔍 Validating {len(self.grasps)} grasps with {os.path.basename(self.gripper_file)}", end="")
        start_idx = 0
        start_time = time.time()
        grasp_sim_buffer = GraspSimBuffer(self.grasps, self.cspace_positions, self.bite_points, self.config.device)
        num_grasps = len(self.grasps)
        batch_count = 0
        self.usd_path = self.get_usd_path(self.object_config.object_file)
        
        # Calculate total batches for progress tracking
        total_batches = (num_grasps + self.config.max_num_envs - 1) // self.config.max_num_envs
        
        while start_idx < num_grasps:
            batch_count += 1
            num_envs = min(self.config.max_num_envs, num_grasps - start_idx)
            print_purple(f"\r  🔄 Preparing Batch {batch_count}/{total_batches}{' ' * 30}", end="", flush=True)
            self.run_grasp_sim(start_idx, num_envs, grasp_sim_buffer, batch_count, total_batches)
            start_idx += num_envs
            # Don't print newline - let the next batch or final message overwrite this line
        
        validation_time = time.time() - start_time
        
        # Clear the batch line (detailed validation summary will be printed by caller)
        print(f"\r", end="", flush=True)
        return grasp_sim_buffer


    # When headed, keep rendering until window closes at the end of the batch (for debugging single grasps)
    def run_grasp_sim(self, start_idx, num_envs, buff, batch_count=None, total_batches=None):
        from graspgen_utils import get_simulation_app
        simulation_app = get_simulation_app(__file__, force_headed=self.force_headed, wait_for_debugger_attach=self.wait_for_debugger_attach)
        # Import Isaac Lab modules after ensuring Isaac Lab is started
        import isaaclab.sim as sim_utils
        from isaaclab.sim import build_simulation_context
        from isaaclab.scene import InteractiveScene
        
        # Setup PVD recording if needed (requires Isaac Lab to be started)
        self._setup_pvd_recording()
        
        # Initialize the simulation context with safer PhysX memory configuration
        # Reduce large GPU allocations that may be contaminated by Isaac Sim 5.0
        sim_cfg = sim_utils.SimulationCfg(
            device=self.config.device, # Set sim device seperate from config.device?
            dt=1.0/self.config.fps,  # Set physics timestep based on FPS
            physx=sim_utils.PhysxCfg(
                min_position_iteration_count=64,
                #min_velocity_iteration_count=0,
                # Reduce from 2**30 (1GB) to 2**26 (64MB) to avoid contamination
                #gpu_collision_stack_size=2**26,  # 64MB instead of 1GB
                # Add explicit memory settings to prevent corruption
                #gpu_max_rigid_contact_count=2**20,  # 1M contacts instead of default
                gpu_max_rigid_patch_count=2**19,   # 512K patches instead of default
                #gpu_found_lost_pairs_capacity=2**18,  # 256K pairs
                #gpu_heap_capacity=2**24,  # 16MB heap
                enable_ccd=self.config.enable_ccd if self.config.device == "cpu" else False,
            )
        )
        with build_simulation_context(device=self.config.device, gravity_enabled = False, auto_add_lighting=True, sim_cfg=sim_cfg) as sim:
            # build simulation context does some kit printing stuff that often leaves stdout in a bad print color state.
            print("\033[0m", end="")
            sim._app_control_on_stop_handle = None

            # Set main camera to look at the first env
            env_offset = ( math.sqrt(float(num_envs))/2.0 - 0.5)*self.config.env_spacing
            sim.set_camera_view(eye=((env_offset-0.09488407096425105, -env_offset-0.4248694091778259, 0.4521177672484193)),
                                target=(env_offset, -env_offset, 0.15))
            
            scene_cfg = self.build_grasp_sim_scene_cfg(num_envs)
            scene = InteractiveScene(scene_cfg)
            # this next line should match the collision filtering if logic, but not.
            if not((not scene.cfg.replicate_physics and scene.cfg.filter_collisions) or (int(self.kit_major_version) >= 107 and scene.device == "cpu")):
                # when cpu the scene is already filtered by the InteractiveScene constructor
                scene.filter_collisions()
            run_start_time = time.time()
            sim.reset()
            
            do_render = not simulation_app.DEFAULT_LAUNCHER_CONFIG['headless']
            # Get scene entities
            gripper = scene["gripper"]
            object = scene["object"]
            contact_forces0 = scene["contact_forces0"]
            contact_forces1 = scene["contact_forces1"]
            
            # Define simulation stepping
            sim_dt = sim.get_physics_dt()
            sim_time = 0.0
            
            initial_grasp_duration = self.config.initial_grasp_duration  # seconds for initial grasp
            grasp_start_time = 0.0
            force_start_time = grasp_start_time + initial_grasp_duration
            
            if not (self.open_limit == "upper" or self.open_limit == "lower"):
                print_red(f"Invalid open limit: {self.open_limit}")
            grasp_mode = 0 if self.open_limit == "upper" else 1
            
            # Pre-compute world frame force tensors for each direction
            Gs = self.config.force_magnitude
            _, gravity_mag = sim.get_physics_context().get_gravity()
            acceleration = Gs * gravity_mag
            force_magnitude = acceleration * object.data.default_mass[0]
            
            # Create force tensors for each sequence
            # TODO Move this to only be done once in validate_grasps, or validate_config?
            world_forces = {}
            wp_world_forces = {}
            force_end_time = force_start_time
            for i, (duration, direction, scale) in enumerate(self.tug_sequences):
                force_end_time += duration
                force = torch.zeros(num_envs, 3, device=self.config.device)
                world_force = [direction[0] * force_magnitude * scale, direction[1] * force_magnitude * scale, direction[2] * force_magnitude * scale]
                wp_world_forces[i] = wp.vec3(world_force[0], world_force[1], world_force[2])
                force[:, 0] = world_force[0]
                force[:, 1] = world_force[1]
                force[:, 2] = world_force[2]
                world_forces[i] = force.unsqueeze(1)
            # unsqueeze to match force shape (env, 1, 3)
            zero_torque = torch.zeros(num_envs, 3, device=self.config.device).unsqueeze(1)
            wp_world_forces_working = wp.zeros(shape=(num_envs, 1), dtype=wp.vec3, device=self.config.device)
            
            # Simulation loop
            wall_time = start_time = time.time()
            data_done = False
            times_to_print = {}
            times_to_print["run_start"] = time.time() - run_start_time
            while_start_time = time.time()
            count = 0
            
            while simulation_app.is_running():
                # use the first frame to get the body positions from the initial joint positions
                if count == 0:
                    gripper_state = gripper.data.default_root_state.clone()
                    if not USE_ORIGIN_PLACEMENT:
                        gripper_state[:, :3] += scene.env_origins
                    gripper.write_root_pose_to_sim(gripper_state[:, :7])
                    gripper.write_root_velocity_to_sim(gripper_state[:, 7:])

                    # Set object poses from grasp file
                    object_state = object.data.default_root_state.clone()
                    if not USE_ORIGIN_PLACEMENT:
                        object_state[:, :3] += scene.env_origins
                    object.write_root_pose_to_sim(object_state[:, :7])
                    object.write_root_velocity_to_sim(object_state[:, 7:])
                    
                    # Apply motion to gripper
                    # joint state
                    joint_pos = gripper.data.default_joint_pos.clone()#gripper.data.soft_joint_pos_limits[..., 1-grasp_mode].clone()##gripper.data.soft_joint_pos_limits[..., grasp_mode].clone() #gripper.data.default_joint_pos.clone()
                    joint_vel = gripper.data.default_joint_vel.clone() #TODO get rid of the clone when we don't need to do it
                    self.get_initial_joint_pos(scene, joint_pos, num_envs, start_idx, buff)
                    # NOTE:if the joint velocity limits are too low, then the joint positions won't jump to the correct
                    # position.  They will have to slowly move there, even though write_joint_state_to_sim is supposed
                    # to be a teleport command.
                    gripper.set_joint_position_target(joint_pos)
                    gripper.write_joint_state_to_sim(joint_pos, joint_vel)
                    # Use joint_vel_limits if available, otherwise fall back to joint_velocity_limits
                    if hasattr(gripper.data, 'joint_vel_limits'):
                        temp_vel_limits = gripper.data.joint_vel_limits.clone()
                        vel_limits = wp.array(gripper.data.joint_vel_limits.clone(), dtype=wp.float32, device=self.config.device)
                    else:
                        temp_vel_limits = gripper.data.joint_velocity_limits.clone()
                        vel_limits = wp.array(gripper.data.joint_velocity_limits.clone(), dtype=wp.float32, device=self.config.device)
                    vel_limits.fill_(1000000000.0)
                    gripper.write_joint_velocity_limit_to_sim(wp.to_torch(vel_limits, requires_grad=False))
                    scene.reset()

                    # clear internal buffers

                    for i in range(2):
                        scene.write_data_to_sim()
                        # Perform step
                        sim.step(render=False)
                        # Update buffers
                        scene.update(sim_dt)
                        #print(".", end="", flush=True)
                    
                    gripper.write_joint_velocity_limit_to_sim(temp_vel_limits)
                    gripper.write_joint_velocity_to_sim(joint_vel)
                    
                    #print("")

                elif count == 1:
                    #zero_vel = torch.zeros(num_envs, 6, device=self.config.device)
                    #object.write_root_com_velocity_to_sim(zero_vel)
                    # Set the initial conditions for the simulation.
                    # zero the velocity of the object
                    sim_time = 0.0
                    gripper_state = gripper.data.default_root_state.clone()
                    if not USE_ORIGIN_PLACEMENT:
                        gripper_state[:, :3] += scene.env_origins
                    gripper.write_root_pose_to_sim(gripper_state[:, :7])
                    gripper.write_root_velocity_to_sim(gripper_state[:, 7:])

                    # Set object poses from grasp file
                    root_state = object.data.default_root_state.clone()
                    wp.launch(
                        kernel=transform_inverse_kernel,
                        dim=num_envs,
                        inputs=[start_idx, 0, buff.pregrasp_transforms, root_state[:, :7], True],
                        device=self.config.device)
                    if not USE_ORIGIN_PLACEMENT:
                        root_state[:, :3] += scene.env_origins
                    object.write_root_pose_to_sim(root_state[:, :7])
                    object.write_root_velocity_to_sim(root_state[:, 7:])

                
                    # gripper should always be closing *
                    if self.config.disable_sim:
                        joint_pos_target = joint_pos
                    else:
                        # Debug: Print joint limits and names for comparison between versions
                        joint_pos_target = gripper.data.soft_joint_pos_limits[..., grasp_mode].clone()  # Closed position
                    
                    # Set joint position target
                    gripper.set_joint_position_target(joint_pos_target)

                    # clear internal buffers
                    scene.reset()
                    times_to_print["reset"] = time.time() - while_start_time
                    while_start_time = time.time()         # -- write data to sim
                    
                # Apply force to object if in force phase
                if not data_done and sim_time >= force_start_time and sim_time < force_end_time and not self.config.disable_sim:
                    # Calculate current sequence index based on time
                    time_since_force_start = sim_time - force_start_time
                    current_sequence = 0
                    total_duration = 0.0
                    
                    for i, (duration, _, _) in enumerate(self.tug_sequences):
                        if time_since_force_start < total_duration + duration:
                            current_sequence = i
                            break
                        total_duration += duration
                    wp.launch(
                        kernel=world_to_object_force_kernel,
                        dim=num_envs,
                        inputs=[object.data.root_quat_w, wp_world_forces[current_sequence], wp_world_forces_working], 
                        device=self.config.device)
                    
                    # Apply force to object (in local frame)
                    object.set_external_force_and_torque(wp.to_torch(wp_world_forces_working,requires_grad=False), zero_torque)

                elif sim_time >= force_end_time and not data_done and not self.config.disable_sim:
                    times_to_print["while_loop"] = time.time() - while_start_time
                    write_data_time = time.time()
                    cdata0 = contact_forces0.data
                    cdata1 = contact_forces1.data
                    # Check if all finger tips have non-zero normal forces
                    non_zero_forces0 = torch.all(torch.norm(cdata0.net_forces_w, dim=-1) > 0, dim=1)
                    non_zero_forces1 = torch.all(torch.norm(cdata1.net_forces_w, dim=-1) > 0, dim=1)

                    # This is an output
                    non_zero_forces = non_zero_forces0 & non_zero_forces1
                    wp.launch(
                        kernel=set_is_success_kernel,
                        dim=num_envs,
                        inputs=[0, start_idx, non_zero_forces, buff.is_success],
                        device=self.config.device)

                    object_pos_rel = object.data.root_pos_w
                    object_quat_rel = object.data.root_quat_w
                    gripper_pos_w = gripper.data.root_pos_w
                    gripper_quat_w = gripper.data.root_quat_w
                    wp.launch(
                        kernel=compute_relative_pos_and_rot_kernel,
                        dim =num_envs,
                        inputs=[0, start_idx, object_pos_rel, object_quat_rel, gripper_pos_w, gripper_quat_w],
                        outputs=[buff.transforms],
                        device=self.config.device)
                    wp.launch(
                        kernel=get_joint_pos_kernel,
                        dim=(num_envs, len(self.cspace_joint_names)),
                        inputs=[0, start_idx, gripper.data.joint_pos, self.cspace_joint_indices],
                        outputs=[buff.cspace_positions],
                        device=self.config.device)
                    if buff.bite_points is not None:
                        finger_idx = self.bite_body_idx#gripper.data.body_names.index(self.finger_colliders[0])
                        #bite_points_before = buff.bite_points.numpy()[start_idx:start_idx+num_envs].tolist()
                        bite_point = wp.vec3(self.bite_point[0], self.bite_point[1], self.bite_point[2]) # TODO maybe don't bother with bite points, I'm not sure they are correct.
                        wp.launch(
                            kernel=get_body_pos_kernel,
                            dim=num_envs,
                            inputs=[0, start_idx, finger_idx, bite_point, gripper.data.body_link_pos_w, gripper_pos_w],
                            outputs=[buff.bite_points],
                            device=self.config.device)
                        #bite_points_after = buff.bite_points.numpy()[start_idx:start_idx+num_envs].tolist()

                    # we want to keep simming if the UI is open so the UI does not freeze
                    data_done = True
                    if do_render and self.force_headed:                    
                        print_purple(f"In {__file__}, waiting for Isaac Lab to close...", flush=True)
            
                        while simulation_app.is_running():
                            sim.step(render=do_render)

                    print("\033[0m", end="")
                    return
                    

                scene.write_data_to_sim()
                # Perform step
                sim.step(render=do_render)
                count += 1
                # Update buffers
                scene.update(sim_dt)
                # Update simulation time
                sim_time += sim_dt
                current_time = time.time()
                if not data_done and int(current_time) > int(wall_time):
                    # Show simulation timing with batch info, overwriting the previous line cleanly
                    if batch_count is not None and total_batches is not None:
                        # Calculate percentage: (sim_time / force_end_time) * 100
                        percentage = int((sim_time / force_end_time) * 100) if force_end_time > 0 else 0
                        # Pad single-digit percentages to keep "Sim:" aligned
                        percentage_str = f" {percentage}%" if percentage < 10 else f"{percentage}%"
                        # Clear the line completely and print fresh to avoid leftover characters
                        print_purple(f"\r  🔄 Batch {batch_count}/{total_batches}: {percentage_str}    Sim: {sim_time:.1f}s / Real: {time.time()-start_time:.1f}s{' ' * 15}", end="", flush=True)
                    else:
                        print(f"\r  ⏱️  Sim: {sim_time:.1f}s / Real: {time.time()-start_time:.1f}s{' ' * 20}", end="", flush=True)
                    wall_time = current_time

def main(args):
    # Initialize simulation_app when needed
    simulation_app = start_isaac_lab_if_needed(file_name=__file__, headless = False if args.force_headed else args.headless, wait_for_debugger_attach=args.wait_for_debugger_attach)
    
    grasp_sim_cfg = GraspingSimulationConfig(
                 max_num_envs = args.max_num_envs, env_spacing = args.env_spacing, fps = args.fps, force_magnitude = args.force_magnitude, initial_grasp_duration = args.initial_grasp_duration, tug_sequences = args.tug_sequences,
                 start_with_pregrasp_cspace_position = args.start_with_pregrasp_cspace_position, open_limit = args.open_limit,
                 disable_sim = args.disable_sim, record_pvd = args.record_pvd, debug_single_index = args.debug_single_index,
                 output_failed_grasp_locations = args.output_failed_grasp_locations, flip_input_grasps = args.flip_input_grasps, enable_ccd = args.enable_ccd, device=args.device, max_num_grasps=args.max_num_grasps, grasp_file= args.grasp_file, grasp_file_args=args)
    grasp_sim = GraspingSimulation(grasp_sim_cfg, force_headed=args.force_headed, wait_for_debugger_attach=args.wait_for_debugger_attach)
    grasp_sim_buffer = grasp_sim.validate_grasps()
    save_to_folder = os.path.join(os.environ.get('GRASP_DATASET_DIR', ''), "grasp_sim_data")
    if grasp_sim_buffer is not None:
        isaac_grasp_data, file_name = grasp_sim.create_isaac_grasp_data(grasp_sim_buffer, save_to_folder=save_to_folder)
    
    # Don't close the simulation app if force_headed is used, let it run until user closes it
    if not args.force_headed:
        simulation_app.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp validation through simulation part of Grasp Gen Data Generation.")
    add_grasp_sim_args(parser, globals(), **collect_grasp_sim_args(globals()))
    args_cli = parser.parse_args()
    
    # Apply gripper configuration if specified
    apply_gripper_configuration(args_cli)

    print(f"enable_ccd: {args_cli.enable_ccd}")
    
    main(args_cli)

