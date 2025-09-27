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
import numpy as np
from grasp_guess import GraspGuessGenerator, GuessObject, add_grasp_guess_args, collect_grasp_guess_args
from grasp_sim import  GraspingSimulation, add_grasp_sim_args, collect_grasp_sim_args
from gripper_configurations import get_gripper_config, apply_gripper_config_to_args
from gripper import Gripper
import json
import os
import time
import torch
from graspgen_utils import print_blue, print_yellow, print_green, print_red, add_arg_to_group
import yaml
from usd_tools import transform_to_matrix

default_graspgen_source = os.path.expanduser(os.environ.get('GRASP_GEN_CODE_DIR', '~/github/GraspGen'))
default_object_scales_json = 'objects/graspgen_example.json'
default_object_dataset = os.path.expanduser(os.environ.get('OBJECT_DATASET_DIR', 'objects'))
default_grasp_dataset = os.path.expanduser(os.environ.get('GRASP_DATASET_DIR', 'results'))

default_num_collision_free_grasps = 64
default_num_colliding_grasps = int(default_num_collision_free_grasps / 4)

default_valid_set_percentage = 0.2

default_guess_only = False
default_overwrite_grasps = False
default_fill_grasps = False
default_load_guesses = False

default_gripper_config = "robotiq_2f_85"

def make_parser(param_dict):
    parser = argparse.ArgumentParser(description="Grasp Data Generation Example for https://github.com/NVlabs/GraspGen")
    
    # Register argument groups since we'll be adding arguments to them
    from graspgen_utils import register_argument_group
    register_argument_group(parser, 'graspgen', 'graspgen', 'GraspGen data generation options')
    
    # Add datagen-specific arguments
    add_arg_to_group('graspgen', parser, "--graspgen_source", default=default_graspgen_source, type=str, 
                     help="Path to the GraspGen source code")
    add_arg_to_group('graspgen', parser, "--object_scales_json", default=default_object_scales_json, type=str, 
                     help="Path to JSON file containing object scales. Format: {\"path/to/object.obj\": scale, ...}")
    add_arg_to_group('graspgen', parser, "--object_dataset", default=default_object_dataset, type=str, 
                     help="Root folder path for the objects in the JSON file")
    add_arg_to_group('graspgen', parser, "--guess_only", action="store_true", default=default_guess_only, 
                     help="Only generate grasp guess data and skip validation (default: validate grasps with sim)")
    add_arg_to_group('graspgen', parser, "--overwrite_grasps", action="store_true", default=default_overwrite_grasps, 
                     help="Overwrite existing grasp data files (default: skip existing files)")
    add_arg_to_group('graspgen', parser, "--fill_grasps", action="store_true", default=default_fill_grasps, 
                     help="Fill existing grasp files with additional grasps if needed (default: skip if file exists)")
    add_arg_to_group('graspgen', parser, "--load_guesses", action="store_true", default=default_load_guesses, 
                     help="Load grasps from existing grasp files instead of generating new ones (default: generate new grasps)")
    add_arg_to_group('graspgen', parser, "--overwrite_gripper_config", action="store_true", default=False, 
                     help="Overwrite existing gripper config files (default: use existing files)")
    add_arg_to_group('graspgen', parser, "--grasp_dataset", default=default_grasp_dataset, type=str, 
                     help="Root folder path for the grasp data output")
    add_arg_to_group('graspgen', parser, "--num_collision_free_grasps", default=default_num_collision_free_grasps, type=int, 
                     help="Number of collision free grasps to generate and test with the simulator")
    add_arg_to_group('graspgen', parser, "--num_colliding_grasps", default=default_num_colliding_grasps, type=int, 
                     help="Number of colliding grasps to generate and test with the simulator")
    add_arg_to_group('graspgen', parser, "--valid_set_percentage", default=default_valid_set_percentage, type=float, 
                     help="Size of the valid set as a fraction of the total number of grasps")
    
    add_grasp_guess_args(parser, param_dict, **collect_grasp_guess_args(param_dict))
    add_grasp_sim_args(parser, param_dict, **collect_grasp_sim_args(param_dict))

    return parser

def create_graspgen_config_files(args, gripper, gripper_name, graspgen_source, overwrite):
    """
    Create GraspGen configuration files for a gripper.
    
    This function generates two configuration files in the GraspGen format:
    1. A YAML file containing gripper parameters for grasp sampling
    2. A Python file containing gripper model and control point definitions
    
    Args:
        args: Arguments object
        gripper: Gripper object containing gripper properties and configuration
        gripper_name (str): Name to use for the gripper configuration files
        graspgen_source (str): Path to the GraspGen source code directory
        overwrite (bool): Whether to overwrite existing configuration files
    """
    
    # Extract gripper properties
    gripper_file = gripper.config.gripper_file
    
    # Check if approach axis is z-axis (2), warn if not
    if gripper.approach_axis != 2:
        print_yellow(f"⚠️  Warning: Gripper approach axis is {gripper.approach_axis}, but _create_graspgen_yaml_config assumes z-axis (2) approach. Generated YAML may be incorrect.")
    
    # Calculate gripper dimensions from bite points
    # TODO Make this work with single ginger gripper
    # TODO Deal with single open configuration?
    width = 2.0*abs(gripper.bite_point[gripper.open_axis])
    depth = abs(gripper.bite_point[gripper.approach_axis])
    
    rotation_xyzw = [0.0, 0.0, 0.0, 1.0] if gripper.open_axis == 0 else [0, 0, 0.7071, 0.7071]
    # Create YAML content
    yaml_content = {
        "name": gripper_name,
        "file_name": gripper_file,
        "width": width,
        "depth": depth,
        "transform_offset_from_asset_to_graspgen_convention": [[0.0, 0.0, 0.0], rotation_xyzw],
        "symmetric_antipodal": True
    }

    # Expand the graspgen_source path
    graspgen_source = os.path.expanduser(graspgen_source)
    config_base_dir = graspgen_source
    if not os.path.exists(config_base_dir):
        config_base_dir = args.grasp_dataset
        print_yellow(f"⚠️  Warning: GraspGen source code directory {graspgen_source} does not exist, writing gripper config to {args.grasp_dataset}")

    config_dir = os.path.join(config_base_dir, 'config', 'grippers')
    
    # Create config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    yaml_path = os.path.join(config_dir, f'{gripper_name}.yaml')
    py_path = os.path.join(config_dir, f'{gripper_name}.py')
    
    # Check if files exist and overwrite is False
    if not overwrite and (os.path.exists(yaml_path) and os.path.exists(py_path)):
        print_blue(f"GraspGen config files in {config_dir} already exist for {gripper_name}, skipping...")
        
        # Check if the existing YAML content is different from the new content
        try:
            with open(yaml_path, 'r') as f:
                existing_yaml_content = yaml.safe_load(f)
            
            # Compare the relevant fields (ignore comments and formatting)
            if existing_yaml_content != yaml_content:
                print_yellow(f"⚠️  Warning: Existing YAML config for {gripper_name} differs from the current gripper configuration.")
                print_yellow(f"   Existing: {existing_yaml_content}")
                print_yellow(f"   Current:  {yaml_content}")
                print_yellow(f"   Consider using --overwrite_gripper_config to update the configuration files.")
        except (yaml.YAMLError, FileNotFoundError) as e:
            print_yellow(f"⚠️  Warning: Could not read existing YAML file {yaml_path}: {e}")
        
        return yaml_content
    
    # Generate configuration files
    print_blue(f"Creating GraspGen config files for {gripper_name} in {config_dir}")
    _create_graspgen_config(yaml_content, yaml_path, py_path)
    return yaml_content


def _create_graspgen_config(yaml_content, yaml_path, py_path):
    """Create the YAML configuration file for GraspGen."""

    # Write YAML file
    with open(yaml_path, 'w') as f:
        f.write("# Generated by GraspDataGen\n")
        yaml.dump(yaml_content, f)

    # Create Python content
    py_content = f'''# Generated by GraspDataGen
import torch
import numpy as np
import trimesh
from grasp_gen.robot import get_canonical_gripper_control_points

class GripperModel(object):
    def __init__(self, _=None):
        self.mesh = trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))

    def get_gripper_collision_mesh(self):
        return self.mesh

    def get_gripper_visual_mesh(self):
        return self.mesh

def load_control_points() -> torch.Tensor:
    control_points = get_canonical_gripper_control_points({yaml_content['width']}, {yaml_content['depth']})
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T

def load_control_points_for_visualization(): 
    control_points = get_canonical_gripper_control_points({yaml_content['width']}, {yaml_content['depth']})
    mid_point = (control_points[0] + control_points[1]) / 2
    control_points = [
        control_points[-2], control_points[0], mid_point,
        [0, 0, 0], mid_point, control_points[1], control_points[-1]
    ]
    return [control_points, ]
'''
    # Write Python file
    with open(py_path, 'w') as f:
        f.write(py_content)

def create_test_valid_set_files(splits_data_write_path, objects_and_scales, valid_set_percentage, overwrite=False):
    # create the test/valid set files
    train_file_path = os.path.join(splits_data_write_path, "train.txt")
    valid_file_path = os.path.join(splits_data_write_path, "valid.txt")
    
    # Check if files exist and if they contain all current objects
    should_recreate = False
    if not overwrite and os.path.exists(train_file_path) and os.path.exists(valid_file_path):
        # Check if existing files contain all objects from objects_and_scales
        existing_train_objects = set()
        existing_valid_objects = set()
        
        try:
            with open(train_file_path, 'r') as f:
                existing_train_objects = set(line.strip() for line in f if line.strip())
            with open(valid_file_path, 'r') as f:
                existing_valid_objects = set(line.strip() for line in f if line.strip())
            
            current_objects = set(objects_and_scales.keys())
            existing_objects = existing_train_objects | existing_valid_objects
            
            if current_objects != existing_objects:
                print_blue(f"Existing train/valid files don't contain all current objects. Recreating...")
                should_recreate = True
            else:
                print_blue(f"Test/valid set files already exist in {splits_data_write_path} and contain all objects, skipping...")
                return
        except Exception as e:
            print_blue(f"Error reading existing train/valid files: {e}. Recreating...")
            should_recreate = True
    
    if overwrite or should_recreate or not os.path.exists(train_file_path) or not os.path.exists(valid_file_path):
        print_blue(f"Creating test/valid set files in {splits_data_write_path}")
        
        # Convert objects_and_scales to list for shuffling
        object_list = list(objects_and_scales.keys())
        
        # Shuffle for random split
        import random
        random.shuffle(object_list)
        
        # Calculate split sizes
        total_objects = len(object_list)
        valid_set_size = max(1, int(total_objects * valid_set_percentage))
        train_set_size = total_objects - valid_set_size
        
        # Split objects
        train_objects = object_list[:train_set_size]
        valid_objects = object_list[train_set_size:]
        
        # Write train file
        with open(train_file_path, 'w') as f:
            for object_path in train_objects:
                f.write(object_path + "\n")
        
        # Write valid file
        with open(valid_file_path, 'w') as f:
            for object_path in valid_objects:
                f.write(object_path + "\n")
        
        print_blue(f"Created train set with {len(train_objects)} objects and valid set with {len(valid_objects)} objects")


def setup_gripper_and_generator(args):
    """Setup gripper configuration, create gripper instance, and initialize guess generator."""
    # Apply gripper configuration if gripper_config is specified
    if args.gripper_config:
        print_blue(f"Using gripper config '{args.gripper_config}'")
        gripper_config = get_gripper_config(args.gripper_config)
        apply_gripper_config_to_args(args, gripper_config)
        print_blue(f"Gripper file: {args.gripper_file}")
    
    # Use gripper file name (without folders or extensions) as folder name if not config
    gripper_name = args.gripper_config if args.gripper_config else os.path.splitext(os.path.basename(args.gripper_file))[0]
    gripper = Gripper.from_args(args)
    guess_generator = GraspGuessGenerator.from_args(args, gripper)

    # create the yaml and py config files for the gripper
    gripper_yaml_config = create_graspgen_config_files(args, gripper, gripper_name, args.graspgen_source, args.overwrite_gripper_config)
    
    return gripper_name, gripper, guess_generator, gripper_yaml_config


def setup_paths_and_data(args, gripper_name):
    """Setup output paths, load object data, and create file mappings."""
    # prepare the grasp data write out folders.
    grasp_data_write_path = os.path.join(os.path.expanduser(args.grasp_dataset), "grasp_data", gripper_name)
    splits_data_write_path = os.path.join(os.path.expanduser(args.grasp_dataset), "splits", gripper_name)
    os.makedirs(grasp_data_write_path, exist_ok=True)
    os.makedirs(splits_data_write_path, exist_ok=True)

    # load the object scales json file to match objects to grasp files
    if not os.path.exists(args.object_scales_json):
        print_red(f"Error: Object scales json file {args.object_scales_json} does not exist")
        return None, None, None, None
    with open(args.object_scales_json, 'r') as f:
        objects_and_scales = json.load(f)

    # create the grasp file paths and uuid map
    grasp_file_paths = []
    uuid_map = {}
    for i, (object_path, scale) in enumerate(objects_and_scales.items()):
        # graspgen does not use the nvidia grasp file format.  It uses a json file with object, scale, gripper, and grasps info. 
        #  grasps have transforms, object_in_gripper.
        # we will generate those files with these names.
        grasp_file_path = f"{os.path.basename(object_path)}.{scale}.grasps.json"
        grasp_file_paths.append(grasp_file_path)
        uuid_map[object_path] = grasp_file_path

    full_uuid_map_file_path = os.path.join(grasp_data_write_path, "map_uuid_to_path.json")
    with open(full_uuid_map_file_path, 'w') as f:
        json.dump(uuid_map, f, indent=2, separators=(',', ': '))

    # create the test/valid set files
    create_test_valid_set_files(splits_data_write_path, objects_and_scales, args.valid_set_percentage)
    
    return grasp_data_write_path, splits_data_write_path, objects_and_scales, grasp_file_paths


def check_existing_grasp_file(full_grasp_file_path, args):
    """Check if existing grasp file exists and determine if more grasps are needed."""
    existing_data = None
    need_more_grasps = True
    
    if os.path.exists(full_grasp_file_path) and not args.overwrite_grasps and not args.load_guesses:
        if not args.fill_grasps:
            print_yellow(f" ⏭️  Skipped (file exists)")
            need_more_grasps = False
        else:
            try:
                with open(full_grasp_file_path, 'r') as f:
                    existing_data = json.load(f)
                print_blue(f"Loading existing grasp data from {full_grasp_file_path}")
                
                # Check if we need to generate more grasps
                if existing_data and 'grasps' in existing_data:
                    existing_grasps_count = len(existing_data['grasps'].get('transforms', []))
                    # get the number of successful and failed grasps
                    successful_grasps_count = sum(existing_data['grasps'].get('object_in_gripper', []))
                    failed_grasps_count = existing_grasps_count - successful_grasps_count
                    
                    if successful_grasps_count >= args.num_collision_free_grasps and failed_grasps_count >= args.num_colliding_grasps:
                        print_yellow(f" ⏭️  Skipped (sufficient grasps: {successful_grasps_count}/{args.num_collision_free_grasps} success, {failed_grasps_count}/{args.num_colliding_grasps} fail)")
                        need_more_grasps = False
                        
            except (json.JSONDecodeError, KeyError) as e:
                print_yellow(f"Warning: Could not load existing grasp file {full_grasp_file_path}: {e}")
                existing_data = None
    
    return existing_data, need_more_grasps


def generate_grasp_guesses(args, guess_generator, object, full_grasp_file_path):
    """Generate or load grasp guesses for an object."""
    # Check if we should load grasps from file instead of generating new ones
    if args.load_guesses and os.path.exists(full_grasp_file_path):
        print_blue(f"Loading grasps from existing file: {full_grasp_file_path}")
        # Load all grasps from file and select the best ones
        grasp_guess_buffer = guess_generator.load_grasps(full_grasp_file_path, object, args.num_collision_free_grasps, args.num_colliding_grasps)
        
        if grasp_guess_buffer is None:
            print_yellow(f"Failed to load grasps from {full_grasp_file_path}, falling back to generation")
            grasp_guess_buffer = guess_generator.generate_grasps(object, args.num_collision_free_grasps, args.num_colliding_grasps)
    else:
        grasp_guess_buffer = guess_generator.generate_grasps(object, args.num_collision_free_grasps, args.num_colliding_grasps)
    
    return grasp_guess_buffer


def run_grasp_simulation(args, grasp_guess_buffer, existing_data):
    """Run grasp simulation to validate grasps."""
    # Check if we already have enough grasps and can skip simulation
    skip_simulation = False
    if existing_data and 'grasps' in existing_data:
        existing_successful = sum(existing_data['grasps'].get('object_in_gripper', []))
        existing_failed = len(existing_data['grasps'].get('object_in_gripper', [])) - existing_successful
        
        if existing_successful >= args.num_collision_free_grasps and existing_failed >= args.num_colliding_grasps:
            skip_simulation = True
            print_blue(f" ⏭️  Skipping simulation - already have sufficient grasps ({existing_successful}/{args.num_collision_free_grasps} success, {existing_failed}/{args.num_colliding_grasps} fail)")

    if not args.guess_only and not skip_simulation:
        # Run simulation to validate grasps
        validation_start_time = time.time()
        grasp_sim = GraspingSimulation.from_args(args, grasp_guess_buffer)
        grasp_sim_buffer = grasp_sim.validate_grasps()
        validation_time = time.time() - validation_start_time
    else:
        grasp_sim_buffer = None
        validation_time = 0.0
    
    return grasp_sim_buffer, validation_time


def process_grasp_transforms(grasp_sim_buffer, grasp_guess_buffer):
    """Process grasp transforms from simulation or guess buffers."""
    if grasp_sim_buffer is not None:
        transforms_np = grasp_sim_buffer.transforms.numpy()
        is_success_np = grasp_sim_buffer.is_success.numpy()
    else:
        if grasp_guess_buffer.succ_buff is None or grasp_guess_buffer.num_successes == 0:
            print_yellow("No successful grasps to process")
            transforms_np = np.array([])
            is_success_np = []
        else:
            transforms_np = grasp_guess_buffer.succ_buff.transforms.numpy()[:grasp_guess_buffer.num_successes]
            is_success_np = [1] * grasp_guess_buffer.num_successes
    
    # Append failed transforms and their success indicators
    if grasp_guess_buffer.num_fails:
        fail_transforms = grasp_guess_buffer.fail_buff.transforms.numpy()[:grasp_guess_buffer.num_fails]
        if len(transforms_np) > 0:
            transforms_np = np.concatenate([transforms_np, fail_transforms])
            is_success_np = np.concatenate([is_success_np, [0] * grasp_guess_buffer.num_fails])
        else:
            transforms_np = fail_transforms
            is_success_np = [0] * grasp_guess_buffer.num_fails
    
    transforms_list = []
    if len(transforms_np) > 0:
        for i, transform in enumerate(transforms_np):
            # Convert wp.transform to 4x4 matrix
            # wp.transforms are [x, y, z, qx, qy, qz, qw]
            transform_matrix = transform_to_matrix(transform)  # TODO GPU instead of CPU
            transforms_list.append(transform_matrix.tolist())
    else:
        print_yellow("No grasps (successful or failed) to save")
    
    return transforms_list, is_success_np


def select_grasps_by_type(transforms, success_flags, max_successful, max_failed):
    """Helper function to select grasps by type up to the specified limits."""
    successful_indices = [i for i, success in enumerate(success_flags) if success]
    failed_indices = [i for i, success in enumerate(success_flags) if not success]
    
    # Select up to the limits
    selected_successful = successful_indices[:max_successful]
    selected_failed = failed_indices[:max_failed]
    
    # Combine and sort by original order
    selected_indices = sorted(selected_successful + selected_failed)
    
    return [transforms[i] for i in selected_indices], [success_flags[i] for i in selected_indices]


def combine_existing_and_new_grasps(existing_data, transforms_list, is_success_np, args):
    """Combine existing grasps with new ones, filtering to only what's needed."""
    all_transforms = []
    all_success_flags = []
    
    if existing_data and 'grasps' in existing_data:
        existing_transforms = existing_data['grasps'].get('transforms', [])
        existing_object_in_gripper = existing_data['grasps'].get('object_in_gripper', [])
        
        # Count existing grasps
        existing_successful = sum(existing_object_in_gripper)
        existing_failed = len(existing_object_in_gripper) - existing_successful
        
        # Calculate how many more we need
        needed_successful = max(0, args.num_collision_free_grasps - existing_successful)
        needed_failed = max(0, args.num_colliding_grasps - existing_failed)
        
        # Start with existing grasps
        all_transforms = existing_transforms
        all_success_flags = existing_object_in_gripper
        
        # Only add new grasps of the type we need
        if needed_successful > 0 or needed_failed > 0:
            new_transforms = transforms_list
            new_success_flags = [bool(success) for success in is_success_np]
            
            # Filter new grasps to only include what we need
            filtered_new_transforms = []
            filtered_new_success_flags = []
            
            for i, (transform, success_flag) in enumerate(zip(new_transforms, new_success_flags)):
                if success_flag and needed_successful > 0:
                    filtered_new_transforms.append(transform)
                    filtered_new_success_flags.append(success_flag)
                    needed_successful -= 1
                elif not success_flag and needed_failed > 0:
                    filtered_new_transforms.append(transform)
                    filtered_new_success_flags.append(success_flag)
                    needed_failed -= 1
            
            # Combine existing with filtered new grasps
            all_transforms = existing_transforms + filtered_new_transforms
            all_success_flags = existing_object_in_gripper + filtered_new_success_flags
            
            print_blue(f"Combined {len(existing_transforms)} existing grasps with {len(filtered_new_transforms)} new grasps (filtered from {len(new_transforms)} available)")
        else:
            print_blue(f"Using {len(existing_transforms)} existing grasps (sufficient)")
    else:
        # No existing data, use all new grasps
        all_transforms = transforms_list
        all_success_flags = [bool(success) for success in is_success_np]
    
    return all_transforms, all_success_flags


def create_grasp_data_structure(object_path, scale, gripper_name, args, gripper_yaml_config):
    """Create the basic grasp data structure for JSON output."""
    return {
        "object": {
            "file": object_path,
            "scale": scale
        },
        "gripper": {
            "name": gripper_name,
            "file_name": args.gripper_file,
            "width": gripper_yaml_config['width'],
            "depth": gripper_yaml_config['depth'],
            "transform_offset_from_asset_to_graspgen_convention": gripper_yaml_config['transform_offset_from_asset_to_graspgen_convention']
        },
        "grasps": {
            "transforms": [],
            "object_in_gripper": []
        }
    }


def calculate_validation_stats(grasp_sim_buffer, total_grasps, success_count):
    """Calculate validation statistics."""
    validation_successes = 0
    validation_fails = 0
    if grasp_sim_buffer is not None:
        # Convert to tensor if it's an array
        is_success_tensor = torch.tensor(grasp_sim_buffer.is_success) if not isinstance(grasp_sim_buffer.is_success, torch.Tensor) else grasp_sim_buffer.is_success
        validation_successes = torch.sum(is_success_tensor).item()
        validation_fails = len(grasp_sim_buffer.is_success) - validation_successes
    
    # Calculate total fails (validation fails + guess fails)
    fail_count = total_grasps - success_count
    total_fails = validation_fails + (fail_count - validation_fails)
    
    return validation_successes, validation_fails, total_fails


def process_single_object(args, object_path, scale, grasp_file_path, grasp_data_write_path, 
                         gripper_name, gripper_yaml_config, guess_generator, i, total_objects):
    """Process a single object to generate grasp data."""
    start_time = time.time()
    
    # Construct full path by joining object_dataset with the relative path from JSON
    full_object_path = os.path.join(os.path.expanduser(args.object_dataset), object_path)
    if not os.path.exists(full_object_path):
        print_red(f"Error: Object file {full_object_path} does not exist")
        return None, 0, 0, 0  # skipped, time, processed, skipped_count

    # graspgen does not use the nvidia grasp file format.  It uses a json file with object, scale, gripper, and grasps info.  grasps have transforms, object_in_gripper.
    full_grasp_file_path = os.path.join(grasp_data_write_path, grasp_file_path)
    
    # Show progress and output filename early
    progress_pct = (i + 1) / total_objects * 100
    print(f"[{i+1}/{total_objects}] ({progress_pct:.1f}%)", end="", flush=True)
    print_blue(f" Processing: {full_grasp_file_path}", end="")
    
    # Check existing grasp file
    existing_data, need_more_grasps = check_existing_grasp_file(full_grasp_file_path, args)
    
    if not need_more_grasps:
        return "skipped", 0, 0, 1  # skipped, time, processed, skipped_count
    
    print()  # New line after the processing line
    object = GuessObject.from_file(full_object_path, scale, args=args)
    if object is None:
        return "skipped", 0, 0, 1  # skipped, time, processed, skipped_count
    
    # Generate grasp guesses
    grasp_guess_buffer = generate_grasp_guesses(args, guess_generator, object, full_grasp_file_path)
    
    # Run simulation
    grasp_sim_buffer, validation_time = run_grasp_simulation(args, grasp_guess_buffer, existing_data)
    
    # Create grasp data structure
    grasp_data = create_grasp_data_structure(object_path, scale, gripper_name, args, gripper_yaml_config)
    
    # Process transforms
    transforms_list, is_success_np = process_grasp_transforms(grasp_sim_buffer, grasp_guess_buffer)
    
    # Combine existing and new grasps
    all_transforms, all_success_flags = combine_existing_and_new_grasps(existing_data, transforms_list, is_success_np, args)
    
    # Select exactly the number of grasps we need
    final_transforms, final_success_flags = select_grasps_by_type(
        all_transforms, all_success_flags, 
        args.num_collision_free_grasps, args.num_colliding_grasps
    )
    
    # Log if we limited the grasps
    if len(final_transforms) < len(all_transforms):
        successful_count = sum(final_success_flags)
        failed_count = len(final_success_flags) - successful_count
        print_blue(f"Limited to {len(final_transforms)} grasps from {len(all_transforms)} available ({successful_count} success, {failed_count} failed)")
    
    grasp_data["grasps"]["transforms"] = final_transforms
    grasp_data["grasps"]["object_in_gripper"] = final_success_flags
    
    # Save the grasp data to JSON file
    with open(full_grasp_file_path, 'w') as f:
        json.dump(grasp_data, f, indent=2)
    
    end_time = time.time()
    object_time = end_time - start_time
    
    # Calculate final counts including both simulation and guess results
    total_grasps = len(grasp_data['grasps']['transforms'])
    success_count = sum(grasp_data['grasps']['object_in_gripper'])
    
    # Calculate validation statistics
    validation_successes, validation_fails, total_fails = calculate_validation_stats(grasp_sim_buffer, total_grasps, success_count)
    
    print_green(f" ✓ Validation complete: {validation_time:.1f}s : Successes {validation_successes}, Fails {validation_fails} ({total_fails} total fails) , and {total_grasps} total grasps in ⏱️  {object_time:.1f}s")
    
    return "processed", object_time, 1, 0  # status, time, processed, skipped_count


def process_all_objects(args, objects_and_scales, grasp_file_paths, grasp_data_write_path,
                       gripper_name, gripper_yaml_config, guess_generator):
    """Process all objects in the dataset."""
    total_time = 0.0
    num_processed = 0
    num_skipped = 0
    count_zero = -1  # for debugging, set to -1 if you want to run all objects in the json file, or 5 if you only want to run the first 5
    total_objects = len(objects_and_scales)
    
    for i, ((object_path, scale), grasp_file_path) in enumerate(zip(objects_and_scales.items(), grasp_file_paths)):
        if not count_zero:
            break
        count_zero -= 1

        status, object_time, processed_count, skipped_count = process_single_object(
            args, object_path, scale, grasp_file_path, grasp_data_write_path,
            gripper_name, gripper_yaml_config, guess_generator, i, total_objects
        )
        
        if status == "skipped":
            num_skipped += skipped_count
        elif status == "processed":
            total_time += object_time
            num_processed += processed_count
            
            # Print average every 10 processed objects
            if num_processed % 10 == 0:
                avg_time = total_time / num_processed
                print_blue(f"  📊 Average time per processed object: {avg_time:.1f}s (processed: {num_processed}, skipped: {num_skipped})")
    
    return total_time, num_processed, num_skipped


def print_final_summary(total_objects, num_processed, num_skipped, total_time):
    """Print final processing summary."""
    print_blue(f"\n📊 Final Summary:")
    print_blue(f"  Total objects: {total_objects}")
    print_blue(f"  Processed: {num_processed}")
    print_blue(f"  Skipped: {num_skipped}")
    if num_processed > 0:
        print_blue(f"  Average time per processed object: {total_time / num_processed:.1f}s")
    print_blue(f"  Total processing time: {total_time:.1f}s")


def graspgen_main(args):
    
    # Setup gripper and generator
    gripper_name, gripper, guess_generator, gripper_yaml_config = setup_gripper_and_generator(args)

    # Setup paths and load data
    result = setup_paths_and_data(args, gripper_name)
    if result[0] is None:  # Error in setup
        return
    grasp_data_write_path, splits_data_write_path, objects_and_scales, grasp_file_paths = result


    # Process all objects
    total_time, num_processed, num_skipped = process_all_objects(
        args, objects_and_scales, grasp_file_paths, grasp_data_write_path,
        gripper_name, gripper_yaml_config, guess_generator
    )
    
    # Print final summary
    print_final_summary(len(objects_and_scales), num_processed, num_skipped, total_time)

if __name__ == "__main__":
    parser = make_parser(globals())
    args = parser.parse_args()
    graspgen_main(args)