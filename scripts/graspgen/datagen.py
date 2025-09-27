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
from grasp_guess import GraspGuessGenerator, GuessObject, add_grasp_guess_args, collect_grasp_guess_args
from grasp_sim import  GraspingSimulation, add_grasp_sim_args, collect_grasp_sim_args
from gripper_configurations import get_gripper_config, apply_gripper_config_to_args
import json
import os
import time
from graspgen_utils import print_blue, print_purple, grasp_data_exists, add_arg_to_group

default_object_scales_json = 'objects/datagen_example.json'
default_object_root = os.environ.get('OBJECT_DATASET_DIR', 'objects')

default_sim_output_folder = os.path.join(os.environ.get('GRASP_DATASET_DIR', ''), 'datagen_sim_data')
default_guess_output_folder = ''

default_num_grasps = 16

def make_parser(param_dict):
    parser = argparse.ArgumentParser(description="Grasp Gen Data Generation.")
    
    # Register argument groups since we'll be adding arguments to them
    from graspgen_utils import register_argument_group
    register_argument_group(parser, 'datagen', 'datagen', 'Data generation options')
    
    # Add datagen-specific arguments
    add_arg_to_group('datagen', parser, "--object_scales_json", default=default_object_scales_json, type=str, 
                     help="Path to JSON file containing object scales. Format: {\"path/to/object.obj\": scale, ...}")
    add_arg_to_group('datagen', parser, "--object_root", default=default_object_root, type=str, 
                     help="Root folder path for the objects in the JSON file")
    add_arg_to_group('datagen', parser, "--overwrite_existing", action="store_true", default=False, 
                     help="Overwrite existing simulation data files (default: skip existing files)")
    add_arg_to_group('datagen', parser, "--sim_output_folder", default=default_sim_output_folder, type=str, 
                     help="Output folder for simulation data (default: grasp_sim_data)")
    add_arg_to_group('datagen', parser, "--guess_output_folder", default=default_guess_output_folder, type=str, 
                     help="Output folder for grasp guess data (default: grasp_guess_data)")
    
    add_grasp_guess_args(parser, param_dict, **collect_grasp_guess_args(param_dict))
    add_grasp_sim_args(parser, param_dict, **collect_grasp_sim_args(param_dict))

    return parser

def datagen_main(args):
    
    # Apply gripper configuration if gripper_config is specified
    if args.gripper_config:
        print_blue(f"Using gripper config '{args.gripper_config}'")
        gripper_config = get_gripper_config(args.gripper_config)
        apply_gripper_config_to_args(args, gripper_config)
        print_blue(f"Gripper file: {args.gripper_file}")
    
    with open(args.object_scales_json, 'r') as f:
        objects_and_scales = json.load(f)
    
    guess_generator = GraspGuessGenerator.from_args(args)
    
    count_zero = -1 # for debugging, set to -1 if you want to run all objects in the json file, or 5 if you only want to run the first 5
    
    # Set output folders based on arguments
    sim_save_to_folder = args.sim_output_folder if args.sim_output_folder else default_sim_output_folder
    guess_save_to_folder = args.guess_output_folder if args.guess_output_folder else default_guess_output_folder
    
    # Use gripper file name (without folders or extensions) as folder name
    gripper_name = os.path.splitext(os.path.basename(args.gripper_file))[0]

    total_time = 0.0
    num_processed = 0
    file_name_prefix = ""
    if args.device == "cpu":
        file_name_prefix = "cpu"
    for i, (object_path, scale) in enumerate(objects_and_scales.items()):
        start_time = time.time()
        if not count_zero:
            break
        count_zero -= 1
        file_extension_prefix= "" if scale == 1.0 else f"{scale}"

        # Construct full path by joining object_root with the relative path from JSON
        full_object_path = os.path.join(args.object_root, object_path)
        
        # Check if simulation data already exists
        if not args.overwrite_existing:
            if predicted_sim_file := grasp_data_exists(gripper_name, full_object_path, sim_save_to_folder, file_name_prefix, file_extension_prefix):
                print_blue(f"✓ Simulation data already exists: {predicted_sim_file}")
                continue
        
        print_purple(f"[{i+1}/{len(objects_and_scales)}] Processing object: {full_object_path} with scale: {scale}")

        object = GuessObject.from_file(full_object_path, scale, args=args)
        grasp_guess_buffer = guess_generator.generate_grasps(object, args.num_grasps, 0)
        if guess_save_to_folder:
            _, _ = grasp_guess_buffer.create_isaac_grasp_data(save_successes=True, save_to_folder=guess_save_to_folder, 
                                                              file_name_prefix=file_name_prefix, file_extension_prefix=file_extension_prefix)

        grasp_sim = GraspingSimulation.from_args(args, grasp_guess_buffer)

        grasp_sim_buffer = grasp_sim.validate_grasps()
        if sim_save_to_folder:
            _, _ = grasp_sim.create_isaac_grasp_data(grasp_sim_buffer, only_driven_joints=True, save_successes=True, save_fails=True, save_to_folder=sim_save_to_folder, 
                                                     file_name_prefix=file_name_prefix, file_extension_prefix=file_extension_prefix)

        end_time = time.time()
        print(f"Time for object {i+1}: {end_time - start_time} seconds")
        total_time += end_time - start_time
        num_processed += 1

    print_blue(f"Total time: {total_time} seconds")
    print_blue(f"Average time per object: {total_time / max(num_processed, 1)} seconds")

if __name__ == "__main__":
    parser = make_parser(globals())
    args = parser.parse_args()
    datagen_main(args)