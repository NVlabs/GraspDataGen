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

# flake8: noqa: E303, E111
import argparse
import os
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import trimesh
import warp as wp
import os

from grasp_constants import GraspState
from gripper import Gripper, add_gripper_args, collect_gripper_args, apply_gripper_configuration
from object import add_object_args, ObjectConfig, collect_object_args
#wp.config.verify_cuda = True
from warp_kernels import (
    triangle_area, compute_transforms_from_random_samples, intersect_mesh_along_negative_normal,find_collision_axis_lengths, ingest_grasp_guess_data_kernel,
    get_closest_offset_transforms_kernel, intersect_other_body_with_offsets, set_offsets_acronym, compute_acronym_transforms_from_random_samples, 
    intersect_with_offsets, get_finger1_success_count, get_finger1_successes, random_mesh_sample, get_body_transforms_acronym,
    get_body_transforms, body_to_object_raycast, center_transform_between_distances, invert_and_orient_grasps, copy_vec3, fill_are_offsets_invalid_kernel,
    intersect_the_offsets_with_offsets, find_widest_valid_opening_kernel
)
import numpy as np
from datetime import datetime
import trimesh
#from typing import List, Tuple, Dict, Any #TODO prompt : "how can I use typing on List, Tupl, and Dict to make things more clear in grasp_guess?"
from graspgen_utils import save_yaml, print_yellow, add_arg_to_group, print_green, print_red, add_isaac_lab_args_if_needed, print_blue, print_purple


default_seed = int(np.random.random()*2147483647.0)
default_num_grasps = 1024
default_num_orientations = 1
default_percent_random_guess_angle = 0.75
default_standoff_distance = 0.001
default_num_offsets = 16
default_do_not_center_finger_opening = False
default_use_acronym_grasp_guess = False
default_correct_acronym_approach = False
default_max_guess_tries = 100

default_save_collision_mesh_folder = ""

def collect_grasp_guess_args(input_dict):
    desired_keys = [
        "default_seed",
        "default_num_grasps",
        "default_num_orientations",
        "default_percent_random_guess_angle",
        "default_standoff_distance",
        "default_num_offsets",
        "default_do_not_center_finger_opening",
        "default_use_acronym_grasp_guess",
        "default_correct_acronym_approach",
        "default_max_guess_tries",
        "default_save_collision_mesh_folder"]
    kwargs = {}
    # No backward compatibility mappings
    for key in desired_keys:
        if key in input_dict:
            kwargs[key] = input_dict[key]
        else:
            # Use local default if not provided in input_dict
            kwargs[key] = globals()[key]
    return kwargs

def add_grasp_guess_args(parser, param_dict, default_seed=default_seed, default_num_grasps=default_num_grasps, default_num_orientations=default_num_orientations,
                         default_percent_random_guess_angle=default_percent_random_guess_angle, default_standoff_distance=default_standoff_distance,
                         default_num_offsets=default_num_offsets, default_do_not_center_finger_opening=default_do_not_center_finger_opening,
                         default_use_acronym_grasp_guess=default_use_acronym_grasp_guess, default_correct_acronym_approach=default_correct_acronym_approach,
                         default_max_guess_tries=default_max_guess_tries, default_save_collision_mesh_folder=default_save_collision_mesh_folder):

    # Register argument groups since we'll be adding arguments to them
    from graspgen_utils import register_argument_group
    register_argument_group(parser, 'grasp_guess', 'grasp_guess', 'Grasp guess options')
    
    add_gripper_args(parser, param_dict, **collect_gripper_args(param_dict))
    add_object_args(parser, param_dict, **collect_object_args(param_dict))

    # warp args
    add_arg_to_group('grasp_guess', parser, "--seed", type=int, default=default_seed, help="Seed for the random number generator.")

    # Simulation args
    add_arg_to_group('grasp_guess', parser, "--num_grasps", type=int, default=default_num_grasps, help="Number of grasps to generate.")
    add_arg_to_group('grasp_guess', parser, "--num_orientations", type=int, default=default_num_orientations, help="Number of orientations to try for each grasp point.")
    add_arg_to_group('grasp_guess', parser, "--percent_random_guess_angle", type=float, default=default_percent_random_guess_angle, help="This percentage of the sample will be random, and the rest will be 0, 90, 180, or 270 degree rotations about the approach axis.")
    add_arg_to_group('grasp_guess', parser, "--standoff_distance", type=float, default=default_standoff_distance, help="Distance to standoff from the surface.")
    # The number of offsets determines how far the finger will move away from the object when looking for collision free positions.
    add_arg_to_group('grasp_guess', parser, "--num_offsets", type=int, default=default_num_offsets, help="Number of offsets to test in initial finger pad placement.")
    add_arg_to_group('grasp_guess', parser, "--do_not_center_finger_opening", action="store_true", default=default_do_not_center_finger_opening,
        help="Do not center the finger opening so fingers are equal distance from the object (default: False)")
    add_arg_to_group('grasp_guess', parser, "--use_acronym_grasp_guess", action="store_true", default=default_use_acronym_grasp_guess,
                        help="Use the original ACRONYM grasp guess method of one opening width tested for collision.")
    add_arg_to_group('grasp_guess', parser, "--correct_acronym_approach", action="store_true", default=default_correct_acronym_approach,
                        help="Correct the ACRONYM transform to move along the approach axis to be the correct distance from the object.")
    add_arg_to_group('grasp_guess', parser, "--max_guess_tries", type=int, default=default_max_guess_tries,
                        help="Maximum total number of attempts to generate grasps before giving up. Set to 0 for unlimited tries (default: 100).")

    # This one is just for debugging when creating the GuessObject, maybe it should be a parameter to the GuessObject class?
    add_arg_to_group('grasp_guess', parser, "--save_collision_mesh_folder", type=str, default=default_save_collision_mesh_folder,
                        help="If not empty, saves the collision mesh as an OBJ file in this directory with '.collision.obj' appended to the input filename")
    
    add_isaac_lab_args_if_needed(parser)

class GraspGuessData:
    def __init__(self, gripper, object, num_grasps, offsets, pregrasp_offsets, transforms, is_invalid, idx_map):
        self.gripper = gripper
        self.object = object
        self.num_grasps = num_grasps
        self.offsets = offsets
        self.pregrasp_offsets = pregrasp_offsets
        self.transforms = transforms
        self.is_invalid = is_invalid
        if idx_map is not None:
            self.idx_map = idx_map
        else:
            indxs = [i for i in range(num_grasps)]
            self.idx_map = wp.array(indxs, dtype=wp.int32, device=transforms.device)

class GraspGuessBuffer:
    class GuessBuffer:
        def __init__(self, max_num_grasps, device):
            self.num_grasps = wp.zeros(1, dtype=wp.int32, device=device)
            self.transforms = wp.array(shape=max_num_grasps, dtype=wp.transform, device=device)
            self.offsets = wp.array(shape=max_num_grasps, dtype=wp.int32, device=device)
            self.pregrasp_offsets = wp.array(shape=max_num_grasps, dtype=wp.int32, device=device)
            self.idx_map = wp.array(shape=max_num_grasps, dtype=wp.int32, device=device)

        def shrink_to_actual_size(self, num_grasps):
            num_grasps = int(num_grasps)

            old_transforms = self.transforms
            old_offsets = self.offsets
            old_pregrasp_offsets = self.pregrasp_offsets
            old_idx_map = self.idx_map

            # Get device from existing arrays
            device = old_transforms.device

            self.transforms = wp.array(shape=num_grasps, dtype=wp.transform, device=device)
            self.offsets = wp.array(shape=num_grasps, dtype=wp.int32, device=device)
            self.pregrasp_offsets = wp.array(shape=num_grasps, dtype=wp.int32, device=device)
            self.idx_map = wp.array(shape=num_grasps, dtype=wp.int32, device=device)

            # Copy transforms (only the first num_grasps elements)
            wp.copy(self.transforms, old_transforms[:num_grasps])
            # Copy offsets (only the first num_grasps elements)
            wp.copy(self.offsets, old_offsets[:num_grasps])
            # Copy pregrasp offsets (only the first num_grasps elements)
            wp.copy(self.pregrasp_offsets, old_pregrasp_offsets[:num_grasps])
            # Copy idx_map (only the first num_grasps elements)
            wp.copy(self.idx_map, old_idx_map[:num_grasps])
            # Set count
            wp.copy(self.num_grasps, wp.array([num_grasps], dtype=wp.int32, device=device))


    def __init__(self, gripper, object, max_num_successes, max_num_fails, device):
        self.gripper = gripper
        self.object = object
        self.max_num_successes = max_num_successes
        self.num_successes = 0
        self.max_num_fails = max_num_fails
        self.num_fails = 0
        self.device = device
        self.succ_buff = GraspGuessBuffer.GuessBuffer(max_num_successes, self.device)# if max_num_successes > 0 else None
        self.fail_buff = GraspGuessBuffer.GuessBuffer(max_num_fails, self.device)# if max_num_fails > 0 else None

    def ingest_grasp_guess_data(self, ggd):
        num_successes_before_ingestion = self.num_successes
        num_fails_before_ingestion = self.num_fails

        need_successes = self.num_successes < self.max_num_successes
        need_fails = self.num_fails < self.max_num_fails

        if need_successes or need_fails:
            wp.launch(ingest_grasp_guess_data_kernel, 
                      dim=ggd.num_grasps,
                      inputs=[ggd.transforms, ggd.offsets, ggd.pregrasp_offsets, ggd.is_invalid, ggd.idx_map,
                              self.max_num_successes if need_successes else 0, self.max_num_fails if need_fails else 0,
                              self.succ_buff.num_grasps, self.succ_buff.transforms, self.succ_buff.offsets, self.succ_buff.pregrasp_offsets, 
                              self.succ_buff.idx_map,
                              self.fail_buff.num_grasps, self.fail_buff.transforms, self.fail_buff.offsets, self.fail_buff.pregrasp_offsets,
                              self.fail_buff.idx_map],
                      device=self.device)
            num_tried_successes = self.succ_buff.num_grasps.numpy()[0] if need_successes else self.max_num_successes
            num_tried_fails = self.fail_buff.num_grasps.numpy()[0] if need_fails else self.max_num_fails
            self.num_successes = min(self.max_num_successes, num_tried_successes)
            self.num_fails = min(self.max_num_fails, num_tried_fails)
        else:
            print("GraspGuessBuffer: ingest_grasp_guess_data: GuessBuffer is full")

        num_successes_added = self.num_successes - num_successes_before_ingestion
        num_fails_added = self.num_fails - num_fails_before_ingestion
        #print(f"num added succ/fail: {num_successes_added}/{num_fails_added}")

        return num_successes_added, num_fails_added

    def shrink_to_actual_size(self):
        """
        Shrink the GraspGuessBuffer to only contain the actual number of grasps found.
        Creates new arrays with the correct size and copies the data.
        
        Returns:
            GraspGuessBuffer: Same buffer with the arrays shrunk to the actual number of grasps
        """
        if self.max_num_successes > self.num_successes:
            self.max_num_successes = self.num_successes
            if self.num_successes == 0:
                self.succ_buff = None
            else:
                self.succ_buff.shrink_to_actual_size(self.num_successes)

        if self.max_num_fails > self.num_fails:
            self.max_num_fails = self.num_fails
            if self.num_fails == 0:
                self.fail_buff = None
            else:
                self.fail_buff.shrink_to_actual_size(self.num_fails)

        return self

    def create_isaac_grasp_data(self, save_successes = True, save_fails = True, only_driven_joints = True, save_to_folder = None, file_name_prefix = "", file_extension_prefix = ""):
        #return None
        succ_transforms = self.succ_buff.transforms.numpy()[:self.num_successes] if save_successes else None
        succ_offsets = self.succ_buff.offsets.numpy()[:self.num_successes] if save_successes else None
        succ_pregrasp_offsets = self.succ_buff.pregrasp_offsets.numpy()[:self.num_successes] if save_successes else None
        succ_idx_map = self.succ_buff.idx_map.numpy()[:self.num_successes] if save_successes else None
        fail_transforms = self.fail_buff.transforms.numpy()[:self.num_fails] if save_fails else None
        fail_offsets = self.fail_buff.offsets.numpy()[:self.num_fails] if save_fails else None
        fail_pregrasp_offsets = self.fail_buff.pregrasp_offsets.numpy()[:self.num_fails] if save_fails else None
        fail_idx_map = self.fail_buff.idx_map.numpy()[:self.num_fails] if save_fails else None

        object_scale = self.object.config.object_scale

        isaac_grasp_data = {
            "format": "isaac_grasp",
            "format_version": "1.0",
            "created_with": "grasp_guess",
            "created_at": datetime.now().isoformat(),
            "object_file": self.object.config.object_file,
            "object_scale": object_scale,
            "gripper_file": self.gripper.config.gripper_file,
            "gripper_frame_link": str(self.gripper.config.base_frame),
            "open_limit": str(self.gripper.open_limit),
            "finger_colliders": self.gripper.config.finger_colliders,
            "base_length": float(self.gripper.base_length),
            "approach_axis": self.gripper.approach_axis,
            "bite_point": list(self.gripper.bite_point),
            "bite_body_idx": self.gripper.finger_indices[0],
            "grasps": {
            }
        }
        driven_joints = list(self.gripper.driven_joints.values())
        cspace_positions = self.gripper.joint_cspace_pos.numpy().tolist()
        pregrasp_cspace_positions = self.gripper.joint_cspace_pos.numpy().tolist()
        bite_points = self.gripper.bite_points.numpy().tolist()
        grasps = {}
        def get_grasp_key(grasp_idx, offset):
            key = f"grasp_{grasp_idx}_{offset}"
            keyid = 0
            while key in grasps:
                key = f"grasp_{grasp_idx}_{offset}_{keyid}"
                keyid += 1
            return key
        def create_isaac_grasp_grasps(isaac_grasp_data, transforms, offsets, pregrasp_offsets, idx_map, only_driven_joints, are_successes, at_idx):
            for i in range(len(transforms)):
                #grasp_idx = idx_map[i]
                grasp_key = get_grasp_key(idx_map[i], offsets[i])
                #if are_successes:
                grasps[grasp_key] = {
                    "confidence": 1.0 if are_successes else 0.0,
                    "position": transforms[i][:3].tolist(),
                    "orientation": {
                        "w": float(transforms[i][6]),
                        "xyz": transforms[i][3:6].tolist()
                    },
                    "cspace_position": {},
                    "bite_point": list(bite_points[offsets[i]]),
                    "pregrasp_cspace_position": {},
                    "pregrasp_bite_point": list(bite_points[pregrasp_offsets[i]]),
                }

                for cspace_pos, pregrasp_cspace_pos, joint_name in zip(cspace_positions[offsets[i]], pregrasp_cspace_positions[pregrasp_offsets[i]], self.gripper.joint_names):
                    if not only_driven_joints or joint_name in driven_joints:
                        grasps[grasp_key]["cspace_position"][str(joint_name)] = float(cspace_pos)
                        grasps[grasp_key]["pregrasp_cspace_position"][str(joint_name)] = float(pregrasp_cspace_pos)

        if save_successes:
            create_isaac_grasp_grasps(isaac_grasp_data, succ_transforms, succ_offsets, succ_pregrasp_offsets, succ_idx_map, only_driven_joints, are_successes=True, at_idx=0)
        if save_fails:
            at_idx = self.num_successes if save_successes else 0
            create_isaac_grasp_grasps(isaac_grasp_data, fail_transforms, fail_offsets, fail_pregrasp_offsets, fail_idx_map, only_driven_joints, are_successes=False, at_idx=at_idx)

        # Sort grasps dictionary by keys
        for key in sorted(grasps.keys()):
            if grasps[key]["confidence"] == 1.0:
                isaac_grasp_data["grasps"][key] = grasps[key]
        
        for key in sorted(grasps.keys()):
            if grasps[key]["confidence"] == 0.0:
                isaac_grasp_data["grasps"][key] = grasps[key]
        
        #isaac_grasp_data["grasps"] = {k: grasps[k] for k in sorted(grasps.keys())}

        #print(f"created {self.num_successes if save_successes else 0} successes and {self.num_fails if save_fails else 0} fails")
        output_file = None
        if save_to_folder is not None:
            from graspgen_utils import predict_grasp_data_filepath
            gripper_name = os.path.splitext(os.path.basename(self.gripper.config.gripper_file))[0]
            output_file = predict_grasp_data_filepath(
                gripper_name,
                self.object.config.object_file,
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


def convert_usd_to_obj_direct(usd_path: str, scale: float = 1.0) -> Optional[str]:
    """
    Convert a USD file to OBJ format using direct Python function calls.
    
    Args:
        usd_path: Path to the USD file
        scale: Scale factor to apply
        
    Returns:
        Path to the converted OBJ file, or None if conversion failed
    """
    try:
        # Check if OBJ file already exists in the same directory
        usd_dir = os.path.dirname(os.path.abspath(usd_path))
        if scale == 1.0:
            obj_filename = os.path.splitext(os.path.basename(usd_path))[0] + ".obj"
        else:
            obj_filename = os.path.splitext(os.path.basename(usd_path))[0] + f".{scale}.obj"
        obj_path = os.path.join(usd_dir, obj_filename)
        
        if os.path.exists(obj_path):
            print_blue(f"✓ Using existing OBJ file: {obj_path}")
            return obj_path
        
        # If OBJ doesn't exist, convert USD to OBJ
        print_blue(f"Converting {usd_path} to {obj_path}")
        
        # Import the conversion function directly
        from usd_to_obj_converter import convert_usd_to_obj
        
        # Perform the conversion
        success = convert_usd_to_obj(usd_path, obj_path, scale)
        
        if success and os.path.exists(obj_path):
            print_blue(f"✓ Successfully converted: {usd_path} -> {obj_path}")
            return obj_path
        else:
            print_yellow(f"USD to OBJ conversion failed")
            return None
            
    except Exception as e:
        print_yellow(f"Error in USD to OBJ conversion: {e}")
        return None


class GuessObject:
    def __init__(self, object_config, device = "cuda", save_collision_mesh_folder = ""):
        self.device = device
        self.config = object_config
        
        # Handle USD files by converting to OBJ first
        # Using local vars so we don't actually change the config, we just have the guess use the obj
        # if a usd was input
        object_file = self.config.object_file
        object_scale = self.config.object_scale
        if object_file.endswith((".usd", ".usda", ".usdc")):
            #print_blue(f"Converting USD file to OBJ: {self.config.object_file}")
            obj_path = convert_usd_to_obj_direct(
                object_file, 
                1.0 # object_scale
            )
            if obj_path is None:
                raise RuntimeError(f"Failed to convert USD file to OBJ: {object_file}")
            
            # Update the config to use the OBJ file
            object_file = obj_path
        
        self.points, self.indices = self.load_obj_or_stl(object_file, object_scale)
        self.points = wp.array(self.points, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(self.indices, dtype=wp.int32, device=self.device).flatten()

            
        self.mesh = wp.Mesh(self.points, self.indices)
        self.nt = len(self.indices)//3
        # to randomly sample the mesh surface it is efficient to keep the cumsum of the triangle area 
        self.cumsum_area_faces = wp.array(shape=self.nt, dtype=wp.float32, device=self.device)
        wp.launch(kernel=triangle_area,
                dim=self.nt,
                inputs=[self.indices, self.points],
                outputs=[self.cumsum_area_faces],
                device=self.device)

        wp.utils.array_scan(self.cumsum_area_faces, self.cumsum_area_faces)  # TODO: Fix this - wp.utils.array_scan doesn't exist

        if save_collision_mesh_folder:
            self.save_collision_mesh(save_collision_mesh_folder)

    @classmethod
    def from_file(cls, file_path, scale=1.0, args=None, device = "cuda", save_collision_mesh_folder = ""):
        if args is not None:
            object_file = args.object_file
            object_scale = args.object_scale
            obj2usd_use_existing_usd = args.obj2usd_use_existing_usd
            obj2usd_collision_approximation = args.obj2usd_collision_approximation
            device = args.device
            save_collision_mesh_folder = args.save_collision_mesh_folder

        # Check if the file exists before creating the object
        if not os.path.exists(file_path):
            print_yellow(f"Object file does not exist: {file_path}")
            return None

        object_config = ObjectConfig.from_file(file_path, scale, args=args)
        return cls(object_config, device, save_collision_mesh_folder)

    def save_collision_mesh(self, output_dir, num_subdivisions=0):
        """Save the collision mesh as an OBJ file."""
        import os
        import trimesh
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the full filename
        filename = os.path.basename(self.config.object_file)
        # Get the base name without extension
        base_name = os.path.splitext(filename)[0]
        
        # Handle different file extensions for output
        if filename.endswith(".obj"):
            # For .obj files, add "collision" before .obj to avoid clash
            output_filename = f"{base_name}.collision.obj"
        elif filename.endswith((".usd", ".stl")):
            # For .usd or .stl files, convert to .obj
            output_filename = f"{base_name}.obj"
        else:
            # For other extensions, add .obj
            output_filename = f"{base_name}.obj"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Create a trimesh object from the points and indices
        vertices = self.points.numpy()
        faces = self.indices.numpy().reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if num_subdivisions > 0:
            submesh = trimesh.remesh.subdivide_loop(mesh.vertices, mesh.faces, iterations=num_subdivisions)
            mesh = trimesh.Trimesh(vertices=submesh[0], faces=submesh[1])
        # Save the mesh
        mesh.export(output_path)
        print(f"Saved collision mesh to {output_path}")

    def load_obj_or_stl(self, file_path, scale):
        mesh = trimesh.load(file_path, validate=True)
        mesh.apply_scale(scale)
        return mesh.vertices, mesh.faces

# TODO: BUG WARNING: Is the transform pointing in the open_axis always such that the second finger pad points toward the first finger on that axis?
# if we need to check what that transform is, then maybe the direction we cast rays for the finger1 collision check against the object
# while running center_finger_opening_kernel is not the correct direction?
class GraspGuessConfig:
    def __init__(self, seed, num_grasps, num_orientations, percent_random_guess_angle, standoff_distance, num_offsets, 
                 do_not_center_finger_opening, use_acronym_grasp_guess, correct_acronym_approach, max_guess_tries, device):
        self.seed = seed
        self.num_grasps = num_grasps
        self.num_orientations = num_orientations
        self.percent_random_guess_angle = percent_random_guess_angle
        self.standoff_distance = standoff_distance
        self.num_offsets = num_offsets
        self.do_not_center_finger_opening = do_not_center_finger_opening
        self.use_acronym_grasp_guess = use_acronym_grasp_guess
        self.correct_acronym_approach = correct_acronym_approach
        self.max_guess_tries = max_guess_tries
        self.device = device

class GraspGuessGenerator:
    def __init__(self, config, gripper: Gripper):
        self.seed = config.seed # self.seed can change, and needs to, but config.seed is there to show what the original input parameter was, and should not change.
        self.config = config
        self.num_grasps = config.num_grasps # num_grasps can chance because of the orientation adjustment, so we need to keep a local copy
        self.gripper = gripper
        self.seed_counter = 0  # Counter to track how many seeds we've generated
        self.validate_config()

    def gen_seed(self):
        """
        Generate a new seed that is reproducible but varied.
        Uses a hash of the original seed combined with a counter to ensure
        each call produces a different but deterministic seed.
        """
        # Use a simple hash function to combine the original seed with the counter
        # This ensures reproducibility while providing variation
        combined = hash((self.config.seed, self.seed_counter)) & 0x7FFFFFFF  # Ensure positive 32-bit integer
        self.seed_counter += 1
        return combined

    @classmethod
    def from_args(cls, args, gripper = None):
        gg_config = GraspGuessConfig(args.seed, args.num_grasps, args.num_orientations, args.percent_random_guess_angle, args.standoff_distance, args.num_offsets, args.do_not_center_finger_opening, args.use_acronym_grasp_guess, args.correct_acronym_approach, args.max_guess_tries, args.device)
        if gripper is None:
            gripper = Gripper.from_args(args)
        if gripper is None:
            raise ValueError("Gripper must be provided to GraspGuessGenerator, but could not be created from args")
        return cls(gg_config, gripper)

    def validate_config(self):
        if self.gripper is None:
            raise ValueError("Gripper must be provided to GraspGuessGenerator")
        if self.config.use_acronym_grasp_guess:
            if len(self.gripper.config.open_configuration) == 0:
                print_yellow("Warning: The original ACRONYM grasp guess calculation was designed " \
                      "to work with a pre-grasp configuration, so without it the maximum open " \
                      "width will be used, and results may be off from expectations.")
        # Enforce a minimum number of guesses to ensure robust sampling
        if self.num_grasps < default_num_grasps:
            print_yellow(f"Warning: Increased num_grasps from {self.num_grasps} to minimum {default_num_grasps}")
            self.num_grasps = default_num_grasps
        # Adjusts num_grasps to be a multiple of num_orientations.
        remainder = self.num_grasps % self.config.num_orientations
        if remainder != 0:
            self.num_grasps = self.num_grasps + (self.config.num_orientations - remainder)
            print_yellow(f"Warning: Adjusted num_grasps from {self.config.num_grasps} to {self.num_grasps} to be a multiple of num_orientations {self.config.num_orientations}")

        self.num_true_random_grasps = self.num_grasps//self.config.num_orientations
    
    def center_finger_opening(self, object, finger1_successes_work_transform, finger1_successes_offset, distances0, distances1, num_transforms):
        distances0.fill_(wp.inf)
        distances1.fill_(wp.inf)
        # First cast rays from finger0 along the open axis to the object
        ray_dir = [0.0, 0.0, 0.0]
        ray_dir[self.gripper.open_axis] = 1.0
        ray_dir = wp.vec3f(ray_dir)
        finger0_points = self.gripper.body_meshes[self.gripper.finger_indices[0]].points
        wp.launch(kernel=body_to_object_raycast,
                      dim=(num_transforms, len(finger0_points)),
                      inputs=[object.mesh.id, ray_dir, finger1_successes_work_transform, 
                              self.gripper.body_transforms[self.gripper.finger_indices[0], :], 
                              finger1_successes_offset, finger0_points, distances0],
                      device=self.config.device)
        
        # Next cast rays from finger1 along the negative open axis to the object
        ray_dir = [0.0, 0.0, 0.0]
        ray_dir[self.gripper.open_axis] = -1.0
        ray_dir = wp.vec3f(ray_dir)
        finger1_points = self.gripper.body_meshes[self.gripper.finger_indices[1]].points
        wp.launch(kernel=body_to_object_raycast,
                      dim=(num_transforms, len(finger1_points)),
                      inputs=[object.mesh.id, ray_dir, finger1_successes_work_transform, 
                              self.gripper.body_transforms[self.gripper.finger_indices[1], :], 
                              finger1_successes_offset, finger1_points, distances1],
                      device=self.config.device)
        
        # Move the work_transform along the negative work axis a distance
        ray_dir = [0.0, 0.0, 0.0]
        ray_dir[self.gripper.open_axis] = -1.0
        ray_dir = wp.vec3f(ray_dir)
        wp.launch(kernel=center_transform_between_distances,
                      dim=num_transforms,
                      inputs=[ray_dir, finger1_successes_work_transform, distances0, distances1],
                      device=self.config.device)
    
    def check_gripper_body_collisions(self, object, bodies_to_collision_check, transform, offset, is_invalid, num_transforms):
        """
        Check every body except the fingers for collision against the object in transforms positions.
        is_invalid[i] set to Grasp.IN_COLLISION if the object is in collision with any of the bodies.
        """
        for b_idx in bodies_to_collision_check:
            b_mesh = self.gripper.body_meshes[b_idx]
            wp.launch(kernel=intersect_other_body_with_offsets,
                      dim=(num_transforms, len(b_mesh.indices)//3),
                      inputs=[b_mesh.points, b_mesh.indices,
                              object.mesh.id, object.mesh.points, object.mesh.indices,
                              self.gripper.body_transforms[b_idx, :], transform, offset],
                      outputs=[is_invalid],
                      device=self.config.device)

    def load_grasps(self, grasp_file_path, object, num_successes, num_fails):
        """
        Load grasps from a JSON file created by graspgen.py.
        
        The function loads all grasps from the file and selects the best ones based on
        the object_in_gripper field. It takes up to num_successes successful grasps
        and up to num_fails failed grasps, prioritizing successful grasps.
        
        Args:
            grasp_file_path (str): Path to the JSON grasp file
            object: GuessObject instance
            num_successes (int): Number of successful grasps needed
            num_fails (int): Number of failed grasps needed
            
        Returns:
            GraspGuessBuffer: Buffer containing loaded grasps
        """
        import json
        
        if not os.path.exists(grasp_file_path):
            print_yellow(f"Grasp file not found: {grasp_file_path}")
            return None
            
        try:
            with open(grasp_file_path, 'r') as f:
                if grasp_file_path.endswith('.json'):
                    grasp_data = json.load(f)
                elif grasp_file_path.endswith('.yaml') or grasp_file_path.endswith('.yml'):
                    import yaml
                    grasp_data = yaml.safe_load(f)
                    # Convert YAML format to JSON format
                    grasp_data = self._convert_yaml_to_json_format(grasp_data)
                else:
                    # Try JSON first, then YAML
                    try:
                        f.seek(0)
                        grasp_data = json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0)
                        import yaml
                        grasp_data = yaml.safe_load(f)
                        grasp_data = self._convert_yaml_to_json_format(grasp_data)
        except (json.JSONDecodeError, yaml.YAMLError, FileNotFoundError) as e:
            print_yellow(f"Error loading grasp file {grasp_file_path}: {e}")
            return None
        
        # Validate the file structure
        if 'grasps' not in grasp_data or 'transforms' not in grasp_data['grasps']:
            print_yellow(f"Invalid grasp file format in {grasp_file_path}")
            return None
        
        transforms_list = grasp_data['grasps']['transforms']
        object_in_gripper_list = grasp_data['grasps'].get('object_in_gripper', [])
        
        if len(transforms_list) == 0:
            print_yellow(f"No grasps found in {grasp_file_path}")
            return None

        # Convert transforms from 4x4 matrices back to wp.transform format
        # We need to convert from the JSON format back to wp.transform format
        from usd_tools import matrix_to_transform
        
        # Separate successful and failed grasps based on object_in_gripper field
        successful_transforms_raw = []
        failed_transforms_raw = []
        
        for i, transform_matrix in enumerate(transforms_list):
            try:
                # Convert 4x4 matrix back to wp.transform
                wp_transform = matrix_to_transform(np.array(transform_matrix))
                
                # Check if this grasp was successful (object_in_gripper field)
                is_successful = object_in_gripper_list[i] if i < len(object_in_gripper_list) else True
                
                if is_successful:
                    successful_transforms_raw.append(wp_transform)
                else:
                    failed_transforms_raw.append(wp_transform)
                    
            except Exception as e:
                print_yellow(f"Error converting transform {i}: {e}")
                continue
        
        if len(successful_transforms_raw) == 0 and len(failed_transforms_raw) == 0:
            print_yellow(f"No valid transforms found in {grasp_file_path}")
            return None
        
        # Select the requested number of grasps
        successful_transforms = successful_transforms_raw[:num_successes]
        failed_transforms = failed_transforms_raw[:num_fails]
        
        # Create GraspGuessBuffer and populate it
        grasp_guess_buffer = GraspGuessBuffer(self.gripper, object, len(successful_transforms), len(failed_transforms), self.config.device)
        
        # Convert to warp arrays
        if successful_transforms:
            succ_transforms_wp = wp.array(successful_transforms, dtype=wp.transform, device=self.config.device)
            # Copy the transforms into the buffer
            wp.copy(grasp_guess_buffer.succ_buff.transforms, succ_transforms_wp)
            # Set the count using wp.copy
            wp.copy(grasp_guess_buffer.succ_buff.num_grasps, wp.array([len(successful_transforms)], dtype=wp.int32, device=self.config.device))
            grasp_guess_buffer.num_successes = len(successful_transforms)
        
        if failed_transforms:
            fail_transforms_wp = wp.array(failed_transforms, dtype=wp.transform, device=self.config.device)
            # Copy the transforms into the buffer
            wp.copy(grasp_guess_buffer.fail_buff.transforms, fail_transforms_wp)
            # Set the count using wp.copy
            wp.copy(grasp_guess_buffer.fail_buff.num_grasps, wp.array([len(failed_transforms)], dtype=wp.int32, device=self.config.device))
            grasp_guess_buffer.num_fails = len(failed_transforms)
        
        print_blue(f"Loaded {len(successful_transforms)} successful and {len(failed_transforms)} failed grasps from {grasp_file_path}")
        print(f"Requested: {num_successes} successes, {num_fails} fails | Actual: {len(successful_transforms)} successes, {len(failed_transforms)} fails")
        
        # Shrink the buffer to actual size and return
        return grasp_guess_buffer.shrink_to_actual_size()

    def _convert_yaml_to_json_format(self, yaml_data):
        """
        Convert YAML grasp format to JSON format expected by load_grasps.
        
        YAML format has grasps as a dictionary with grasp names as keys.
        JSON format has grasps as a list with transforms and object_in_gripper arrays.
        """
        if 'grasps' not in yaml_data:
            raise ValueError("YAML file does not contain 'grasps' section")
        
        transforms_list = []
        object_in_gripper_list = []
        
        for grasp_name, grasp_data in yaml_data['grasps'].items():
            # Convert position and orientation to 4x4 transformation matrix
            position = grasp_data.get('position', [0, 0, 0])
            orientation = grasp_data.get('orientation', {})
            
            # Convert quaternion to rotation matrix
            w = orientation.get('w', 1.0)
            xyz = orientation.get('xyz', [0, 0, 0])
            
            # Create 4x4 transformation matrix
            transform_matrix = self._quaternion_to_matrix(position, xyz, w)
            transforms_list.append(transform_matrix)
            
            # Determine if grasp was successful based on confidence
            confidence = grasp_data.get('confidence', 1.0)
            object_in_gripper_list.append(confidence > 0.0)
        
        # Create JSON format structure
        json_data = {
            'grasps': {
                'transforms': transforms_list,
                'object_in_gripper': object_in_gripper_list
            }
        }
        
        return json_data

    def _quaternion_to_matrix(self, position, xyz, w):
        """
        Convert quaternion and position to 4x4 transformation matrix.
        """
        import numpy as np
        
        # Normalize quaternion
        norm = np.sqrt(w*w + xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])
        if norm > 0:
            w /= norm
            xyz = [x/norm for x in xyz]
        
        # Convert quaternion to rotation matrix
        x, y, z = xyz
        
        # First row
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - w*z)
        r02 = 2*(x*z + w*y)
        
        # Second row
        r10 = 2*(x*y + w*z)
        r11 = 1 - 2*(x*x + z*z)
        r12 = 2*(y*z - w*x)
        
        # Third row
        r20 = 2*(x*z - w*y)
        r21 = 2*(y*z + w*x)
        r22 = 1 - 2*(x*x + y*y)
        
        # Create 4x4 matrix
        matrix = [
            [r00, r01, r02, position[0]],
            [r10, r11, r12, position[1]],
            [r20, r21, r22, position[2]],
            [0, 0, 0, 1]
        ]
        
        return matrix

    def generate_grasps(self, object, num_successes, num_fails):
        # Early return if no grasps are requested
        if num_successes <= 0 and num_fails <= 0:
            print("No grasps requested (num_successes=0, num_fails=0). Returning empty buffer.")
            return GraspGuessBuffer(self.gripper, object, 0, 0, self.config.device)
        
        generator_function = self.generate_grasps_acronym_sampler if self.config.use_acronym_grasp_guess else self.generate_grasps_finger_pad_placement

        grasp_guess_buffer = GraspGuessBuffer(self.gripper, object, num_successes, num_fails, self.config.device)

        grasps_left = [num_successes, num_fails]

        #print(f"Generating {num_successes} successes and {num_fails} fails.")
        tries = 0 # TODO Limit the tries to some reasonable number if not producing after a while... see the server_batch version I think.
        max_tries = 10  # Safety limit to prevent infinite loops
        max_total_tries = self.config.max_guess_tries if self.config.max_guess_tries > 0 else float('inf')  # Total tries limit, 0 means unlimited
        total_tries = 0
        gen_start_time = time.time()
        while grasps_left[0] > 0 or grasps_left[1] > 0:
            # Check total tries limit first
            if total_tries >= max_total_tries or tries >= max_tries:
                print(f"\nWarning: Reached maximum total tries ({max_total_tries}) or consecutive tries ({max_tries}) without finding enough grasps. Stopping generation.")
                print(f"Still needed: {grasps_left[0]} successes, {grasps_left[1]} fails")
                break
                
            grasp_guess_data = generator_function(object)
            #ggd_cpu = grasp_guess_data.is_invalid.numpy().tolist()# = f"grasp_guess_data/{object.config.object_file.split('/')[-1]}"
            num_successes_added, num_fails_added = grasp_guess_buffer.ingest_grasp_guess_data(grasp_guess_data)
            grasps_left[0] -= num_successes_added
            grasps_left[1] -= num_fails_added
            # Seed is now generated dynamically via self.gen_seed() for each warp call
            tries += 1
            total_tries += 1
            
            # Reset consecutive tries counter if we found any grasps
            if num_successes_added > 0 or num_fails_added > 0:
                tries = 0
                
            success_percent = (float(num_successes-grasps_left[0])/float(num_successes))*100.0
            print_purple(f"\r  🔄 Try {total_tries}: +{num_successes_added} success, +{num_fails_added} fail ({success_percent:.1f}% complete)", end="", flush=True)
        gen_time = time.time() - gen_start_time
        actual_successes = grasp_guess_buffer.num_successes
        actual_fails = grasp_guess_buffer.num_fails
        print_green(f"\r ✓ Generated {actual_successes} success + {actual_fails} fail grasps in {gen_time:.1f}s ({total_tries} tries)")

        # Shrink the buffer to actual size and return
        return grasp_guess_buffer.shrink_to_actual_size()
    
    def find_widest_valid_opening(self, object, do_not_center_finger_opening, work_transform, offsets, is_invalid, num_transforms):
        are_offsets_invalid = wp.array(shape=(self.gripper.num_openings, num_transforms), dtype=wp.int32, device=self.config.device)
        wp.launch(kernel=fill_are_offsets_invalid_kernel,
                  dim=(self.gripper.num_openings, num_transforms),
                  inputs=[offsets],
                  outputs=[are_offsets_invalid],
                  device=self.config.device)
        
        for b_idx in self.gripper.finger_indices:
            b_mesh = self.gripper.body_meshes[b_idx]
            wp.launch(kernel=intersect_the_offsets_with_offsets,
                    dim=(self.gripper.num_openings, num_transforms,len(b_mesh.indices)//3),
                    inputs=[do_not_center_finger_opening, offsets, b_mesh.points, b_mesh.indices,
                            object.mesh.id, object.mesh.points, object.mesh.indices,
                            self.gripper.body_transforms[b_idx, :], work_transform,
                            are_offsets_invalid],
                    device=self.config.device)
        wp.launch(kernel=find_widest_valid_opening_kernel,
                  dim=num_transforms,
                  inputs=[self.gripper.num_openings, are_offsets_invalid, is_invalid, offsets],
                  device=self.config.device)
        # offsets are now set to the widest valid opening for the fingers

    def get_random_mesh_samples(self, object, work_points, work_normals):
        wp.launch(kernel=random_mesh_sample,
                  dim=self.num_true_random_grasps,
                  inputs=[object.mesh.id, object.nt, object.cumsum_area_faces, self.gen_seed()],
                  outputs=[work_points, work_normals],
                  device=self.config.device)

    def _get_cache_filepath(self):
        """Get the filepath for the debug cache."""
        cache_dir = "debug_output"
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "random_mesh_samples_cache.npz")

    def _try_load_cached_samples(self, object, work_points, work_normals):
        """Try to load cached samples. Returns True if successfully loaded, False otherwise."""
        cache_file = self._get_cache_filepath()
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            cache_data = np.load(cache_file)
            cached_object_file = str(cache_data['object_file'])
            cached_object_scale = float(cache_data['object_scale'])
            cached_points = cache_data['points']
            cached_normals = cache_data['normals']
            
            # Check if cache matches current object, scale, and has enough samples
            if (cached_object_file == object.config.object_file and 
                cached_object_scale == object.config.object_scale and
                len(cached_points) >= self.num_true_random_grasps):
                
                # Copy cached data to work arrays (only the amount we need)
                points_to_copy = cached_points[:self.num_true_random_grasps]
                normals_to_copy = cached_normals[:self.num_true_random_grasps]
                
                # Convert to warp arrays and copy to device
                cached_points_wp = wp.array(points_to_copy, dtype=wp.vec3, device=self.config.device)
                cached_normals_wp = wp.array(normals_to_copy, dtype=wp.vec3, device=self.config.device)
                
                wp.copy(work_points, cached_points_wp)
                wp.copy(work_normals, cached_normals_wp)
                
                print_green(f"Loaded {self.num_true_random_grasps} cached random samples from {cache_file}")
                return True
            else:
                print_yellow(f"Cache exists but doesn't match requirements. Object: {cached_object_file} vs {object.config.object_file}, "
                      f"object_scale: {cached_object_scale} vs {object.config.object_scale}, "
                      f"Samples: {len(cached_points)} vs {self.num_true_random_grasps}")
                return False
                
        except Exception as e:
            print_red(f"Failed to load cache: {e}")
            return False

    def _save_cached_samples(self, object, work_points, work_normals):
        """Save the current samples to cache."""
        cache_file = self._get_cache_filepath()
        
        try:
            # Convert warp arrays to numpy
            points_np = work_points.numpy()
            normals_np = work_normals.numpy()
            
            # Save to cache with scale parameter
            np.savez(cache_file,
                    object_file=object.config.object_file,
                    object_scale=object.config.object_scale,
                    points=points_np,
                    normals=normals_np)
            
            print(f"Saved {len(points_np)} random samples to cache: {cache_file}")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def generate_grasps_acronym_sampler(self, object):
        if not self.gripper.transform_body_frame == self.gripper.finger_indices[0]:
            self.gripper.set_transform_body_frame(self.gripper.finger_indices[0])
        
        work_transforms = wp.array(shape=self.num_grasps, dtype=wp.mat44, device=self.config.device)
        work_offsets = wp.array(shape=self.num_grasps, dtype=wp.int32, device=self.config.device)
        work_points = wp.array(shape=self.num_true_random_grasps, dtype=wp.vec3, device=self.config.device)
        work_normals = wp.array(shape=self.num_true_random_grasps, dtype=wp.vec3, device=self.config.device)
        work_lengths = wp.array(shape=self.num_true_random_grasps, dtype=wp.float32, device=self.config.device)    
        root_transforms = wp.array(shape=self.num_grasps, dtype=wp.transform, device=self.config.device)
        is_invalid = wp.array(shape=self.num_grasps, dtype=wp.int32, device=self.config.device)

        # acronym-pipeline/graspsampling-py/graspsampling/sampling.py ACRONYM Sampler
        # extra parameters:
        # maximum_aperture: maximum width of the gripper, probably won't worry about this one, but 
        #   I could check bite_at_root[open_axis]*2 and cull out any grasp offsets that are higher than that.
        # num_orientations: each random sample can spin around the closing axis to try new grasps.
        # friction_cone: TODO? utilities.sample_spherical_cap

        # get random object samples and normals from the object
        self.get_random_mesh_samples(object, work_points, work_normals)

        # The biggest difference between the ACRONYM sampler and the finger pad placement sampler is when it checks for
        # collisions, and how it finds the center of the closing axis.
        # The ACRONYM sampler does not do collision until after it gets all the samples, and not while it is finding samples.
        # center of the closing axis is based an the collision line segment from the sample point.  That line segment is
        # found by casting a ray from the sample point in the direction of the -normal, and then finding nearest and furthest point.
        # It uses the center of those points as the point that is pointed to by the approach direction.
        wp.launch(kernel=find_collision_axis_lengths,
                  dim=self.num_true_random_grasps,
                  inputs=[object.mesh.id, work_points, work_normals],
                  outputs=[work_lengths],
                  device=self.config.device)
        wp.launch(kernel=compute_acronym_transforms_from_random_samples,
                  dim=self.num_true_random_grasps,
                  inputs=[work_points, work_normals, self.gen_seed(), self.config.percent_random_guess_angle,
                          work_lengths, self.gripper.num_openings, self.gripper.open_widths_reverse, self.gripper.open_configuration_offset, self.config.correct_acronym_approach,
                          self.gripper.open_axis, True],
                  outputs=[work_transforms, work_offsets],
                  device=self.config.device)
        wp.launch(kernel=invert_and_orient_grasps,
                  dim=(self.num_true_random_grasps, self.config.num_orientations),
                  inputs=[work_transforms, self.num_true_random_grasps, self.config.num_orientations, self.gripper.open_axis, self.gen_seed()],
                  device=self.config.device)
        
        wp.launch(kernel=set_offsets_acronym,
                  dim=(self.num_true_random_grasps, self.config.num_orientations),
                  inputs=[self.num_true_random_grasps, work_offsets],
                  device=self.config.device)

        wp.launch(kernel=get_body_transforms_acronym,
                  dim=self.num_grasps,
                  inputs=[self.gripper.body_transforms[self.gripper.base_idx, :], work_transforms, work_offsets],
                  outputs=[root_transforms],
                  device=self.config.device)
        # the ACRONYM version only checks one opening for collision
        orig_work_offsets = wp.clone(work_offsets)
        if self.config.correct_acronym_approach:
            work_offsets.fill_(self.gripper.open_configuration_offset)

        is_invalid.fill_(GraspState.VALID)
        bodies_to_collision_check = [i for i in range(len(self.gripper.body_meshes))]
        self.check_gripper_body_collisions(object, bodies_to_collision_check, work_transforms, work_offsets, is_invalid, self.num_grasps)

        return GraspGuessData(self.gripper, object, self.num_grasps, orig_work_offsets, work_offsets, root_transforms, is_invalid, None)
    
    def generate_grasps_finger_pad_placement(self, object):
        DEBUG_PRINT_DATA_HERE = False
        if not self.gripper.transform_body_frame == self.gripper.finger_indices[0]:
            self.gripper.set_transform_body_frame(self.gripper.finger_indices[0])
        work_transforms = wp.array(shape=self.num_grasps, dtype=wp.mat44, device=self.config.device)
        work_points = wp.array(shape=self.num_true_random_grasps, dtype=wp.vec3, device=self.config.device)
        work_normals = wp.array(shape=self.num_true_random_grasps, dtype=wp.vec3, device=self.config.device)
        idx_map = wp.array(shape=self.num_grasps, dtype=wp.int32, device=self.config.device)
        is_invalid = wp.array(shape=(max(self.gripper.num_openings, self.config.num_offsets), self.num_grasps), dtype=wp.int32, device=self.config.device)
        is_invalid.fill_(GraspState.VALID)

        num_successes = wp.zeros(shape=1, dtype=wp.int32, device=self.config.device)
        # is_invalid will eventually hold the final Grasp.IN_COLLISION state if there is one in all offsets tested
        # sample random points


        # get random object samples and normals from the object
        self.get_random_mesh_samples(object, work_points, work_normals)

        wp.launch(kernel=compute_transforms_from_random_samples,
                  dim=self.num_true_random_grasps,
                  inputs=[work_points, work_normals, self.gen_seed(), self.config.percent_random_guess_angle, self.config.standoff_distance, self.gripper.open_axis, True],
                  outputs=[work_transforms],
                  device=self.config.device)
        if DEBUG_PRINT_DATA_HERE:
            work_transforms_cpu = work_transforms.numpy()
            print(f"work_transforms_cpu: {work_transforms_cpu}")
        wp.launch(kernel=invert_and_orient_grasps,
                  dim=(self.num_true_random_grasps, self.config.num_orientations),
                  inputs=[work_transforms, self.num_true_random_grasps, self.config.num_orientations, self.gripper.open_axis, self.gen_seed()],
                  device=self.config.device)
        if DEBUG_PRINT_DATA_HERE:
            invert_and_orient_grasps_cpu = work_transforms.numpy()
            print(f"invert_and_orient_grasps_cpu: {invert_and_orient_grasps_cpu}")

        # Test finger0 collision and back it off till no collision
        finger0_mesh = self.gripper.body_meshes[self.gripper.finger_indices[0]]
        finger0_nt = len(finger0_mesh.indices)//3
        wp.launch(kernel=intersect_mesh_along_negative_normal,
                dim=(self.config.num_offsets, self.num_grasps, finger0_nt),
                inputs=[self.config.standoff_distance, self.gripper.open_axis, finger0_mesh.points, finger0_mesh.indices,
                        object.mesh.id, object.mesh.points, object.mesh.indices, work_transforms, is_invalid],
                device=self.config.device)
        if DEBUG_PRINT_DATA_HERE:
            is_invalid_cpu = is_invalid.numpy()
            print(f"is_invalid_cpu: {is_invalid_cpu}")
        wp.launch(kernel=get_closest_offset_transforms_kernel,
                dim=self.num_grasps,
                inputs=[self.config.num_offsets, self.config.standoff_distance, self.gripper.open_axis, is_invalid, work_transforms, num_successes, idx_map],
                device=self.config.device)
        num_finger0_successes = int(num_successes.numpy()[0])
        is_invalid.fill_(GraspState.VALID) # Use this again

        # it looks complex, but it's there if you want to check more objects, like len(self.gripper.body_meshes),
        # That did not make speed better when I did try it.
        for b_idx in [self.gripper.finger_indices[1]]:
            if b_idx == self.gripper.finger_indices[0]:
                continue
            b_mesh = self.gripper.body_meshes[b_idx]
            wp.launch(kernel=intersect_with_offsets,
                      dim=(self.gripper.num_openings, num_finger0_successes, len(b_mesh.indices)//3),
                  inputs=[b_mesh.points, b_mesh.indices,
                          object.mesh.id, object.mesh.points, object.mesh.indices,
                          self.gripper.body_transforms[b_idx, :], work_transforms, idx_map,
                          is_invalid],
                  device=self.config.device)
        
        num_successes.zero_()
        wp.launch(kernel=get_finger1_success_count,
                  dim=(self.gripper.num_openings, num_finger0_successes),
                  inputs=[self.gripper.num_openings, idx_map, is_invalid, num_successes],
                  device=self.config.device)
        num_finger1_successes = int(num_successes.numpy()[0])
        num_successes.zero_()
        finger1_successes_offset = wp.array(shape=num_finger1_successes, dtype=wp.int32, device=self.config.device)
        finger1_successes_work_transform = wp.array(shape=num_finger1_successes, dtype=wp.mat44, device=self.config.device)
        finger1_successes_idx = wp.array(shape=num_finger1_successes, dtype=wp.int32, device=self.config.device)
        wp.launch(kernel=get_finger1_successes,
                  dim=(self.gripper.num_openings, num_finger0_successes),
                  inputs=[self.gripper.num_openings, idx_map, is_invalid, num_successes, work_transforms, finger1_successes_offset, finger1_successes_work_transform, finger1_successes_idx],
                  device=self.config.device)
    
        #self.gripper.save_scene("finger1_successes", finger1_successes_work_transform, finger1_successes_offset, object_mesh=object.mesh)#, only_do_these_bodies = [self.gripper.finger_indices[0]])
        if self.config.do_not_center_finger_opening:
            bodies_to_collision_check = [i for i in range(len(self.gripper.body_meshes)) if i not in self.gripper.finger_indices]
        else:
            bodies_to_collision_check = [i for i in range(len(self.gripper.body_meshes))]
            distances0 = wp.array(shape=num_finger1_successes, dtype=wp.float32, device=self.config.device)
            distances1 = wp.array(shape=num_finger1_successes, dtype=wp.float32, device=self.config.device)
            self.center_finger_opening(object, finger1_successes_work_transform, finger1_successes_offset, distances0, distances1, num_finger1_successes)

        # find the widest opening that has no finger collision

        is_invalid = wp.array(shape=num_finger1_successes, dtype=wp.int32, device=self.config.device)
        is_invalid.fill_(GraspState.VALID)
        widest_offsets = wp.clone(finger1_successes_offset)

        # These can become invalid if centered and fingers are in collision without a way to back out.
        self.find_widest_valid_opening(object, self.config.do_not_center_finger_opening, finger1_successes_work_transform, widest_offsets, is_invalid, num_finger1_successes)

        
        #self.gripper.save_scene("centered", finger1_successes_work_transform, finger1_successes_offset, object_mesh=object.mesh)#, only_do_these_bodies = [self.gripper.finger_indices[0]])


        is_invalid = wp.array(shape=num_finger1_successes, dtype=wp.int32, device=self.config.device)
        is_invalid.fill_(GraspState.VALID)
        self.check_gripper_body_collisions(object, bodies_to_collision_check, finger1_successes_work_transform, widest_offsets, is_invalid, num_finger1_successes)
        
        root_transforms = wp.array(shape=num_finger1_successes, dtype=wp.transform, device=self.config.device)
        wp.launch(kernel=get_body_transforms,
                  dim=num_finger1_successes,
                  inputs=[self.gripper.body_transforms[self.gripper.base_idx, :], finger1_successes_work_transform, finger1_successes_offset],
                  outputs=[root_transforms],
                  device=self.config.device)

        return GraspGuessData(self.gripper, object, num_finger1_successes, finger1_successes_offset, widest_offsets, root_transforms, is_invalid, finger1_successes_idx)

    def debug_set_random_points(self, work_points, work_normals):
        wp_points = wp.array([[0.17238757014274597, 0.16907328367233276, -0.03870725631713867],[-0.0037364927120506763, -0.05887838453054428, -0.02273072488605976]], dtype=wp.vec3, device=self.config.device)
        wp_normals = wp.array([[0.25167912244796753, -0.9648826718330383, -0.07522616535425186], [0.4326925575733185, -0.13429687917232513, -0.8914827108383179]], dtype=wp.vec3, device=self.config.device)
        wp.launch(kernel=copy_vec3,
                  dim=min(len(wp_points), len(work_points)),
                  inputs=[wp_points],
                  outputs=[work_points],
                  device=self.config.device)
        wp.launch(kernel=copy_vec3,
                  dim=min(len(wp_normals), len(work_normals)),
                  inputs=[wp_normals],
                  outputs=[work_normals],
                  device=self.config.device)
        self.seed = 0
        
        print(f"work_points: {work_points}")
        print(f"work_normals: {work_normals}")

def main(args):
    # Main function to run the grasp guess generator
    # Isaac Lab will be started automatically when needed (for USD files or gripper creation)
    # The force_headed setting will be respected when Isaac Lab is actually started
    
    with wp.ScopedTimer("create_guess_generator"):
        guess_generator = GraspGuessGenerator.from_args(args)
    
    with wp.ScopedTimer("create_object"):
        object = GuessObject.from_file(args.object_file, args=args)
    
    with wp.ScopedTimer("generate_grasps"):
        grasp_guess_buffer = guess_generator.generate_grasps(object, args.num_grasps, 0)#args.num_grasps)
        
    with wp.ScopedTimer("create_isaac_grasp_data"):
        file_name_prefix = "cpu" if args.device == "cpu" else ""
        save_to_folder = os.path.join(os.environ.get('GRASP_DATASET_DIR', ''), "grasp_guess_data")
        _, output_file = grasp_guess_buffer.create_isaac_grasp_data(save_successes=True, save_fails=True, save_to_folder=save_to_folder, file_name_prefix=file_name_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp Guessing part of Grasp Gen Data Generation.")
    add_grasp_guess_args(parser, globals(), **collect_grasp_guess_args(globals()))
    args_cli = parser.parse_args()
    
    # Apply gripper configuration if specified
    apply_gripper_configuration(args_cli)
    
    main(args_cli)