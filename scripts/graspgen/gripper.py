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

import warp as wp
import numpy as np
import os
import json
from warp_kernels import transform_to_mat44_kernel2d, reframe_to_new_body
from graspgen_utils import print_yellow, add_arg_to_group, open_configuration_string_to_dict, add_isaac_lab_args_if_needed, get_simulation_app, print_blue
from grasp_constants import DEFAULT_MEASURE_CONVERGENCE, DEFAULT_CONVERGENCE_ITERATIONS

# Import gripper configuration functionality
from gripper_configurations import get_gripper_config, apply_gripper_config_to_args

default_gripper_file = "bots/onrobot_rg6.usd"
default_gripper_config = ""
default_finger_colliders = ["right_inner_finger", "left_inner_finger"]
default_base_frame = "base_frame"
default_open_configuration = '{}'
default_pinch_width_resolution = 8
default_bite = 0.01

def collect_gripper_args(input_dict):
    desired_keys = [
        "default_gripper_config",
        "default_gripper_file",
        "default_finger_colliders",
        "default_base_frame",
        "default_open_configuration",
        "default_pinch_width_resolution",
        "default_bite"]
    kwargs = {}
    for key in desired_keys:
        if key in input_dict:
            kwargs[key] = input_dict[key]
        else:
            # Use local default if not provided in input_dict
            kwargs[key] = globals()[key]
    return kwargs

def add_gripper_args(parser, param_dict,
                     default_gripper_config=default_gripper_config,
                     default_gripper_file=default_gripper_file,
                     default_finger_colliders=default_finger_colliders,
                     default_base_frame=default_base_frame,
                     default_open_configuration=default_open_configuration,
                     default_pinch_width_resolution=default_pinch_width_resolution,
                     default_bite=default_bite):
    
    # Register the gripper group since we'll be adding arguments to it
    from graspgen_utils import register_argument_group, add_create_gripper_args, collect_create_gripper_args
    register_argument_group(parser, 'gripper', 'gripper', 'Gripper configuration options')
    
    add_create_gripper_args(parser, param_dict, **collect_create_gripper_args(param_dict))

    
    # Add gripper configuration argument
    add_arg_to_group('gripper', parser, "--gripper_config", type=str, default=default_gripper_config,
                     help="Gripper configuration to use. This will automatically set gripper_file, "
                          "finger_colliders, base_frame, and any other overrides defined in the configuration.")
    
    add_arg_to_group('gripper', parser, "--gripper_file", type=str, default=default_gripper_file, help="Path to the gripper file.")
    add_arg_to_group('gripper', parser, "--finger_colliders", type=str, nargs='+',  # TODO: make this default to a property on the gripper
                     default=default_finger_colliders,
                     help="Paths to finger colliders in the gripper USD file")
    add_arg_to_group('gripper', parser, "--base_frame", type=str, default=default_base_frame,  # TODO: make this default to a property on the gripper
                     help="Name of the base frame origin in the gripper")
    add_arg_to_group('gripper', parser, "--bite", type=float, default=default_bite,
                     help="The depth of the bite from the fingertip. It will be used to position finger for grasp generation.")
    add_arg_to_group('gripper', parser, "--pinch_width_resolution", type=int, default=default_pinch_width_resolution, 
                     help="Number of pinch opening widths to test.")
    add_arg_to_group('gripper', parser, "--open_configuration", type=str, default=default_open_configuration,
                     help="Initial state of the original grasp guess.")
    
    add_isaac_lab_args_if_needed(parser)

def apply_gripper_configuration(args):
    """
    Apply gripper configuration if specified in args.
    
    Args:
        args: The parsed arguments object
        
    Returns:
        bool: True if configuration was applied, False otherwise
    """
    if hasattr(args, 'gripper_config') and args.gripper_config:
        print_blue(f"Using gripper config '{args.gripper_config}'")
        try:
            gripper_config = get_gripper_config(args.gripper_config)
            apply_gripper_config_to_args(args, gripper_config)
            print_blue(f"Gripper file: {args.gripper_file}")
            return True
        except ValueError as e:
            print_yellow(f"Warning: {e}")
            return False
    return False

class GripperConfig:
    """Configuration class for gripper creation and loading."""
    
    def __init__(self, gripper_file, finger_colliders, base_frame, bite,
                 pinch_width_resolution, open_configuration, device):
        self.gripper_file = gripper_file
        self.finger_colliders = finger_colliders
        self.base_frame = base_frame
        self.bite = bite
        self.pinch_width_resolution = pinch_width_resolution
        if isinstance(open_configuration, str):
            open_configuration = open_configuration_string_to_dict(open_configuration)
        self.open_configuration = open_configuration
        self.device = device
    
    def to_dict(self):
        return {
            "gripper_file": self.gripper_file,
            "finger_colliders": self.finger_colliders,
            "base_frame": self.base_frame,
            "bite": self.bite,
            "pinch_width_resolution": self.pinch_width_resolution,
            "open_configuration": self.open_configuration,
            "device": self.device,
        }

def create_gripper(gripper_config, headless=True, force_headed=False,
                   wait_for_debugger_attach=False, save_gripper=True,
                   measure_convergence=DEFAULT_MEASURE_CONVERGENCE,
                   convergence_iterations=DEFAULT_CONVERGENCE_ITERATIONS):
    npz_path = os.path.splitext(gripper_config.gripper_file)[0] + '.npz'
    gripper = Gripper.load(gripper_config)
    if gripper is not None:
        return gripper

    # if the gripper file exists, remove it, because it can't be loaded and will be replaced
    if os.path.exists(npz_path):
        os.remove(npz_path)

    # need to do simulation app here with the passed in args, or create_gripper_lab will use args_cli for it.
    get_simulation_app(__file__, force_headed=force_headed, wait_for_debugger_attach=wait_for_debugger_attach)
    from create_gripper_lab import create_gripper_with_lab
    gripper = create_gripper_with_lab(gripper_config, save_gripper, measure_convergence, convergence_iterations, wait_for_debugger_attach=wait_for_debugger_attach)
    if gripper is not None:
        return gripper

    print(f"Failed to create gripper: {npz_path}")
    return None


class Gripper:
    def __init__(self, config, gripper_data=None):
        self.config = config
        if gripper_data is not None:
            num_bodies = gripper_data["body_transforms"].shape[0]
            self.num_openings = gripper_data["num_openings"]
            self.open_configuration_offset = gripper_data["open_configuration_offset"]
            self.bite_point = [gripper_data["bite_point"][0], gripper_data["bite_point"][1], gripper_data["bite_point"][2]]
            self.bite_points = wp.array(
                gripper_data["bite_points"], dtype=wp.vec3,
                device=self.config.device)
            self.open_widths = wp.array(
                gripper_data["open_widths"], dtype=wp.float32,
                device=self.config.device)
            self.open_widths_reverse = wp.array(
                gripper_data["open_widths"][::-1], dtype=wp.float32,
                device=self.config.device)
            self.open_limit = gripper_data["open_limit"]
            self.body_names = gripper_data["body_names"]
            self.joint_names = gripper_data["joint_names"]
            self.driven_joints = gripper_data["driven_joints"]
            self.joint_cspace_pos = wp.array(gripper_data["joints"]["cspace_positions"], dtype=wp.float32, device=self.config.device)  # len(joint_names) x num_openings
            self.approach_axis = gripper_data["approach_axis"]
            self.open_axis = gripper_data["open_axis"]
            self.base_idx = gripper_data["bodies"][self.config.base_frame]
            self.base_length = gripper_data["base_length"]
            self.finger_indices = gripper_data["finger_indices"]
            self.body_transforms = wp.array(shape=(num_bodies, self.num_openings), dtype=wp.mat44, device=self.config.device)
            wp.launch(kernel=transform_to_mat44_kernel2d,
                      dim=(num_bodies, self.num_openings),
                      inputs=[gripper_data["body_transforms"]],
                      outputs=[self.body_transforms],
                      device=self.config.device)
            self.transform_body_frame = gripper_data["transform_body_frame"]  # This is the body id that the transforms are in the reference frame of.
            self.body_meshes = []
            for b_idx in range(num_bodies):
                self.body_meshes.append(wp.Mesh(gripper_data["bodies"][b_idx]["collision_mesh"]["vertices"], gripper_data["bodies"][b_idx]["collision_mesh"]["indices"]))

    @classmethod
    def from_args(cls, args):
        gripper_config = GripperConfig(args.gripper_file, args.finger_colliders, args.base_frame, args.bite, args.pinch_width_resolution, args.open_configuration, args.device)
        measure_convergence = getattr(args, 'measure_convergence', DEFAULT_MEASURE_CONVERGENCE)
        convergence_iterations = getattr(args, 'convergence_iterations', DEFAULT_CONVERGENCE_ITERATIONS)
        return create_gripper(
            gripper_config, 
            headless=False if args.force_headed else args.headless, 
            force_headed=args.force_headed, 
            wait_for_debugger_attach=getattr(args, 'wait_for_debugger_attach', False),
            save_gripper=True if 'save_gripper' not in args else args.save_gripper, 
            measure_convergence=measure_convergence, 
            convergence_iterations=convergence_iterations
        )


    @classmethod
    def load(cls, config, skip_config_validation=False):
        """Load gripper state from disk.
        
        Args:
            config (GripperConfig): Configuration object containing necessary parameters
            skip_config_validation (bool): If True, skip config validation and load data with indifference
            
        Returns:
            Gripper: A new Gripper instance with loaded state, or None if .npz file doesn't exist or args don't match
        """
        # Convert USD/USDA path to .npz path
        npz_path = os.path.splitext(config.gripper_file)[0] + '.npz'
        
        # Return None if .npz file doesn't exist
        if not os.path.exists(npz_path):
            return None
            
        # Load data from file
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading NPZ file {npz_path}: {e}")
            return None

        # Check if all required keys are present
        required_keys = [
            'open_limit', 'open_configuration_offset', 'num_openings', 'open_widths', 'bite_points', 'joint_names', 'driven_joints', 'joint_cspace_pos',
            'body_names', 'base_idx', 'base_length', 'finger_indices',
            'body_transforms', 'body_vertices', 'body_indices',
            'open_axis', 'approach_axis', 'transform_body_frame', 'config', 'bite_point'
        ]
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"NPZ file {npz_path} is missing required keys: {missing_keys}")
            return None

        # Check if saved args match input args (skip if requested)
        if not skip_config_validation:
            try:
                saved_config = data['config'].item()
                if (saved_config['gripper_file'] != config.gripper_file or
                    saved_config['finger_colliders'] != config.finger_colliders or
                    saved_config['base_frame'] != config.base_frame or
                    saved_config['bite'] != config.bite or
                    saved_config['pinch_width_resolution'] != config.pinch_width_resolution or
                    saved_config['open_configuration'] != config.open_configuration):# or
                    # maybe check what it was saved with? saved_config['device'][:3] != config.device[:3]):
                    print_yellow(f"Saved config doesn't match input config.")
                    print_yellow(f"Saved config: {saved_config}")
                    print_yellow(f"Input config: {config.to_dict()}")
                    return None
            except Exception as e:
                print(f"Error validating arguments in NPZ file {npz_path}: {e}")
                return None
        
        # Create new gripper instance
        gripper = cls(config)
        
        try:
            # Restore state
            # remember to move the data that is to be used by warp to a warp array!
            gripper.open_limit = data['open_limit'].item()
            gripper.num_openings = data['num_openings'].item()
            gripper.open_configuration_offset = data['open_configuration_offset'].item()
            # Handle bite_point which might be an array
            bite_point_data = data['bite_point']
            if bite_point_data.shape == ():
                gripper.bite_point = bite_point_data.item()
            else:
                gripper.bite_point = bite_point_data.tolist()
            gripper.bite_points = wp.array(data['bite_points'], dtype=wp.vec3, device=config.device)
            gripper.open_widths = wp.array(data['open_widths'], dtype=wp.float32, device=config.device)
            # this open_widths_reverse business is a hack so we can run lower_bounds, and it would be better to refactor and make 
            # 0 closed and -1 open.
            gripper.open_widths_reverse = wp.array(data['open_widths'][::-1], dtype=wp.float32, device=config.device)
            gripper.body_names = data['body_names'].tolist()
            gripper.joint_names = data['joint_names'].tolist()
            gripper.driven_joints = data['driven_joints'].tolist()
            gripper.joint_cspace_pos = wp.array(data['joint_cspace_pos'], dtype=wp.float32, device=config.device)
            gripper.approach_axis = data['approach_axis'].item()
            gripper.open_axis = data['open_axis'].item()
            gripper.base_idx = data['base_idx'].item()
            gripper.base_length = data['base_length'].item()
            gripper.finger_indices = data['finger_indices'].tolist()
            gripper.body_transforms = wp.array(data['body_transforms'], dtype=wp.mat44, device=config.device)
            # Handle transform_body_frame which might be None or an object
            transform_data = data['transform_body_frame']
            if transform_data.shape == ():
                gripper.transform_body_frame = transform_data.item()
            else:
                gripper.transform_body_frame = None
            bv = data['body_vertices'].tolist()
            bvl = []
            for bidx in bv:
                _bv = wp.array(bv[bidx], dtype=wp.vec3, device=config.device)
                bvl.append(_bv)
            bil = []
            bi = data['body_indices'].tolist()
            for bidx in bi:
                _bi = wp.array(bi[bidx], dtype=wp.int32, device=config.device)
                bil.append(_bi)
            gripper.body_meshes = [wp.Mesh(bv, bi) for bv, bi in zip(bvl, bil)]
        except Exception as e:
            print(f"Error restoring gripper state from NPZ file {npz_path}: {e}")
            return None

        print(f"Loaded gripper state from {npz_path}")
        return gripper
    
    def save(self, filepath):
        """Save gripper state to disk.
        
        Args:
            filepath (str): Path to save the gripper state to
        """
        
        # Create directory if it doesn't exist
        filepath = os.path.splitext(filepath)[0] + '.npz'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert Warp arrays to numpy for saving
        data = {
            'config': {
                'gripper_file': self.config.gripper_file,
                'finger_colliders': self.config.finger_colliders,
                'base_frame': self.config.base_frame,
                'bite': self.config.bite,
                'pinch_width_resolution': self.config.pinch_width_resolution,
                'open_configuration': self.config.open_configuration,
                'device': self.config.device,
            },

            'open_limit': self.open_limit,
            'num_openings': self.num_openings,
            'open_configuration_offset': self.open_configuration_offset,
            'bite_point': self.bite_point,
            'bite_points': self.bite_points.numpy(),
            'open_widths': self.open_widths.numpy(),
            'body_names': self.body_names,
            'base_idx': self.base_idx,
            'base_length': self.base_length,
            'finger_indices': self.finger_indices,
            'joint_names': self.joint_names,
            'driven_joints': self.driven_joints,
            'joint_cspace_pos': self.joint_cspace_pos.numpy(),
            'body_transforms': self.body_transforms.numpy(),
            'body_vertices': {bidx: body.points.numpy() for bidx, body in enumerate(self.body_meshes)},
            'body_indices': {bidx: body.indices.numpy() for bidx, body in enumerate(self.body_meshes)},
            'approach_axis': self.approach_axis,
            'open_axis': self.open_axis,
            'transform_body_frame': self.transform_body_frame,
        }
        
        np.savez(filepath, **data)
        print(f"Saved gripper state to {filepath}")

    def get_npz_path(self, filename, config):

        filepath = os.path.splitext(filename)[0]
        if config.device == "cpu":
            filepath += ".cpu.npz"
        else:
            filepath += ".npz"
        return filepath

    def set_transform_body_frame(self, ref_idx):
        self.transform_body_frame = ref_idx
        ref_transform = self.body_transforms[ref_idx, :]
        # get the reference frame and get the inverse transforms to multiply the others by.
        wp.launch(kernel=reframe_to_new_body,
                  dim=(len(self.body_transforms) - 1, len(ref_transform)),
                  inputs=[ref_idx, self.body_transforms],
                  device=self.config.device)
        ref_transform.fill_(wp.mat44(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    
    def save_scene(self, folder_name, grasp_transforms, offsets=0, object_mesh=None, only_do_these_bodies=None, blips=None):
        """Save scene data for debugging purposes.
        
        Args:
            folder_name: Name of the output folder
            grasp_transforms: Array of grasp transforms
            offsets: Offset indices for grasps (default: 0)
            object_mesh: Optional object mesh to include
            only_do_these_bodies: Optional list of body indices to process
            blips: Optional dictionary of body blip positions
        """
        output_dir = f"debug_output/{folder_name}/"
        # Remove directory if it exists and create a new empty one
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(grasp_transforms, wp.array):
            grasp_transforms = grasp_transforms.numpy()

        if isinstance(offsets, int):
            offsets = [offsets] * len(grasp_transforms)
        elif isinstance(offsets, np.ndarray):
            offsets = offsets.tolist()
        elif isinstance(offsets, wp.array):
            offsets = offsets.numpy().tolist()
        
        if len(offsets) != len(grasp_transforms):
            raise ValueError("offsets must be an int or a list of the same length as grasp_transforms")

        def write_obj_flat_tris(file_path, verts, tris, b_idx=0, blips=None):
            with open(file_path, "w") as f:
                for v in verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                if blips is not None and b_idx in blips:
                    offset = 0.001
                    f.write(f"v {blips[b_idx][0]} {blips[b_idx][1]-offset} {blips[b_idx][2]}\n")
                    f.write(f"v {blips[b_idx][0]-offset} {blips[b_idx][1]+offset} {blips[b_idx][2]-offset}\n")
                    f.write(f"v {blips[b_idx][0]+offset} {blips[b_idx][1]+offset} {blips[b_idx][2]-offset}\n")
                    f.write(f"v {blips[b_idx][0]} {blips[b_idx][1]+offset} {blips[b_idx][2]+offset}\n")
                for t in range(0, len(tris), 3):
                    f.write(f"f {tris[t]+1} {tris[t+1]+1} {tris[t+2]+1}\n")
                if blips is not None and b_idx in blips:
                    f.write(f"f {len(verts)+4} {len(verts)+2} {len(verts)+1}\n")
                    f.write(f"f {len(verts)+4} {len(verts)+3} {len(verts)+2}\n")
                    f.write(f"f {len(verts)+3} {len(verts)+4} {len(verts)+1}\n")
                    f.write(f"f {len(verts)+2} {len(verts)+3} {len(verts)+1}\n")

        if object_mesh is not None:
            name = f"Object"
            obj_file = f"{output_dir}/{name}.obj"
            verts = object_mesh.points.numpy().tolist()
            tris = object_mesh.indices.numpy().tolist()
            write_obj_flat_tris(obj_file, verts, tris)
        body_transforms = self.body_transforms.numpy()
        for b_idx, b_name in enumerate(self.body_names):
            if only_do_these_bodies is not None and b_idx not in only_do_these_bodies:
                continue
            obj_file = f"{output_dir}/{b_name}.obj"
            json_file = f"{output_dir}/{b_name}.json"
            verts = self.body_meshes[b_idx].points.numpy().tolist()
            tris = self.body_meshes[b_idx].indices.numpy().tolist()
            write_obj_flat_tris(obj_file, verts, tris, b_idx, blips)

            with open(json_file, "w") as f:
                out_xforms = []
                for gx_idx, gxform in enumerate(grasp_transforms):
                    env_idx = offsets[gx_idx]
                    bxform = body_transforms[b_idx, env_idx]
                    gxform = wp.mat44(gxform)
                    t = gxform @ wp.mat44(bxform)
                    mat_xform = [[t[0, 0], t[0, 1], t[0, 2], t[0, 3]],
                                 [t[1, 0], t[1, 1], t[1, 2], t[1, 3]],
                                 [t[2, 0], t[2, 1], t[2, 2], t[2, 3]],
                                 [t[3, 0], t[3, 1], t[3, 2], t[3, 3]]]
                    out_xforms.append(mat_xform)

                json.dump(out_xforms, f)

        print("saved scene")
