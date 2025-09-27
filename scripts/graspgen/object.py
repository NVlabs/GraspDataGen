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

from graspgen_utils import add_arg_to_group, add_isaac_lab_args_if_needed, str_to_bool


default_object_file = "objects/mug.obj"
default_object_scale = 1.0
default_obj2usd_use_existing_usd = True
default_obj2usd_collision_approximation = "convexDecomposition"
default_obj2usd_friction = 1.0


def collect_object_args(input_dict):
    desired_keys = [
        "default_object_file",
        "default_object_scale",
        "default_obj2usd_use_existing_usd",
        "default_obj2usd_collision_approximation",
        "default_obj2usd_friction"]
    kwargs = {}
    for key in desired_keys:
        if key in input_dict:
            kwargs[key] = input_dict[key]
        else:
            # Use local default if not provided in input_dict
            kwargs[key] = globals()[key]
    return kwargs


def add_object_args(parser, param_dict,
                    default_object_file=default_object_file,
                    default_object_scale=default_object_scale,
                    default_obj2usd_use_existing_usd=default_obj2usd_use_existing_usd,
                    default_obj2usd_collision_approximation=default_obj2usd_collision_approximation,
                    default_obj2usd_friction=default_obj2usd_friction):

    # Register the object group since we'll be adding arguments to it
    from graspgen_utils import register_argument_group
    register_argument_group(parser, 'object', 'object', 'Object configuration options')

    add_arg_to_group('object', parser, "--object_file", type=str, default=default_object_file, help="Path to the object file.")

    add_arg_to_group('object', parser, "--object_scale", type=float, default=default_object_scale,  # 0.00625, #1.0, # banana is 0.025
                     help="Scale the input object by this amount.")

    add_arg_to_group('object', parser, "--obj2usd_use_existing_usd", type=str_to_bool, nargs='?', const=True,
                     default=default_obj2usd_use_existing_usd,
                     help="Use the USD file if it exists when the input is obj or stl, otherwise create it.")

    # TODO: Make all collision types work: boundingCube, boundingSphere, meshSimplification (only on static?), triangleMesh (only on static?)
    add_arg_to_group('object', parser, "--obj2usd_collision_approximation", type=str, default=default_obj2usd_collision_approximation,  # default="convexDecomposition",
                     choices=["sdf", "convexDecomposition", "convexHull", "sphereFill", "none"],
                     help="Collision approximation method for the object")

    add_arg_to_group('object', parser, "--obj2usd_friction", type=float, default=default_obj2usd_friction,
                     help="Friction coefficient for both static and dynamic friction of the object physics material.")

    add_isaac_lab_args_if_needed(parser)


class ObjectConfig:
    def __init__(self, object_file=default_object_file, object_scale=default_object_scale,
                 obj2usd_use_existing_usd=default_obj2usd_use_existing_usd,
                 obj2usd_collision_approximation=default_obj2usd_collision_approximation,
                 obj2usd_friction=default_obj2usd_friction):
        self.object_file = object_file
        self.object_scale = object_scale
        self.obj2usd_use_existing_usd = obj2usd_use_existing_usd
        self.obj2usd_collision_approximation = obj2usd_collision_approximation
        self.obj2usd_friction = obj2usd_friction

    @classmethod
    def from_file(cls, file_path, scale, args=None):
        object_file = file_path
        object_scale = default_object_scale
        obj2usd_use_existing_usd = default_obj2usd_use_existing_usd
        obj2usd_collision_approximation = default_obj2usd_collision_approximation
        obj2usd_friction = default_obj2usd_friction
        if args is not None:
            # object_file = args.object_file
            object_scale = args.object_scale
            obj2usd_use_existing_usd = args.obj2usd_use_existing_usd
            obj2usd_collision_approximation = args.obj2usd_collision_approximation
            obj2usd_friction = args.obj2usd_friction
        # Apply the scale uniformly to all object types
        object_scale *= scale
        return cls(object_file=object_file, object_scale=object_scale,
                   obj2usd_use_existing_usd=obj2usd_use_existing_usd,
                   obj2usd_collision_approximation=obj2usd_collision_approximation,
                   obj2usd_friction=obj2usd_friction)

    @classmethod
    def from_isaac_grasp_dict(cls, grasp_dict, grasp_file_args=None):
        # Initialize default values
        object_file = default_object_file
        object_scale = default_object_scale
        obj2usd_use_existing_usd = default_obj2usd_use_existing_usd
        obj2usd_collision_approximation = default_obj2usd_collision_approximation
        obj2usd_friction = default_obj2usd_friction

        # First, apply values from grasp file (lowest priority)
        if 'object_file' in grasp_dict:
            object_file = grasp_dict["object_file"]
        if 'object_scale' in grasp_dict:
            object_scale = grasp_dict["object_scale"]

        if 'obj2usd_use_existing_usd' in grasp_dict:
            obj2usd_use_existing_usd = grasp_dict["obj2usd_use_existing_usd"]
        if 'obj2usd_collision_approximation' in grasp_dict:
            obj2usd_collision_approximation = grasp_dict["obj2usd_collision_approximation"]
        if 'obj2usd_friction' in grasp_dict:
            obj2usd_friction = grasp_dict["obj2usd_friction"]

        # Then, override with command line arguments (highest priority)
        if grasp_file_args is not None:
            object_file = grasp_file_args.object_file
            object_scale = grasp_file_args.object_scale
            obj2usd_use_existing_usd = grasp_file_args.obj2usd_use_existing_usd
            obj2usd_collision_approximation = grasp_file_args.obj2usd_collision_approximation
            obj2usd_friction = grasp_file_args.obj2usd_friction

        return cls(object_file=object_file, object_scale=object_scale,
                   obj2usd_use_existing_usd=obj2usd_use_existing_usd,
                   obj2usd_collision_approximation=obj2usd_collision_approximation,
                   obj2usd_friction=obj2usd_friction)
