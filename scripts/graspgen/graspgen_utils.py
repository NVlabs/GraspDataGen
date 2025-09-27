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

"""Utility functions for grasp generation."""

import argparse
import os
import yaml
from grasp_constants import DEFAULT_MEASURE_CONVERGENCE, DEFAULT_CONVERGENCE_ITERATIONS


def reset_gpu_context():
    # GPU Context Reset to avoid interference from other Isaac Sim versions
    import os
    import torch
    try:
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force CUDA context reset
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        print("GPU context reset completed")
    except Exception as e:
        print(f"Warning: Could not fully reset GPU context: {e}")


def str_to_bool(v):
    """Parse a string to a boolean value.
    
    Args:
        v: String value to parse
        
    Returns:
        bool: Parsed boolean value
        
    Raises:
        argparse.ArgumentTypeError: If the string cannot be parsed
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


# Global argument group registry for organized help output
_arg_groups = {}


def set_argument_groups(groups_dict):
    """Set the global argument groups for organized help output.
    
    Args:
        groups_dict: Dictionary mapping group names to ArgumentGroup objects
    """
    global _arg_groups
    _arg_groups = groups_dict

def get_argument_group(group_name, fallback_parser=None):
    """Get an argument group by name, with fallback to main parser.
    
    Args:
        group_name: Name of the group to get
        fallback_parser: Parser to use if group doesn't exist
        
    Returns:
        The argument group or fallback parser
    """
    global _arg_groups
    return _arg_groups.get(group_name, fallback_parser)


def register_argument_group(parser, group_name, title, description=None):
    """Register a new argument group with the global registry.
    
    Args:
        parser: The argparse parser to add the group to
        group_name: Name of the group (used as key in registry)
        title: Title for the argument group
        description: Optional description for the argument group
        
    Returns:
        The created ArgumentGroup object
    """
    global _arg_groups
    
    # Create the argument group
    if group_name not in _arg_groups:
        group = parser.add_argument_group(title, description)
        # Register it globally
        _arg_groups[group_name] = group
    else:
        group = _arg_groups[group_name]
    
    return group


def add_arg_if_not_exists(parser, *args, **kwargs):
    """Add argument only if it doesn't already exist"""
    try:
        parser.add_argument(*args, **kwargs)
        return True
    except argparse.ArgumentError:
        # Argument already exists
        return False


def add_arg_to_group(group_name, fallback_parser, *args, **kwargs):
    """Add an argument to a specific group, creating the group if it doesn't exist.
    
    Args:
        group_name: Name of the group to add the argument to
        fallback_parser: Parser to use if group doesn't exist
        *args: Arguments to pass to add_argument
        **kwargs: Keyword arguments to pass to add_argument
    """
    global _arg_groups
    
    # Get the group, creating it if it doesn't exist
    group = _arg_groups.get(group_name)
    if group is None:
        # Create a default group with the group_name as title
        group = fallback_parser.add_argument_group(group_name)
        _arg_groups[group_name] = group
    
    # Add the argument to the group, but only if it doesn't already exist
    return add_arg_if_not_exists(group, *args, **kwargs)


def add_isaac_lab_args_if_needed(parser):
    # Register the isaac group since we'll be adding arguments to it
    register_argument_group(parser, 'grasp_data_gen', 'grasp_data_gen', 'GraspDataGen configuration options')
    
    # Only add force_headed if it doesn't already exist
    if "--force_headed" not in parser._option_string_actions:
        add_arg_to_group('grasp_data_gen', parser, "--force_headed", action="store_true", default=False,
                         help="Force Isaac Lab to run in headed mode.")
    # Only add wait_for_debugger_attach if it doesn't already exist
    if "--wait_for_debugger_attach" not in parser._option_string_actions:
        add_arg_to_group('grasp_data_gen', parser, "--wait_for_debugger_attach", action="store_true", default=False,
                         help="Wait for a debugger to attach before continuing execution.")
    if "--device" not in parser._option_string_actions:
        # Import AppLauncher only when needed
        try:
            from isaaclab.app import AppLauncher
            # Pass the main parser to AppLauncher, not the argument group
            AppLauncher.add_app_launcher_args(parser)
            # Set headless=True as default, but force_headed will override this
            parser.set_defaults(headless=True)
        except ImportError:
            # AppLauncher not available, skip adding args
            pass

def open_configuration_string_to_dict(input_string):
    if input_string is None:
        return {}
    try:
        import json
        initial_state = json.loads(input_string)
        if not isinstance(initial_state, dict):
            print_yellow("Warning: --open_configuration must be a JSON "
                         "object/dictionary. Example: "
                         "'{\"joint1\": 0.5, \"joint2\": 1.0}'. "
                         f"The open_configuration, {input_string}, "
                         "will be ignored.")
            initial_state = {}
        else:
            for joint_name, angle in initial_state.items():
                if not isinstance(joint_name, str):
                    print_yellow("Warning: Joint names in "
                                 "--open_configuration must be strings. "
                                 f"Invalid joint name: {joint_name}. "
                                 "The open_configuration will be ignored.")
                    initial_state = {}
                    break
                if not isinstance(angle, (int, float)):
                    print_yellow("Warning: Joint angles in "
                                 "--open_configuration must be numbers. "
                                 f"Invalid angle for joint {joint_name}: "
                                 f"{angle}. The open_configuration will be "
                                 "ignored.")
                    initial_state = {}
                    break
    except json.JSONDecodeError:
        print_yellow("Warning: --open_configuration must be a valid JSON "
                     "string containing a dictionary of joint names and "
                     "angles. Example: '{\"joint1\": 0.5, \"joint2\": 1.0}'. "
                     f"The open_configuration, {input_string}, will be "
                     "ignored.")
        initial_state = {}
    return initial_state


def print_blue(*args, **kwargs):
    print_color(94, *args, **kwargs)


def print_yellow(*args, **kwargs):
    print_color(93, *args, **kwargs)


def print_red(*args, **kwargs):
    print_color(91, *args, **kwargs)


def print_green(*args, **kwargs):
    print_color(92, *args, **kwargs)


def print_purple(*args, **kwargs):
    print_color(95, *args, **kwargs)


def print_color(code=0, *args, **kwargs):
    """Print text in colored format and reset to default color.
    
    This function accepts all the same arguments as the built-in print function.
    
    Args:
        code: ANSI color code (0=default, 91=red, 92=green, 93=yellow, 94=blue, 95=purple)
        *args: All positional arguments passed to print
        **kwargs: All keyword arguments passed to print
    """
    # Extract the 'end' parameter if provided, otherwise use default
    end = kwargs.get('end', '\n')
    
    # Convert all args to strings and join them with the separator
    sep = kwargs.get('sep', ' ')
    text = sep.join(str(arg) for arg in args)
    
    # Apply purple color formatting
    colored_text = f"\033[{code}m{text}\033[0m"
    
    # Update kwargs to use our colored text
    kwargs['end'] = end
    
    # Call print with the colored text, handling flush parameter correctly
    print_kwargs = {k: v for k, v in kwargs.items() if k not in ['sep']}
    if 'flush' not in print_kwargs:
        print_kwargs['flush'] = True
    print(colored_text, **print_kwargs)

def save_yaml(grasps, output_file):
    """Save grasps to a YAML file.
    
    Args:
        grasps: Dictionary of grasps to save
        output_file (str): Path to save the grasps to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print_blue(f"Saving grasps to {output_file}...")
    with open(output_file, 'w') as f:
        yaml.dump(grasps, f, default_flow_style=False, sort_keys=False)


lab_starter = None
class LabStarter:
    def __init__(self, file_name, headless=True, wait_for_debugger_attach=False):
        """Initialize the LabStarter.
        
        Args:
            file_name (str): Name of the file that started the lab
            headless (bool): Whether to run in headless mode
            wait_for_debugger_attach (bool): Whether to wait for debugger attachment
                                            before continuing execution
        """
        self.headless = headless
        self.wait_for_debugger_attach = wait_for_debugger_attach
        self.initiating_file = file_name  # Store the full path
        self.headless_string = "headless" if self.headless else "interactive"
        self.simulation_app = None
        self._initialized = False

    @property
    def file_name(self):
        return os.path.basename(self.initiating_file) if self.initiating_file else None

    def get_simulation_app(self):
        """Get the simulation app, initializing it if needed.
        
        Returns:
            The simulation app instance or None if initialization failed
        """
        if not self._initialized:
            try:
                from isaaclab.app import AppLauncher
                app_launcher = AppLauncher(headless=self.headless)
                self.simulation_app = app_launcher.app
                self._initialized = True
                
                # Handle debugger attachment if requested (after app is created)
                if self.wait_for_debugger_attach:
                    print_yellow("Waiting for debugger to attach...")
                    print_yellow("You can now attach your debugger to this process.")
                    print_yellow("Press Enter when ready to continue...")
                    input()
                    print_green("continuing execution...")
                
                print("\033[0m", end="")
                return self.simulation_app
            except Exception as e:
                print_red(f"Failed to initialize Isaac Lab: {str(e)}")
                print("\033[0m", end="")
                return None
        return self.simulation_app

def start_isaac_lab_if_needed(file_name, headless=True, wait_for_debugger_attach=False):
    """Start Isaac Lab if it is not already running.
    
    Args:
        file_name (str): Name of the file that started the lab
        headless (bool): Whether to run in headless mode
        wait_for_debugger_attach (bool): Whether to wait for debugger attachment
                                        before continuing execution
        
    Returns:
        The simulation app instance or None if initialization failed
    """
    global lab_starter
    
    if lab_starter is None:
        lab_starter = LabStarter(file_name=file_name, headless=headless, wait_for_debugger_attach=wait_for_debugger_attach)
        return lab_starter.get_simulation_app()
    else:
        return lab_starter.get_simulation_app()

def get_simulation_app(file_name, force_headed=False, wait_for_debugger_attach=False):
    """Get the current simulation app instance, initializing it if needed.
    
    Args:
        file_name (str): Name of the file calling this function (e.g., __file__)
        force_headed (bool): Whether to force headed mode (overrides headless 
                            default)
        wait_for_debugger_attach (bool): Whether to wait for debugger attachment
                                        before continuing execution
        
    Returns:
        The simulation app instance or None if initialization failed
    """
    global lab_starter
    
    if lab_starter is None:
        # Use force_headed to determine headless setting
        headless = not force_headed
        lab_starter = LabStarter(file_name=file_name, headless=headless, wait_for_debugger_attach=wait_for_debugger_attach)
        return lab_starter.get_simulation_app()
    else:
        # Check if the requested headless setting matches the current one
        requested_headless = not force_headed
        if lab_starter.headless != requested_headless:
            print_yellow(f"Warning: Isaac Lab already started with "
                         f"headless={lab_starter.headless}, "
                         f"but requested headless={requested_headless}. "
                         f"Using existing configuration.")
        # If lab_starter exists but no initiating file was recorded, update it
        if lab_starter.initiating_file is None:
            lab_starter.initiating_file = file_name
        return lab_starter.get_simulation_app()


def get_lab_starter_info():
    """Get information about the lab starter.
    
    Returns:
        dict: Information about the lab starter, or None if not initialized
    """
    global lab_starter
    if lab_starter is None:
        return None
    return {
        'initiating_file': lab_starter.initiating_file,
        'file_name': lab_starter.file_name,
        'headless': lab_starter.headless,
        'initialized': lab_starter._initialized
    }


def predict_grasp_data_filepath(gripper_name, object_file, save_to_folder,
                                file_name_prefix="", file_extension_prefix=""):
    """
    Predict the file path that would be generated by create_isaac_grasp_data 
    functions without actually writing any data.
    
    Args:
        gripper_file (str): Path to the gripper file
        object_file (str): Path to the object file
        save_to_folder (str): Base folder to save to
        file_name_prefix (str): Optional prefix for the filename
        
    Returns:
        str: The predicted file path, or None if save_to_folder is None
    """
    if save_to_folder is None:
        return None
        
    object_name = os.path.splitext(os.path.basename(object_file))[0]
    output_folder = save_to_folder
    if gripper_name:
        output_folder = os.path.join(output_folder, gripper_name)
    extension = f"{file_extension_prefix}.yaml" if file_extension_prefix else "yaml"
    file_name = f"{object_name}.{extension}"
    if file_name_prefix != "":
        file_name = f"{file_name_prefix}.{file_name}"
    return os.path.join(output_folder, file_name)


def grasp_data_exists(gripper_name, object_file, save_to_folder,
                     file_name_prefix="", file_extension_prefix=""):
    """
    Check if grasp data file already exists for the given parameters.
    
    Args:
        gripper_file (str): Path to the gripper file
        object_file (str): Path to the object file
        save_to_folder (str): Base folder to save to
        file_name_prefix (str): Optional prefix for the filename
        
    Returns:
        filename if exists, and "" otherwise
    """
    predicted_file = predict_grasp_data_filepath(
        gripper_name, object_file, save_to_folder, file_name_prefix, file_extension_prefix
    )
    if predicted_file is not None and os.path.exists(predicted_file):
        return predicted_file
    return ""


def collect_create_gripper_args(input_dict):
    """Collect convergence-related arguments for gripper creation."""
    desired_keys = [
        "default_measure_convergence",
        "default_convergence_iterations",
        ]
    kwargs = {}
    for key in desired_keys:
        if key in input_dict:
            kwargs[key] = input_dict[key]
    return kwargs

def add_create_gripper_args(parser, param_dict,
                           default_measure_convergence=DEFAULT_MEASURE_CONVERGENCE,
                           default_convergence_iterations=DEFAULT_CONVERGENCE_ITERATIONS):
    
    from graspgen_utils import register_argument_group, add_arg_to_group, add_isaac_lab_args_if_needed
    register_argument_group(parser, 'create_gripper', 'create_gripper', 'Create gripper options')
    """Add convergence-related arguments to the parser."""
    add_arg_to_group('create_gripper', parser, "--measure_convergence", action="store_true",
                     default=default_measure_convergence,
                     help="Measure convergence of the gripper.")
    add_arg_to_group('create_gripper', parser, "--convergence_iterations", type=int,
                     default=default_convergence_iterations,
                     help="Number of simulation iterations for convergence (only used when running standalone).")
    
    add_isaac_lab_args_if_needed(parser)

