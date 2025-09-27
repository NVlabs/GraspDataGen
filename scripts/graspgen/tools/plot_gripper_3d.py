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
"""
3D visualization of gripper body positions from Isaac Lab simulation.
This script plots gripper.data.body_com_pos_w in 3D space.
"""

import sys
import os

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent directory to path to import modules from graspgen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper import GripperConfig, add_gripper_args, collect_gripper_args, apply_gripper_configuration
from graspgen_utils import print_red, add_arg_to_group, start_isaac_lab_if_needed, add_isaac_lab_args_if_needed, open_configuration_string_to_dict


# Default values for gripper arguments (needed for collect_gripper_args)
default_gripper_file = "bots/onrobot_rg6.original.flat.usd"
default_finger_colliders = ["right_inner_finger", "left_inner_finger"]
default_base_frame = "base_frame"
default_open_configuration = '{}'
default_pinch_width_resolution = 8
default_bite = 0.01

# Default values for convergence arguments (needed for collect_create_gripper_args)
default_measure_convergence = True
default_convergence_iterations = 20


def add_plot_args(parser, param_dict):
    from graspgen_utils import register_argument_group
    register_argument_group(parser, 'plot_gripper_3d', 'plot_gripper_3d', 'Plot gripper 3D options')
    
    add_gripper_args(parser, param_dict, **collect_gripper_args(param_dict))

    add_arg_to_group('plot_gripper_3d', parser, '--save_plot', action='store_true',
                        help='Save plot to file instead of displaying')
    add_arg_to_group('plot_gripper_3d', parser, '--plot_file', type=str, 
                        default='gripper_3d_plot.png',
                        help='Output filename for saved plot')
    add_arg_to_group('plot_gripper_3d', parser, '--show_trajectories', action='store_true',
                        help='Show trajectories connecting body positions across configurations.')
    add_arg_to_group('plot_gripper_3d', parser, '--highlight_fingers', action='store_true',
                        help='Highlight finger bodies with different colors.')
    add_isaac_lab_args_if_needed(parser)


def plot_gripper_3d(gripper_data, body_names, save_plot=False,
                    plot_file='gripper_3d_plot.png', show_trajectories=False,
                    highlight_fingers=False, finger_indices=None):
    """
    Plot gripper body positions in 3D
    
    Args:
        gripper_data: Tensor of shape [num_envs, num_bodies, 3] with body positions
        body_names: List of body names
        save_plot: Whether to save plot to file
        plot_file: Output filename
        show_trajectories: Whether to show trajectories
        highlight_fingers: Whether to highlight finger bodies
        finger_indices: List of finger body indices
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(gripper_data, torch.Tensor):
        positions = gripper_data.cpu().numpy()
    else:
        positions = gripper_data
    
    num_envs, num_bodies, _ = positions.shape
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different bodies
    colors = plt.cm.tab10(np.linspace(0, 1, num_bodies))
    
    # Special colors for fingers if highlighting
    if highlight_fingers and finger_indices:
        for i, finger_idx in enumerate(finger_indices):
            if i == 0:
                colors[finger_idx] = [1, 0, 0, 1]  # Red for first finger
            elif i == 1:
                colors[finger_idx] = [0, 0, 1, 1]  # Blue for second finger
    
    # Plot each body across all configurations
    for body_idx in range(num_bodies):
        if body_idx < len(body_names):
            body_name = body_names[body_idx]
        else:
            body_name = f"Body_{body_idx}"
        
        # Get positions for this body across all environments
        body_positions = positions[:, body_idx, :]  # [num_envs, 3]
        
        # Plot the positions
        ax.scatter(body_positions[:, 0], body_positions[:, 1], 
                   body_positions[:, 2], color=colors[body_idx], 
                   label=body_name, s=50, alpha=0.7)
        
        # Show trajectories if requested
        if show_trajectories:
            ax.plot(body_positions[:, 0], body_positions[:, 1], 
                    body_positions[:, 2], color=colors[body_idx], 
                    alpha=0.3, linestyle='-')
    
    # Add labels for first and last configurations
    if num_envs > 1:
        # First configuration (typically fully open)
        for body_idx in range(num_bodies):
            if body_idx in (finger_indices or []):
                ax.text(positions[0, body_idx, 0], positions[0, body_idx, 1], positions[0, body_idx, 2], 
                       f"{body_names[body_idx]}_open", fontsize=8, alpha=0.7)
        
        # Last configuration (typically fully closed)
        for body_idx in range(num_bodies):
            if body_idx in (finger_indices or []):
                ax.text(positions[-1, body_idx, 0], positions[-1, body_idx, 1], positions[-1, body_idx, 2], 
                       f"{body_names[body_idx]}_closed", fontsize=8, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'Gripper Body Positions in 3D Space\n({num_envs} configurations, {num_bodies} bodies)')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Make axes equal
    def set_axes_equal(ax):
        """Make axes of 3D plot have equal scale"""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        max_range = max(x_range, y_range, z_range)
        
        x_center = np.mean(x_limits)
        y_center = np.mean(y_limits)
        z_center = np.mean(z_limits)
        
        ax.set_xlim3d([x_center - max_range/2, x_center + max_range/2])
        ax.set_ylim3d([y_center - max_range/2, y_center + max_range/2])
        ax.set_zlim3d([z_center - max_range/2, z_center + max_range/2])
    
    set_axes_equal(ax)
    
    # Save or show plot
    if save_plot:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    else:
        plt.show()

def create_and_plot_gripper(config, plot_args):
    """Create gripper and plot its body positions"""
    # Create gripper using the lab simulation
    from create_gripper_lab import GripperCreator
    gripper_creator = GripperCreator(config)
    
    # We need to modify the create_gripper method to return the body_com_pos_w data
    # For now, let's capture it during the simulation
    gripper_data = None
    body_names = None
    finger_indices = None
    
    # Store the original create_gripper method
    original_create_gripper = gripper_creator.create_gripper
    
    def create_gripper_with_data_capture(save_gripper=False):
        nonlocal gripper_data, body_names, finger_indices
        
        # Create gripper normally
        gripper = original_create_gripper(save_gripper)
        
        # The data we need is captured during the simulation
        # We need to access it from the simulation context
        return gripper
    
    # Replace the method
    gripper_creator.create_gripper = create_gripper_with_data_capture
    
    # Create the gripper
    gripper = gripper_creator.create_gripper(save_gripper=False)
    
    # Check if we have a saved gripper file we can load data from
    if hasattr(gripper, 'body_transforms') and gripper.body_transforms is not None:
        # Extract position data from transforms
        transforms = gripper.body_transforms.numpy()  # [num_bodies, num_envs]
        num_bodies, num_envs = transforms.shape[0], transforms.shape[1]
        
        # Convert transforms to positions
        positions = np.zeros((num_envs, num_bodies, 3))
        for env_idx in range(num_envs):
            for body_idx in range(num_bodies):
                # Extract position from transform matrix
                transform = transforms[body_idx, env_idx]
                if hasattr(transform, 'p'):
                    positions[env_idx, body_idx, :] = [transform.p[0], transform.p[1], transform.p[2]]
                else:
                    # If it's a matrix, extract translation part
                    positions[env_idx, body_idx, :] = transform[:3, 3]
        
        body_names = gripper.body_names
        finger_indices = gripper.finger_indices
        
        # Plot the data
        plot_gripper_3d(positions, body_names, 
                       save_plot=plot_args.save_plot,
                       plot_file=plot_args.plot_file,
                       show_trajectories=plot_args.show_trajectories,
                       highlight_fingers=plot_args.highlight_fingers,
                       finger_indices=finger_indices)
    else:
        print_red("Could not extract gripper body position data for plotting")

def main(args):    
    # Convert open configuration string to dict if needed
    open_configuration = args.open_configuration
    if isinstance(open_configuration, str):
        open_configuration = open_configuration_string_to_dict(open_configuration)
    
    # Create gripper configuration
    config = GripperConfig(
        gripper_file=args.gripper_file,
        finger_colliders=args.finger_colliders,
        base_frame=args.base_frame,
        bite=args.bite,
        pinch_width_resolution=args.pinch_width_resolution,
        open_configuration=open_configuration,
        device=args.device
    )
    
    # Create and plot gripper
    create_and_plot_gripper(config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot gripper body positions in 3D")
    add_plot_args(parser, globals())
    args_cli = parser.parse_args()
    
    # Apply gripper configuration if specified
    apply_gripper_configuration(args_cli)
    
    # Initialize simulation_app when needed
    simulation_app = start_isaac_lab_if_needed(file_name=__file__, headless= False if args_cli.force_headed else args_cli.headless, wait_for_debugger_attach=getattr(args_cli, 'wait_for_debugger_attach', False))
    
    main(args_cli)