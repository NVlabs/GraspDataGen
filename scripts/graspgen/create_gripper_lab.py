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
from gripper import apply_gripper_configuration, add_gripper_args, collect_gripper_args
from graspgen_utils import print_red, print_purple, start_isaac_lab_if_needed, add_create_gripper_args, collect_create_gripper_args
from grasp_constants import DEFAULT_MEASURE_CONVERGENCE, DEFAULT_CONVERGENCE_ITERATIONS

# Use centralized constants from grasp_constants.py
default_measure_convergence = DEFAULT_MEASURE_CONVERGENCE
default_convergence_iterations = DEFAULT_CONVERGENCE_ITERATIONS

import torch
import os
import warp as wp
import json
from warp_kernels import get_transforms_kernel, add_constant_kernel, add_2d_translation_kernel

import numpy as np

from gripper import Gripper, GripperConfig
import usd_tools
from graspgen_utils import open_configuration_string_to_dict

class GripperCreator:
    def __init__(self, config, wait_for_debugger_attach=False):        
        # Set number of environments to match number of grasps, but not exceed num_envs
        self.config = config
        self.wait_for_debugger_attach = wait_for_debugger_attach
        self._do_render = None  # Will be set when needed

    @property
    def do_render(self):
        """Get the render setting, initializing it if needed."""
        if self._do_render is None:
            from graspgen_utils import get_simulation_app
            # Get the simulation app without forcing headed mode
            # The headless setting should be determined by the main function
            simulation_app = get_simulation_app(__file__, force_headed=False, wait_for_debugger_attach=self.wait_for_debugger_attach)
            if simulation_app is not None:
                self._do_render = not simulation_app.DEFAULT_LAUNCHER_CONFIG['headless']
            else:
                # Default to headless if Isaac Lab is not started yet
                self._do_render = False
        return self._do_render

    def build_scene_cfg(self):
        # Import Isaac Lab modules after Isaac Lab is started
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from isaaclab.actuators import ImplicitActuatorCfg
        
        @configclass
        class GripperCreatorSceneCfg(InteractiveSceneCfg):
            """Configuration for a grasping scene."""    
            # lights
            dome_light = AssetBaseCfg(
                prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            )
            # gripper
            gripper = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                init_state=ArticulationCfg.InitialStateCfg(),
                actuators={
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=[".*",],
                        stiffness=1000.0,
                        damping=100.0,
                        #effort_limit_sim=500.0,
                        #velocity_limit_sim=.40,
                    ),
                },
            )

        scene_cfg = GripperCreatorSceneCfg(num_envs=self.num_envs, env_spacing=0.0)
        scene_cfg.gripper.spawn = sim_utils.UsdFileCfg(
            usd_path=self.config.gripper_file,
            )
        return scene_cfg

    def save_scene_full(self, folder_name, gripper):
        num_bodies = len(gripper.body_names)
        gripper_bodies = {}
        local_transform_inverses = []
        for b_idx, name in enumerate(gripper.body_names):
            gripper_bodies[b_idx] = {}
            prim_path = gripper.root_physx_view.link_paths[0][b_idx]
            gripper_bodies[b_idx]["name"] = name
            _cm, _lti = usd_tools.get_prim_collision_mesh(
                prim=self.scene.stage.GetPrimAtPath(prim_path), collision_enabled=False, device=self.config.device)
            gripper_bodies[b_idx]["collision_mesh"] = _cm
            gripper_bodies[b_idx]["local_transform_inverse"] = _lti
            local_transform_inverses.append(_lti)

        # This is for convenience, so we can use the body name to index into the gripper_bodies dictionary.
        for b_idx, name in enumerate(gripper.body_names):
            gripper_bodies[name] = b_idx

        # transforms[b_idx, 0] is the OPEN gripper, and transforms[b_idx, -1] is the CLOSED gripper
        body_transforms = wp.zeros(shape=(num_bodies, self.num_envs), dtype=wp.transform, device=self.config.device)
        local_transform_inverses = wp.array(local_transform_inverses, dtype=wp.transform, device=self.config.device)
        # this was all coded as if the lower limit was the open position.  But the upper limit can also be the open position.
        # So collect the transforms so that 0 is the open position, and -1 is the closed position, no matter if it's the lower or upper limit.

        open_limit, approach_axis, open_axis, mid_axis, finger_indices = self.get_open_limit_axes_and_finger_indices(gripper)
        reverse_order = open_limit == "upper"

        wp.launch(get_transforms_kernel,
                dim=(num_bodies, self.num_envs),
                inputs=[gripper.data.body_pos_w, gripper.data.body_quat_w, body_transforms, self.num_envs, reverse_order, local_transform_inverses],
                device=self.config.device
                )

        self.save_scene(folder_name, 0, gripper_bodies, body_transforms, use_blips=True)

    # blips is a dictionary of body names, and locations to draw blips at
    def save_scene(self, folder_name, env_idx, gripper_bodies, body_transforms, only_do_these_bodies=None, use_blips=True):
        """
        Save the scene to a folder.
        If use_blips is True, then the blips, at the local origin, will be saved as a separate object.
        """
        grasp_dataset_dir = os.environ.get('GRASP_DATASET_DIR', '')
        output_dir = os.path.join(grasp_dataset_dir, f"debug_output/{folder_name}/")
        # Remove directory if it exists and create a new empty one
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        gripper = self.scene["gripper"]
        transforms = body_transforms.numpy() #MTC
        for b_idx, b_name in enumerate(gripper.body_names):
            if only_do_these_bodies is not None and b_idx not in only_do_these_bodies:
                continue
            name = b_name
            obj_file = f"{output_dir}/{name}.obj"
            json_file = f"{output_dir}/{name}.json"
            verts = gripper_bodies[b_idx]["collision_mesh"]["vertices"].numpy().tolist()
            tris = gripper_bodies[b_idx]["collision_mesh"]["indices"].numpy().tolist()
            with open(obj_file, "w") as f:
                for v in verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                if use_blips:
                    offset = 0.001
                    f.write(f"v 0.0 {-offset} 0.0\n")
                    f.write(f"v {-offset} {offset} {-offset}\n")
                    f.write(f"v {offset} {offset} {-offset}\n")
                    f.write(f"v 0.0 {offset} {offset}\n")
                for t in range(0, len(tris), 3):
                    f.write(f"f {tris[t]+1} {tris[t+1]+1} {tris[t+2]+1}\n")
                if use_blips:
                    f.write(f"f {len(verts)+4} {len(verts)+2} {len(verts)+1}\n")
                    f.write(f"f {len(verts)+4} {len(verts)+3} {len(verts)+2}\n")
                    f.write(f"f {len(verts)+3} {len(verts)+4} {len(verts)+1}\n")
                    f.write(f"f {len(verts)+2} {len(verts)+3} {len(verts)+1}\n")

            with open(json_file, "w") as f:
                xform = transforms[b_idx, env_idx]
                mat_xform = usd_tools.transform_to_matrix(xform)
                json.dump([mat_xform.tolist()], f)

        print(f"saved scene {output_dir}")

    def get_bite_point(self, collision_mesh, approach_axis, open_axis, mid_axis):
        bbox = collision_mesh["bbox"]
        t_min = bbox[0]#wp.transform_point(local_transform, wp.vec3(bbox[0]))
        t_max = bbox[1]#wp.transform_point(local_transform, wp.vec3(bbox[1]))

        bite_point = wp.vec3()
        if self.config.bite > t_max[approach_axis] - t_min[approach_axis]:
            print_red(f"Warning: Bite is greater than the {['x','y','z'][approach_axis]} axis of the finger, bite: {self.config.bite}, approach axis range: {t_max[approach_axis] - t_min[approach_axis]}")
        bite_point[approach_axis] = t_max[approach_axis] - self.config.bite
        bite_point[open_axis] = t_max[open_axis]
        bite_point[mid_axis] = t_min[mid_axis] + 0.5*(t_max[mid_axis]-t_min[mid_axis])
        # do I need to worry about the approach axis being negative?
        return bite_point

    # Which limit, lower or upper, is the open state for the gripper?  The open_limit
    # returns open_limit, approach_axis, open_axis, mid_axis, finger_indices
    def get_open_limit_axes_and_finger_indices(self, gripper):
        # get the finger positions in both the 0 (lower) and -1 (upper) limits env.
        # the env with the fingers furthest apart is the open_limit.
        # furthermore, the largest vector in the delta is the open_axis, and if that is negative, then you need to switch the finger indices
        finger_idxs = [gripper.body_names.index(self.config.finger_colliders[0]), gripper.body_names.index(self.config.finger_colliders[1])]
        lower_delta = wp.vec3f(gripper.data.body_com_pos_w[0, finger_idxs[1]] - gripper.data.body_com_pos_w[0, finger_idxs[0]])
        upper_delta = wp.vec3f(gripper.data.body_com_pos_w[self.num_envs-1, finger_idxs[1]] - gripper.data.body_com_pos_w[self.num_envs-1, finger_idxs[0]])
        lower_length = wp.length(lower_delta)
        upper_length = wp.length(upper_delta)
        # The tool position is the position between the fingers when the gripper is closed.
        # whichever finger env ()
        if lower_length > upper_length:
            open_limit = "lower"
            delta = lower_delta
            tool_pos = 0.5*wp.vec3f(gripper.data.body_com_pos_w[self.num_envs-1, finger_idxs[1]] + gripper.data.body_com_pos_w[self.num_envs-1, finger_idxs[0]])
        else:
            open_limit = "upper"
            delta = upper_delta
            tool_pos = 0.5*wp.vec3f(gripper.data.body_com_pos_w[0, finger_idxs[1]] + gripper.data.body_com_pos_w[0, finger_idxs[0]])
        # delta is now finger1 - finger0 position in the open limit.  The largest delta is the open_axis.
        open_axis = wp.argmax(wp.abs(delta))
        if delta[open_axis] < 0.0:
            finger_idxs = finger_idxs[::-1]
        approach_delta = tool_pos - wp.vec3f(gripper.data.body_com_pos_w[0, gripper.body_names.index(self.config.base_frame)])
        approach_delta[open_axis] = 0.0
        approach_axis = wp.argmax(wp.abs(approach_delta))
        mid_axis = 3 - approach_axis - open_axis
        # MTC TODO if the approach axis delta is negative, then we use negaive normals when moving away form the object?
        return open_limit, approach_axis, open_axis, mid_axis, finger_idxs

    def create_gripper(self, save_gripper=False, measure_convergence=DEFAULT_MEASURE_CONVERGENCE, convergence_iterations=DEFAULT_CONVERGENCE_ITERATIONS):
        from graspgen_utils import get_simulation_app
        # Get the simulation app without forcing headed mode
        # The headless setting should be determined by the main function
        simulation_app = get_simulation_app(__file__, force_headed=False, wait_for_debugger_attach=self.wait_for_debugger_attach)

        # Import Isaac Lab modules after Isaac Lab is started
        import isaaclab.sim as sim_utils
        from isaaclab.sim import build_simulation_context
        from isaaclab.scene import InteractiveScene
        
        # Initialize the simulation context
        dt = 0.1 * DEFAULT_CONVERGENCE_ITERATIONS / max(DEFAULT_CONVERGENCE_ITERATIONS, convergence_iterations)
        sim_cfg = sim_utils.SimulationCfg(
            device=self.config.device,
            dt=dt,  # Set physics timestep based on FPS
            physx=sim_utils.PhysxCfg()
        )
        with build_simulation_context(device=self.config.device, gravity_enabled = False, auto_add_lighting=True, sim_cfg=sim_cfg) as sim:
            sim._app_control_on_stop_handle = None

            #self.sim = sim_utils.SimulationContext(sim_cfg) 
            # Set main camera
            sim.set_camera_view(eye=(1.7889285412085776, 2.4058293979259333, 2.254122255934516), target=(0.0, 0.0, 0.0))
            
            # Create scene
            # need an environment for the min, max, and open configuration... and as many pinch widths as requested.
            
            self.num_envs = 0 if len(self.config.open_configuration) == 0 else 1
            self.num_envs += max(2, self.config.pinch_width_resolution)
            scene_cfg = self.build_scene_cfg()

            self.scene = InteractiveScene(scene_cfg)
            #self.save_scene_full(f'gripper', gripper)
            # End of used to be in __init__

            # Setup a simulation to put the gripper in an array of open positions, then get the body positions
            # and gripper info after running the sim for the the ik.
            sim.reset()

            sim_dt = sim.get_physics_dt()
            
            # Get scene entities
            gripper = self.scene["gripper"]
            #self.save_scene_full(f'sim_reset', gripper)

            sim_time = 0.0
            count = 0
            gripper_state = gripper.data.default_root_state.clone()
            gripper_state[:, :3] += self.scene.env_origins
            gripper.write_root_pose_to_sim(gripper_state[:, :7])
            gripper.write_root_velocity_to_sim(gripper_state[:, 7:])
            #self.save_scene_full(f'write_root', gripper)
            
            # Apply motion to gripper
            # joint state
            # Create linearly interpolated joint positions between lower and upper limits
            lower_limits = gripper.data.soft_joint_pos_limits[..., 0]  # [self.num_envs, 6]
            upper_limits = gripper.data.soft_joint_pos_limits[..., 1]  # [self.num_envs, 6]
            # get the driven joints, and if the joint is not driven set it's limits to 0
            # MTC TODO Or the value of the first driven joint?  CAn we find it's mimic mirror? and set it to that?

            drive_types_first_env = gripper.root_physx_view.get_drive_types()[0]  # Shape: (J,)
            # Create a mask for non-zero drive types
            non_zero_mask = drive_types_first_env != 0  # Shape: (J,)
            driven_joints = {}
            gripper_joints = {}
            for j_idx, name in enumerate(gripper.data.joint_names):
                gripper_joints[j_idx] = name
                # Filter driven joint names using the mask
                if non_zero_mask[j_idx]:
                    driven_joints[j_idx] = name
        
            for j in range(lower_limits.shape[1]):
                if j not in driven_joints:
                    for i in range(lower_limits.shape[0]):
                        lower_limits[i, j] = 0.0
                        upper_limits[i, j] = 0.0
            
            # Create interpolation weights from 0 to 1, but add an extra at the end for the open configuration
            num_weights = self.num_envs if len(self.config.open_configuration) == 0 else self.num_envs - 1
            weights = torch.linspace(0, 1, num_weights, device=lower_limits.device)
            if len(self.config.open_configuration) > 0:
                weights = torch.cat([weights, torch.tensor([1.0], device=lower_limits.device)])
            weights = weights.view(-1, 1)  # [self.num_envs, 1]
            # Interpolate between lower and upper limits
            joint_pos = lower_limits + weights * (upper_limits - lower_limits)  # [self.num_envs, 6]

            open_configuration_offset = 0
            if len(self.config.open_configuration) > 0:
                prev_existing = None
                for joint_name, joint_angle in self.config.open_configuration.items():
                    joint_idx = gripper.data.joint_names.index(joint_name)
                    joint_pos[-1, joint_idx] = joint_angle
                    # Check if joint_angle already exists in the array
                    # This gets kind of complex to figure out if all the open_configuratition
                    # values already exist in the same place in the array.  If they do, then
                    # we can shrink the transforms to the number of unique open_configuration values.
                    existing_indices = torch.where(joint_pos[:-1, joint_idx] == joint_angle)[0]
                    if len(existing_indices) > 0:
                        candidate_existing = existing_indices.tolist()
                        if prev_existing is None:
                            prev_existing = candidate_existing
                        else:
                            prev_existing = list(set(prev_existing) & set(candidate_existing))
                if prev_existing is not None:
                    open_configuration_offset = prev_existing[0]
                    # shrink_transforms = True
                    # NOTE: THIS IS A BIG DEAL... -1 no longer works for entries into gripper.data.body_com_pos_w etc.
                    # we are shrinking the output transoforms by ignoring the last entry of the input ones.
                    self.num_envs = self.num_envs - 1
                else:
                    # Sort joint_pos based on the values in joint_idx column
                    sorted_indices = torch.argsort(joint_pos[:, joint_idx])
                    open_configuration_offset = torch.where(sorted_indices == (self.num_envs - 1))[0][0].item()
                    joint_pos = joint_pos[sorted_indices]
            
            joint_vel = gripper.data.default_joint_vel.clone()
            gripper.write_joint_state_to_sim(joint_pos, joint_vel)
            #self.save_scene_full(f'write_joint_state', gripper)
            # clear internal buffers
            self.scene.reset()
            #self.save_scene_full(f'scene_reset', gripper)

            # gripper should always be matching the joint positions
            joint_pos_target = joint_pos 

            # Set joint position target
            gripper.set_joint_position_target(joint_pos_target)
            # -- write data to sim
            self.scene.write_data_to_sim()
            # self.save_scene_full(f'write_data_to_sim', gripper)
            # Perform step
            if measure_convergence:
                old_pos = gripper.data.body_com_pos_w.clone()
            for i in range(convergence_iterations):
                sim.step(render=self.do_render)
                count += 1
                # Update buffers
                self.scene.update(sim_dt)
                # Update simulation time
                sim_time += sim_dt
                if measure_convergence:
                    new_pos = gripper.data.body_com_pos_w.clone()
                
                    delta = new_pos - old_pos
                    # Calculate the magnitude of the difference for each env/body
                    delta_magnitude = torch.norm(delta, dim=-1)  # Shape: [num_envs, num_bodies]
                    max_diff = torch.max(delta_magnitude)
                    
                    print(f"Step {i}: Max position difference: {max_diff.item():.8f} {max_diff.item()}")
                    old_pos = new_pos
            self.scene.write_data_to_sim()
            # self.save_scene_full(f'write_data_to_sim_final', gripper)


            # The simulation is done.
            # Gather the information necessary to build a Gripper object
            gripper_data_out = {}
            gripper_data_out["num_openings"] = self.num_envs
            gripper_data_out["base_frame"] = self.config.base_frame
            open_limit, approach_axis, open_axis, mid_axis, finger_indices = self.get_open_limit_axes_and_finger_indices(gripper)
            gripper_data_out["open_limit"] = open_limit # Is the open position of the gripper the lower or upper limit?
            gripper_data_out["approach_axis"] = approach_axis # The axis the gripper points in
            gripper_data_out["open_axis"] = open_axis # The axis the fingers move in when opening.
            gripper_data_out["mid_axis"] = mid_axis # The axis fingers don't move in, assuming the root and fingers move in the approach-open-axis plane.
            gripper_data_out["finger_indices"] = finger_indices # The indices of the fingers that move in the open_axis, finger1-finger0 will be positive in the open_axis direction.
            reverse_order = gripper_data_out["open_limit"] == "upper"
            if reverse_order:
                open_configuration_offset = self.num_envs - 1 - open_configuration_offset
            gripper_data_out["open_configuration_offset"] = open_configuration_offset

            # Get the collision meshes
            num_bodies = len(gripper.body_names)
            gripper_bodies = {}
            local_transform_inverses = []
            for b_idx, name in enumerate(gripper.body_names):
                gripper_bodies[b_idx] = {}
                prim_path = gripper.root_physx_view.link_paths[0][b_idx]
                gripper_bodies[b_idx]["name"] = name
                _cm, _lti = usd_tools.get_prim_collision_mesh(
                    prim=self.scene.stage.GetPrimAtPath(prim_path), collision_enabled=False, device=self.config.device)
                gripper_bodies[b_idx]["collision_mesh"] = _cm
                gripper_bodies[b_idx]["local_transform_inverse"] = _lti
                local_transform_inverses.append(_lti)

            # bbox[1] is max
            root_bbox = gripper_bodies[gripper.body_names.index(self.config.base_frame)]["collision_mesh"]["bbox"]
            gripper_data_out["base_length"] = root_bbox[1][approach_axis] - root_bbox[0][approach_axis]

            # This is for convenience, so we can use the body name to index into the gripper_bodies dictionary.
            for b_idx, name in enumerate(gripper.body_names):
                gripper_bodies[name] = b_idx

            # transforms[b_idx, 0] is the OPEN gripper, and transforms[b_idx, -1] is the CLOSED gripper
            body_transforms = wp.zeros(shape=(num_bodies, self.num_envs), dtype=wp.transform, device=self.config.device)
            local_transform_inverses = wp.array(local_transform_inverses, dtype=wp.transform, device=self.config.device)
            # this was all coded as if the lower limit was the open position.  But the upper limit can also be the open position.
            # So collect the transforms so that 0 is the open position, and -1 is the closed position, no matter if it's the lower or upper limit.
            wp.launch(get_transforms_kernel,
                    dim=(num_bodies, self.num_envs),
                    inputs=[gripper.data.body_pos_w, gripper.data.body_quat_w, body_transforms, self.num_envs, reverse_order, local_transform_inverses],
                    device=self.config.device
                    )

            #self.save_scene("finger0_before_bite", 0, gripper_bodies, body_transforms, only_do_these_bodies=[finger_indices[0]]) #MTC
            #self.save_scene("before_bite", 0, gripper_bodies, body_transforms) #MTC

            # calculate the bite point on finger 0, and then move finger 0 so the bite point is it's origin.
            bite_point = self.get_bite_point(gripper_bodies[finger_indices[0]]["collision_mesh"], approach_axis, open_axis, mid_axis)
            gripper_data_out["bite_point"] = bite_point
            wp.launch(kernel=add_constant_kernel,
                    dim=len(gripper_bodies[finger_indices[0]]["collision_mesh"]["vertices"]), 
                    inputs=[gripper_bodies[finger_indices[0]]["collision_mesh"]["vertices"], -bite_point],
                    device=self.config.device)
            wp.launch(kernel=add_2d_translation_kernel,
                    dim=self.num_envs,
                    inputs=[body_transforms, finger_indices[0], bite_point],
                    device=self.config.device)
            gripper_data_out["bite_points"] = body_transforms[finger_indices[0], :].numpy()[:, :3]
            for i in range(gripper_data_out["bite_points"].shape[0]):
                gripper_data_out["bite_points"][i, mid_axis] = 0.0
                gripper_data_out["bite_points"][i, approach_axis] = abs(gripper_data_out["bite_points"][i, approach_axis])
                gripper_data_out["bite_points"][i, open_axis] = -gripper_data_out["bite_points"][i, open_axis]
            gripper_data_out["open_widths"] = gripper_data_out["bite_points"][:, open_axis]*2.0
            gripper_data_out["bodies"] = gripper_bodies
            gripper_data_out["body_transforms"] = body_transforms
            gripper_data_out["body_names"] = gripper.body_names
            # Get the joint information
            # I wonder if we want to store all the scpace positions instead of just the driven ones?
            # would that extra data be useful?  Could use it in ik without sim or mimic solving.
            # Get the drive types for the first environment (since they're all the same)
            gripper_data_out["driven_joints"] = driven_joints

            for j_idx, name in enumerate(gripper.data.joint_names):
                gripper_joints[name] = j_idx

            # Keep the joint positions in the same order as the transforms
            if gripper_data_out["open_limit"] == "lower":
                gripper_joints["cspace_positions"] = gripper.data.joint_pos[:self.num_envs].clone()
            else:
                gripper_joints["cspace_positions"] = gripper.data.joint_pos[:self.num_envs].clone().flip(0)

            gripper_data_out["joints"] = gripper_joints
            gripper_data_out["joint_names"] = gripper.data.joint_names
            gripper_data_out["transform_body_frame"] = None # Body is probably 0, but if it's not then be safe.

            return_gripper = Gripper(self.config, gripper_data_out)
            print(f"Gripper Created from {self.config.gripper_file}")
            if save_gripper:
                return_gripper.save(self.config.gripper_file)
            if self.do_render:
                print_purple(f"In {__file__}, waiting for Isaac Lab to close...", flush=True)
                while simulation_app.is_running():
                    sim.step(render=self.do_render)
                print("\033[0m", end="")

            return return_gripper

def create_gripper_with_lab(gripper_config, save_gripper=True, measure_convergence=DEFAULT_MEASURE_CONVERGENCE, convergence_iterations=DEFAULT_CONVERGENCE_ITERATIONS, wait_for_debugger_attach=False):
    # Main function to run the grasping simulation
    # TODO Make this work with CPU
    # Check if user specified CPU device
    if gripper_config.device == "cpu":
        print_red("Warning: --device cpu is not expected to work with this script. If you want to run other components with CPU, please run create_gripper_lab.py standalone for your gripper first using cuda so the gripper.npz file is created.")
        
    gripper_creator = GripperCreator(gripper_config, wait_for_debugger_attach=wait_for_debugger_attach)
    gripper = gripper_creator.create_gripper(save_gripper, measure_convergence, convergence_iterations)
    return gripper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a gripper for use with Grasp Gen using Isaac Lab.")
    add_create_gripper_args(parser, globals(), **collect_create_gripper_args(globals()))
    add_gripper_args(parser, globals(), **collect_gripper_args(globals()))
    args_cli = parser.parse_args()
    
    # Apply gripper configuration if specified
    apply_gripper_configuration(args_cli)
    
    # Initialize simulation_app when needed
    simulation_app = start_isaac_lab_if_needed(file_name=__file__, headless= False if args_cli.force_headed else args_cli.headless, wait_for_debugger_attach=getattr(args_cli, 'wait_for_debugger_attach', False))
    
    gripper_file=args_cli.gripper_file
    finger_colliders=args_cli.finger_colliders
    base_frame=args_cli.base_frame
    bite=args_cli.bite
    pinch_width_resolution=args_cli.pinch_width_resolution
    open_configuration=args_cli.open_configuration
    if isinstance(open_configuration, str):
        open_configuration = open_configuration_string_to_dict(open_configuration)
    device=args_cli.device

    config = GripperConfig(gripper_file, finger_colliders, base_frame, bite, pinch_width_resolution, open_configuration, device)
    create_gripper_with_lab(config, save_gripper=True, measure_convergence=args_cli.measure_convergence, convergence_iterations=args_cli.convergence_iterations, wait_for_debugger_attach=getattr(args_cli, 'wait_for_debugger_attach', False))