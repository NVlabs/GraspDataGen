# Gripper Setup Guide

This guide walks you through the process of setting up a custom gripper for use with GraspDataGen. We'll use the Robotiq 2F-85 gripper as an example, but the same principles apply to other grippers.

## Prerequisites

- Isaac Sim installed and configured
- Access to the Isaac Sim USD assets
- Basic familiarity with Isaac Sim interface

## Step 1: Load and Save the Gripper

1. **Load the gripper from Isaac Sim assets:**
   - Open Isaac Sim
   - Load the gripper from: `omniverse://isaac-dev.ov.nvidia.com/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd`

2. **Save the gripper locally:**
   - Use `File -> Save Flattened As...`
   - Save to `bots/robotiq_2f_85.usd`
   - **Important:** Make sure to open the saved file before making any edits

## Step 2: Ungroup the Robot

GraspDataGen does not handle nested robots yet, so we need to move the robot to the root USD path.

1. **Select and ungroup the robot:**
   - Select the `Robotiq_2F_85 (defaultPrim)` prim
   - Right-click and select "Ungroup Selected"
   - **Caution:** Don't select the one under that - you need the IsaacRobotAPI on that level

   <img src="../images/gripper_setup_ungroup.png" alt="Ungroup the robot" width="67%">

2. **Set the default prim:**
   - Right-click on the `Robotiq_2F_85` prim
   - Set it as the defaultPrim from the right-click menu

## Step 3: Add Root Joint

A root joint is needed to hold the hand steady throughout the simulation.

1. **Create a fixed joint:**
   - Select the `Robotiq_2F_85 (defaultPrim)` prim
   - Go to `Create -> Physics -> Joint -> Fixed Joint`
   - Name it `root_joint` or another name that won't clash

   <img src="../images/gripper_setup_add_root.png" alt="Add root joint" width="67%">

2. **Configure the joint:**
   - In the Properties window, set `Physics | Joint | Body 1` to `/Robotiq_2F_85/base_link`

3. **Add `root_joint` to the `IsaacRobotAPI`:**
   - Select the `Robotiq_2F_85 (defaultPrim)` prim
   - On the Property tab find  `IsaacRobotAPI | robotJoints` and click Add Target then select the `root_joint` 

## Step 4: Configure Finger Collision Detection (Optional)

Repeat these steps for both `right_inner_finger` and `left_inner_finger`.

### Enable CCD (Continuous Collision Detection)

1. **Select the finger:**
   - Select `Robotiq_2F_85/right_inner_finger`

2. **Enable CCD:**
   - Check the box for `Physics | Rigid Body | Enable CCD`
   - Note: This won't actually be enabled during GPU simulation, but it's a parameter in `grasp_sim.py` for CPU-based simulation

### Improve Collision Geometry

1. **Make the finger non-instancable:**
   - Select `Robotiq_2F_85/right_inner_finger/visuals`
   - Uncheck the box for "Instancable" directly under the prim path in the Property tab

2. **Change collision approximation:**
   - Find the `Physics | Collider` attribute on the `/Robotiq_2F_85/right_inner_finger/visuals/Defeatured_2F_85_PAD_OPEN_fingertipsstep_01/Defeatured_2F_85_PAD_OPEN_fingertipsstep` prim
   - Change the approximation from "Convex Hull" to "Convex Decomposition"

   <img src="../images/gripper_setup_cd.png" alt="Convex Decomposition setup" width="67%">

3. **Enable shrink wrap:**
   - Check the `Physics | Collider | Advanced | Shrink Wrap` checkbox

### Visualize Collision Objects

You can visualize the collision objects as they will be simulated:

1. Click on the Eye icon at the top of the viewport
2. Check: `Show By Type -> Physics -> Colliders -> Selected`

## Step 5: Repeat for Other Fingers

Apply the same configuration steps to the `left_inner_finger`.

## Step 6: Save and Configure

1. **Save the USD file** after making all changes

2. **Add gripper configuration:**
   - Add your new gripper configuration to `GRIPPER_CONFIGURATIONS` in `scripts/graspgen/gripper_configurations.py`

   Here is an example of what the configuration should look like. At minimum, you need to set `gripper_file`, `finger_colliders`, and `base_frame`. Note that we set `bite` to half the size of the pad to show you can customize all the other [configuration parameters](../api/parameter-system.md):

   ```python    
   'robotiq_2f_85': {
       'gripper_file': 'bots/robotiq_2f_85.usd',
       'finger_colliders': ['right_inner_finger', 'left_inner_finger'],
       'base_frame': 'base_link',
       'bite': 0.0185,  # half of 37mm
   },
   ```

3. **Test the configuration:**
   - Your gripper should now be ready for use with the [Gripper Definition](../components/gripper-definition.md) tool (`create_gripper_lab.py`)

## Important Notes

The Robotiq 2F-85 gripper from the Isaac Sim assets already has several important parameters configured that we didn't need to change:

- **Mass values** are set on the links to real-world masses
- **Force and velocity limits** are set on the driving joint (`finger_joint`)
- **Mimic joints** are properly configured for the remaining joints

These parameters ensure the gripper simulates correctly in the physics engine.

