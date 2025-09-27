# Gripper Definition

The Gripper Definition component reads USD gripper files and extracts the parameters needed for grasp generation. It analyzes the gripper's geometry, kinematics, and collision properties to create a definition that can be used by the grasp guess generation system.

This document describes the gripper definition component and its parameters in detail. There are also some examples of its use:

- **[Gripper Setup Example](../examples/gripper-setup.md)** - Describes in detail how the Robotiq 85 was prepared for use in GraspDataGen
- **[Running the Gripper Definition Component](../examples/gripper-definition.md)** - Visually verify your gripper will work with the simulation

## Overview

The gripper definition process:

1. **Loads a USD gripper file** and analyzes its structure
2. **Identifies finger colliders** and their kinematic relationships
3. **Computes gripper parameters** including approach direction, close direction, and collision bodies
4. **Saves the definition** in a format that can be efficiently loaded by other components

## Basic Usage

The gripper definition code itself, `gripper.py`, does not have an executable path, but the code that creates the definition does.

```bash
python scripts/graspgen/create_gripper_lab.py --gripper_config onrobot_rg6
```

> **Note**: The `--gripper_config` option automatically sets all gripper parameters. For detailed information about gripper configurations and the parameter system, see **[Parameter System](../api/parameter-system.md)**.

## Command Line Arguments

This section explains the command line arguments for the gripper in general, found in `gripper.py`, and those that the gripper creator, `create_gripper_lab.py`, uses. The gripper definition component also uses shared [GraspDataGen Parameters](../api/parameter-system.md#graspdatagen-arguments).

### Gripper Arguments

The `gripper.py` file defines the default arguments for a gripper. These are the core gripper configuration parameters:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gripper_config` | str | `None` | Predefined gripper configuration to use |
| `--gripper_file` | str | `"bots/onrobot_rg6.usd"` | Path to gripper USD file |
| `--finger_colliders` | list | `["right_inner_finger", "left_inner_finger"]` | Names of finger collider bodies |
| `--base_frame` | str | `"base_frame"` | Name of the base frame origin |
| `--bite` | float | `0.01` | Depth of bite from fingertip (meters) |
| `--pinch_width_resolution` | int | `8` | Number of pinch opening widths to test |
| `--open_configuration` | str | `"{}"` | Initial joint configuration as JSON |

### Gripper Creation Arguments

The `create_gripper_lab.py` component defines additional arguments/parameters specific to gripper creation:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--measure_convergence` | flag | `False` | Measure convergence of the gripper |
| `--convergence_iterations` | int | `20` | Number of simulation iterations for convergence |

## Gripper Parameters

The gripper definition system uses several key parameters that are shared across all components:

- **`gripper_file`** - Path to the USD gripper file (must be flattened for best compatibility)
- **`finger_colliders`** - Names of the two rigid bodies that serve as the gripper's fingers
- **`base_frame`** - Name of the single rigid body prim that the fingers attach to (should be placed at the origin)
- **`bite`** - Minimum depth from the fingertip used to start grasp location (meters)
- **`pinch_width_resolution`** - Number of pinch opening widths to test during grasp generation
- **`open_configuration`** - Initial joint configuration as JSON string

### Parameter Details

#### Gripper File Requirements
GraspDataGen has only been tested with gripper USD files that have been flattened so all their properties are in one file with no external references.

#### Finger Colliders
The names you want to use are the names of the two rigid bodies that are the fingers of the gripper. When setting up the gripper, make sure these names are unique.

#### Base Frame
The single rigid body prim that the fingers attach to. When setting up the gripper, this should be placed at the origin of the scene. All the grasp transforms will be relative to this prim. There should be no transforms attached to this base.

#### Bite Depth
The bite is the minimum length from the end of the fingertip that is used to start the location of the guessed grasp. Consider your gripper grasping a sphere. A bite of 0.0 would mean that the fingertip would be exactly touching the sphere. In the image below, the gripper on the left has a bite that is half the height of the fingertip, and the one on the right has a bite of 0.0.

<img src="../images/bite.png" alt="The bite defines the minimum depth a finger will reach in to grab an object. On the right is a bite of 0" width="55%">

#### Pinch Width Resolution

The pinch width resolution is the number of pinch opening sizes (widths) that the grasp guess generator will test when trying to create collision-free grasps. Consider the image below: with a `pinch_width_resolution` of 8 (left), there will be 8 opening sizes tested; with a `pinch_width_resolution` of 2 (right), only the closed and open positions are tested.

<img src="../images/pinch_width_resolution.png" alt="The number or opening widths tested per location when guessing grasps" width="55%">

#### Open Configuration

Aside from the max open and close positions, and the even spacings in between based on the `pinch_width_resolution`, you can add a custom opening size to test with the open configuration parameter.
The `open_configuration` is a string representation of a JSON map of joint names to angles. For example, running the following command will create a gripper definition with a specific open width that will be guaranteed to be tested:

```bash
python scripts/graspgen/create_gripper_lab.py --force_headed --gripper_config onrobot_rg6 --pinch_width_resolution 2 --open_configuration '{"finger_joint": -0.31}'

```

<img src="../images/open_configuration.png" alt="Force a specific open configuration to be used" width="55%">

## Create Gripper Parameters

One of the things the gripper creator does is set the gripper joint positions to various opening sizes, and then solve the inverse kinematics for the individual link world-space positions. This process can be problematic if the IK solver can't converge. To monitor convergence, and to enforce smaller steps in the solve if necessary, there are two parameters specific to the gripper creator.

- **`measure_convergence`** - Measure IK convergence and print the results to the terminal.
- **`convergence_iterations`** - Set a higher number of steps to solve for convergence.

### Parameter Details

#### Measure Convergence
Output the convergence of the IK solve in terms of maximum change in link position each frame. The result is printed to the terminal.

#### Convergence Iterations
If the IK is not converging, you can increase the number of steps used when solving, which will in turn decrease the timestep size and improve stability.

## Integration with Other Components

The gripper definition component is used by several other components in the GraspDataGen system:

- **[Grasp Guess Generation](../components/grasp-guess.md)** - Uses gripper parameters to generate geometrically plausible grasps
- **[Grasp Simulation](../components/grasp-sim.md)** - The physics-based validation uses the same USD **`--gripper_file`** used in the grasp guess generation
- **[Batch Data Generation](../workflows/datagen.md)** - Processes multiple objects using gripper configurations

### Gripper Configurations

GraspDataGen includes predefined gripper configurations that automatically set all gripper parameters. These configurations are defined in `scripts/graspgen/gripper_configurations.py` and can be used to create new **`--gripper_config`** arguments:

- **`robotiq_2f_85`** - Robotiq 2F-85 parallel gripper
- **`onrobot_rg6`** - OnRobot RG6 gripper  
- **`franka_panda`** - Franka Panda gripper

For more information about the parameter system and configurations, see **[Parameter System](../api/parameter-system.md)**.

## Output

The gripper definition process creates:

1. **`.npz` file** - Binary format containing all gripper parameters and collision data for grasp guessing
2. **Debug output** - OBJ files and JSON transforms for [debug visualization](../tools/utility-tools.md#debug-visualization)
