# Grasp Guess Generation

The Grasp Guess Generation component creates geometrically plausible grasps for triangle mesh objects based on a given gripper model. It uses collision detection and geometric analysis to generate grasps that are likely to be successful.

This document describes the grasp guess generation component and its parameters in detail. There are also some examples of its use:

- **[As part of a full datagen pipeline](../workflows/datagen.md)** - Use the grasp_guess functions instead of calling `grasp_guess.py` directly
- **[As a standalone app](../examples/grasp-guess.md)** - Use `grasp_guess.py` as a standalone app

## Overview

The grasp guess generation process:

1. **Loads gripper NPZ** file, or creates one if necessary
1. **Loads object mesh** and converts USD meshes to OBJ if necessary
1. **Samples surface points** on the object
1. **Generates grasp poses** based on surface normals and gripper geometry
1. **Tests collision-free positions** for finger placement
1. **Outputs grasp candidates** in Isaac Grasp format

## Basic Usage

The grasp guess generation code needs a gripper and an object as input.  As a standalone tool you could run:

```bash
python scripts/graspgen/grasp_guess.py \
    --gripper_config onrobot_rg6 \
    --object_file objects/banana.obj
```

The grasp guess generator can also be used as a component in custom Python code instead of being called directly, as described in the [datagen example](../workflows/datagen.md).

## Command Line Arguments

This section explains the command line arguments for objects in general, found in `object.py`, and those that the grasp guess generator uses. The grasp guess generator also uses shared [GraspDataGen Parameters](../api/parameter-system.md#graspdatagen-arguments) and [Gripper Parameters](../components/gripper-definition.md#gripper-arguments).


### Object Arguments

The `object.py` file defines the default arguments for object processing:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--object_file` | str | `"objects/mug.obj"` | Path to object file (USD/OBJ/STL) |
| `--object_scale` | float | `1.0` | Scale of object (applied uniformly to all object types) |
| `--obj2usd_use_existing_usd` | bool | `True` | Use existing USD file if available |
| `--obj2usd_collision_approximation` | str | `"convexDecomposition"` | Collision approximation method |
| `--obj2usd_friction` | float | `1.0` | Friction coefficient |

### Grasp Guess Generation Arguments

The `grasp_guess.py` component defines the core arguments for grasp generation:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | Random | Seed for random number generator |
| `--num_grasps` | int | `1024` | Number of grasps to generate |
| `--num_orientations` | int | `1` | Number of orientations to try per grasp point |
| `--percent_random_guess_angle` | float | `0.75` | Percentage of random vs. aligned rotations |
| `--standoff_distance` | float | `0.001` | Distance to standoff from surface |
| `--num_offsets` | int | `16` | Number of offsets to test for finger placement |
| `--do_not_center_finger_opening` | flag | `False` | Don't center finger opening |
| `--use_acronym_grasp_guess` | flag | `False` | Use ACRONYM grasp method |
| `--correct_acronym_approach` | flag | `False` | Correct ACRONYM approach direction |
| `--max_guess_tries` | int | `100` | Maximum attempts before giving up (0 = unlimited) |
| `--save_collision_mesh_folder` | str | `""` | Save collision mesh to folder |

## Object Parameters

The object parameters control how the object is used in grasp guess generation and the [`grasp_sim`](./grasp-sim.md) module:

- **`object_file`** - Path to a USD, OBJ, or STL triangle mesh object
- **`object_scale`** - The amount to scale the input object by

The following three parameters are used only in [`grasp_sim`](./grasp-sim.md) when the input object needs to be converted to USD:

- **`obj2usd_use_existing_usd`** - If the USD is cached use it.
- **`obj2usd_collision_approximation`** - The type of collision object used
- **`obj2usd_friction`** - The friction used when simulating the object

The parameters prefixed by `obj2usd` are used only in the [`grasp_sim`](./grasp-sim.md) module. They will not be used if the input object is already USD so that user-tuned USD files can be used. You can also tune the USD settings per object for OBJ/STL inputs by converting the objects, editing the created USD, and then ensuring **`--obj2usd_use_existing_usd`** is true (which it is by default).

### Parameter Details

#### Object File
The path to the object mesh file. Supported formats include USD, OBJ, and STL. The grasp guess generator only needs a triangle mesh without physics properties because it is only checking for geometrically plausible grasps, so it will automatically convert USD files to OBJ format for processing if USD is used as input. Similarly, the grasp validation with simulation module needs a USD file for simulation, so if the input is an OBJ or STL file, it will use the `obj2usd_`-prefixed parameters to convert to a USD file.

#### Object Scale
Uniform scaling factor applied to the object. This allows testing grasps on different sized versions of the same object without modifying the original mesh file.

#### USD Conversion Parameters
##### Use Existing USD
When true, reuses existing USD files instead of regenerating them from OBJ/STL files. This can be used in two ways. The first is simple caching, where the automatically converted file will be converted once and then used for each run thereafter. The second allows a user to create a USD object that is different from the input OBJ version of the object. It could be different by just having default physics parameters (like with simple caching), or it could be an entirely different object like a simulation-ready deformable object.

This argument accepts explicit boolean values: `--obj2usd_use_existing_usd true|false` (also accepts `yes/no`, `y/n`, `1/0`). It can also be used as a flag without a value to imply `True`.


##### Collision Approximation
There are several different collision approximations that PhysX uses when simulating dynamic objects. The following subset can be used when converting an OBJ file into a USD for simulation:

  - **`"sdf"`** - Signed Distance Field approximation. The SDF approximation creates both a scaled signed distance field of the object and a high-resolution triangle mesh of that field. This is probably the most accurate representation, but it creates many contact points and **`--max_num_envs`** will likely have to be set rather low to use this representation.
  - **`"convexDecomposition"`** - Convex decomposition of the mesh (default). This approximation breaks the object into pieces and uses shrink-wrapped convex hull approximations of each of those pieces combined for collision.
  - **`"convexHull"`** - Creates a single shrink-wrapped convex hull approximation.
  - **`"sphereFill"`** - Sphere-based collision approximation using a set of spheres to approximate the object.
  - **`"none"`** - No collision approximation, mostly used for debugging purposes, the object will not be present in the simulation if this is used.

##### Friction
This value (default 1.0) will be used for both dynamic and static friction of the object.


## Grasp Guess Generation Parameters

The grasp guess generation process uses the parameters from the gripper and the object to create a random sampling of guesses at what good collision-free grasps may be. Generally, the grasp guess generation algorithm proceeds as follows:
1. Find a random point on the object surface and place the bite point of a finger collider a small distance away from that point; ensure that finger is not colliding with the object.
1. Check a number of positions of the second finger at different opening widths for collision to collect candidate grasps.
1. Possibly adjust the grasp based on parameters and make sure it's collision-free.

There are several parameters that control the above guess generation algorithm:

- **`seed`** - Random seed for reproducible results
- **`num_grasps`** - Number of grasp candidates to generate
- **`num_orientations`** - Number of orientations to test per grasp point
- **`percent_random_guess_angle`** - Percentage of grasps using random vs aligned orientations
- **`standoff_distance`** - The distance the bite point starts from the surface for the initial guess.
- **`num_offsets`** - The number of times to try to back the finger away from the surface if the initial `standoff_distance` causes a collision
- **`do_not_center_finger_opening`** - Disable centering of finger opening after generating the guess
- **`use_acronym_grasp_guess`** - Use the ACRONYM grasp generation method used in the original ACRONYM pipeline
- **`correct_acronym_approach`** - Correct approach distance when using the ACRONYM grasp guesses to account for opening width vs approach direction
- **`max_guess_tries`** - Maximum number of attempts to generate grasps before giving up (0 means unlimited tries)
- **`save_collision_mesh_folder`** - Folder to save collision mesh for debugging


### Parameter Details

#### Seed

The random seed used to initialize the random number generator. Random choices are needed for the initial random point scatter on the object and choosing an orientation for the grasp, as well as which of those angles will actually be completely random or aligned with one of four random axes.

#### Num Grasps
The number of collision-free guesses to generate. If `num_grasps` is not a multiple of `num_orientations`, then it will be increased until it is.

#### Num Orientations

Num orientations is used when you don't want truly random points but instead want each random point to also have a rotation of grasps around that point. The following images, created with the [grasp_display tool](../tools/utility-tools.md#grasp-display), give an example of 8 grasps on a sphere while setting `num_orientations` to 1 (left) or 8 (right).

<img src="../images/orient_1.png" alt="8 random samples, with `num_orientations = 1`" width="30%">
<img src="../images/orient_8.png" alt="8 random samples, with `num_orientations = 8`" width="30%">

#### Percent Random Guess Angle
Sometimes there are objects that naturally grasp well along orthogonal axes. For example, when picking up a pipe, it would make sense to grasp it perpendicular to its axis. The `percent_random_guess_angle` parameter controls the random rotation around the normal of the placed finger. Consider these images, created with the [visualize_grasp_data tool](../tools/utility-tools.md#grasp-data-visualization). The images are of 1024 grasps generated for a square thin board. On the left, we have set **`--percent_random_guess_angle 0.0`**, so all gripper rotations are axis-aligned. For fully random rotations, set **`--percent_random_guess_angle 1.0`**, as in the image on the right.

<img src="../images/rand_guess_angle_0.png" alt="1024 grasps generated for a flat board with `--percent_random_guess_angle = 0.0`" width="30%">
<img src="../images/rand_guess_angle_1.png" alt="1024 grasps generated for a flat board with `--percent_random_guess_angle = 1.0`" width="30%">

#### Standoff Distance
When the gripper finger is first placed at the random point on the surface, it is backed off a small amount so it won't be in collision with the object. The amount it is backed off away from the object, along the surface normal, is the `standoff_distance` parameter.

#### Num Offsets
When placing the first finger at the random point, the finger is moved away from the surface along the object's surface normal by some multiple of `standoff_distance`. The `num_offsets` parameter is used to decide how many times to back off and test for collision before giving up. The test moves further and further from the surface, starting at `standoff_distance` and ending at `standoff_distance * num_offsets`.

#### Do Not Center Finger Opening
When generating grasp guesses, a first finger is placed, often close to the surface based on `standoff_distance` and `num_offsets`. Then the finger opening at each of the [`pinch_width_resolution`s](../components/gripper-definition.md#pinch-width-resolution) and the [`open_configuration`](../components/gripper-definition.md#open-configuration) is tested for collision. That means the distance from the object of the two fingers may be quite different. By default, the grasp is centered about the object by averaging these distances. If you do not want to have that object centering happen, use the **`--do_not_center_finger_opening`** flag.

**Note**: The `do_not_center_finger_opening` parameter does not work with the **`--use_acronym_grasp_guess`** flag.

#### Use ACRONYM Grasp Guess

GraspDataGen was initially created to reproduce the results from [ACRONYM: A Large-Scale Grasp Dataset Based on Simulation](https://arxiv.org/abs/2011.09584) in IsaacLab. To allow backward compatibility, you can use the algorithm used in that paper with the **`--use_acronym_grasp_guess`** flag. To keep full backward compatibility, you want to take a few other parameters into account:

- The ACRONYM grasp guess generation tests a single grasp opening. You can set this grasp opening specifically with an [open_configuration](../components/gripper-definition.md#open-configuration), and if one is not defined, then the widest gripper opening limit will be used. A value of **`--open_configuration '{"finger_joint": -0.31}'`** works well for the `onrobot_rg6` configuration.

- The ACRONYM algorithm centers the gripper around the object and keeps the gripper's distance from the object along the approach vector the same no matter how much space is between the gripper fingers and the object. This distance can be a problem if the fingers are far from the object and the gripper fingers move along the approach direction as they close, like many pinch grippers that are not antipodal in nature. This shortcoming of the original algorithm is still in place but can be addressed with the **`--correct_acronym_approach`** flag.

#### Correct ACRONYM Approach
The `correct_acronym_approach` parameter, when true, will use the different gripper openings from [`pinch_width_resolution`](../components/gripper-definition.md#pinch-width-resolution) and the [`open_configuration`](../components/gripper-definition.md#open-configuration) to test which opening is most closed but not intersecting the object; it will then move the gripper base away from the object appropriately so that the grasp will close on the object as near to the original random sample location as possible.

#### Max Guess Tries
The `max_guess_tries` parameter controls the maximum total number of attempts the grasp guess algorithm will make before giving up. The algorithm generates grasps in batches and continues until it has found the requested number of successful and failed grasps. If the total number of attempts reaches `max_guess_tries`, the algorithm will stop and report how many grasps were still needed.

- **Default value**: `100` - Allows up to 100 total attempts before giving up
- **Value of `0`**: Unlimited tries - the algorithm will continue indefinitely until the requested number of grasps are found

This parameter is useful for preventing infinite loops when generating grasps for objects that are difficult to grasp or when using restrictive parameters that make grasp generation challenging.

**Note**: Guessing will also stop if 10 consecutive attemps are made without a success.


#### Save Collision Mesh Folder
When debugging, it may be helpful to save the collision mesh for the object. This is especially helpful if inputting a USD mesh so you can ensure that the OBJ generated is correct. You can save the object mesh by specifying a folder name, e.g., **`--save_collision_mesh_folder=debug_mesh`**.
