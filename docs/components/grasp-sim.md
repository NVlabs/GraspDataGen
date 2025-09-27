# Grasp Validation with Simulation

The Grasp Simulation component validates grasp candidates through physics-based simulation using PhysX. It tests whether grasps can successfully hold objects under various disturbance forces and conditions.

Grasp validation can be run on user generated grasps in the Isaac Grasp format, or grasps from the grasp guess generator.

This document describes the grasp validation component and its parameters in detail. There are also some examples of its use:

- **[Verify Grasps with Simulation](../examples/grasp-sim.md)** - Verify user-defined grasps with simulation
- **[Complete Data Generation Pipeline](../workflows/datagen.md)** - Use the grasp validation component as a piece of a pipeline

## Overview

The grasp simulation process:

1. **Loads grasp candidates** from grasp guess generation or user input
2. **Sets up physics simulation** with gripper, object, and environment
3. **Executes grasp sequence** including close, and disturbance phases
4. **Evaluates grasp success** based on object contact with gripper fingers
5. **Outputs validated grasps** with success/failure classification

## Basic Usage

The easiest way to run grasp simulation is by inputting an Isaac Grasp YAML file generated with `grasp_guess.py`, using the **`--grasp_file`** parameter:

```bash

python scripts/graspgen/grasp_sim.py --grasp_file grasp_guess_data/onrobot_rg6/mug.yaml
```

## Command Line Arguments

This section explains the command-line arguments for the grasp validation through simulation component. The grasp simulator also uses shared [GraspDataGen Parameters](../api/parameter-system.md#graspdatagen-arguments), [Gripper Parameters](../components/gripper-definition.md#gripper-arguments), and [Object Parameters](../components/grasp-guess.md#object-arguments).

### Grasp Simulation Arguments

The `grasp_sim.py` component defines the core simulation parameters:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--grasp_file` | str | `"grasp_guess_data/onrobot_rg6/mug.yaml"` | Path to grasp file |
| `--max_num_envs` | int | `1024` | Maximum number of environments to spawn |
| `--max_num_grasps` | int | `0` | Maximum number of grasps to process (0 = all) |
| `--env_spacing` | float | `1.0` | Spacing between environments |
| `--fps` | float | `250.0` | Simulation FPS (physics timestep = 1/FPS) |
| `--force_magnitude` | float | `1.0` | Magnitude of disturbance force (N) |
| `--initial_grasp_duration` | float | `1.0` | Initial grasp duration in seconds |
| `--tug_sequences` | str | `"[[0.5, [0, 0, 1], 1.0], [0.5, [0, 2, 1], 1.0], [0.50, [0, -2, 1], 1.0], [0.50, [2, 0, 1], 1.0], [0.50, [-2, 0, 1], 1.0]]"` | JSON-formatted tug sequences |
| `--start_with_pregrasp_cspace_position` | bool | `True` | Start with pregrasp position |
| `--open_limit` | str | `""` | Open position limit ("lower" or "upper") |
| `--grasp_file_cspace_position` | str | `{}` | JSON mapping joint name (str) -> position (float) used when the grasp file lacks `cspace_position` |
| `--enable_ccd` | bool | `True` | Enable continuous collision detection |
| `--output_failed_grasp_locations` | flag | `False` | Output failed grasp locations |
| `--flip_input_grasps` | flag | `False` | Flip input grasps 180° around approach axis |
| `--disable_sim` | flag | `False` | Disable simulation |
| `--record_pvd` | flag | `False` | Record PVD files for debugging |
| `--debug_single_index` | int | `0` | Load single grasp for debugging |

## Grasp Simulation Parameters

The grasp simulation validates each candidate by running a short physics episode:
1. Initializes an Isaac Lab scene with the specified gripper and object.
2. For each grasp, sets the object pose from the grasp transform and seeds the gripper joints from either the pregrasp or grasp c-space (configurable).
3. Holds the initial configuration for stabilization, then commands a grasp close sequence.
4. Applies a series of tug forces to the object defined by `tug_sequences` and scaled by `force_magnitude`.
5. Determines success if the object and gripper fingers are in contact after the simulation.

The grasp simulation parameters control the physics validation process and testing conditions:

- **`grasp_file`** - Path to grasp file in Isaac Grasp format
- **`max_num_envs`** - Maximum number of parallel simulation environments
- **`max_num_grasps`** - Maximum number of grasps to process (0 = all)
- **`env_spacing`** - Spacing between parallel simulation environments
- **`fps`** - Simulation frames per second (determines physics timestep)
- **`force_magnitude`** - Magnitude of disturbance forces applied during testing
- **`initial_grasp_duration`** - Duration to hold grasp before applying disturbances
- **`tug_sequences`** - JSON array defining disturbance force sequences
- **`start_with_pregrasp_cspace_position`** - Start simulation from pregrasp position
- **`open_limit`** - Open position limit ("lower" or "upper")
- **`grasp_file_cspace_position`** - JSON dictionary mapping joint names to joint positions, used as a fallback when loading grasp files
- **`enable_ccd`** - Enable continuous collision detection for stability
- **`output_failed_grasp_locations`** - Save positions of failed grasps for analysis
- **`flip_input_grasps`** - Flip input grasps 180° around the approach axis (debug)
- **`disable_sim`** - Disable physics simulation (setup/debug only)
- **`record_pvd`** - Record PhysX PVD debug files (headless only)
- **`debug_single_index`** - Load and test only a single grasp index

### Parameter Details

#### Grasp File
Path to a YAML file containing grasp candidates in Isaac Grasp format. This file is typically generated by the grasp guess component or manually created.

##### When not generated by grasp_guess
If you author the YAML yourself, include at least the following:

- Top-level fields:
  - `object_file` (recommended) and `object_scale` (optional) – or pass via CLI
  - `gripper_file` – path to the gripper USD (can be overridden via CLI)
  - `finger_colliders` – list of two collider prim names on the gripper used to detect contact (CLI override available)
  - `open_limit` – either `"upper"` or `"lower"` to indicate which joint limit is the open pose for the gripper
  - Optional: `approach_axis` (int; default 2), `bite_point` (vec3), `bite_body_idx` (int)
- `grasps` – mapping from grasp name to grasp data. Each grasp requires:
  - `position`: [x, y, z]
  - `orientation`: `{ w: float, xyz: [qx, qy, qz] }` (unit quaternion)
  - Either `cspace_position` or `pregrasp_cspace_position`: dict mapping joint name (str) -> joint position (float)
  - Optional: `bite_point` and `pregrasp_bite_point` (vec3), `confidence` (float; 0.0 marks a failed grasp that will be skipped)

Minimal example:

```yaml
format: isaac_grasp
format_version: "1.0"
object_file: objects/mug.obj
object_scale: 1.0
gripper_file: bots/onrobot_rg6.usd
finger_colliders: [right_inner_finger, left_inner_finger]
open_limit: lower
grasps:
  grasp_0:
    position: [0.10, 0.00, 0.15]
    orientation: { w: 1.0, xyz: [0.0, 0.0, 0.0] }
    cspace_position: { finger_joint: -0.31 }
    bite_point: [0.0, 0.0, 0.0]
```

CLI overrides and fallbacks (useful when fields are missing in the YAML):

- `object_file`, `object_scale`, `gripper_file`, `finger_colliders`, and `open_limit {upper|lower}` have CLI overrides
- **`--start_with_pregrasp_cspace_position`** selects `pregrasp_cspace_position` instead of `cspace_position` from the file when present (default True).
- **`--grasp_file_cspace_position '{"finger_joint": 0.31}'`** provides a fallback mapping of joint name -> position if the file lacks `cspace_position`/`pregrasp_cspace_position`. Keys must match the gripper joint names.
- **`--flip_input_grasps`** flips grasps 180° around the approach axis (debug). Note: flip currently applies to **`--grasp_file`** inputs only.

Notes and caveats:

- If `open_limit` is missing, the sim will default to treating open as `lower` and print a warning. Set this explicitly for correct behavior.
- `confidence: 0.0` grasps in the YAML are filtered out.
- When converting OBJ/STL inputs to USD for sim, USD material/friction and collision approximation are controlled by the object parameters (see below) and can be set in the YAML or via CLI.

#### Max Num Envs
Maximum number of parallel simulation environments spawned per batch. Higher values increase throughput but also GPU memory use. Use `max_num_envs` to limit concurrency for complex environments.

#### Max Num Grasps
Limit the total number of grasps processed. `0` means use all grasps from the file/buffer.

#### Env Spacing
World spacing between parallel environments. This should purely be a visual tool, as environments should not collide with one another even if they are on top of each other. However, it's recommended to increase spacing if assets overlap across environment boundaries.

#### FPS
Simulation frames per second. The physics timestep is `1/FPS`. Higher FPS improves stability but reduces throughput.

#### Force Magnitude
After the gripper closes on the object, it will try to pull the object out of the gripper's grasp to make sure it has a secure hold. The `force_magnitude` parameter is the magnitude of that pull force expressed in multiples of gravity (Gs). The actual force is `Gs * |g| * object_mass`.

#### Initial Grasp Duration
Time in seconds to close on the object and hold it before tugging on the object. This may have to be increased if your gripper closes slowly. You can use **`--force_headed`** if you suspect your object is escaping the grasp because of a slow-closing gripper.

#### Tug Sequences
JSON list of `[duration, [x, y, z], scale]` entries. Direction `[x, y, z]` is normalized; force applied is `force_magnitude * scale` along that direction for `duration` seconds. Example:

```bash
--tug_sequences '[[0.5,[0,0,1],1.0],[0.5,[2,0,1],1.0]]'
```

#### Start With Pregrasp C-Space Position
When true, set the position of the joints using `pregrasp_cspace_position` if present; otherwise use `cspace_position`. This is helpful when using grasps that were generated by the [`grasp_guess`](../components/grasp-guess.md) component, because they supply the maximum open joint positions as pregrasp c-space positions and the most closed gripper position as the c-space positions.

This argument accepts explicit boolean values: `--start_with_pregrasp_cspace_position true|false` (also accepts `yes/no`, `y/n`, `1/0`). It can also be used as a flag without a value to imply `True`.

#### Open Limit
Grippers have upper and lower joint limits that indicate open and closed positions. Which limit maps to open vs closed is not standardized. `open_limit` indicates which joint limit is the open configuration of the gripper: `upper` or `lower`. This is required for correct close direction. You can use **`--force_headed`** if you suspect your gripper is letting an object go instead of grasping it.

#### Grasp File C-Space Position
Some grasp files only have transforms in them, but `grasp_sim` needs the joint positions as well.  The `grasp_file_cspace_position` parameter is a JSON dictionary mapping joint names to joint positions, used as a fallback when loading grasp files when the grasp entries lack `cspace_position`/`pregrasp_cspace_position`. Example:

```bash
--grasp_file_cspace_position '{"finger_joint": 0.31}'
```

#### Enable CCD
Continuous collision detection (CCD) prevents fast-moving objects from tunneling through colliders. This argument accepts explicit boolean values: `--enable_ccd true|false` (also accepts `yes/no`, `y/n`, `1/0`) and can be used as a flag without a value to imply `True`. CCD currently only works when running with **`--device cpu`**.

#### Output Failed Grasp Locations
When set, failed grasps are written back with their simulated post-disturbance pose so you can visualize failures. Their confidence is set to 0.0.

#### Flip Input Grasps
Rotate each input grasp 180° around the approach axis (debugging aid). Applies to file-based inputs. The flipped grasps will not be saved as the pregrasp positions if output is written.

#### Disable Sim
When the `disable_sim` flag is used, the simulation will not move the gripper, and there will be no collisions. This effectively skips physics stepping but still sets up the scene and joints. Useful for fast debugging of loading/joint positions.

**Note**: if you just want to visualize grasps with the USD gripper try [`grasp_display.py`](../tools/utility-tools.md#grasp-display).

#### Record PVD
Record [PhysX PVD capture](https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/VisualDebugger.html) (headless-only). Files are written under `/tmp/pvdout2/`.

#### Debug Single Index
Load and simulate only a single grasp index from the file (>0). Combine with `--max_num_envs 1` and `--force_headed` to view it interactively.

