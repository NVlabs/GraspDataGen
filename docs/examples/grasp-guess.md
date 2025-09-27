# Single-Object Grasp Guess Example

This example shows how to generate collision-free grasp guesses for a single object using `grasp_guess.py`. It assumes you have a working gripper configuration and an object mesh.

## Prerequisites

- A valid gripper USD and configuration (e.g., `onrobot_rg6`)
- An object mesh file in USD, OBJ, or STL format

## Quick Start

```bash
python scripts/graspgen/grasp_guess.py \
    --gripper_config onrobot_rg6 \
    --object_file objects/banana.obj \
    --num_grasps 1024
```

Notes:
- If the gripper definition `.npz` does not exist yet, it will be created automatically.
- OBJ/STL inputs are used directly. USD inputs are converted to OBJ internally for grasp guessing.

## Common Options

- `--seed <int>`: Make results reproducible.
- `--num_orientations <int>`: Try multiple rotations per surface point.
- `--percent_random_guess_angle <float 0..1>`: Mix of axis-aligned vs random rotations.
- `--standoff_distance <float>` and `--num_offsets <int>`: Control initial finger placement and retries.

Example with more control:

```bash
python scripts/graspgen/grasp_guess.py \
    --gripper_config onrobot_rg6 \
    --object_file objects/banana.obj \
    --seed 123 \
    --num_grasps 2048 \
    --num_orientations 8 \
    --percent_random_guess_angle 0.25 \
    --standoff_distance 0.0015 \
    --num_offsets 24
```

## Output

Results are written in Isaac Grasp YAML format, typically under `grasp_guess_data/<gripper>/object.yaml`. Each grasp entry contains a transform and joint `cspace_position`/`pregrasp_cspace_position` values. See the component docs for details.

## Visualization

- Web-based, fast: `visualize_grasp_data.py` (recommended)

```bash
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths grasp_guess_data/onrobot_rg6/banana.yaml \
    --object-root .
```

- Full USD gripper: `grasp_display.py` (slower with many grasps)

```bash
python scripts/graspgen/grasp_display.py \
    --grasp_file grasp_guess_data/onrobot_rg6/banana.yaml \
    --max_num_grasps 100
```
