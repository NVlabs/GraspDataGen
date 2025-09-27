# Batch Data Generation

The Batch Data Generation workflow example, `datagen.py`, processes multiple objects to create training datasets for machine learning systems. This workflow is designed for large-scale data generation and can be customized for specific training requirements.

## Overview

The batch generation workflow:

1. **Processes multiple objects** from a JSON configuration file
2. **Applies consistent settings** across all objects
3. **Generates grasp candidates** for each object
4. **Validates grasps** through physics simulation
5. **Creates organized datasets** with success/failure labels

## Usage

### Basic Batch Processing

To generate grasps for a set of objects, you need the base folder where you keep the objects (**`--object_root`**) and a JSON file with scales and object file names relative to that root.

### JSON Configuration Format

The `object_scales_json` file should contain object paths (relative to `object_root`) and their scales:

```json
{
    "objects/banana.obj": 0.75,
    "objects/apple.obj": 0.8,
    "objects/orange.usd": 1.2,
    "objects/peach.stl": 0.5
}
```

### Directory Structure

```
object_root/
├── objects/
│   ├── banana.obj
│   ├── apple.obj
│   ├── orange.usd
│   └── peach.stl
```

### Standard Dataset Generation Example

```bash
python scripts/graspgen/datagen.py \
    --gripper_config onrobot_rg6 \
    --object_scales_json objects/datagen_example.json \
    --object_root objects \
    --num_grasps 1024 
```

This would create the Isaac Grasp data files:

```
datagen_sim_data/
└── onrobot_rg6/
    ├── banana.0.75.yaml          # Banana object at 0.75 scale
    ├── banana.1.0.yaml           # Banana object at 1.0 scale
    ├── Sphere.0.05.yaml          # Sphere object at 0.05 scale
    └── threelayer.0.005.yaml     # Threelayer object at 0.005 scale
```


### Multi-Gripper Dataset

```bash
# Generate dataset for multiple grippers
gripper_configs=("robotiq_2f_85" "onrobot_rg6" "franka_panda")
for gripper_config in "${gripper_configs[@]}"; do
    python scripts/graspgen/datagen.py \
        --gripper_config "$gripper_config" \
        --object_scales_json objects/datagen_example.json \
        --object_root objects \
        --num_grasps 1024 
done
```

### Custom Output Directories

```bash
python scripts/graspgen/datagen.py \
    --gripper_config onrobot_rg6 \
    --object_scales_json objects/datagen_example.json \
    --object_root objects \
    --sim_output_folder custom_sim_data \
    --guess_output_folder custom_guess_data \
    --num_grasps 1024 \
    --overwrite_existing
```
