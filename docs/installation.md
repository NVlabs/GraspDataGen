# Installation Guide

This guide covers how to install and set up GraspDataGen for different environments.

## Prerequisites

GraspDataGen has the same system requirements as [IsaacLab](https://github.com/isaac-sim/IsaacLab). Please refer to the IsaacLab installation documentation for detailed system requirements and prerequisites.

**Additional Requirements:**
- **Meshcat** (optional): Required only if you want to visualize grasps. This is automatically included in the Docker container.

## Installation Methods

### Method 1: Docker (Recommended)

The easiest way to get started with GraspDataGen is using the provided Dockerfile, which includes all dependencies and IsaacLab.

Run the following commands from the GraspDataGen source folder.

```bash
# Pull the IsaacLab Docker image and install the GraspDataGen code
./docker/build.sh
./docker/run.sh --grasp_dataset <path_to_write_grasps_to> --object_dataset <path_to_objects>
```

Once inside the container, GraspDataGen is ready to use.

#### ./docker/run.sh parameters

All parameters are optional. The script accepts the following arguments:

- `--grasp_dataset <path>` (optional): Path to the grasp dataset directory where all output from the scripts will be written. If not provided, the current folder (`.`) will be used to create folders like `grasp_guess_data`, etc.

- `--object_dataset <path>` (optional): Path to the object dataset directory. This will be prefixed onto the `object_root` in the datagen.py example and used as `object_dataset` in the graspgen example.  If not set, the default will be used. Can be overridden by the CLI args.

- `--grasp_gen_code <path>` (optional): Path to the GraspGen code directory. This is only used in the graspgen.py example and is used to add the gripper config to the right place in the GraspGen config folder. If not defined, it will be set to the grasp_dataset folder and users will need to move it to the `<GraspGen>/config/grippers` folder themselves.

- `--graspdatagen_code <path>` (optional): Path to the GraspDataGen code directory. This is mostly used for debugging and when users wish to make live updates to the code in their container.

**Test Installation:**

Once the docker container is build, you can test the instalation by generating some grasps and then validating a few of them on an object already in the GraspDataGen repo.

```bash
cd GraspDataGen

./docker/run.sh

# You should not be inside a docker container in the /code/GraspDataGen folder with a # prompt.
# Run the command to generate the guesses with the Robotiq_2f_85 gripper.
python scripts/graspgen/grasp_guess.py --gripper_config robotiq_2f_85

# That command will open up a headless IsaacLab session to build the gripper definition, and then it will generate 1024 grasps.
# Run the command the validate the grasps, and visualize the simulation as it runs.
python scripts/graspgen/grasp_sim.py --grasp_file /grasp_dataset/grasp_guess_data/robotiq_2f_85/mug.yaml --force_headed --max_num_grasps 16
```

**Note**: If you are having trouble opening an IsaacLab GUI, remove the `--force_headed` flag from the last command above.  You can then visualize the grasps with meshcat using the following command:

```bash
# Start a meshcat server to visualise the results, instead of watching the simulation in the IsaacLab UI
python scripts/graspgen/tools/visualize_grasp_data.py --grasp-paths /grasp_dataset/grasp_sim_data/robotiq_2f_85/mug.yaml
```


### Method 2: With Existing IsaacLab pip Installation

If you already have IsaacLab installed in your Python environment, you can use GraspDataGen directly with Python:

```bash
# Navigate to your GraspDataGen directory
cd /path/to/GraspDataGen

# Run scripts directly with Python
python scripts/graspgen/datagen.py --gripper_config onrobot_rg6 --object_scales_json objects/datagen_example.json
```

### Method 3: IsaacLab Integration with Symbolic Links

You can integrate GraspDataGen with your IsaacLab source installation from the [IsaacLab](https://github.com/isaac-sim/IsaacLab) repo using symbolic links:

```bash
# Navigate to your IsaacLab directory
cd /path/to/IsaacLab

# Create symbolic links to GraspDataGen components
ln -s /path/to/GraspDataGen/bots .
ln -s /path/to/GraspDataGen/objects .
cd scripts
ln -s /path/to/GraspDataGen/scripts/graspgen .
```

After creating the symbolic links, you can use IsaacLab's launcher:

```bash
# Use IsaacLab's Python environment
./isaaclab.sh -p scripts/graspgen/datagen.py \
    --gripper_config onrobot_rg6 \
    --object_scales_json objects/datagen_example.json
```

### Getting Help

- Check the [IsaacLab documentation](https://github.com/isaac-sim/IsaacLab) for IsaacLab-specific issues
- Review the [GraspDataGen documentation overview](README.md) for usage examples
- Check the [component documentation](components/) for detailed parameter information

## Next Steps

Once installation is complete, you can:

1. Start with the [Quick Start Guide](README.md#quick-start) in the main documentation
2. Explore [component examples](examples/) to understand individual components
3. Follow [workflow guides](workflows/) for complete pipelines
4. Use [utility tools](tools/) for analysis and debugging
