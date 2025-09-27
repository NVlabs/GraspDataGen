
# GraspDataGen

GraspDataGen is a standalone data generation tool but it can also be used to generate data for training new [Grasp Gen](https://github.com/NVlabs/GraspGen) models. It is designed to create collision-free, geometrically plausible grasps for triangle mesh objects and USD grippers, then validate them through physics simulation.

## Overview

The system consists of three main components:

1. **[Gripper Definition](docs/components/gripper-definition.md)** - Reads USD gripper files and extracts parameters needed for grasp generation
2. **[Grasp Guess Generation](docs/components/grasp-guess.md)** - Creates geometrically plausible grasps for objects
3. **[Grasp Simulation](docs/components/grasp-sim.md)** - Validates grasps through physics simulation using PhysX

Each component can be used independently or as part of a complete workflow.

## Installation

For detailed installation instructions, see the **[Installation Guide](docs/installation.md)**. The guide covers:

- **Docker installation** (recommended) - Includes all dependencies and IsaacLab
- **Integration with existing IsaacLab** - Use with your current IsaacLab setup

## Using Datagen

The easiest way to get started using the full GraspDataGen pipeline is by using predefined gripper configurations (**see**: `scripts/graspgen/gripper_configurations.py`) and the `datagen.py` script:

```bash
python scripts/graspgen/datagen.py \
    --gripper_config onrobot_rg6 \
    --object_scales_json objects/datagen_example.json \
    --object_root objects \
    --num_grasps 1024 \
    --max_num_envs 256
```

The `datagen.py` script is an example of using the 3 components together to generate and verify grasps for multiple objects defined in a JSON file.

**Note**: The `datagen.py` script handles objects differently: it loads multiple objects from a JSON file instead of a single object file. In contrast, individual components such as `grasp_guess.py` and `grasp_sim.py` use the **`--object_file`** argument for single objects.

### Available Gripper Configurations

The system includes predefined configurations for common grippers:

- **`robotiq_2f_85`** - Robotiq 2F-85 parallel gripper
- **`onrobot_rg6`** - OnRobot RG6 gripper  
- **`franka_panda`** - Franka Panda gripper

Use **`--gripper_config <name>`** to configure all gripper parameters based on the user definitions in `scripts/graspgen/gripper_configurations.py`.

You can override any configuration parameter by providing it explicitly on the command line:

```bash
# Use onrobot_rg6 configuration but with custom parameters
python scripts/graspgen/datagen.py \
    --gripper_config onrobot_rg6 \
    --gripper_file bots/custom_gripper.usd \
    --bite 0.02 \
    --object_scales_json objects/datagen_example.json \
    --object_root objects
```

This will use the onrobot_rg6 configuration as a base but override the gripper file and bite depth with your custom values.

**Note**: Read more about the **[parameter override system](docs/api/parameter-system.md#parameter-override-systems)** that applies to all components of GraspDataGen.

Alternatively, you can add any new gripper configuration needed by adding to `GRIPPER_CONFIGS` in `scripts/graspgen/gripper_configurations.py`:

```python
    'Robotiq_2F_85_msJul21': {
        'gripper_file': 'bots/Robotiq_2F_85_msJul21.usd',
        'finger_colliders': ['right_inner_finger', 'left_inner_finger'],
        'base_frame': 'base_link',
        'bite': 0.0185,  # half of 37mm
        'convergence_iterations': 172,
    },
```

This custom config was used when experimenting with a gripper that had a stiff physics setup and needed more **`--convergence_iterations`** to get a proper gripper definition.

#### Datagen Documentation
- **[Complete Data Generation Pipeline](docs/workflows/datagen.md)** - Generate and verify grasps for a list of objects and scales

## Using each component individually

This section describes the three main components of the GraspDataGen code base in the simplest standalone mode, and supplies documentation for a more detailed explanation and example.

### Gripper definition

The gripper definition module is used to read the USD of a gripper and prepare it for grasp generation and validation. The minimum you need to create a gripper definition is a USD file, and the names of the finger and base prims, and you can create the definition with the `create_gripper_lab.py` script.

```bash
python scripts/graspgen/create_gripper_lab.py \
  --gripper_file bots/onrobot_rg6.usd \
  --finger_colliders right_inner_finger left_inner_finger \
  --base_frame base_frame
```

#### Documentation
- **[Overview and parameters](docs/components/gripper-definition.md)** - Details on what the component does and its parameters
- **[Gripper Setup Example](docs/examples/gripper-setup.md)** - Describes in detail how the Robotiq 85 was prepared for use in GraspDataGen
- **[Running the Gripper Definition Component](docs/examples/gripper-definition.md)** - Visually verify your gripper will work with the simulation

### Generate grasp guesses

You can generate geometrically plausible, collision-free grasps with the `grasp_guess.py` script.

```bash
python scripts/graspgen/grasp_guess.py \
    --gripper_config onrobot_rg6 \
    --object_file objects/banana.obj
```

#### Documentation
- **[Overview and parameters](docs/components/grasp-guess.md)** - Details on what the component does and its parameters
- **[Single Object Example](docs/examples/grasp-guess.md)** - Create collision free grasps for a single object


### Validate grasps through simulation

If the grasps you want to validate have been generated by the grasp_guess module, then the only parameter you need to set when running grasp_sim is **`--grasp_file`**. The object and gripper settings will be gathered from the grasp file.

```bash
python scripts/graspgen/grasp_sim.py \
    --grasp_file grasp_guess_data/onrobot_rg6/banana.yaml
```

#### Documentation
- **[Overview and parameters](docs/components/grasp-sim.md)** - Details on what the component does and its parameters
- **[Verify Grasps with Simulation](docs/examples/grasp-sim.md)** - Verify user defined grasps with simulation

## Documentation

📚 **Comprehensive documentation is available in the [docs/](docs/) directory:**

- **[Documentation Overview](docs/README.md)** - Complete guide to the system
- **[Installation Guide](docs/installation.md)** - Docker installation and IsaacLab integration
- **[Component Documentation](docs/components/)** - Detailed guides for each component
  - [Gripper Definition](docs/components/gripper-definition.md)
  - [Grasp Guess Generation](docs/components/grasp-guess.md)
  - [Grasp Simulation](docs/components/grasp-sim.md)
- **[Workflow Documentation](docs/workflows/)** - Complete workflow guides
  - [Batch Data Generation](docs/workflows/datagen.md)
  - [Using with Grasp Gen](docs/workflows/graspgen.md)
- **[Tools Documentation](docs/tools/)** - Utility tools for analysis, debugging, and data processing
  - [Tools Overview](docs/tools/README.md)
  - [Compare Tools](docs/tools/compare-tools.md) - Compare grasp simulations and gripper configurations
  - [Debug Tools](docs/tools/utility-tools.md#debug-visualization) - Debugging and troubleshooting tools
  - [Utility Tools](docs/tools/utility-tools.md) - Data processing and workflow management
- **[Examples](docs/examples/)** - Examples to model your own workflow after.
  - [Gripper Setup](docs/examples/gripper-setup.md) - Create a gripper definition and check it visually
  - [Running the Gripper Definition Component](docs/examples/gripper-definition.md) - Visually verify your gripper will work with the simulation
  - [Single Object Grasp Guess Generation](docs/examples/grasp-guess.md) - Create collision-free grasps for a single object
  - [Verify Grasps with Simulation](docs/examples/grasp-sim.md) - Verify user-defined grasps with simulation
- **[API Reference](docs/api/)** - Configuration and technical details
  - [Args and Parameters](docs/api/parameter-system.md)
