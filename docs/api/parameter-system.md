# Parameter System

The GraspDataGen system uses a comprehensive parameter system that allows components to share common arguments while maintaining clear separation of concerns. This document describes the parameter system architecture and the shared arguments used across all components.

The parameters are defined in each component, and there are command-line argument equivalents for each parameter. The parameter default values, defined at the top of each component, are used if no override or command-line argument is given. The default values can be overridden by adding them to new or existing components or workflows that use the arguments.

**Note**: Use **`--help`** or **`-h`** on any script to see the parameters available for each module.

## GraspDataGen Arguments

There are a few arguments that are always present. Not all components use IsaacLab, but most do. When IsaacLab is used, all IsaacLab and IsaacSim arguments will also be present. The default values for these are hard-coded and cannot be overridden by the default override system.

These GraspDataGen arguments are shared across all components and control Isaac Lab and IsaacSim execution:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | `"cuda"` | Device to use for computation (cuda/cpu) |
| `--headless` | flag | `True` | Run Isaac Lab in headless mode |
| `--force_headed` | flag | `False` | Force Isaac Lab to run in headed mode |
| `--wait_for_debugger_attach` | flag | `False` | Wait for debugger to attach before continuing execution |

### GraspDataGen Argument Details

#### Device Selection
- **`--device`** - Controls which device to use for computation. Use `"cuda"` for GPU acceleration or `"cpu"` for CPU-only execution.

The `device` parameter originated as a command-line argument in IsaacLab and, when used, will be passed to all instances of IsaacLab that need it. GraspDataGen also uses the `device` parameter for components that don't actually use IsaacLab, like grasp guess generation. In that case, `device` is used when running Warp kernels.

#### Display Mode
- **`--headless`** - Default behavior runs IsaacLab without a display window. The `headless` parameter comes from IsaacSim.
- **`--force_headed`** - Overrides headless mode to force a display window; useful for debugging, visualization, and when you're not sure what the `headless` parameter is set to but you know you want a display window.

#### Debugging
- **`--wait_for_debugger_attach`** - Pauses execution until a debugger attaches, enabling step-by-step debugging of the code

This parameter only works when an instance of IsaacLab is created, and the pause happens right after the `simulation_app` is created.

## Parameter System Organization

The parameter system is organized into logical groups that are shared across components:

- **[GraspDataGen Arguments](#graspdatagen-arguments)** - Common arguments for all of GraspDataGen's configuration and execution
- **[Gripper Arguments](../components/gripper-definition.md#gripper-arguments)** - Gripper-specific configuration parameters, used to create the gripper definition, create grasp guesses, and grasp simulation
- **[Object Arguments](../components/grasp-guess.md#object-arguments)** - Object processing and collision approximation parameters, used by grasp guess generation and grasp simulation
- **Component-Specific Arguments** - Arguments unique to each component
  - **[Gripper Definition Creation](../components/gripper-definition.md#gripper-creation-arguments)** - Parameters used when creating the gripper definition with `create_gripper_lab.py`
  - **[Grasp Guess Generation](../components/grasp-guess.md#grasp-guess-generation-arguments)** - Parameters controlling how potential grasp poses are generated and sampled in `grasp_guess.py`
  - **[Grasp Validation with Simulation](../components/grasp-sim.md#grasp-simulation-arguments)** - Parameters for physics simulation settings and validation criteria in `grasp_sim.py`

## Parameter Override Systems

Each component in GraspDataGen has a set of parameters that affect it. To keep each piece self-contained, those parameters and their defaults are defined at the top of each component, and there is a layered system that can override those defaults.

Parameters are applied in the following order (lowest to highest priority):

1. **System defaults** - Fallback values
1. **Component defaults** - Hard-coded default values at the top of any module
1. **Gripper configurations** - Values from `--gripper_config`
1. **Command-line arguments** - Explicitly specified values

### System Defaults

The default values for the parameters are stored at the top of the Python modules for each component and have the structure `default_<parameter>`. So the `bite` parameter used in the gripper module has a default value of `default_bite`, defined at the top of `gripper.py`.

### Component Defaults

The component default system provides a mechanism to change the default values of each component. This is useful when you want to create your own module with your own defaults, or when you want to debug or run existing modules without typing a long string of command-line arguments. During development, this was also very helpful for creating standalone demos, bug repros, or iterating on a particular parameter without having to change the source code of the module that the parameter was in.

Each module has a pair of add/collect args functions that can be used when you create a parser, and they are used together with Python's `globals()` function to ensure that the default values at the top of a module are used instead of the system defaults (see: `datagen.py` pipeline module for an example).

The following code would create a module that uses the parameters in the `grasp_guess` module and overrides the value for the `bite` parameter system default value (`default_bite = 0.01` defined at the top of the `gripper` module) with its own component-level default value.

```python
default_bite = 0.042

parser = argparse.ArgumentParser(description="My Cool Data Generation Pipeline.")
add_grasp_guess_args(parser, globals(), **collect_grasp_guess_args(globals()))
args = parser.parse_args()
```

### Gripper Configurations

The override system includes predefined gripper configurations that automatically set all gripper parameters, as well as any other parameter put into the config:

```python
GRIPPER_CONFIGS = {
    'robotiq_2f_85': {
        'gripper_file': 'bots/robotiq_2f_85.usd',
        'finger_colliders': ['right_inner_finger', 'left_inner_finger'],
        'base_frame': 'base_link',
        'bite': 0.0185,
        'convergence_iterations': 200,
    },
    'onrobot_rg6': {
        'gripper_file': 'bots/onrobot_rg6.usd',
        'finger_colliders': ['right_inner_finger', 'left_inner_finger'],
        'base_frame': 'base_frame',
    },
    'franka_panda': {
        'gripper_file': 'bots/franka_panda.usd',
        'finger_colliders': ['panda_rightfinger', 'panda_leftfinger'],
        'base_frame': 'panda_hand',
    },
}
```

When using `--gripper_config <name>`, the system automatically:

1. **Loads the configuration** from predefined settings
2. **Applies all parameters** (gripper_file, finger_colliders, base_frame, etc.)
3. **Overrides defaults** with configuration-specific values
4. **Preserves command-line overrides** if explicitly provided



### Command-Line Arguments

The command-line arguments have the final say in what a parameter will be.  The parameter name and the command-line argument string are the same. For example, the `bite` parameter corresponds to the command-line argument **`--bite`**.
