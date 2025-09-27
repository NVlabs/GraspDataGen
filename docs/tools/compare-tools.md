# Compare Tools

The compare tools help you analyze differences between grasp simulations and gripper configurations. These tools are essential for understanding how different grippers perform and for debugging variations in grasp results.

## Compare Grasp Sims

The `compare_grasp_sims.py` script compares two grasp simulation YAML files and analyzes their confidence values with a compact ASCII visualization.

### Overview

This tool compares grasp simulation results between two YAML files that were run with the same pregrasp parameters and object (or reports if objects differ). The gripper will typically be different between the files.

### Features

- **Visual Comparison**: ASCII art bars showing the distribution of confidence values
- **Comprehensive Analysis**: Categorizes grasps into different confidence patterns
- **Object Validation**: Warns if the objects being compared are different
- **Detailed Breakdown**: Optional verbose mode shows individual grasp names
- **Summary Statistics**: Percentage breakdowns and agreement metrics

### Usage

```bash
python scripts/graspgen/tools/compare_grasp_sims.py <file1.yaml> <file2.yaml> [--verbose]
```

### Examples

```bash
# Basic comparison
python scripts/graspgen/tools/compare_grasp_sims.py \
    grasp_sim_data/robotiq_2f_85/cuda.banana.yaml \
    grasp_sim_data/Robotiq_2F_85_msJul21/cuda.banana.yaml

# Detailed comparison with grasp names
python scripts/graspgen/tools/compare_grasp_sims.py \
    grasp_sim_data/robotiq_2f_85/cuda.banana.yaml \
    grasp_sim_data/Robotiq_2F_85_msJul21/cuda.banana.yaml \
    --verbose
```

### Output Categories

The tool categorizes grasps into the following groups:

- **✅ BOTH CONFIDENT (1.0)**: Grasps that have confidence 1.0 in both files
- **❌ BOTH FAILED (0.0)**: Grasps that have confidence 0.0 in both files  
- **⚠️ MISMATCHED**: Grasps where one file has confidence 1.0 and the other has 0.0
- **📄 ONLY IN FILE**: Grasps that exist in only one of the files

### Sample Output

```
================================================================================
                    GRASP SIMULATION COMPARISON
================================================================================

📁 File 1: cuda.banana.yaml
📁 File 2: cuda.banana.yaml

📊 TOTAL GRASPS: 1024

────────────────────────────────────────────────────────────
                    CONFIDENCE COMPARISON
────────────────────────────────────────────────────────────

✅ BOTH CONFIDENT (1.0): 656
   █████████████████████████░░░░░░░░░░░░░░░ 64.1%

❌ BOTH FAILED (0.0): 154
   ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 15.0%

⚠️  MISMATCHED - cuda.banana.yaml confident, cuda.banana.yaml failed: 2
   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.2%

⚠️  MISMATCHED - cuda.banana.yaml failed, cuda.banana.yaml confident: 212
   ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 20.7%

────────────────────────────────────────────────────────────
                    SUMMARY TABLE
────────────────────────────────────────────────────────────
Category                            Count    Percentage  
────────────────────────────────────────────────────────────
Both Confident (1.0)                656      64.1        %
Both Failed (0.0)                   154      15.0        %
File1 1.0, File2 0.0                2        0.2         %
File1 0.0, File2 1.0                212      20.7        %
────────────────────────────────────────────────────────────
TOTAL AGREEMENT                     810      79.1        %
TOTAL DISAGREEMENT                  214      20.9        %
============================================================
```

### File Format

The script expects YAML files in the Isaac Grasp format with the following structure:

```yaml
format: isaac_grasp
object_file: objects/banana.obj
gripper_file: bots/robotiq_2f_85.usd
grasps:
  grasp_0_4:
    confidence: 1.0
    position: [...]
    orientation: {...}
    # ... other grasp data
```

### Notes

- The script automatically detects if the objects being compared are different and warns the user
- Confidence values are expected to be either 0.0 or 1.0
- The tool handles cases where grasps exist in only one of the files
- The ASCII art bars are scaled to show relative proportions

## Compare Grippers

The `compare_grippers.py` script will compare two gripper NPZ configuration files and analyze their differences.

### Overview

This tool compares gripper configuration files (NPZ format) created by the gripper definition component. It's useful for understanding differences between gripper configurations and ensuring consistency.

### Features

- **Detailed Comparison**: Compares all parameters in gripper configuration files
- **Array Analysis**: Handles numpy arrays with proper tolerance for floating-point comparisons
- **Verbose Output**: Optional detailed output showing all differences
- **Type Handling**: Properly handles different data types and shapes

### Usage

```bash
python scripts/graspgen/tools/compare_grippers.py <file1.npz> <file2.npz> [--verbose]
```

### Examples

```bash
# Basic comparison
python scripts/graspgen/tools/compare_grippers.py \
    gripper_configs/robotiq_2f_85.npz \
    gripper_configs/robotiq_2f_85_modified.npz

# Detailed comparison with verbose output
python scripts/graspgen/tools/compare_grippers.py \
    gripper_configs/robotiq_2f_85.npz \
    gripper_configs/robotiq_2f_85_modified.npz \
    --verbose
```

### Sample Output

```
Comparing gripper configurations:
File 1: gripper_configs/robotiq_2f_85.npz
File 2: gripper_configs/robotiq_2f_85_modified.npz

✅ Configuration files are IDENTICAL

All parameters match between the two gripper configurations.
```

Or for files with differences:

```
Comparing gripper configurations:
File 1: gripper_configs/robotiq_2f_85.npz
File 2: gripper_configs/robotiq_2f_85_modified.npz

❌ Configuration files are DIFFERENT

Differences found:

bite: Arrays differ:
  File 1: 0.02 (scalar array, dtype: float64)
  File 2: 0.025 (scalar array, dtype: float64)

finger_colliders: Arrays differ:
  File 1: ['right_finger' 'left_finger'] (shape: (2,), dtype: object)
  File 2: ['right_finger_collider' 'left_finger_collider'] (shape: (2,), dtype: object)
```

### File Format

The script expects NPZ files created by the gripper definition component, which contain:

- **bite**: Bite depth parameter
- **finger_colliders**: Names of finger collider bodies
- **base_frame**: Base frame name
- **gripper_file**: Path to the gripper USD file
- **body_names**: Array of body names
- **body_com_pos_w**: Body center of mass positions
- **joint_names**: Array of joint names
- **joint_positions**: Joint position limits
- **collision_meshes**: Collision mesh data

### Use Cases

1. **Configuration Validation**: Ensure gripper configurations are consistent across different runs
2. **Parameter Comparison**: Compare different gripper parameter settings
3. **Debugging**: Identify differences when grasp results vary unexpectedly
4. **Quality Assurance**: Verify that gripper configurations meet specifications

### Tips

- Use `--verbose` flag to see detailed differences when files don't match
- The tool handles floating-point tolerances automatically
- Large arrays are truncated in output for readability
- The tool provides clear indication of whether files are identical or different 


