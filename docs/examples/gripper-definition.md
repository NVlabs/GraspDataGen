# Running the [Gripper Definition](../components/gripper-definition.md) Component

This example demonstrates how to use the `create_gripper_lab.py` script to create gripper definitions and verify they work correctly with GraspDataGen. This is a practical guide that shows you what to expect when things go right and how to troubleshoot when they don't.

## Overview

The `create_gripper_lab.py` script is the main tool for creating gripper definitions. It:

1. **Loads a USD gripper file** and analyzes its structure
2. **Simulates the gripper** through various opening configurations
3. **Extracts kinematic data** including body transforms and joint limits
4. **Saves the definition** as an `.npz` file for use by other components
5. **Provides visual feedback** when run in headed mode

## Basic Usage

### Using Predefined Gripper Configurations

The easiest way to create a gripper definition is using a [predefined configuration](../api/parameter-system.md):

```bash
# Create gripper definition for OnRobot RG6
python scripts/graspgen/create_gripper_lab.py --gripper_config onrobot_rg6 --force_headed
```

This will:
- Load the OnRobot RG6 USD file from `bots/onrobot_rg6.usd`
- Use the predefined finger colliders and base frame
- Create a gripper definition with 8 pinch width resolutions (default)
- Save the result as `bots/onrobot_rg6.npz`

### Manual Configuration

You can also specify gripper parameters manually:

```bash
# Manual configuration example
python scripts/graspgen/create_gripper_lab.py \
    --gripper_file bots/robotiq_2f_85.usd \
    --finger_colliders right_inner_finger left_inner_finger \
    --base_frame base_link \
    --bite 0.0185 \
    --pinch_width_resolution 8 \
    --force_headed
```

## What Success Looks Like

### Expected Output

When the script runs successfully, you should see output like this:

```
Using gripper config 'onrobot_rg6'
  Overriding gripper_file: bots/onrobot_rg6.usd
  Overriding finger_colliders: ['right_inner_finger', 'left_inner_finger']
  Overriding base_frame: base_frame
Gripper file: bots/onrobot_rg6.usd
Gripper Created from bots/onrobot_rg6.usd
```

### Visual Verification

When running with **`--force_headed`**, you should see a "fan shape" visualization in Isaac Sim showing the gripper in different opening configurations:

<img src="../images/flower_shape_onrobot.png" alt="Gripper configuration example showing fan shape pattern" width="55%">

This visualization shows:
- **Multiple gripper instances** overlapping at the same origin
- **Different opening widths** from fully closed to fully open
- **Smooth transitions** between configurations (no gaps or jumps)

### Generated Files

The script creates several output files:

1. **`.npz` file** - Main gripper definition (e.g., `bots/onrobot_rg6.npz`)
2. **Debug output** - OBJ files and JSON transforms in `debug_output/` folder (if enabled) for use with the [debug visualization](../tools/utility-tools.md#debug-visualization) tool

## Troubleshooting Common Issues

### Issue 1: Missing or Incorrect Finger Colliders

**Symptoms:**
- Error: `ValueError: finger_colliders not found in gripper`
- Script fails to identify finger bodies

**Solution:**
1. Check the USD file structure using a USD viewer
2. Verify finger collider names match exactly:

```bash
# List available body names (if supported by your USD viewer)
# Or check the USD file directly for body names
```

3. Update the finger colliders parameter:

```bash
python scripts/graspgen/create_gripper_lab.py \
    --gripper_file bots/your_gripper.usd \
    --finger_colliders correct_finger_name_1 correct_finger_name_2 \
    --base_frame your_base_frame
```

### Issue 2: Incorrect Base Frame

**Symptoms:**
- Gripper appears in wrong position or orientation
- Transform errors in subsequent components

**Solution:**
1. Ensure the base frame is the root frame of the gripper
2. Verify the base frame has no transforms attached to it
3. Check that the base frame is at the origin (0,0,0)

### Issue 3: Poor Convergence

**Symptoms:**
- Convergence measurements show large position differences
- Gripper doesn't reach target positions
- Inconsistent opening widths

**Solution:**
1. Enable convergence measurement:

```bash
python scripts/graspgen/create_gripper_lab.py \
    --gripper_config onrobot_rg6 \
    --measure_convergence \
    --force_headed
```

2. Look for convergence output like:
```
Step 0: Max position difference: 0.00123456
Step 1: Max position difference: 0.00098765
Step 2: Max position difference: 0.00001234
```

3. If convergence is poor, increase iterations:

```bash
python scripts/graspgen/create_gripper_lab.py \
    --gripper_config onrobot_rg6 \
    --measure_convergence \
    --convergence_iterations 50 \
    --force_headed
```

### Issue 4: No Fan Pattern

**Symptoms:**
- Gripper configurations don't show smooth transitions
- Gaps or jumps between opening widths
- Inconsistent spacing

**Solution:**
1. Increase convergence iterations to ensure proper positioning
2. Check that the gripper has proper joint limits defined
3. Verify the USD file contains valid articulation data

```bash
# Increase iterations for better convergence
python scripts/graspgen/create_gripper_lab.py \
    --gripper_config onrobot_rg6 \
    --measure_convergence \
    --convergence_iterations 100 \
    --force_headed
```

### Issue 5: USD File Format Issues

**Symptoms:**
- Script fails to load USD file
- Missing articulation or collision data
- Unexpected behavior

**Solution:**
1. Ensure the USD file is flattened (no external references)
2. Verify the file contains proper articulation data
3. Check that collision meshes are properly defined
4. Make sure MassAPI is applied only to components with RigidBodyAPI or CollisionAPI.  The physics solver will ignore all other mass.

## Advanced Usage

### Custom Open Configuration

You can specify a custom open configuration to test specific joint positions:

```bash
# Test a specific finger joint position
python scripts/graspgen/create_gripper_lab.py \
    --gripper_config onrobot_rg6 \
    --open_configuration '{"finger_joint": -0.31}' \
    --force_headed
```

This will create an additional gripper configuration with the specified joint positions.

### Convergence Analysis

For detailed convergence analysis:

```bash
# Full convergence analysis with more iterations
python scripts/graspgen/create_gripper_lab.py \
    --gripper_config onrobot_rg6 \
    --measure_convergence \
    --convergence_iterations 100 \
    --force_headed
```

Look for convergence output that shows decreasing position differences over time.

### Debug Output

For advanced debugging of gripper creation issues, you can modify the `create_gripper_lab.py` code to save intermediate scene data at various points during the gripper creation process, and view that data with the [debug visualizer](../tools/utility-tools.md#debug-visualization)

#### How Debug Output Works

The gripper creator includes two debugging functions:

- **`save_scene_full(folder_name, gripper)`** - Saves all gripper bodies with their collision meshes and transforms
- **`save_scene(folder_name, env_idx, gripper_bodies, body_transforms)`** - Saves specific bodies at a particular environment state

#### Enabling Debug Output

To enable debug output, you need to uncomment the debugging calls in `create_gripper_lab.py`. Look for lines like:

```python
#self.save_scene_full(f'gripper', gripper)
#self.save_scene_full(f'sim_reset', gripper)
#self.save_scene_full(f'write_root', gripper)
#self.save_scene_full(f'write_joint_state', gripper)
#self.save_scene_full(f'scene_reset', gripper)
#self.save_scene_full(f'write_data_to_sim', gripper)
#self.save_scene_full(f'write_data_to_sim_final', gripper)
```

Uncomment the specific debug points you want to investigate:

```python
self.save_scene_full(f'write_joint_state', gripper)  # Debug joint positioning
self.save_scene_full(f'write_data_to_sim_final', gripper)  # Debug final state
```

#### Debug Output Files

When enabled, debug output creates a `debug_output/` folder containing:

```
debug_output/
├── write_joint_state/
│   ├── base_link.obj          # Base frame collision mesh
│   ├── base_link.json         # Base frame transform matrix
│   ├── right_inner_finger.obj # Right finger collision mesh
│   ├── right_inner_finger.json# Right finger transform matrix
│   ├── left_inner_finger.obj  # Left finger collision mesh
│   └── left_inner_finger.json # Left finger transform matrix
└── write_data_to_sim_final/
    ├── base_link.obj
    ├── base_link.json
    └── ...
```

Each folder represents a different point in the gripper creation process.

#### Viewing Debug Output

To visualize the debug output:

1. **OBJ Files** - Contains the collision mesh geometry for each gripper body
2. **JSON Files** - Contains the 4x4 transformation matrix for positioning each body

**Recommended: Use the [Visualization Tool](../tools/utility-tools.md#debug-visualization)**

GraspDataGen includes a dedicated visualization tool for debug output:

```bash
# Visualize all debug folders
python scripts/graspgen/tools/visualize_debug.py --path debug_output/

# Visualize a specific debug folder
python scripts/graspgen/tools/visualize_debug.py --path debug_output/write_joint_state/

# Include normal vectors and mesh edges for detailed analysis
python scripts/graspgen/tools/visualize_debug.py --path debug_output/ --draw-normals --draw-edges
```

The tool provides:
- **Web-based 3D visualization** using Meshcat
- **Automatic color coding** for different gripper bodies
- **Transform application** - Meshes are positioned correctly using JSON transforms
- **Multiple debug points** - Compare different stages side-by-side
- **Optional normals and edges** - For detailed mesh analysis

**Alternative Viewers:**

- **Blender** - Import OBJ files and manually apply transforms from JSON
- **MeshLab** - View OBJ geometry only (no transforms)
- **Custom Python Script** - Load and visualize both geometry and transforms

#### Example Debug Workflow

1. **Identify the issue** - Gripper links appear in wrong positions
2. **Enable debug output** - Uncomment relevant `save_scene_full()` calls
3. **Run gripper creation**:
   ```bash
   python scripts/graspgen/create_gripper_lab.py --gripper_config onrobot_rg6 --force_headed
   ```
4. **Visualize debug output**:
   ```bash
   python scripts/graspgen/tools/visualize_debug.py --path debug_output/
   ```
5. **Examine results** - Open the web browser to view the 3D visualization
6. **Compare debug points** - Check different stages (sim_reset, write_joint_state, etc.)
7. **Identify problems** - Look for misaligned bodies, incorrect joint positions, or transform issues
8. **Fix issues** - Adjust USD file, joint limits, or gripper parameters
9. **Repeat** - Re-run creation and visualization until issues are resolved

#### Common Debug Points

- **`sim_reset`** - Initial gripper state after simulation reset
- **`write_joint_state`** - After setting joint positions (check joint limits)
- **`write_data_to_sim`** - After writing data to simulation
- **`write_data_to_sim_final`** - Final state after convergence

#### Debugging Tips

1. **Start with final state** - Enable `write_data_to_sim_final` first
2. **Check joint limits** - Ensure joints can reach target positions
3. **Verify collision meshes** - Make sure OBJ files contain expected geometry
4. **Compare transforms** - Check if JSON transforms match expected positions
5. **Use multiple debug points** - Compare states before and after problematic steps

This debugging approach was essential during the development of the gripper creator and continues to be valuable when preparing new grippers for use with GraspDataGen.

## Verification Checklist

Before using your gripper with other GraspDataGen components, verify:

- [ ] Script runs without errors
- [ ] `.npz` file is created successfully
- [ ] Visual verification shows proper fan pattern
- [ ] Convergence measurements show small position differences (< 1e-6)
- [ ] Gripper configurations show smooth transitions
- [ ] No gaps or jumps in opening widths

## Next Steps

Once your gripper definition is working correctly, you can:

1. **Generate grasp guesses** using `grasp_guess.py`
2. **Validate grasps** using `grasp_sim.py`
3. **Run batch processing** using `datagen.py`

For more information about gripper parameters and configurations, see **[Gripper Definition](../components/gripper-definition.md)**.
