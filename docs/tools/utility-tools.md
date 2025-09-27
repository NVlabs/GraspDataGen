# Utility Tools

The utility tools provide essential functionality for data processing, visualization, and workflow management in GraspDataGen. These tools help you manipulate data, create visualizations, and automate common tasks.

## Debugging with debugpy

GraspDataGen supports debugging using debugpy, allowing you to debug scripts running inside Docker containers with your IDE.

### Overview

The Docker container includes debugpy pre-installed, enabling remote debugging of GraspDataGen scripts. This is particularly useful for development and troubleshooting complex grasp generation workflows.

### Features

- **Remote Debugging**: Debug scripts running inside Docker containers
- **IDE Integration**: Works with VS Code, PyCharm, and other IDEs that support debugpy
- **Live Code Changes**: Use `--graspdatagen_code` to mount local code for immediate debugging

### Usage

#### 1. Start Container with Code Mounting

```bash
# Mount local code for live debugging
./docker/run.sh --graspdatagen_code .
```

#### 2. Run Script with Debugpy

Inside the container, run your script with debugpy:

```bash
cd /code/GraspDataGen
python -m debugpy --wait-for-client --listen 0.0.0.0:5678 ./scripts/graspgen/datagen.py --object_scales_json objects/datagen_example.json --object_root objects --gripper_file bots/onrobot_rg6.usd --num_grasps 10 --device cpu
```

#### 3. Connect from IDE

Connect your IDE's debugger to `localhost:5678`:

- **VS Code**: Use "Python: Remote Attach" configuration
- **PyCharm**: Use "Attach to Local Process" with port 5678
- **Other IDEs**: Configure remote debugging to localhost:5678

## Grasp Display

The `grasp_display.py` tool displays grasps directly in IsaacLab by spawning a single object and rendering the full gripper USD at each grasp location.

### Overview

This tool loads a grasp YAML (from Grasp Guess or otherwise), spawns the referenced object once, and then replicates the gripper articulation at each grasp pose. Joint positions are initialized from either the grasp's `cspace_position` or the `pregrasp_cspace_position`.

### Features

- **Full USD gripper rendering**: Shows the actual gripper model at each grasp pose
- **Single object instance**: Displays one object mesh with proper scaling
- **Subset selection**: Limit by `--max_num_grasps` or explicit `--grasp_indices`
- **Pregrasp option**: Start joints from `pregrasp_cspace_position`

> Note: Displaying many grippers at the same time can take a while due to scene setup and rendering overhead in IsaacLab. Prefer using `--max_num_grasps` or specific `--grasp_indices` when exploring large datasets.

### Usage

```bash
python scripts/graspgen/grasp_display.py \
    --grasp_file <path/to/grasp.yaml> \
    [--max_num_grasps <N>] \
    [--grasp_indices i j k] \
    [--start_with_pregrasp_cspace_position] \
    [--device cuda]
```

### Arguments

- `--grasp_file`: Path to a grasp YAML file (required)
- `--max_num_grasps`: Maximum number of grasps to display (0 uses all)
- `--grasp_indices`: Explicit list of grasp indices to display
- `--start_with_pregrasp_cspace_position`: Initialize from `pregrasp_cspace_position` instead of `cspace_position`
- IsaacLab runtime options (added automatically):
  - `--device`: Compute device, e.g., `cuda` or `cpu`
  - `--wait_for_debugger_attach`: Delay startup for attaching a debugger

### Examples

```bash
# Display all grasps (may be slow if the dataset is large)
python scripts/graspgen/grasp_display.py \
    --grasp_file grasp_guess_data/onrobot_rg6/banana.yaml

# Display first 50 grasps
python scripts/graspgen/grasp_display.py \
    --grasp_file grasp_guess_data/onrobot_rg6/banana.yaml \
    --max_num_grasps 50

# Display specific grasp indices with pregrasp joint positions
python scripts/graspgen/grasp_display.py \
    --grasp_file grasp_guess_data/onrobot_rg6/banana.yaml \
    --grasp_indices 0 5 42 101 \
    --start_with_pregrasp_cspace_position
```

## Plot Gripper 3D

The `plot_gripper_3d.py` tool creates a 3D visualization of gripper body positions from Isaac Lab gripper creation simulation.

### Overview

This tool creates 3D visualizations of gripper configurations, showing the positions of all gripper bodies across different configurations. It's useful for understanding gripper geometry and validating configurations.

### Features

- **3D Visualization**: Interactive 3D plots of gripper body positions
- **Trajectory Display**: Show trajectories connecting body positions
- **Finger Highlighting**: Highlight finger bodies with different colors
- **Configuration Comparison**: Compare different gripper configurations
- **Export Options**: Save plots as image files

### Usage

```bash
python scripts/graspgen/tools/plot_gripper_3d.py \
    --gripper_config <gripper_config> \
    [--save_plot] \
    [--plot_file <filename>] \
    [--show_trajectories] \
    [--highlight_fingers]
```

### Examples

```bash
# Basic 3D visualization
python scripts/graspgen/tools/plot_gripper_3d.py \
    --gripper_config onrobot_rg6

# Save plot with finger highlighting
python scripts/graspgen/tools/plot_gripper_3d.py \
    --gripper_config onrobot_rg6 \
    --save_plot \
    --plot_file gripper_visualization.png \
    --highlight_fingers

# Show trajectories between configurations
python scripts/graspgen/tools/plot_gripper_3d.py \
    --gripper_config robotiq_2f_85 \
    --show_trajectories
```

### Visualization Features

- **Body Positions**: Each gripper body is shown as a point in 3D space
- **Color Coding**: Different bodies are shown in different colors
- **Finger Highlighting**: Finger bodies can be highlighted in red/blue
- **Trajectories**: Lines connecting body positions across configurations
- **Labels**: Body names and configuration labels

### Use Cases

1. **Configuration Validation**: Verify gripper geometry and positioning
2. **Debugging**: Visualize expected gripper open/close behavior during simulation
3. **Documentation**: Create visual documentation of gripper configurations
4. **Analysis**: Understand gripper kinematics and movement patterns

## Debug Visualization

The `visualize_debug.py` script creates a web-based 3D visualization from `create_gripper_lab.py` debug output files using Meshcat.

### Overview

This tool provides a clean, user-friendly interface for visualizing the OBJ/JSON debug output generated by the gripper creation process in `create_gripper_lab.py`. It uses Meshcat for interactive web-based 3D visualization, making it easy to inspect gripper geometry and debug issues.

### Features

- **Web-based Visualization**: Interactive 3D viewer accessible through web browser
- **Multi-folder Support**: Visualize all debug folders or specific ones
- **Mesh Rendering**: Display collision meshes with proper transforms
- **Normal Vectors**: Optional visualization of mesh face normals
- **Edge Rendering**: Optional wireframe display of mesh edges
- **Color Coding**: Different folders get unique colors for easy identification
- **Clean Interface**: Simple command-line interface with helpful error messages

### Usage

```bash
python scripts/graspgen/tools/visualize_debug.py \
    --path <debug_path> \
    [--draw-normals] \
    [--draw-edges]
```

### Arguments

- `--path, -p`: Path to debug output directory or specific folder (default: debug_output/)
- `--draw-normals`: Draw normal vectors for each mesh
- `--draw-edges`: Draw edges for each mesh

### Examples

```bash
# Visualize all debug folders
python scripts/graspgen/tools/visualize_debug.py --path debug_output/

# Visualize a specific debug folder
python scripts/graspgen/tools/visualize_debug.py --path debug_output/write_joint_state/

# Include normals and edges for detailed inspection
python scripts/graspgen/tools/visualize_debug.py --path debug_output/ --draw-normals --draw-edges

# Visualize with custom path
python scripts/graspgen/tools/visualize_debug.py --path /path/to/debug_output/
```

### Prerequisites

Before using this tool, you need to enable debug output in create_gripper_lab.py:

1. Open `scripts/graspgen/create_gripper_lab.py`
2. Uncomment the desired `save_scene_full()` calls:
   ```python
   # Uncomment these lines to enable debug output
   self.save_scene_full(f'write_joint_state', gripper)
   self.save_scene_full(f'write_data_to_sim_final', gripper)
   ```
3. Run create_gripper_lab.py to generate debug data

### Debug Output Structure

The tool expects debug output in the following structure:
```
debug_output/
├── write_joint_state/
│   ├── base_link.obj          # Base frame collision mesh
│   ├── base_link.json         # Base frame transform matrix
│   ├── right_inner_finger.obj # Right finger collision mesh
│   ├── right_inner_finger.json # Right finger transform matrix
│   └── ...
└── write_data_to_sim_final/
    ├── base_link.obj
    ├── base_link.json
    └── ...
```

### Visualization Features

- **Collision Meshes**: Shows the actual collision geometry used in simulation
- **Transform Matrices**: Applies the correct positioning from JSON files
- **Color Coding**: Each debug folder gets a unique color
- **Normal Vectors**: Visualize surface normals (useful for debugging collision issues)
- **Wireframe Edges**: Show mesh topology and structure
- **Interactive Controls**: Pan, zoom, and rotate using mouse controls

### Common Debug Points

Enable different debug points by uncommenting the corresponding lines in create_gripper_lab.py:

- **`sim_reset`**: Initial gripper state after simulation reset
- **`write_joint_state`**: After setting joint positions (check joint limits)
- **`write_data_to_sim`**: After writing data to simulation  
- **`write_data_to_sim_final`**: Final state after convergence

### Use Cases

1. **Gripper Validation**: Verify that gripper links are positioned correctly
2. **Joint Limit Debugging**: Check if joints can reach target positions
3. **Collision Debugging**: Inspect collision meshes and their positioning
4. **Convergence Analysis**: Compare before/after states to verify simulation convergence
5. **Documentation**: Create visual documentation of gripper behavior

### Dependencies

```bash
pip install meshcat trimesh
```

### Troubleshooting

**No debug output found**: Make sure you've enabled debug output in create_gripper_lab.py by uncommenting the save_scene_full() calls.

**Meshes appear in wrong positions**: Check that the JSON transform files contain valid 4x4 transformation matrices.

**Visualization doesn't open**: The tool will print a URL (typically http://127.0.0.1:7000/static/). Copy this URL into your web browser.

## Grasp Data Visualization

The `visualize_grasp_data.py` script creates a web-based 3D visualization of grasp data from YAML files using meshcat.

### Overview

This tool provides a comprehensive visualization of grasp datasets, including object meshes and grasp poses. It shows Y-shaped gripper line drawings with approach direction-based coloring and confidence-based pass/fail categorization.

### Features

- **Web-based Visualization**: Interactive 3D viewer accessible through web browser
- **Multi-dataset Support**: Visualize multiple YAML files in a grid layout
- **Object Mesh Rendering**: Display object meshes with proper scaling and positioning
- **Multi-format Support**: Native support for OBJ, STL, USD, USDA, and USDC files
- **Grasp Visualization**: Y-shaped gripper line drawings with approach direction coloring
- **Confidence Categorization**: Color-coded grasps based on confidence thresholds
- **Grid Layout**: Automatic grid arrangement for multiple datasets
- **Fallback Support**: Multiple visualization methods for robustness

### Usage

```bash
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths <grasp_paths> \
    --object-root <object_root> \
    [--gripper-name <gripper_name>] \
    [--max-grasps <max_grasps>] \
    [--draw-normals] \
    [--draw-edges] \
    [--no-grasps]
```

### Arguments

- `--grasp-paths, -p`: Paths to YAML files containing grasp data or directories containing YAML files (required)
- `--object-root, -o`: Root directory for object mesh files (default: "objects/")
- `--gripper-name`: Name of the gripper for visualization (default: "onrobot_rg6")
- `--max-grasps`: Maximum number of grasps to visualize per dataset (default: no limit)
- `--draw-normals`: Draw normal vectors for each mesh
- `--draw-edges`: Draw edges for each mesh
- `--no-grasps`: Skip grasp visualization, only show object meshes

### Examples

```bash
# Visualize all YAML files in a directory
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths grasp_data/ \
    --object-root objects/

# Visualize specific YAML files
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths file1.yaml file2.yaml \
    --object-root objects/

# Limit number of grasps per dataset
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths grasp_data/ \
    --max-grasps 50

# Include normals and edges
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths grasp_data/ \
    --draw-normals \
    --draw-edges

# Custom object root path
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths grasp_data/ \
    --object-root /path/to/objects/

# Use multiple dataset directories
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths /home/mcarlson/gitlab-master/GraspDataGen/datagen_guess_data/onrobot_rg6 \
                 /home/mcarlson/gitlab-master/GraspDataGen/datagen_sim_data/onrobot_rg6 \
    --object-root /home/mcarlson/gitlab-master/GraspDataGen/ \
    --gripper-name onrobot_rg6
```

### Data Format

The tool expects YAML files in the Isaac grasp format:

```yaml
format: isaac_grasp
format_version: '1.0'
object_file: objects/banana.1.0.obj  # Supports OBJ, STL, USD, USDA, USDC
object_scale: 1.0
base_length: 0.16676733328495175
approach_axis: 2
bite_point: [0.024600457400083542, -2.2378750145435333e-05, 0.05639537051320076]
grasps:
  grasp_0_5:
    confidence: 1.0
    position: [-0.23065072298049927, 0.11876945197582245, -0.07466025650501251]
    orientation:
      w: 0.1255687028169632
      xyz: [0.5989119410514832, 1.271737488650615e-07, 0.7909089922904968]
    bite_point: [0.027983803302049637, 0.0, 0.2679715156555176]
```

**Supported Object Formats:**
- **OBJ files**: Standard mesh format with full support
- **STL files**: Stereolithography format with full support  
- **USD files**: Universal Scene Description format with full support
- **USDA/USDC files**: ASCII and compressed USD variants

### Visualization Features

- **Object Meshes**: Loaded from OBJ files with proper scaling and positioning
- **Grasp Poses**: Y-shaped gripper line drawings showing grasp geometry
- **Approach Direction Coloring**: Rainbow colors based on approach vector direction
- **Confidence Categorization**: 
  - Green: High confidence (>0.8) - "pass"
  - Inverted colors: Low confidence (≤0.8) - "fail"
- **Grid Layout**: Automatic arrangement of multiple datasets
- **Interactive Controls**: Pan, zoom, and rotate using mouse controls

### Technical Details

The tool creates visualizations using:

1. **Render Points**: Creates base and bite points using `approach_axis` and `base_length`
2. **Y-Shape Geometry**: Generates 7 vertices forming a Y-shaped gripper representation
3. **Color Calculation**: Uses approach direction-based coloring for visual variety
4. **Confidence Handling**: Applies color inversion for failed grasps
5. **Fallback System**: Multiple visualization methods for robustness

### Use Cases

1. **Grasp Analysis**: Visualize and analyze grasp quality and distribution
2. **Dataset Comparison**: Compare different grasp datasets (guess vs. simulation)
3. **Debugging**: Identify issues with grasp generation or data format
4. **Documentation**: Create visual documentation of grasp datasets
5. **Quality Assessment**: Evaluate grasp confidence and approach directions

### Dependencies

```bash
pip install meshcat trimesh pyyaml numpy
```

### Troubleshooting

**No grasps visualized**: Check that the YAML files contain valid grasp data in the expected format.

**Objects not visible**: Verify that object mesh files exist at the specified paths and are valid OBJ files.

**Visualization doesn't open**: The tool will print a URL (typically http://127.0.0.1:7001/static/). Copy this URL into your web browser.

**Memory issues with large datasets**: Use `--max-grasps` to limit the number of grasps visualized per dataset.

## Convert YAML to JSON

The `convert_yaml_to_json.py` script converts Isaac Lab YAML grasp files to the JSON format expected by GraspGen.

### Overview

This tool bridges the gap between Isaac Lab's grasp data format and GraspGen's expected input format. It handles the conversion of quaternion orientations to transformation matrices and maps confidence scores to success flags.

### Features

- **Format Conversion**: Converts Isaac Lab YAML format to GraspGen JSON format
- **Quaternion Handling**: Converts quaternion orientations to 4x4 transformation matrices
- **Confidence Mapping**: Maps confidence scores to success/failure flags
- **Gripper Configuration**: Supports custom gripper configuration files

### Usage

```bash
python scripts/graspgen/tools/convert_yaml_to_json.py \
    <yaml_file> \
    [json_file] \
    [max_grasps] \
    [gripper_config]
```

### Arguments

- `yaml_file`: Path to Isaac Lab YAML grasp file (required)
- `json_file`: Output JSON file path (optional, defaults to YAML filename with .json extension)
- `max_grasps`: Maximum number of grasps to convert (optional, no limit by default)
- `gripper_config`: Path to gripper configuration YAML file (optional)

### Examples

```bash
# Convert with default JSON filename
python scripts/graspgen/tools/convert_yaml_to_json.py grasp_data.yaml

# Convert with custom JSON filename
python scripts/graspgen/tools/convert_yaml_to_json.py grasp_data.yaml output.json

# Convert only first 100 grasps
python scripts/graspgen/tools/convert_yaml_to_json.py grasp_data.yaml output.json 100

# Convert with custom gripper configuration
python scripts/graspgen/tools/convert_yaml_to_json.py grasp_data.yaml output.json 100 config/grippers/robotiq_2f_85.yaml

# Convert multiple files
for file in grasp_guess_data/onrobot_rg6/*.yaml; do
    python scripts/graspgen/tools/convert_yaml_to_json.py "$file"
done
```

### Input Format

The tool expects Isaac Lab YAML files in the following format:

```yaml
object_file: objects/banana.obj
object_scale: 1.0
gripper_file: grippers/onrobot_rg6.usd
grasps:
  grasp_0:
    confidence: 1.0
    position: [0.1, 0.2, 0.3]
    orientation:
      xyz: [0.0, 0.0, 0.0]
      w: 1.0
  grasp_1:
    confidence: 0.5
    position: [0.2, 0.3, 0.4]
    orientation:
      xyz: [0.1, 0.0, 0.0]
      w: 0.9
```

### Output Format

The tool generates GraspGen JSON files in the following format:

```json
{
  "object": {
    "file": "objects/banana.obj",
    "scale": 1.0
  },
  "gripper": {
    "name": "robotiq_2f_85",
    "file_name": "grippers/onrobot_rg6.usd",
    "width": 0.08709684014320374,
    "depth": 0.12992018461227417,
    "transform_offset_from_asset_to_graspgen_convention": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  },
  "grasps": {
    "transforms": [
      [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
      [[0.98, 0, 0.2, 0.2], [0, 1, 0, 0.3], [-0.2, 0, 0.98, 0.4], [0, 0, 0, 1]]
    ],
    "object_in_gripper": [true, false]
  }
}
```

### Gripper Configuration

The tool supports custom gripper configuration files from GraspGen to override default values:

```yaml
width: 0.08709684014320374
depth: 0.12992018461227417
transform_offset_from_asset_to_graspgen_convention:
  - [0.0, 0.0, 0.0]  # position
  - [1.0, 0.0, 0.0, 0.0]  # quaternion [w, x, y, z]
```

## Analyze Grasp Data

The `analyze_grasp_data.py` script provides comprehensive analysis of grasp data files, supporting both Isaac Lab YAML and GraspGen JSON formats.

### Overview

This tool analyzes grasp data files and provides detailed statistics including success/failure rates, confidence distributions, and quality metrics. It can analyze individual files or entire directories, making it perfect for dataset validation and quality assessment.

### Features

- **Multi-format Support**: Works with both Isaac Lab YAML and GraspGen JSON formats
- **Comprehensive Statistics**: Success rates, confidence distributions, quality metrics
- **Batch Analysis**: Analyze entire directories of grasp files
- **Detailed Output**: Optional detailed mode with raw data
- **Export Results**: Save analysis results to JSON files
- **Error Handling**: Graceful handling of invalid files with detailed error reporting

### Usage

```bash
python scripts/graspgen/tools/analyze_grasp_data.py \
    <file_or_directory> \
    [--detailed] \
    [--output results.json] \
    [--quiet]
```

### Arguments

- `path`: Path to grasp file or directory containing grasp files (required)
- `--detailed`: Include detailed output with raw data and confidence distributions
- `--output, -o`: Save analysis results to JSON file
- `--min-success`: Minimum number of successful grasps required per file
- `--min-failed`: Minimum number of failed grasps required per file
- `--min-total`: Minimum total number of grasps required per file
- `--quiet`: Suppress console output (useful when saving to file)
- `--no-progress`: Disable progress bar (useful for automated scripts)

### Examples

```bash
# Analyze a single YAML file
python scripts/graspgen/tools/analyze_grasp_data.py \
    grasp_sim_data/onrobot_rg6/banana.yaml

# Analyze a single JSON file
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/banana.json

# Analyze all files in a directory
python scripts/graspgen/tools/analyze_grasp_data.py \
    grasp_sim_data/onrobot_rg6/

# Analyze with detailed output
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/banana.json \
    --detailed

# Analyze directory and save results
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/ \
    --output analysis_results.json

# Analyze with criteria (minimum 1000 successful, 500 failed grasps per file)
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/ \
    --min-success 1000 \
    --min-failed 500

# Analyze with minimum total grasps requirement
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/ \
    --min-total 2000

# Analyze with progress bar disabled (for scripts)
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/ \
    --no-progress

# Quiet analysis (no console output, no progress bar)
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/ \
    --output results.json \
    --quiet
```

### Supported Formats

#### Isaac Lab YAML Format

The tool expects YAML files in the Isaac Lab grasp format:

```yaml
object_file: objects/banana.obj
object_scale: 1.0
gripper_file: grippers/onrobot_rg6.usd
grasps:
  grasp_0:
    confidence: 1.0
    position: [0.1, 0.2, 0.3]
    orientation:
      xyz: [0.0, 0.0, 0.0]
      w: 1.0
  grasp_1:
    confidence: 0.5
    position: [0.2, 0.3, 0.4]
    orientation:
      xyz: [0.1, 0.0, 0.0]
      w: 0.9
```

#### GraspGen JSON Format

The tool expects JSON files in the GraspGen format:

```json
{
  "object": {
    "file": "objects/banana.obj",
    "scale": 1.0
  },
  "gripper": {
    "name": "robotiq_2f_85",
    "file_name": "grippers/onrobot_rg6.usd",
    "width": 0.08709684014320374,
    "depth": 0.12992018461227417
  },
  "grasps": {
    "transforms": [
      [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
      [[0.98, 0, 0.2, 0.2], [0, 1, 0, 0.3], [-0.2, 0, 0.98, 0.4], [0, 0, 0, 1]]
    ],
    "object_in_gripper": [true, false]
  }
}
```

### Analysis Output

#### Single File Analysis

For individual files, the tool provides:

- **Basic Statistics**: Total grasps, successful grasps, failed grasps, success rate
- **Confidence Statistics**: Mean, median, standard deviation, min/max, quartiles
- **Quality Metrics**: High confidence (≥0.8), medium confidence (0.3-0.8), low confidence (<0.3)
- **Metadata**: Object file, scale, gripper information
- **Detailed Mode**: Raw confidence values and success flags

#### Directory Analysis

For directories, the tool provides:

- **Progress Bar**: Visual progress indicator with percentage completion and ETA
- **Overall Summary**: Total grasps across all files, overall success rate
- **Per-File Averages**: Average grasps per file, average successes/failures per file with standard deviations
- **Per-File Ranges**: Min/max values for grasps, successes, and failures across files
- **Aggregate Statistics**: Combined confidence statistics across all files
- **Error Reporting**: Files that couldn't be analyzed with error details


#### Criteria Analysis

When criteria are specified (using `--min-success`, `--min-failed`, or `--min-total`), the tool provides:

- **Criteria Summary**: How many files meet vs. fail the specified criteria
- **Completion Statistics**: For files that need more grasps, shows current progress and completion percentage (capped at 100%)
- **Failed Files Summary**: Shows count of failing files with limited examples (first 3 + count of remaining)
- **Progress Tracking**: Shows how close files are to meeting requirements

### Sample Output

```
📊 Analysis: banana.yaml
   Format: YAML
   Object: objects/banana.obj
   Scale: 1.0
   Gripper: grippers/onrobot_rg6.usd
   Total Grasps: 1000
   Successful: 750 (75.0%)
   Failed: 250
   Confidence - Mean: 0.750, Std: 0.433
   Confidence - Min: 0.000, Max: 1.000
   Quality - High: 600, Medium: 200, Low: 200

📈 Directory Summary: graspgen_data
   Files Analyzed: 50
   Total Grasps: 50000
   Successful: 37500 (75.0%)
   Failed: 12500
   Overall Confidence - Mean: 0.750, Std: 0.433

📊 Per-File Averages:
   Avg Grasps per File: 1000.0 ± 0.0
   Avg Successful per File: 750.0 ± 0.0
   Avg Failed per File: 250.0 ± 0.0
   Avg Success Rate per File: 75.0% ± 0.0%

📊 Per-File Ranges:
   Grasps per File: 1000 - 1000
   Successful per File: 750 - 750
   Failed per File: 250 - 250

🎯 Criteria Analysis:
   Criteria: ≥1000 successful, ≥500 failed
   Files Meeting Criteria: 45 (90.0%)
   Files Failing Criteria: 5
   Success Completion: 75.0%
   Failed Completion: 50.0%

❌ Files Failing Criteria: 5 files
   apple.json: <500 failed (S:800, F:200, T:1000)
   orange.json: <1000 successful (S:600, F:400, T:1000)
   ... and 3 more files

📋 Individual File Results: 50 files analyzed
```
