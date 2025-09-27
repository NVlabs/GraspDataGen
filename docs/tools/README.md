# Tools Overview

GraspDataGen includes a set of utility tools to help with data generation, analysis, debugging, and comparison tasks. These tools are located in the `scripts/graspgen/tools/` directory and can be used independently or as part of your workflow.

## Tool Categories

### 🔍 Compare Tools
Tools for comparing and analyzing differences between grasp simulations and gripper configurations:

- **[compare_grasp_sims.py](compare-tools.md#compare-grasp-sims)** - Compare two grasp simulation YAML files with confidence analysis
- **[compare_grippers.py](compare-tools.md#compare-grippers)** - Compare gripper NPZ configuration files

### 🛠️ Utility Tools
General utility scripts for data processing and workflow management:

- **[grasp_display.py](utility-tools.md#grasp-display)** - Display grasps in IsaacLab with full gripper USDs
- **[plot_gripper_3d.py](utility-tools.md#plot-gripper-3d)** - 3D visualization of gripper body positions
- **[visualize_debug.py](utility-tools.md#debug-visualization)** - Web-based 3D visualization of debug output from `create_gripper_lab.py`
- **[visualize_grasp_data.py](utility-tools.md#grasp-data-visualization)** - Web-based 3D visualization of grasp data from YAML files
- **[convert_yaml_to_json.py](utility-tools.md#convert-yaml-to-json)** - Convert Isaac Lab YAML grasp files to GraspGen JSON format
- **[analyze_grasp_data.py](utility-tools.md#analyze-grasp-data)** - Analyze grasp data files and provide comprehensive statistics

## Quick Start

Most tools can be run directly from the command line. Here are some common examples:

```bash
# Display grasps in IsaacLab (may be slow with many grasps)
python scripts/graspgen/grasp_display.py \
    --grasp_file grasp_guess_data/onrobot_rg6/banana.yaml \
    --max_num_grasps 50

# Compare two grasp simulation results
python scripts/graspgen/tools/compare_grasp_sims.py \
    grasp_sim_data/gripper1/object.yaml \
    grasp_sim_data/gripper2/object.yaml

# Compare two grippers
python scripts/graspgen/tools/compare_grippers.py \
    gripper_configs/robotiq_2f_85.npz \
    gripper_configs/robotiq_2f_85_modified.npz

# Visualize gripper configuration in 3D
python scripts/graspgen/tools/plot_gripper_3d.py \
    --gripper_config onrobot_rg6 \
    --save_plot

# Visualize debug output from gripper creation
python scripts/graspgen/tools/visualize_debug.py \
    --path debug_output/ \
    --draw-normals

# Visualize grasp data from YAML files
python scripts/graspgen/tools/visualize_grasp_data.py \
    --grasp-paths grasp_data/ \
    --object-root objects/ \
    --max-grasps 100

# Filter specific grasps for debugging
python scripts/graspgen/tools/filter_grasp_yaml.py \
    input_grasps.yaml output_debug.yaml \
    --indices 10 25 42 67

# Convert Isaac Lab YAML to GraspGen JSON format
python scripts/graspgen/tools/convert_yaml_to_json.py \
    grasp_sim_data/onrobot_rg6/banana.yaml \
    graspgen_data/banana.json \
    1000 \
    config/grippers/onrobot_rg6.yaml

# Analyze grasp data files for statistics
python scripts/graspgen/tools/analyze_grasp_data.py \
    graspgen_data/banana.json \
    --detailed
```

## Tool Documentation

Each tool category has detailed documentation:

- **[Compare Tools](compare-tools.md)** - Detailed guides for comparison tools
- **[Utility Tools](utility-tools.md)** - General utility scripts
