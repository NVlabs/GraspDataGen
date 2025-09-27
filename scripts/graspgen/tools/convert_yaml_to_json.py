#!/usr/bin/env python3
"""
Convert YAML grasp file to JSON graspgen format.

This tool converts Isaac Lab YAML grasp files to the JSON format expected by graspgen.
It handles the conversion of quaternion orientations to transformation matrices and
maps confidence scores to success flags.

Usage:
    python convert_yaml_to_json.py <yaml_file> [json_file] [max_grasps]
    
Examples:
    # Convert with default JSON filename
    python convert_yaml_to_json.py grasp_data.yaml
    
    # Convert with custom JSON filename
    python convert_yaml_to_json.py grasp_data.yaml output.json
    
    # Convert only first 100 grasps
    python convert_yaml_to_json.py grasp_data.yaml output.json 100
"""

import yaml
import json
import numpy as np
import sys
import os
from pathlib import Path

def quaternion_to_matrix(position, xyz, w):
    """Convert quaternion and position to 4x4 transformation matrix."""
    # Convert quaternion to rotation matrix
    x, y, z = xyz
    w = w
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Convert to rotation matrix
    rotation_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    # Create 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    
    return transform_matrix.tolist()

def convert_yaml_to_json(yaml_path, json_path, max_grasps=None, gripper_config_path=None):
    """Convert YAML grasp file to JSON graspgen format."""
    
    print(f"Loading YAML file: {yaml_path}")
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Extract object information
    object_file = yaml_data['object_file']
    object_scale = yaml_data['object_scale']
    
    # Extract gripper information
    gripper_file = yaml_data['gripper_file']
    gripper_name = "robotiq_2f_85"  # Extract from gripper file name
    
    # Load gripper config if provided
    gripper_config = {}
    if gripper_config_path and os.path.exists(gripper_config_path):
        print(f"Loading gripper config: {gripper_config_path}")
        with open(gripper_config_path, 'r') as f:
            gripper_config = yaml.safe_load(f)
    
    # Extract gripper parameters from config or use defaults
    width = gripper_config.get('width', 0.08709684014320374)
    depth = gripper_config.get('depth', 0.12992018461227417)
    transform_offset_raw = gripper_config.get('transform_offset_from_asset_to_graspgen_convention', 
                                            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    
    # Convert nested list format to flat list format
    if isinstance(transform_offset_raw, list) and len(transform_offset_raw) == 2:
        # Format: [[x, y, z], [qx, qy, qz, qw]] -> convert to 4x4 matrix then flatten
        position = transform_offset_raw[0]
        quaternion = transform_offset_raw[1]
        
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = quaternion
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = position
        transform_offset = transform_matrix.flatten().tolist()
    else:
        # Already in flat format
        transform_offset = transform_offset_raw
    
    # Convert grasps
    grasps = yaml_data['grasps']
    transforms_list = []
    object_in_gripper_list = []
    
    print(f"Processing {len(grasps)} grasps...")
    
    count = 0
    for grasp_name, grasp_data in grasps.items():
        if max_grasps and count >= max_grasps:
            break
            
        # Extract position and orientation
        position = np.array(grasp_data['position'])
        orientation = grasp_data['orientation']
        xyz = np.array(orientation['xyz'])
        w = orientation['w']
        
        # Convert to transformation matrix
        transform_matrix = quaternion_to_matrix(position, xyz, w)
        transforms_list.append(transform_matrix)
        
        # Determine success based on confidence
        confidence = grasp_data.get('confidence', 1.0)
        object_in_gripper_list.append(confidence > 0.0)
        
        count += 1
        if count % 100 == 0:
            print(f"  Processed {count} grasps...")
    
    print(f"Converted {count} grasps")
    
    # Create JSON structure
    json_data = {
        "object": {
            "file": object_file,
            "scale": object_scale
        },
        "gripper": {
            "name": gripper_name,
            "file_name": gripper_file,
            "width": width,
            "depth": depth,
            "transform_offset_from_asset_to_graspgen_convention": transform_offset
        },
        "grasps": {
            "transforms": transforms_list,
            "object_in_gripper": object_in_gripper_list
        }
    }
    
    # Save JSON file
    print(f"Saving JSON file: {json_path}")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Conversion complete!")
    print(f"  Input: {yaml_path}")
    print(f"  Output: {json_path}")
    print(f"  Grasps: {len(transforms_list)}")
    print(f"  Successful: {sum(object_in_gripper_list)}")
    print(f"  Failed: {len(object_in_gripper_list) - sum(object_in_gripper_list)}")
    print(f"  Gripper config: {gripper_config_path if gripper_config_path else 'default values'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_yaml_to_json.py <yaml_file> [json_file] [max_grasps] [gripper_config]")
        print("\nExamples:")
        print("  python convert_yaml_to_json.py grasp_data.yaml")
        print("  python convert_yaml_to_json.py grasp_data.yaml output.json")
        print("  python convert_yaml_to_json.py grasp_data.yaml output.json 100")
        print("  python convert_yaml_to_json.py grasp_data.yaml output.json 100 config/grippers/robotiq_2f_85.yaml")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        json_path = sys.argv[2]
    else:
        # Create JSON filename from YAML filename
        yaml_file = Path(yaml_path)
        json_path = yaml_file.with_suffix('.json')
    
    max_grasps = None
    if len(sys.argv) >= 4:
        max_grasps = int(sys.argv[3])
    
    gripper_config_path = None
    if len(sys.argv) >= 5:
        gripper_config_path = sys.argv[4]
    
    convert_yaml_to_json(yaml_path, json_path, max_grasps, gripper_config_path)

if __name__ == "__main__":
    main()
