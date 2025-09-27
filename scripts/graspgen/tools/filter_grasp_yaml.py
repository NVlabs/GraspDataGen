# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
Utility script to filter and rename grasp entries from a YAML file.

Takes a YAML file with grasp data and creates a new one with only the selected
grasps (based on provided indices) renamed as grasp_0, grasp_1, grasp_2, etc.

This is good for debugging.  When looking at the env index of failed grasps, keep
track of them, and then run this file to create a debug grasp file with just those
as the grasps, then use package_debug_bundle.py to package the debug grasps into a bundle
for the debug team.
"""

import argparse
import yaml
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file '{file_path}': {e}")
        sys.exit(1)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """Save data to YAML file, creating directory if it doesn't exist."""
    try:
        # Create output directory if it doesn't exist
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"Successfully saved filtered grasps to '{file_path}'")
    except Exception as e:
        print(f"Error saving YAML file '{file_path}': {e}")
        sys.exit(1)


def filter_and_rename_grasps(input_file: str, output_file: str,
                             indices: List[int]) -> None:
    """
    Filter and rename grasps based on provided indices.
    
    Args:
        input_file: Path to input YAML file
        output_file: Path to output YAML file
        indices: List of indices to select from the original grasps
    """
    # Load the original YAML file
    data = load_yaml(input_file)
    
    if 'grasps' not in data:
        print("Error: Input YAML file does not contain 'grasps' section")
        sys.exit(1)
    
    # Get all grasp entries as a list (preserve order)
    original_grasps = list(data['grasps'].items())
    total_grasps = len(original_grasps)
    
    print(f"Original file contains {total_grasps} grasps")
    print(f"Requested indices: {indices}")
    
    # Validate indices
    invalid_indices = [idx for idx in indices
                       if idx < 0 or idx >= total_grasps]
    if invalid_indices:
        print(f"Error: Invalid indices {invalid_indices}. "
              f"Must be in range [0, {total_grasps-1}]")
        sys.exit(1)
    
    # Create new grasps dictionary with filtered and renamed entries
    new_grasps = {}
    for new_idx, original_idx in enumerate(indices):
        original_name, grasp_data = original_grasps[original_idx]
        new_name = f"grasp_{new_idx}"
        new_grasps[new_name] = grasp_data
        print(f"  {original_name} (index {original_idx}) -> {new_name}")
    
    # Create the output data structure
    output_data = data.copy()
    output_data['grasps'] = new_grasps
    
    # Update metadata to reflect the filtering NOTE: this is not a good idea, keep the original name
    if 'created_with' in output_data:
        output_data['created_with'] = f"{output_data['created_with']}"
    
    # Add a comment about the filtering
    output_data['filter_info'] = {
        'original_file': input_file,
        'original_grasp_count': total_grasps,
        'filtered_grasp_count': len(indices),
        'selected_indices': indices
    }
    
    # Save the filtered YAML
    save_yaml(output_data, output_file)
    
    print(f"Filtered {len(indices)} grasps from {total_grasps} total grasps")


def main():
    parser = argparse.ArgumentParser(
        description="Filter and rename grasp entries from a YAML file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter specific grasps by indices
  python filter_grasp_yaml.py input.yaml output.yaml --indices 64 160 231
  
  # Filter first 10 grasps
  python filter_grasp_yaml.py input.yaml output.yaml --indices 0 1 2 3 4 5 6 7 8 9
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input YAML file containing grasps'
    )
    
    parser.add_argument(
        'output_file',
        help='Path to output YAML file for filtered grasps'
    )
    
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        required=True,
        help='List of indices to select from the original grasps (0-based)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Remove duplicates and sort indices for consistent output
    unique_indices = sorted(set(args.indices))
    if len(unique_indices) != len(args.indices):
        print("Note: Removed duplicate indices. "
              f"Using: {unique_indices}")
    
    filter_and_rename_grasps(args.input_file, args.output_file,
                            unique_indices)


if __name__ == '__main__':
    main() 