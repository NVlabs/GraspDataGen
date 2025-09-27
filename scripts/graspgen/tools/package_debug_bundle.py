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
Package debugging bundle for grasp simulation.

Creates a zip file containing:
- The input YAML file (with folder structure)
- The object_file and gripper_file referenced in the YAML
- All Python dependencies needed to run grasp_sim.py
"""

import argparse
import ast
import yaml
import zipfile
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file '{file_path}': {e}")
        sys.exit(1)


def get_minimal_dependencies() -> List[str]:
    """Get the minimal set of Python files needed for grasp_sim.py."""
    return [
        "scripts/graspgen/grasp_sim.py",
        "scripts/graspgen/graspgen_utils.py",
        "scripts/graspgen/gripper.py",
        "scripts/graspgen/object.py",
        "scripts/graspgen/warp_kernels.py",
        "scripts/graspgen/warp_kernels_debug.py"
    ]


def parse_python_imports(file_path: str) -> Set[str]:
    """
    Parse a Python file and extract local module imports.
    
    Returns:
        Set of module names that are imported locally
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        imports = set()
        
        # Walk the AST to find import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        return imports
    except Exception as e:
        print(f"Warning: Could not parse imports from {file_path}: {e}")
        return set()


def find_local_dependencies(main_file: str,
                            visited: Optional[Set[str]] = None) -> List[str]:
    """
    Find all local Python dependencies for a given Python file.
    
    Args:
        main_file: Path to the main Python file
        visited: Set of already processed files (to avoid circular deps)
        
    Returns:
        List of file paths for local dependencies
    """
    if visited is None:
        visited = set()
    
    main_path = Path(main_file).resolve()
    main_file_str = str(main_path)
    
    if not main_path.exists():
        print(f"Error: Main file {main_file} does not exist")
        sys.exit(1)
    
    # Avoid infinite recursion from circular imports
    if main_file_str in visited:
        return []
    
    visited.add(main_file_str)
    
    # Get the directory containing the main file
    base_dir = main_path.parent
    
    # Parse imports from the main file
    imports = parse_python_imports(main_file_str)
    
    # Find local dependencies
    dependencies = [main_file_str]  # Include the main file itself
    
    for import_name in imports:
        # Handle dotted imports (e.g., package.module)
        module_name = import_name.split('.')[0]
        
        # Check if this is a local module (exists as .py file in same dir)
        potential_file = base_dir / f"{module_name}.py"
        if potential_file.exists():
            dep_path = str(potential_file.resolve())
            if dep_path not in visited:
                # Recursively find dependencies of this dependency
                sub_deps = find_local_dependencies(dep_path, visited)
                for sub_dep in sub_deps:
                    if sub_dep not in dependencies:
                        dependencies.append(sub_dep)
    
    return dependencies


def convert_to_relative_path(abs_path: str) -> str:
    """Convert absolute path to relative path from current working dir."""
    try:
        abs_path_obj = Path(abs_path)
        cwd = Path.cwd()
        return str(abs_path_obj.relative_to(cwd))
    except ValueError:
        # If path is not relative to cwd, return as-is
        return abs_path


def get_grasp_sim_dependencies(use_full_deps: bool = False) -> List[str]:
    """Get list of Python files that grasp_sim.py depends on."""
    if not use_full_deps:
        # Use minimal dependencies (the original hardcoded list)
        dependencies = get_minimal_dependencies()
        print(f"Using minimal dependencies ({len(dependencies)} files):")
    else:
        # Use full dynamic detection
        grasp_sim_path = "scripts/graspgen/grasp_sim.py"
        
        if not Path(grasp_sim_path).exists():
            print(f"Error: {grasp_sim_path} not found")
            sys.exit(1)
        
        dependencies = find_local_dependencies(grasp_sim_path)
        
        # Convert absolute paths to relative paths
        relative_dependencies = []
        for dep in dependencies:
            relative_path = convert_to_relative_path(dep)
            relative_dependencies.append(relative_path)
        
        dependencies = relative_dependencies
        dependencies.sort()
        print(f"Using full dependencies ({len(dependencies)} files):")
    
    for dep in dependencies:
        print(f"  - {dep}")
    
    return dependencies


def validate_files_exist(files: List[str], yaml_file: str) -> None:
    """Validate that all files exist before creating the package."""
    missing_files = []
    
    for file_path in files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: The following files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print(f"\nReferenced from YAML file: {yaml_file}")
        sys.exit(1)


def resolve_output_path(yaml_file: str, output_zip: Optional[str] = None) -> str:
    """
    Resolve the output zip file path, handling cases where output_zip is a directory.
    
    Args:
        yaml_file: Path to input YAML file (used for auto-generated names)
        output_zip: User-provided output path (file or directory, optional)
        
    Returns:
        Final zip file path
    """
    if output_zip is None:
        # Auto-generate filename in current directory
        yaml_name = Path(yaml_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"debug_package_{yaml_name}_{timestamp}.zip"
    
    output_path = Path(output_zip)
    
    # Check if the provided path is a directory or should be treated as one
    if (output_path.is_dir() or 
        (not output_path.suffix and not output_path.exists())):
        # It's a directory (existing or to be created)
        # Generate filename inside the directory
        yaml_name = Path(yaml_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_package_{yaml_name}_{timestamp}.zip"
        return str(output_path / filename)
    else:
        # It's a file path
        return str(output_path)


def create_debug_package(yaml_file: str, 
                         output_zip: Optional[str] = None,
                         use_full_deps: bool = True) -> None:
    """
    Create a debug package containing all necessary files.
    
    Args:
        yaml_file: Path to the input YAML file
        output_zip: Path to output zip file or directory (optional)
        use_full_deps: Whether to include all dependencies or just minimal
    """
    # Load the YAML file to get referenced files
    yaml_data = load_yaml(yaml_file)
    
    # Extract referenced files
    object_file = yaml_data.get('object_file', '')
    gripper_file = yaml_data.get('gripper_file', '')
    
    if not object_file or not gripper_file:
        print("Error: YAML file must contain 'object_file' and 'gripper_file'")
        sys.exit(1)
    
    # Get Python dependencies
    print("Analyzing Python dependencies...")
    python_deps = get_grasp_sim_dependencies(use_full_deps)
    
    # List of all files to include in the package
    files_to_package = [
        yaml_file,
        object_file,
        gripper_file
    ] + python_deps
    
    # Validate all files exist
    validate_files_exist(files_to_package, yaml_file)
    
    # Resolve the final output path
    final_output_zip = resolve_output_path(yaml_file, output_zip)
    
    # Create output directory if it doesn't exist
    output_path = Path(final_output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the zip file
    print(f"\nCreating debug package: {final_output_zip}")
    print(f"Input YAML: {yaml_file}")
    print(f"Object file: {object_file}")
    print(f"Gripper file: {gripper_file}")
    print(f"Python dependencies: {len(python_deps)} files")
    
    try:
        with zipfile.ZipFile(final_output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files while preserving folder structure
            for file_path in files_to_package:
                if Path(file_path).exists():
                    # Add file to zip with its full path to preserve structure
                    zipf.write(file_path, file_path)
                    print(f"  Added: {file_path}")
                else:
                    print(f"  Warning: File not found: {file_path}")
            
            # Add a README file with instructions
            readme_content = f"""# Debug Package for Grasp Simulation

This package contains all files needed to run grasp simulation debugging.

## Contents:
- Input YAML file: {yaml_file}
- Object file: {object_file}
- Gripper file: {gripper_file}
- Python dependencies: {len(python_deps)} files

## Python Dependencies:
{chr(10).join(f'- {dep}' for dep in python_deps)}

## Usage:
1. Extract this zip file
2. Navigate to the extracted directory
3. Run the simulation:
   python3 scripts/graspgen/grasp_sim.py --grasp_file {yaml_file} --force_headed

## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
## Dependencies: {'Full (auto-detected)' if use_full_deps else 'Minimal (core only)'}
"""
            zipf.writestr("README.txt", readme_content)
            print("  Added: README.txt")
    
    except Exception as e:
        print(f"Error creating zip file: {e}")
        sys.exit(1)
    
    print(f"\nDebug package created successfully: {final_output_zip}")
    print(f"Package size: {Path(final_output_zip).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Package debugging bundle for grasp simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Package with auto-generated filename
  python package_debug_bundle.py grasp_debug_data/mug_m1.Robotiq.yaml
  
  # Package in specific directory (auto-generates filename)
  python package_debug_bundle.py grasp_debug_data/mug_m1.Robotiq.yaml -o debug_packages/
  
  # Package with custom filename
  python package_debug_bundle.py grasp_debug_data/mug_m1.Robotiq.yaml -o debug_bundle.zip
        """
    )
    
    parser.add_argument(
        'yaml_file',
        help='Path to input YAML file containing grasp data'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output zip file path or directory (optional, auto-generated if not provided)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.yaml_file).exists():
        print(f"Error: Input YAML file '{args.yaml_file}' does not exist")
        sys.exit(1)
    
    create_debug_package(args.yaml_file, args.output)


if __name__ == '__main__':
    main() 