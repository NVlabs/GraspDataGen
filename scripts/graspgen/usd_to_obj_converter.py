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
USD to OBJ Batch Converter

This module provides functionality to batch convert USD files to OBJ format
without requiring Isaac Lab. It uses the installed pxr package directly.

Usage:
    # Single file conversion
    python usd_to_obj_converter.py --usd_file path/to/file.usd --scale 1.0 --output_dir /path/to/output
    
    # Batch conversion
    python usd_to_obj_converter.py --object_scales_json path/to/scales.json \
        --object_root /path/to/objects --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple
import numpy as np
import trimesh

try:
    from pxr import Usd, UsdGeom
except ImportError as e:
    print("Error: Failed to import pxr. Please install USD "
          "(e.g. via `pip install usd-core`).")
    print(f"Original error: {e}")
    sys.exit(1)


def extract_mesh_from_usd(usd_path: str, scale: float = 1.0) -> Optional[trimesh.Trimesh]:
    """
    Extract mesh data from a USD file and return as a trimesh object.
    
    Args:
        usd_path: Path to the USD file
        scale: Scale factor to apply to the mesh
        
    Returns:
        trimesh.Trimesh object or None if no mesh found
    """
    try:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"Warning: Could not open USD file: {usd_path}")
            return None
        
        # Collect all mesh data from the stage
        vertices = []
        faces = []
        vertex_offset = 0
        
        def process_prim(prim, transform_matrix=None):
            nonlocal vertices, faces, vertex_offset
            
            if transform_matrix is None:
                transform_matrix = np.eye(4)
            
            # Get transform for this prim
            if prim.HasAttribute("xformOp:transform"):
                local_transform = np.array(
                    prim.GetAttribute("xformOp:transform").Get())
                transform_matrix = transform_matrix @ local_transform
            elif prim.GetTypeName() == "Xform":
                xform = UsdGeom.Xform(prim)
                local_transform = np.array(xform.GetLocalTransformation())
                transform_matrix = transform_matrix @ local_transform
            
            # Process mesh prims
            if prim.GetTypeName() == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                
                # Get mesh data
                mesh_vertices = np.array(
                    mesh.GetPointsAttr().Get(), dtype=np.float32)
                face_vertex_indices = np.array(
                    mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                face_vertex_counts = np.array(
                    mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
                
                # Apply transform to vertices
                if len(mesh_vertices) > 0:
                    # Convert to homogeneous coordinates
                    homogeneous_vertices = np.column_stack(
                        [mesh_vertices, np.ones(len(mesh_vertices))])
                    transformed_vertices = (
                        transform_matrix @ homogeneous_vertices.T).T[:, :3]
                    
                    # Apply scale
                    transformed_vertices *= scale
                    
                    # Convert face data to triangles
                    current_vertex = 0
                    for count in face_vertex_counts:
                        if count == 3:
                            # Triangle
                            face = face_vertex_indices[
                                current_vertex:current_vertex + 3]
                            faces.append(face + vertex_offset)
                        elif count == 4:
                            # Quad - split into two triangles
                            face = face_vertex_indices[
                                current_vertex:current_vertex + 4]
                            faces.append([face[0], face[1], face[2]] +
                                         vertex_offset)
                            faces.append([face[0], face[2], face[3]] +
                                         vertex_offset)
                        else:
                            # Polygon - triangulate (simple fan triangulation)
                            face = face_vertex_indices[
                                current_vertex:current_vertex + count]
                            for i in range(1, count - 1):
                                faces.append([face[0], face[i], face[i + 1]] +
                                             vertex_offset)
                        
                        current_vertex += count
                    
                    vertices.extend(transformed_vertices)
                    vertex_offset += len(transformed_vertices)
            
            # Recursively process children
            for child in prim.GetChildren():
                process_prim(child, transform_matrix)
        
        # Start processing from the root
        root_prim = stage.GetPrimAtPath("/")
        process_prim(root_prim)
        
        if not vertices:
            print(f"Warning: No mesh data found in USD file: {usd_path}")
            return None
        
        # Create trimesh object
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Clean up the mesh (remove duplicate vertices, etc.)
        mesh.process()
        
        return mesh
        
    except Exception as e:
        print(f"Error processing USD file {usd_path}: {e}")
        return None

def convert_usd_to_obj(usd_path: str, obj_path: str, scale: float = 1.0) -> bool:
    """Convert a single USD file to OBJ format.
    
    Args:
        usd_path: Path to input USD file
        obj_path: Path to output OBJ file
        scale: Scale factor to apply
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Extract mesh from USD
        mesh = extract_mesh_from_usd(usd_path, scale)
        if mesh is None:
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        
        # Export to OBJ
        mesh.export(obj_path)
        
        print(f"✓ Converted: {usd_path} -> {obj_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {usd_path}: {e}")
        return False


def batch_convert_usd_to_obj(
    object_scales_json: str,
    object_root: str,
    output_dir: str,
    overwrite_existing: bool = False,
    max_files: Optional[int] = None
) -> Tuple[int, int]:
    """
    Batch convert USD files to OBJ format based on a scales JSON file.
    
    Args:
        object_scales_json: Path to JSON file containing object paths and scales
        object_root: Root directory for the objects in the JSON file
        output_dir: Output directory for OBJ files
        overwrite_existing: Whether to overwrite existing OBJ files
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        Tuple of (successful_conversions, total_files)
    """
    # Load the scales JSON file
    try:
        with open(object_scales_json, 'r') as f:
            objects_and_scales = json.load(f)
    except Exception as e:
        print(f"Error loading scales JSON file {object_scales_json}: {e}")
        return 0, 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful_conversions = 0
    total_files = 0
    
    print(f"Starting batch conversion of {len(objects_and_scales)} USD files...")
    print(f"Object root: {object_root}")
    print(f"Output directory: {output_dir}")
    
    for i, (object_path, scale) in enumerate(objects_and_scales.items()):
        if max_files is not None and i >= max_files:
            break
            
        total_files += 1
        
        # Construct full paths
        full_usd_path = os.path.join(object_root, object_path)
        
        # Determine output OBJ path
        obj_filename = os.path.splitext(os.path.basename(object_path))[0] + f".{scale}.obj"
        obj_path = os.path.join(output_dir, obj_filename)
        
        # Check if output already exists
        if os.path.exists(obj_path) and not overwrite_existing:
            print(f"⏭️  Skipping (exists): {obj_path}")
            continue
        
        # Check if input file exists
        if not os.path.exists(full_usd_path):
            print(f"✗ Input file not found: {full_usd_path}")
            continue
        
        # Convert the file
        if convert_usd_to_obj(full_usd_path, obj_path, scale):
            successful_conversions += 1
    
    print("\nBatch conversion complete!")
    print(f"Successful conversions: {successful_conversions}/{total_files}")
    print(f"Output directory: {output_dir}")
    
    return successful_conversions, total_files


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert USD files to OBJ format using pxr package"
    )
    
    # Single file conversion arguments
    parser.add_argument(
        "--usd_file",
        type=str,
        help="Path to single USD file to convert (for single file conversion)"
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the mesh (for single file conversion)"
    )
    
    # Batch conversion arguments
    parser.add_argument(
        "--object_scales_json",
        type=str,
        help="Path to JSON file containing object paths and scales (for batch conversion)"
    )
    
    parser.add_argument(
        "--object_root",
        type=str,
        help="Root directory for the objects in the JSON file (for batch conversion)"
    )
    
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output directory for OBJ files"
    )
    
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        default=False,
        help="Overwrite existing OBJ files (default: skip existing files)"
    )
    
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing, batch mode only)"
    )
    
    args = parser.parse_args()
    
    # Determine conversion mode
    if args.usd_file:
        # Single file conversion mode
        if not os.path.exists(args.usd_file):
            print(f"Error: USD file not found: {args.usd_file}")
            sys.exit(1)
        
        # Determine output path
        if args.scale == 1.0:
            obj_filename = os.path.splitext(os.path.basename(args.usd_file))[0] + ".obj"
        else:
            obj_filename = os.path.splitext(os.path.basename(args.usd_file))[0] + f".{args.scale}.obj"
        obj_path = os.path.join(args.output_dir, obj_filename)
        
        # Perform single file conversion
        success = convert_usd_to_obj(args.usd_file, obj_path, args.scale)
        
        if not success:
            print("Conversion failed!")
            sys.exit(1)
            
    elif args.object_scales_json and args.object_root:
        # Batch conversion mode
        if not os.path.exists(args.object_scales_json):
            print(f"Error: Object scales JSON file not found: "
                  f"{args.object_scales_json}")
            sys.exit(1)
        
        if not os.path.exists(args.object_root):
            print(f"Error: Object root directory not found: {args.object_root}")
            sys.exit(1)
        
        # Perform batch conversion
        successful, total = batch_convert_usd_to_obj(
            object_scales_json=args.object_scales_json,
            object_root=args.object_root,
            output_dir=args.output_dir,
            overwrite_existing=args.overwrite_existing,
            max_files=args.max_files
        )
        
        if successful == 0 and total > 0:
            print("No files were successfully converted!")
            sys.exit(1)
        elif successful < total:
            print(f"Warning: Only {successful}/{total} files were converted "
                  f"successfully.")
            sys.exit(1)
    else:
        print("Error: Must specify either --usd_file (for single file) or "
              "--object_scales_json and --object_root (for batch conversion)")
        sys.exit(1)


if __name__ == "__main__":
    main() 
