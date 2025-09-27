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

# Visualization tools for grasp generation debugging.
#
# This module provides functions to visualize object meshes and normals
# at random sample points in meshcat, useful for debugging grasp generation.
#
# Usage:
#     from visualize_tools import visualize_object_and_normals
#     visualize_object_and_normals(object, work_points, work_normals, clear_view=True)
#     visualize_object_and_normals(object, work_points, work_normals, clear_view=False)  # Update existing view

import numpy as np
import trimesh

# Try to import meshcat, but don't fail if it's not available
try:
    import meshcat
    import meshcat.geometry as mcg
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    print("Warning: meshcat not available. Visualization functions will be disabled.")
    print("To enable visualization, install: pip install meshcat")


def visualize_object_and_normals(object, work_points, work_normals, 
                                 clear_view=True, normal_length=0.01, 
                                 object_color=None, normal_color=None,
                                 vis=None):
    """Visualize object mesh and normals at random sample points in meshcat.
    
    Args:
        object: GuessObject instance containing the object mesh
        work_points: Warp array of shape (N, 3) containing random sample points
        work_normals: Warp array of shape (N, 3) containing normals at sample points
        clear_view: If True, clear the existing meshcat view before adding new objects
        normal_length: Length of normal vectors to display
        object_color: RGB color for object mesh as [r, g, b] values in [0, 1]. Default: [0.7, 0.7, 0.7]
        normal_color: RGB color for normal vectors as [r, g, b] values in [0, 1]. Default: [1.0, 0.0, 0.0] (red)
        vis: Existing meshcat visualizer instance. If None, creates a new one.
    
    Returns:
        meshcat.Visualizer: The visualizer instance used, or None if meshcat not available
    """
    if not MESHCAT_AVAILABLE:
        print("Warning: meshcat not available. Skipping visualization.")
        return None
    # Set default colors
    if object_color is None:
        object_color = [0.7, 0.7, 0.7]  # Default gray
    if normal_color is None:
        normal_color = [1.0, 0.0, 0.0]  # Default red
    
    # Create or use existing visualizer
    if vis is None:
        vis = meshcat.Visualizer()
        print(f"Created new meshcat visualizer at: {vis.url()}")
    else:
        print(f"Using existing meshcat visualizer at: {vis.url()}")
    
    # Clear view if requested
    if clear_view:
        vis.delete()
        print("Cleared existing meshcat view")
    
    # Convert warp arrays to numpy for visualization
    points_np = work_points.numpy()
    normals_np = work_normals.numpy()
    
    # Visualize the object mesh
    _visualize_object_mesh(vis, object, object_color)
    
    # Visualize normals at sample points
    _visualize_sample_normals(vis, points_np, normals_np, normal_length, normal_color)
    
    print(f"Visualized object with {len(points_np)} sample points and normals")
    return vis


def _visualize_object_mesh(vis, object, color):
    """Helper function to visualize the object mesh.
    
    Args:
        vis: Meshcat visualizer instance
        object: GuessObject instance
        color: RGB color as [r, g, b] values in [0, 1]
    """
    if not MESHCAT_AVAILABLE:
        return
    # Convert warp mesh to trimesh for easier visualization
    vertices = object.points.numpy()
    faces = object.indices.numpy().reshape(-1, 3)
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Create meshcat geometry
    geometry = mcg.TriangularMeshGeometry(
        vertices=mesh.vertices,
        faces=mesh.faces
    )
    
    # Create material with color
    material = mcg.MeshLambertMaterial(
        color=int(color[0] * 255) << 16 | int(color[1] * 255) << 8 | int(color[2] * 255)
    )
    
    # Set object in visualizer
    vis["object_mesh"].set_object(geometry, material)
    print(f"Added object mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")


def _visualize_sample_normals(vis, points, normals, length, color):
    """Helper function to visualize normals at sample points.
    
    Args:
        vis: Meshcat visualizer instance
        points: Numpy array of shape (N, 3) containing sample points
        normals: Numpy array of shape (N, 3) containing normals
        length: Length of normal vectors
        color: RGB color as [r, g, b] values in [0, 1]
    """
    if not MESHCAT_AVAILABLE:
        return
    positions = []
    colors = []
    
    for i, (point, normal) in enumerate(zip(points, normals)):
        # Normalize the normal vector
        normal_norm = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Start and end points of normal vector
        start = point
        end = point + normal_norm * length
        
        positions.extend([start, end])
        
        # Use uniform color for all normals
        colors.extend([color, color])
    
    if len(positions) > 0:
        # Create line segments for normals
        vis["sample_normals"].set_object(
            mcg.LineSegments(
                mcg.PointsGeometry(
                    position=np.array(positions).T.astype(np.float32),
                    color=np.array(colors).T.astype(np.float32)
                ),
                mcg.LineBasicMaterial(vertexColors=True)
            )
        )
        print(f"Added {len(points)} normal vectors at sample points")


def visualize_points_only(points, clear_view=True, point_color=None, point_size=0.005, vis=None):
    """Visualize only the sample points as spheres in meshcat.
    
    Args:
        points: Warp array or numpy array of shape (N, 3) containing sample points
        clear_view: If True, clear the existing meshcat view before adding new objects
        point_color: RGB color for points as [r, g, b] values in [0, 1]. Default: [0.0, 1.0, 0.0] (green)
        point_size: Size of point spheres
        vis: Existing meshcat visualizer instance. If None, creates a new one.
    
    Returns:
        meshcat.Visualizer: The visualizer instance used, or None if meshcat not available
    """
    if not MESHCAT_AVAILABLE:
        print("Warning: meshcat not available. Skipping visualization.")
        return None
    if point_color is None:
        point_color = [0.0, 1.0, 0.0]  # Default green
    
    # Create or use existing visualizer
    if vis is None:
        vis = meshcat.Visualizer()
        print(f"Created new meshcat visualizer at: {vis.url()}")
    else:
        print(f"Using existing meshcat visualizer at: {vis.url()}")
    
    # Clear view if requested
    if clear_view:
        vis.delete()
        print("Cleared existing meshcat view")
    
    # Convert warp array to numpy if needed
    if hasattr(points, 'numpy'):
        points_np = points.numpy()
    else:
        points_np = np.array(points)
    
    # Create spheres for each point
    for i, point in enumerate(points_np):
        # Create sphere geometry
        sphere = mcg.Sphere(point_size)
        
        # Create material with color
        material = mcg.MeshLambertMaterial(
            color=int(point_color[0] * 255) << 16 | int(point_color[1] * 255) << 8 | int(point_color[2] * 255)
        )
        
        # Set sphere in visualizer with position
        vis[f"sample_point_{i}"].set_object(sphere, material)
        # Position the sphere at the sample point
        transform = np.eye(4)
        transform[:3, 3] = point
        vis[f"sample_point_{i}"].set_transform(transform)
    
    print(f"Added {len(points_np)} sample points as spheres")
    return vis


def create_visualizer():
    """Create and return a new meshcat visualizer instance.
    
    Returns:
        meshcat.Visualizer: New visualizer instance, or None if meshcat not available
    """
    if not MESHCAT_AVAILABLE:
        print("Warning: meshcat not available. Cannot create visualizer.")
        return None
    vis = meshcat.Visualizer()
    print(f"Created new meshcat visualizer at: {vis.url()}")
    return vis


def clear_visualizer(vis):
    """Clear all objects from the meshcat visualizer.
    
    Args:
        vis: Meshcat visualizer instance
    """
    if not MESHCAT_AVAILABLE:
        print("Warning: meshcat not available. Cannot clear visualizer.")
        return
    vis.delete()
    print("Cleared meshcat visualizer")


def visualize_object_and_transforms(object, transforms,
                                   clear_view=True, axis_length=0.02,
                                   object_color=None, vis=None):
    """Visualize object mesh and coordinate frames at transform locations in meshcat.
    
    Args:
        object: GuessObject instance containing the object mesh
        transforms: Warp array or numpy array of shape (N, 4, 4) containing 4x4 transformation matrices
        clear_view: If True, clear the existing meshcat view before adding new objects
        axis_length: Length of coordinate frame axes
        object_color: RGB color for object mesh as [r, g, b] values in [0, 1]. Default: [0.7, 0.7, 0.7]
        vis: Existing meshcat visualizer instance. If None, creates a new one.
    
    Returns:
        meshcat.Visualizer: The visualizer instance used, or None if meshcat not available
    """
    if not MESHCAT_AVAILABLE:
        print("Warning: meshcat not available. Skipping visualization.")
        return None
    
    # Set default colors
    if object_color is None:
        object_color = [0.7, 0.7, 0.7]  # Default gray
    
    # Create or use existing visualizer
    if vis is None:
        vis = meshcat.Visualizer()
        print(f"Created new meshcat visualizer at: {vis.url()}")
    else:
        print(f"Using existing meshcat visualizer at: {vis.url()}")
    
    # Clear view if requested
    if clear_view:
        vis.delete()
        print("Cleared existing meshcat view")
    
    # Convert warp arrays to numpy for visualization
    if hasattr(transforms, 'numpy'):
        transforms_np = transforms.numpy()
    else:
        transforms_np = np.array(transforms)
    
    # Visualize the object mesh
    _visualize_object_mesh(vis, object, object_color)
    
    # Visualize coordinate frames at transform locations
    _visualize_transform_frames(vis, transforms_np, axis_length)
    
    print(f"Visualized object with {len(transforms_np)} transform coordinate frames")
    return vis


def _visualize_transform_frames(vis, transforms, axis_length):
    """Helper function to visualize coordinate frames at transform locations.
    
    Args:
        vis: Meshcat visualizer instance
        transforms: Numpy array of shape (N, 4, 4) containing transformation matrices
        axis_length: Length of coordinate frame axes
    """
    if not MESHCAT_AVAILABLE:
        return
    
    for i, transform in enumerate(transforms):
        # Extract translation and rotation from transform matrix
        translation = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        
        # Create coordinate frame name
        frame_name = f"transform_frame_{i}"
        
        # X-axis (red)
        x_axis_end = translation + rotation_matrix[:, 0] * axis_length
        vis[f"{frame_name}/x_axis"].set_object(
            mcg.Line(
                mcg.PointsGeometry(np.array([translation, x_axis_end]).T),
                mcg.MeshBasicMaterial(color=0xFF0000)  # Red
            )
        )
        
        # Y-axis (green)
        y_axis_end = translation + rotation_matrix[:, 1] * axis_length
        vis[f"{frame_name}/y_axis"].set_object(
            mcg.Line(
                mcg.PointsGeometry(np.array([translation, y_axis_end]).T),
                mcg.MeshBasicMaterial(color=0x00FF00)  # Green
            )
        )
        
        # Z-axis (blue)
        z_axis_end = translation + rotation_matrix[:, 2] * axis_length
        vis[f"{frame_name}/z_axis"].set_object(
            mcg.Line(
                mcg.PointsGeometry(np.array([translation, z_axis_end]).T),
                mcg.MeshBasicMaterial(color=0x0000FF)  # Blue
            )
        )
    
    print(f"Added {len(transforms)} coordinate frames at transform locations")


if __name__ == "__main__":
    # Example usage
    print("This module provides visualization tools for grasp generation debugging.")
    print("Import and use the functions in your code:")
    print("  from visualize_tools import visualize_object_and_normals")
    print("  vis = visualize_object_and_normals(object, work_points, work_normals)")
