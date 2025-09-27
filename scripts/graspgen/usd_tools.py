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

import numpy as np
import warp as wp
import trimesh
from warp_kernels import add_constant_kernel, concatenate_kernel, transform_points_kernel


def validate_transform(xform):
    """
    Validate a transform array for NaN, infinite, or other invalid values.
    
    Args:
        xform: Transform array [x, y, z, qx, qy, qz, qw]
        
    Returns:
        bool: True if valid, False if invalid
    """
    if xform is None:
        return False
    
    # Check for NaN values
    if np.isnan(xform).any():
        return False
    
    # Check for infinite values
    if np.isinf(xform).any():
        return False
    
    # Check quaternion normalization (last 4 elements)
    if len(xform) >= 7:
        qx, qy, qz, qw = xform[3], xform[4], xform[5], xform[6]
        quat_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        # Allow some tolerance for floating point precision
        if quat_norm < 1e-6 or quat_norm > 1e6:
            return False
    
    return True


def transform_to_matrix(xform):
    # Validate input transform
    if not validate_transform(xform):
        print(f"DEBUG transform_to_matrix: Invalid transform detected: {xform}")
        # Return identity matrix for invalid input
        return np.eye(4)
    
    xform = wp.transform((xform[0], xform[1], xform[2]), (xform[3], xform[4], xform[5], xform[6]))
    p = wp.transform_get_translation(xform)
    q = wp.transform_get_rotation(xform)
    rot = wp.quat_to_matrix(q)
    # fmt: off
    wpmat = wp.mat44(
        rot[0][0], rot[0][1], rot[0][2], p[0],
        rot[1][0], rot[1][1], rot[1][2], p[1],
        rot[2][0], rot[2][1], rot[2][2], p[2],
        0.0, 0.0, 0.0, 1.0,
    )
    r = np.eye(4)
    r[0, 0] = wpmat[0, 0]
    r[0, 1] = wpmat[0, 1]
    r[0, 2] = wpmat[0, 2]
    r[0, 3] = wpmat[0, 3]
    r[1, 0] = wpmat[1, 0]
    r[1, 1] = wpmat[1, 1]
    r[1, 2] = wpmat[1, 2]
    r[1, 3] = wpmat[1, 3]
    r[2, 0] = wpmat[2, 0]
    r[2, 1] = wpmat[2, 1]
    r[2, 2] = wpmat[2, 2]
    r[2, 3] = wpmat[2, 3]
    r[3, 3] = wpmat[3, 3] 

    return r

def matrix_to_transform(matrix):
    """
    Convert a 4x4 transformation matrix to wp.transform format.

    Args:
        matrix: 4x4 numpy array or list representing transformation matrix

    Returns:
        wp.transform: Transform in warp format [x, y, z, qx, qy, qz, qw]
    """
    matrix = np.array(matrix, dtype=np.float32)

    # Extract translation (last column, first 3 elements)
    translation = matrix[:3, 3]

    # Extract rotation matrix (top-left 3x3)
    rotation_matrix = matrix[:3, :3]

    # Convert rotation matrix to quaternion
    quat = wp.quat_from_matrix(wp.mat33(rotation_matrix.flatten()))

    # Create wp.transform
    return wp.transform(translation, quat)

def get_prim_transform(prim, transpose_rotation=True):
    # Ensure Isaac Lab is started before importing pxr
    from graspgen_utils import start_isaac_lab_if_needed
    # Start Isaac Lab in headless mode for USD operations
    start_isaac_lab_if_needed(file_name=__file__, headless=True)
    
    from pxr import UsdGeom
    xform = UsdGeom.Xform(prim)
    mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    if transpose_rotation:
        rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    else:
        rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].flatten()))
    pos = mat[3, :3]
    scale = np.ones(3, dtype=np.float32)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale = np.array(op.Get(), dtype=np.float32)
    return wp.transform(pos, rot), scale

def get_prim_collision_mesh(prim, collision_enabled, incoming_transform=None, incoming_scale=None, mesh_data=None, device="cuda"):
    """
    Get the collision mesh for a primitive in the USD stage.
    
    Args:
        prim (pxr.UsdPrim): The primitive to get the collision mesh for
        collision_enabled (bool): Whether the collision mesh is enabled
        incoming_transform (wp.transform): The transform to apply to the mesh vertices
        incoming_scale (np.ndarray): The scale to apply to the mesh vertices
        mesh_data (dict): Dictionary to store the mesh data with keys:
            - 'vertices': List of vertex positions
            - 'indices': List of triangle indices
        device (str): Device to use for warp arrays (default: "cuda")
    
    Returns:
        dict: The updated mesh_data dictionary containing the collision mesh information
    """
    # Ensure Isaac Lab is started before importing pxr
    from graspgen_utils import start_isaac_lab_if_needed
    # Start Isaac Lab in headless mode for USD operations
    start_isaac_lab_if_needed(file_name=__file__, headless=True)
    
    from pxr import UsdGeom
    
    # We only want to return the mesh on the original call, the other calls pass through parameters
    return_mesh = False
    if mesh_data is None:
        mesh_data = {"vertices": wp.array(dtype=wp.vec3, device=device), "indices": wp.array(dtype=wp.int32, device=device)}
        return_mesh = True
    type_name = str(prim.GetTypeName())
    path = str(prim.GetPath())
    schemas = set(prim.GetAppliedSchemas())

    # get the transform and scale of the primitive
    local_transform, local_scale = get_prim_transform(prim)
    if incoming_transform is None:
        xform = wp.transform()
    else:
        xform = wp.transform_multiply(incoming_transform, local_transform)
    if incoming_scale is None:
        scale = local_scale # need local scale but not local transform on first pass?
    else:
        scale = incoming_scale * local_scale

    # we really only want to collect meshes if collision has been enabled in ancestors
    if "PhysicsCollisionAPI" in schemas:
        collision_enabled = prim.GetAttribute("physics:collisionEnabled").Get()

    # The children of instanced prims are not gotten through GetChildren(), but through GetPrototype().GetChildren()
    if prim.IsInstance():
        children_refs = prim.GetPrototype().GetChildren()
    else:
        children_refs = prim.GetChildren()

    # for now the only collision objects we support are meshes
    if collision_enabled and type_name == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        points = wp.array(np.array(mesh.GetPointsAttr().Get(), dtype=np.float32), dtype=wp.vec3, device=device)
        indices = wp.array(np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32), dtype=wp.int32, device=device)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

        # Verify all faces are triangles
        if not np.all(counts == 3):
            non_tri_faces = np.where(counts != 3)[0]
            raise ValueError(f"Mesh at {prim.GetPath()} contains non-triangular faces at indices {non_tri_faces}. All faces must be triangles.")

    if collision_enabled and (type_name == "Mesh" or type_name == "Mesh"): # "Cube" here if you can get the above working.

        # Move points to GPU and perform scaling and transformation in-place
        wp.launch(transform_points_kernel, dim=len(points), inputs=[points, xform, wp.vec3(scale)], device=device)
        # Append the new vertices and indices
        new_vertices = wp.array(dtype=wp.vec3, shape=len(mesh_data['vertices']) + len(points), device=device)
        wp.copy(new_vertices, mesh_data['vertices'])
        wp.launch(concatenate_kernel, dim=len(points), inputs=[new_vertices, points, len(mesh_data['vertices'])], device=device)
        
        new_indices = wp.array(dtype=wp.int32, shape=len(mesh_data['indices']) + len(indices), device=device)
        wp.copy(new_indices, mesh_data['indices'])
        start_idx = len(mesh_data['vertices'])
        wp.launch(kernel=add_constant_kernel, dim=len(indices), inputs=[indices, start_idx], device=device)
        wp.launch(concatenate_kernel, dim=len(indices), inputs=[new_indices, indices, len(mesh_data['indices'])], device=device)

        mesh_data['vertices'] = new_vertices
        mesh_data['indices'] = new_indices

    for child in children_refs:
        get_prim_collision_mesh(child, collision_enabled, xform, scale, mesh_data, device)

    if return_mesh:
        # for some reason, when importing to usd, there are a ton of duplicate vertices
        # so we merge them here before returning

        # This will put the mesh in the same frame of reference needed to use approach axis, open axis, and mid axis
        # it will also allow us to use to calculate the bite point, and update the body_transforms to the correct frame
        local_transform = wp.transform(wp.vec3(0, 0, 0), wp.transform_get_rotation(local_transform))
        wp.launch(transform_points_kernel, 
                  dim=len(mesh_data['vertices']), 
                  inputs=[mesh_data['vertices'], local_transform, wp.vec3(local_scale)],
                  device=device)
        verts = mesh_data['vertices'].numpy()
        faces = mesh_data['indices'].numpy()
        # Reshape faces from [n*3] to [n, 3]
        faces = faces.reshape(-1, 3)
        #print(f'before num verts: {len(verts)}')
        #print(f'before num faces: {len(faces)}')
        mesh = trimesh.Trimesh(verts, faces)
        #print(f'after num verts: {len(mesh.vertices)}')
        #print(f'after num faces: {len(mesh.faces)}')
        mesh_data['bbox'] = mesh.bounds
        mesh_data['vertices'] = wp.array(mesh.vertices, dtype=wp.vec3, device=device)
        mesh_data['indices'] = wp.array(mesh.faces, dtype=wp.int32, device=device).flatten()
        return mesh_data, wp.transform_inverse(local_transform) # we need the local_transform_inverse to update the body_transforms to the correct frame
    else:
        return None
    
def get_usd_mesh(source, scale, device="cuda"):
    """Get the mesh for a USD file.
    
    Args:
        source: Path to USD file or USD stage object
        scale: Scale factor to apply to the mesh
        device: Device to use for warp arrays (default: "cuda")
        
    Returns:
        dict: Dictionary containing mesh data with keys:
            - 'vertices': Warp array of vertex positions
            - 'indices': Warp array of triangle indices
            - 'bbox': Mesh bounding box
    """
    # Ensure Isaac Lab is started before importing pxr
    from graspgen_utils import start_isaac_lab_if_needed
    # Start Isaac Lab in headless mode for USD operations
    start_isaac_lab_if_needed(file_name=__file__, headless=True)
    
    try:
        from pxr import Usd
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e
    if isinstance(source, str):
        stage = Usd.Stage.Open(source, Usd.Stage.LoadAll)
    else:
        stage = source
    mesh, _ = get_prim_collision_mesh(
        prim=stage.GetPrimAtPath("/"), collision_enabled=True, incoming_scale=scale, device=device)
    return mesh
