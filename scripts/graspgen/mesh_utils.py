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

"""Utility functions for mesh conversion and USD creation."""

import os
from pathlib import Path
import numpy as np
import trimesh
from typing import Optional, Tuple

def create_prim_from_vertices(
    stage,
    prim_path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    translation: Optional[Tuple[float, float, float]] = None,
    orientation: Optional[Tuple[float, float, float, float]] = None,
    scale: Optional[Tuple[float, float, float]] = None,
    mass: Optional[float] = None,
    collision_approximation: str = "convexHull",
    visual_material = None,
    physics_material = None,
    instanceable: bool = True,
) -> None:
    # Import USD and Isaac Lab modules after Isaac Lab is started
    from pxr import Usd, UsdPhysics, PhysxSchema, UsdShade, UsdGeom
    """Create a USD prim from vertex and triangle data with various collision options.

    This function creates a USD prim with a mesh defined from vertices and triangles. It performs the
    following steps:

    1. Creates a USD Xform prim at the path :obj:`prim_path`
    2. Creates a USD Mesh prim with the input vertices and triangles at :obj:`{prim_path}/geometry/mesh`
    3. Sets up collision properties with the specified approximation method
    4. Optionally applies visual and physics materials
    5. Optionally sets up rigid body properties if mass is provided
    6. Optionally makes the geometry instanceable for better performance

    Args:
        prim_path: The path to the primitive to be created
        vertices: Array of vertex positions (Nx3)
        triangles: Array of triangle indices (Mx3)
        translation: The translation of the prim. Defaults to None
        orientation: The orientation of the prim in quaternion format (w,x,y,z). Defaults to None
        scale: The scale of the prim. Defaults to None
        mass: The mass of the rigid body. If None, no rigid body properties are set. Defaults to None
        collision_approximation: The method used for approximating collision mesh. Valid options are:
            "sdf" - Signed Distance Field collision
            "convexDecomposition" - Convex decomposition of the mesh
            "convexHull" - Convex hull of the mesh
            "boundingCube" - Axis-aligned bounding box
            "boundingSphere" - Bounding sphere
            "meshSimplification" - Simplified mesh collision using PhysxTriangleMeshSimplificationCollisionAPI
            "sphereFill" - Sphere fill collision using PhysxSphereFillCollisionAPI
            "triangleMesh" - Exact triangle mesh collision
            "none" - No collision
            Defaults to "convexHull"
        visual_material: The visual material to apply. Defaults to None
        physics_material: The physics material to apply. Defaults to None
        instanceable: Whether to make the geometry instanceable. Defaults to True
    """
    # Create parent prim
    root_prim = UsdGeom.Xform.Define(stage, prim_path)
    root_prim = stage.GetPrimAtPath(prim_path)
    stage.SetDefaultPrim(root_prim)
    xform_root_prim = UsdGeom.Xform(root_prim)
    if translation:
        xform_root_prim.AddTranslateOp().Set(translation)
    if orientation:
        xform_root_prim.AddRotateOp().Set(orientation)
    if scale:
        xform_root_prim.AddScaleOp().Set(scale)

    # Set up rigid body properties if mass is provided
    if mass is not None:
        # Add rigid body API
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_api.GetEnableCCDAttr().Set(True)
        UsdPhysics.MassAPI.Apply(root_prim)
        root_prim.GetAttribute("physics:mass").Set(mass)

    # Create geometry prim
    geom_prim_path = f"{prim_path}/geometry"
    geom_prim = UsdGeom.Xform.Define(stage, geom_prim_path)
    geom_prim = stage.GetPrimAtPath(geom_prim_path)
    
    # Make geometry instanceable if requested
    if instanceable:
        geom_prim.SetInstanceable(True)
        #geom_prim.SetKind("component")

    # Create mesh prim
    mesh_prim_path = f"{geom_prim_path}/mesh"
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_prim_path)
    mesh_prim = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_prim_path))
    mesh_prim.GetPointsAttr().Set(vertices)
    mesh_prim.GetFaceVertexIndicesAttr().Set(triangles.flatten())
    mesh_prim.GetFaceVertexCountsAttr().Set(np.asarray([3] * len(triangles)))
    mesh_prim.GetSubdivisionSchemeAttr().Set("none")
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)

    # Set up collision properties
    if collision_approximation != "none":
        # Add base collision API schemas
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
        #    "PhysicsCollisionAPI",
        #    "PhysxCollisionAPI",
        
        # Add specific collision API based on approximation type
        # TODO: Make all collision types work, boundingCube and boundingSphere, at least, are not working
        if collision_approximation == "sdf":
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
        elif collision_approximation == "convexDecomposition":
            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_prim.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)
            mesh_prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(128)
        elif collision_approximation == "convexHull":
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        #elif collision_approximation == "boundingCube":
        #    PhysxSchema.PhysxBoxCollisionAPI.Apply(mesh_prim)
        elif collision_approximation == "boundingSphere":
            PhysxSchema.PhysxSphereCollisionAPI.Apply(mesh_prim)
        elif collision_approximation == "meshSimplification":
            PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        elif collision_approximation == "sphereFill":
            PhysxSchema.PhysxSphereFillCollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        elif collision_approximation == "triangleMesh":
            collision_approximation = "none" # Not sure why this needs to be, triangleMesh is sort of default like this
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        
        # Set physics approximation attribute
        mesh_prim.GetAttribute("physics:approximation").Set(collision_approximation)
        #mesh_prim.GetAttribute("physxRigidBody:enableCCD").Set(True)

    # Apply visual material if provided
    if visual_material is not None:
        visual_material_path = f"{geom_prim_path}/visualMaterial"
        visual_material.func(visual_material_path, visual_material)
        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        mesh_prim.GetAttribute("material:binding").Set(visual_material_path)

    # Apply physics material if provided
    if physics_material is not None:
        physics_material_path = f"{prim_path}/physicsMaterial"
        material = UsdShade.Material.Define(stage, physics_material_path)
        physics_material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material_api.CreateRestitutionAttr().Set(physics_material.restitution)
        physics_material_api.CreateStaticFrictionAttr().Set(physics_material.static_friction)
        physics_material_api.CreateDynamicFrictionAttr().Set(physics_material.dynamic_friction)
        #xform_root_prim.GetReferences().AddReference(physics_material_path)


        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
        mat_binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants)#, physics_material_path)
        """
        physics_material_path = f"{prim_path}/physicsMaterial"
        UsdShade.Material.Define(stage, physics_material_path)
        physics_material_prim = UsdShade.Material(stage.GetPrimAtPath(physics_material_path))
        #UsdPhysics.MaterialAPI.Apply(physics_material_prim)
        cfg = physics_material.to_dict()
        if "dynamic_friction" in cfg:
            physics_material_prim.CreateDynamicFrictionAttr().Set(cfg["dynamic_friction"])
        if "static_friction" in cfg:
            physics_material_prim.CreateStaticFrictionAttr().Set(cfg["static_friction"])
        if "restitution" in cfg:
            physics_material_prim.GetRestitutionAttr().Set(cfg["restitution"])
        """
        #UsdShade.Material.Define(stage, prim_path)
        #UsdPhysics.Material.Define(stage, physics_material_path)
        #physics_material.func(physics_material_path, physics_material)
        #UsdPhysics.MaterialAPI.Apply(root_prim)
        #root_prim.GetAttribute("material:binding").Set(physics_material_path)


def convert_mesh_to_usd(
    usd_path: str,
    mesh_path: str, 
    overwrite: bool = True,
    vertex_scale: Optional[float] = None,
    translation: Optional[Tuple[float, float, float]] = None,
    orientation: Optional[Tuple[float, float, float, float]] = None,
    scale: Optional[Tuple[float, float, float]] = None,
    mass: Optional[float] = None,
    collision_approximation: str = "convexHull",
    visual_material = None,
    physics_material = None,
) -> str:
    # Import USD and Isaac Lab modules after Isaac Lab is started
    from pxr import Usd
    """Convert a mesh file (OBJ/STL) to USD format.
    
    Args:
        usd_path: Path to the USD file to create
        mesh_path: Path to the mesh file
        overwrite: Whether to overwrite an existing USD file
        See: create_prim_from_vertices for more details
        
    Returns:
        Path to the created USD file
    """
    # Expand user path (handle ~ in file paths)
    mesh_path = os.path.expanduser(mesh_path)
    
    # Check if file exists
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    # If USD file exists and we're not overwriting, return its path
    if os.path.exists(usd_path) and not overwrite:
        return usd_path
    
    # Load the mesh file
    mesh = trimesh.load(mesh_path, validate=True)
    if vertex_scale is not None:
        mesh.apply_scale(vertex_scale)
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.faces)

    stage = Usd.Stage.CreateNew(usd_path)

    # Create the USD prim
    create_prim_from_vertices(
        stage=stage,
        prim_path="/object",
        vertices=vertices,
        triangles=triangles,
        translation=translation,
        orientation=orientation,
        scale=scale,
        mass=mass,
        collision_approximation=collision_approximation,
        visual_material=visual_material,
        physics_material=physics_material,
        instanceable=True,
    )
    
    stage.Save()
    stage = None
    #stage.Close()
    
    # Check if USD file was created
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not created: {usd_path}")
    
    return usd_path