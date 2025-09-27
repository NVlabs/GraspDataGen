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

from grasp_constants import GraspState
import warp as wp
from typing import Any
from warp_functions import (
    wp_inverse_rigid_transform,
    wp_plane_transform_axis,
    compute_offset_along_negative_normal,
    triangle_mesh_intersect,
    mat44_to_transform,
    transform_to_mat44,
)


# fill all the more closed position with too closed.
# closed position is -1, and open is 0
@wp.kernel
def fill_are_offsets_invalid_kernel(
    offsets: wp.array(dtype=wp.int32),
    is_invalid: wp.array2d(dtype=wp.int32),
):
    offset_idx, tid = wp.tid()
    offset = offsets[tid]
    # if we don't center the fingers, they have already bee checked for collision and are valid

    if offset_idx > offset:
        is_invalid[offset_idx, tid] = GraspState.TOO_CLOSED
    else:
        is_invalid[offset_idx, tid] = GraspState.VALID


@wp.kernel
def intersect_the_offsets_with_offsets(
    do_not_center_finger_opening: int,
    offsets: wp.array(dtype=wp.int32),
    verts0: wp.array(dtype=wp.vec3),
    tris0: wp.array(dtype=wp.int32),
    mesh1: wp.uint64,
    verts1: wp.array(dtype=wp.vec3),
    tris1: wp.array(dtype=wp.int32),
    xform0: wp.array(dtype=wp.mat44),  # offset_idx
    xform1: wp.array(dtype=wp.mat44),  # idx
    is_invalid: wp.array2d(dtype=wp.int32),
):
    offset_idx, idx, face = wp.tid()
    # if  we didn't center the fingers, then the finger at offset is valid
    # remember, offsets is a idx lenght array of offset indices.
    offset = offsets[idx]
    if do_not_center_finger_opening and offset_idx == offset:
        return
    if is_invalid[offset_idx, idx] != GraspState.VALID:
        return
    xform = xform1[idx] @ xform0[offset_idx]
    had_intersection = triangle_mesh_intersect(face, xform, verts0, tris0, mesh1, verts1, tris1)
    if had_intersection:
        is_invalid[offset_idx, idx] = GraspState.IN_COLLISION

@wp.kernel
def find_widest_valid_opening_kernel(
    num_openings: int,
    are_offsets_invalid: wp.array2d(dtype=wp.int32),
    is_invalid: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    offset = offsets[tid]
    if are_offsets_invalid[offset, tid] != GraspState.VALID:
        is_invalid[tid] = GraspState.IN_COLLISION
    else:
        for i in range(offset, -1, -1):
            if i > 0:
                if are_offsets_invalid[i-1, tid] != GraspState.VALID:
                    break
        offsets[tid] = i

@wp.kernel
def get_body_transforms_acronym(
    body_transforms: wp.array(dtype=wp.mat44),
    transforms: wp.array(dtype=wp.mat44),
    offsets: wp.array(dtype=wp.int32),
    transforms_out: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    offset_idx = offsets[tid]
    trans = transforms[tid] @ body_transforms[offset_idx]
    xform = mat44_to_transform(trans)
    transforms_out[tid] = xform

@wp.kernel
def compute_acronym_transforms_from_random_samples(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    seed: wp.int32,
    percent_random: wp.float32,
    axes_lengths: wp.array(dtype=wp.float32),
    num_openings: wp.int32,
    open_widths_reverse: wp.array(dtype=wp.float32),
    open_configuration_offset: wp.int32,
    correct_acronym_approach: wp.bool,
    open_axis: wp.int32,
    negative_normal: wp.bool,
    transforms: wp.array(dtype=wp.mat44),
    offsets: wp.array(dtype=wp.int32)):
    """
    Sample a triangle mesh and return a list of transforms that
    are, the transform of the sample transform in the object frame.

    Args:
        points: The points to sample
        normals: The normals of the points
        seed: The seed for the random number generator
        percent_random: The percentage of random samples
        standoff_distance: The distance to standoff from the surface
        open_axis: The axis around which to rotate (0=x, 1=y, 2=z)
        transforms: The array to store the transforms
    """
    id = wp.tid()
    rng = wp.rand_init(seed, id)
    x = points[id]
    n = normals[id]

    #offset = num_openings - 1 - offset_reverse
    # offset_reverse: 3, offset: 5, length: 0.081
    # open_widths: [0.1514, 0.1402, 0.1286, 0.1246, 0.1048, 0.0817, 0.0558, 0.0285, -0.0002]
    #wp.printf("offset_reverse: %d, offset: %d, length: %f\n", offset_reverse, offset, length)
    length = axes_lengths[id]
    if correct_acronym_approach:
        offset_reverse = wp.lower_bound(open_widths_reverse, length)
        offsets[id] = num_openings - 1 - offset_reverse
    else:
        offsets[id] = open_configuration_offset
        offset_reverse = num_openings - 1 - open_configuration_offset

    #offset_reverse = num_openings - 1 - open_configuration_offset

    # in this case, the standoff is half open_width - length
    standoff = 0.50*(open_widths_reverse[offset_reverse] - length)
    x = x + n * standoff
    if negative_normal:
        n = -n

    contact_transform = wp_plane_transform_axis(x, n, open_axis)

    angle = wp.randf(rng) * wp.tau - wp.pi
    random_chance = wp.randf(rng)
    if random_chance >= percent_random:
        which_angle = id % 4
        if which_angle == 0:
            angle = -wp.pi
        elif which_angle == 1:
            angle = -wp.pi / 2.0
        elif which_angle == 2:
            angle = 0.0
        else:
            angle = wp.pi / 2.0
    
    # Create rotation matrix around the specified axis
    rot_trans = wp.mat44(1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0)

    # Apply rotation around the specified axis and set standoff distance
    #rot_trans[open_axis, 3] = standoff
    #rot_trans[approach_axis, 3] = standoff
    if open_axis == 0:  # x-axis
        rot_trans[1, 1] = wp.cos(angle)
        rot_trans[1, 2] = -wp.sin(angle)
        rot_trans[2, 1] = wp.sin(angle)
        rot_trans[2, 2] = wp.cos(angle)
        #rot_trans[0, 3] = standoff  # Apply standoff along x-axis
    elif open_axis == 1:  # y-axis
        rot_trans[0, 0] = wp.cos(angle)
        rot_trans[0, 2] = wp.sin(angle)
        rot_trans[2, 0] = -wp.sin(angle)
        rot_trans[2, 2] = wp.cos(angle)
        #rot_trans[1, 3] = standoff  # Apply standoff along y-axis
    else:  # z-axis (default)
        rot_trans[0, 0] = wp.cos(angle)
        rot_trans[0, 1] = -wp.sin(angle)
        rot_trans[1, 0] = wp.sin(angle)
        rot_trans[1, 1] = wp.cos(angle)
        #rot_trans[2, 3] = standoff  # Apply standoff along z-axis
    
    T = rot_trans @ contact_transform
    transforms[id] = T

@wp.kernel
def find_collision_axis_lengths(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    lengths: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    x = points[tid]
    n = normals[tid]
    offset = 2.0 # 2 meters should do it, coming from the other side of the object.
    ray_origin = x - n * offset 
    ray_dir = n

    query = wp.mesh_query_ray(mesh, ray_origin, ray_dir, wp.inf)
    """
        result (bool): Whether a hit is found within the given constraints.
        sign (float32): A value > 0 if the ray hit in front of the face, returns < 0 otherwise.
        face (int32): Index of the closest face.
        t (float32): Distance of the closest hit along the ray.
        u (float32): Barycentric u coordinate of the closest hit.
        v (float32): Barycentric v coordinate of the closest hit.
        normal (vec3f): Face normal.
    """
    length = offset - query.t
    lengths[tid] = length

@wp.kernel
def set_offsets_acronym(
    num_true_random: wp.int32,
    offsets: wp.array(dtype=wp.int32)
):
    random_idx, orientation_idx = wp.tid()
    idx = orientation_idx*num_true_random + random_idx
    offsets[idx] = offsets[random_idx]

@wp.kernel
def copy_vec3(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    dst[tid] = src[tid]

# change the basis of all the bodies except ref_idx.
@wp.kernel
def reframe_to_new_body(ref_idx: int, transforms: wp.array2d(dtype=wp.mat44)):
    b_idx, tid = wp.tid()
    if b_idx >= ref_idx:
        b_idx += 1
    inv_frame_xform = wp_inverse_rigid_transform(transforms[ref_idx, tid])
    transforms[b_idx, tid] = inv_frame_xform @ transforms[b_idx, tid]

@wp.kernel
def add_constant_kernel(arr: wp.array(dtype=Any), constant: Any):
    """Add a constant value to each element in the array.
    
    Args:
        arr: Input array to modify
        constant: Value to add to each element
    """
    thread_id = wp.tid()
    arr[thread_id] += constant

@wp.kernel
def multiply_constant_kernel(arr: wp.array(dtype=Any), constant: Any):
    """Multiply a constant value to each element in the array.
    
    Args:
        arr: Input array to modify
        constant: Value to multiply to each element
    """
    thread_id = wp.tid()
    arr[thread_id] *= constant

@wp.kernel
def add_translation_kernel(arr: wp.array(dtype=wp.transform), translation: wp.vec3):
    """Add a translation to each element in the array.
    
    Args:
        arr: Input array to modify
        translation: Value to add to each element
    """
    thread_id = wp.tid()
    p = wp.transform_get_translation(arr[thread_id]) + translation
    arr[thread_id] = wp.transform(p, wp.transform_get_rotation(arr[thread_id]))

@wp.kernel
def add_2d_translation_kernel(arr: wp.array2d(dtype=wp.transform), idx: wp.int32, translation: wp.vec3):
    """Add a translation to each element in the array.
    
    Args:
        arr: Input array to modify
        translation: Value to add to each element
    """
    thread_id = wp.tid()
    trans = arr[idx, thread_id]
    p = wp.transform_get_translation(trans) + translation
    arr[idx, thread_id] = wp.transform(p, wp.transform_get_rotation(trans))

@wp.kernel
def body_to_object_raycast(
    mesh: wp.uint64,
    ray_dir: wp.vec3,
    transforms: wp.array(dtype=wp.mat44),
    body_transforms: wp.array(dtype=wp.mat44),
    body_transform_offsets: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32),
):
    tid, pid = wp.tid() # Trasform idx, and point idx
    offset_idx = body_transform_offsets[tid]
    body_trans = body_transforms[offset_idx]
    body_xform = mat44_to_transform(body_trans)
    v = wp.transform_point(body_xform, vertices[pid])

    trans = transforms[tid]
    xform = mat44_to_transform(trans)
    ray_origin = wp.transform_point(xform, v)
    ray_end = wp.transform_point(xform, v + ray_dir)
    ray_dir = ray_end - ray_origin
    query = wp.mesh_query_ray(mesh, ray_origin, ray_dir, wp.inf)
    """
        result (bool): Whether a hit is found within the given constraints.
        sign (float32): A value > 0 if the ray hit in front of the face, returns < 0 otherwise.
        face (int32): Index of the closest face.
        t (float32): Distance of the closest hit along the ray.
        u (float32): Barycentric u coordinate of the closest hit.
        v (float32): Barycentric v coordinate of the closest hit.
        normal (vec3f): Face normal.
    """
    if query.result:
        wp.atomic_min(distances, tid, query.t)

@wp.kernel
def center_transform_between_distances(
    ray_dir: wp.vec3,
    work_transforms: wp.array(dtype=wp.mat44),
    distances0: wp.array(dtype=wp.float32),
    distances1: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # Only apply centering if both distances are finite (not Inf or NaN)
    if wp.isfinite(distances0[tid]) and wp.isfinite(distances1[tid]):
        trans = work_transforms[tid]
        quat = wp.transform_get_rotation(mat44_to_transform(trans))
        ray_dir = wp.quat_rotate(quat, ray_dir)
        delta = 0.5 * (distances1[tid] - distances0[tid])
        trans[0, 3] += ray_dir[0] * delta
        trans[1, 3] += ray_dir[1] * delta
        trans[2, 3] += ray_dir[2] * delta
        work_transforms[tid] = trans

@wp.kernel
def compute_transforms_from_random_samples(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    seed: wp.int32,
    percent_random: wp.float32,
    standoff_distance: wp.float32,
    open_axis: wp.int32,
    negative_normal: wp.bool,
    transforms: wp.array(dtype=wp.mat44)):
    """
    Sample a triangle mesh and return a list of transforms that
    are, the transform of the sample transform in the object frame.

    Args:
        points: The points to sample
        normals: The normals of the points
        seed: The seed for the random number generator
        percent_random: The percentage of random samples
        standoff_distance: The distance to standoff from the surface
        open_axis: The axis around which to rotate (0=x, 1=y, 2=z)
        transforms: The array to store the transforms
    """
    id = wp.tid()
    rng = wp.rand_init(seed, id) #seed*2 here DEBUG MTC
    x = points[id]
    n = normals[id]
    if negative_normal:
        n = -n
    
    contact_transform = wp_plane_transform_axis(x, n, open_axis)
    rand_num = wp.randf(rng)
    angle = rand_num * wp.tau - wp.pi

    #if id == 0:
    #    wp.printf("\nseed=%i: ", seed)
    
    #if wp.abs(n[0]) > 0.9:
    #    wp.printf("%f, ", rand_num)
    random_chance = wp.randf(rng)
    if random_chance >= percent_random:
        which_angle = id % 4
        if which_angle == 0:
            angle = -wp.pi
        elif which_angle == 1:
            angle = -wp.pi / 2.0
        elif which_angle == 2:
            angle = 0.0
        else:
            angle = wp.pi / 2.0
    
    # Create rotation matrix around the specified axis
    rot_trans = wp.mat44(1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0)
    
    # negaitve normal means postive standoff? check this
    if negative_normal:
        standoff = standoff_distance
    else:
        standoff = -standoff_distance

    # Apply rotation around the specified axis and set standoff distance
    if open_axis == 0:  # x-axis
        rot_trans[1, 1] = wp.cos(angle)
        rot_trans[1, 2] = -wp.sin(angle)
        rot_trans[2, 1] = wp.sin(angle)
        rot_trans[2, 2] = wp.cos(angle)
        rot_trans[0, 3] = standoff  # Apply standoff along x-axis
    elif open_axis == 1:  # y-axis
        rot_trans[0, 0] = wp.cos(angle)
        rot_trans[0, 2] = wp.sin(angle)
        rot_trans[2, 0] = -wp.sin(angle)
        rot_trans[2, 2] = wp.cos(angle)
        rot_trans[1, 3] = standoff  # Apply standoff along y-axis
    else:  # z-axis (default)
        rot_trans[0, 0] = wp.cos(angle)
        rot_trans[0, 1] = -wp.sin(angle)
        rot_trans[1, 0] = wp.sin(angle)
        rot_trans[1, 1] = wp.cos(angle)
        rot_trans[2, 3] = standoff  # Apply standoff along z-axis
    
    T = rot_trans @ contact_transform
    transforms[id] = T#wp_inverse_rigid_transform(T)

@wp.kernel
def concatenate_kernel(
    dest: wp.array(dtype=Any),
    src: wp.array(dtype=Any),
    start_idx: int
):
    """Concatenate source array into destination array starting at specified index.
    
    Args:
        dest: Destination array to write into
        src: Source array to read from
        start_idx: Starting index in destination array
    """
    tid = wp.tid()
    dest[start_idx + tid] = src[tid]

@wp.kernel
def get_body_close_open_pos_kernel(
    num_bodies: int,
    num_envs: int,
    f0_idx: int,
    f1_idx: int,
    root_idx: int,
    transforms: wp.array2d(dtype=wp.transform),
    body_close_open_pos: wp.array(dtype=wp.transform)
):
    # root_pos = gripper_bodies["transforms"][self.args.base_frame][-1].p # CLOSED is the last transform, upper limit
    # trans = transforms[root_idx, num_envs-1]
    body_close_open_pos[0] = transforms[root_idx, num_envs - 1]
    # finger_pos0 = gripper_bodies[f0_idx]["transforms"][-1].p
    body_close_open_pos[1] = transforms[f0_idx, num_envs - 1]
    # finger_pos1 = wp.transform_point(transforms[f1_idx, num_envs-1], finger_pos1)
    body_close_open_pos[2] = transforms[f1_idx, num_envs - 1]
    # finger_pos0 = gripper_bodies[f0_idx]["transforms"][0].p
    body_close_open_pos[3] = transforms[f0_idx, 0]
    # finger_pos1 = gripper_bodies[f1_idx]["transforms"][0].p
    body_close_open_pos[4] = transforms[f1_idx, 0]

@wp.kernel
def get_closest_offset_transforms_kernel(
    num_offsets: int,
    standoff_distance: wp.float32,
    axis: wp.int32,
    is_invalid: wp.array2d(dtype=wp.int32),# num_offsets X num_grasps
    transforms: wp.array(dtype=wp.mat44),
    num_successes: wp.array(dtype=wp.int32),
    idx_map: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    for i in range(num_offsets):
        if is_invalid[i, tid] == GraspState.VALID:
            idx = wp.atomic_add(num_successes, 0, 1)
            idx_map[idx] = tid
            if i == 0:
                break
            trans = transforms[tid]
            offset = float(i) * standoff_distance
            v_offset = compute_offset_along_negative_normal(offset, trans, axis)
            trans[0, 3] += v_offset[0]
            trans[1, 3] += v_offset[1]
            trans[2, 3] += v_offset[2]
            transforms[tid] = trans
            is_invalid[0, tid] = GraspState.VALID
            break

@wp.kernel
def get_offset_positions(
    standoff_distance: wp.float32,
    axis: wp.int32,
    num_offsets: int,
    num_transforms: int,
    transforms: wp.array(dtype=wp.mat44),
    offset_positions: wp.array2d(dtype=wp.vec3)
):
    offset, tid = wp.tid()
    trans = transforms[tid]
    v_offset = compute_offset_along_negative_normal(float(offset) * standoff_distance, trans, axis)
    trans[0, 3] += v_offset[0]
    trans[1, 3] += v_offset[1]
    trans[2, 3] += v_offset[2]
    offset_positions[offset, tid] = wp.vec3(trans[0, 3], trans[1, 3], trans[2, 3])

@wp.kernel # bodies x envs
def get_transforms_kernel(
    pos: wp.array2d(dtype=wp.vec3),
    quat: wp.array2d(dtype=wp.vec4),
    transforms: wp.array2d(dtype=wp.transform),
    num_envs: int,
    reverse_order: bool,
    local_transform_inverses: wp.array(dtype=wp.transform)
):
    """
    This kernel gets the transforms from 0 (fully open) to -1 (fully closed),
    and all interpolated ones in the environments for each of the bodies in the gripper.
    It also takes into account the local transform of the body, so that the transforms
    are in the correct frame of reference the meshes were collected in.
    """
    b_idx, env_idx = wp.tid()
    out_env_idx = env_idx
    if reverse_order:
        out_env_idx = num_envs - 1 - env_idx
    p = pos[env_idx, b_idx]
    # input quats are [w, x, y, z]
    q4 = quat[env_idx, b_idx]
    # warp quats are [x, y, z, w]
    q = wp.quat(q4[1], q4[2], q4[3], q4[0])
    trans = wp.transform(p, q)
    local_transform = local_transform_inverses[b_idx]
    xform = wp.transform_multiply(trans, local_transform)
    transforms[b_idx, out_env_idx] = xform

@wp.kernel
def get_finger1_success_count(
    num_offsets: int,
    idx_map: wp.array(dtype=wp.int32),
    is_invalid: wp.array2d(dtype=wp.int32),
    num_successes: wp.array(dtype=wp.int32),
):
    offset_idx, tid = wp.tid()
    idx = idx_map[tid]
    if is_invalid[offset_idx, idx] == GraspState.VALID:
        if offset_idx == num_offsets - 1 or is_invalid[offset_idx + 1, idx] != GraspState.VALID:
            wp.atomic_add(num_successes, 0, 1)

@wp.kernel
def get_finger1_successes(
    num_offsets: int,
    idx_map: wp.array(dtype=wp.int32),
    is_invalid: wp.array2d(dtype=wp.int32),
    num_successes: wp.array(dtype=wp.int32),  # should start as 0 again
    work_transforms: wp.array(dtype=wp.mat44),  # idx
    finger1_successes_offset: wp.array(dtype=wp.int32),  # int with offset index
    finger1_successes_work_transform: wp.array(dtype=wp.mat44),  # mat44 with work transform
    finger1_successes_idx: wp.array(dtype=wp.int32),  # int with original index of the random point
):
    offset_idx, tid = wp.tid()
    idx = idx_map[tid]
    if is_invalid[offset_idx, idx] == GraspState.VALID:
        if offset_idx == num_offsets - 1 or is_invalid[offset_idx + 1, idx] != GraspState.VALID:
            i = wp.atomic_add(num_successes, 0, 1)
            finger1_successes_offset[i] = offset_idx
            finger1_successes_work_transform[i] = work_transforms[idx]
            finger1_successes_idx[i] = idx

@wp.kernel
def get_body_transforms(
    body_transforms: wp.array(dtype=wp.mat44),
    transforms: wp.array(dtype=wp.mat44),
    offsets: wp.array(dtype=wp.int32),
    transforms_out: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    offset_idx = offsets[tid]
    trans = transforms[tid] @ body_transforms[offset_idx]
    xform = mat44_to_transform(trans)
    transforms_out[tid] = xform

@wp.kernel
def closest_finger0_object_raycast(
    mesh: wp.uint64,
    in_ray_dir: wp.vec3,
    transforms: wp.array(dtype=wp.mat44),
    vertices: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32),
    ray_directions: wp.array(dtype=wp.vec3),
):
    tid, v_idx = wp.tid()
    trans = transforms[tid]
    xform = mat44_to_transform(trans)
    ray_origin = wp.transform_point(xform, vertices[v_idx])
    ray_dir = wp.quat_rotate(wp.transform_get_rotation(xform), in_ray_dir)
    ray_directions[tid] = ray_dir
    query = wp.mesh_query_ray(mesh, ray_origin, ray_dir, wp.inf)
    """
        result (bool): Whether a hit is found within the given constraints.
        sign (float32): A value > 0 if the ray hit in front of the face, returns < 0 otherwise.
        face (int32): Index of the closest face.
        t (float32): Distance of the closest hit along the ray.
        u (float32): Barycentric u coordinate of the closest hit.
        v (float32): Barycentric v coordinate of the closest hit.
        normal (vec3f): Face normal.
    """
    if query.result:
        wp.atomic_min(distances, tid, query.t)

@wp.kernel
def intersect_mesh_along_negative_normal(
    standoff_distance: wp.float32,
    open_axis: wp.int32,
    verts0: wp.array(dtype=wp.vec3),
    tris0: wp.array(dtype=wp.int32),
    mesh1: wp.uint64,
    verts1: wp.array(dtype=wp.vec3),
    tris1: wp.array(dtype=wp.int32),
    xforms: wp.array(dtype=wp.mat44),
    is_invalid: wp.array2d(dtype=wp.int32),  # size of (num_offsets, num_transforms)
):
    offset_idx, idx, face = wp.tid()
    if is_invalid[offset_idx, idx] != GraspState.VALID:
        return
    # mesh_0 is assumed to be the query mesh, we launch one thread
    # for each face in mesh_0 and test it against the opposing mesh's BVH
    # transforms from mesh_0 -> mesh_1 space
    xform = xforms[idx]
    # compute the offset, xform[0:3,open_axis] is the normal
    v_offset = compute_offset_along_negative_normal(float(offset_idx) * standoff_distance, xform, open_axis)
    xform[0, 3] += v_offset[0]
    xform[1, 3] += v_offset[1]
    xform[2, 3] += v_offset[2]
    # load query triangles points and transform to mesh_1's space
    # v0 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 1.0, 0.0))
    # v1 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 0.0, 1.0))
    # v2 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 0.0, 0.0))
    # xform = wp.transpose(xform)
    had_intersection = triangle_mesh_intersect(face, xform, verts0, tris0, mesh1, verts1, tris1)
    if had_intersection:
        is_invalid[offset_idx, idx] = GraspState.IN_COLLISION

@wp.kernel
def intersect_other_body_with_offsets(
    verts0: wp.array(dtype=wp.vec3),
    tris0: wp.array(dtype=wp.int32),
    mesh1: wp.uint64,
    verts1: wp.array(dtype=wp.vec3),
    tris1: wp.array(dtype=wp.int32),
    xform0: wp.array(dtype=wp.mat44),  # offset_idx
    xform1: wp.array(dtype=wp.mat44),  # tid
    offsets: wp.array(dtype=wp.int32),
    is_invalid: wp.array(dtype=wp.int32),  # tid
):
    tid, face = wp.tid()
    if is_invalid[tid] != GraspState.VALID:
        return
    offset_idx = offsets[tid]
    xform = xform1[tid] @ xform0[offset_idx]
    had_intersection = triangle_mesh_intersect(face, xform, verts0, tris0, mesh1, verts1, tris1)
    if had_intersection:
        is_invalid[tid] = GraspState.IN_COLLISION

@wp.kernel
def intersect_with_offsets(
    verts0: wp.array(dtype=wp.vec3),
    tris0: wp.array(dtype=wp.int32),
    mesh1: wp.uint64,
    verts1: wp.array(dtype=wp.vec3),
    tris1: wp.array(dtype=wp.int32),
    xform0: wp.array(dtype=wp.mat44),  # offset_idx
    xform1: wp.array(dtype=wp.mat44),  # idx
    idx_map: wp.array(dtype=wp.int32),
    is_invalid: wp.array2d(dtype=wp.int32),
):
    offset_idx, tid, face = wp.tid()
    idx = idx_map[tid]
    if is_invalid[offset_idx, idx] != GraspState.VALID:
        return
    xform = xform1[idx] @ xform0[offset_idx]
    had_intersection = triangle_mesh_intersect(face, xform, verts0, tris0, mesh1, verts1, tris1)
    if had_intersection:
        is_invalid[offset_idx, idx] = GraspState.IN_COLLISION

# given random grasps in [0-num_true_random) of transforms,
# rotate them by (orientation_idx) * (360/num_orientations) around open_axis
@wp.kernel
def invert_and_orient_grasps(
    transforms: wp.array(dtype=wp.mat44),
    num_true_random: wp.int32,
    num_orientations: wp.int32,
    open_axis: wp.int32,
    seed: wp.int32,
):
    random_idx, orientation_idx = wp.tid()
    trans = transforms[random_idx]
    idx = orientation_idx*num_true_random + random_idx
    if orientation_idx == 0:
        T = trans
    else:
        angle_delta = (wp.pi*2.0)/float(num_orientations)
        angle = float(orientation_idx) * angle_delta
        rng = wp.rand_init(seed, idx)
        # randomize the angle
        rand_angle = wp.randf(rng)
        old_angle = angle
        angle += rand_angle*angle_delta - angle_delta/2.0

        rot_trans = wp.mat44(1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0)
        
        # Apply rotation around the specified axis
        if open_axis == 0:  # x-axis
            rot_trans[1, 1] = wp.cos(angle)
            rot_trans[1, 2] = -wp.sin(angle)
            rot_trans[2, 1] = wp.sin(angle)
            rot_trans[2, 2] = wp.cos(angle)
        elif open_axis == 1:  # y-axis
            rot_trans[0, 0] = wp.cos(angle)
            rot_trans[0, 2] = wp.sin(angle)
            rot_trans[2, 0] = -wp.sin(angle)
            rot_trans[2, 2] = wp.cos(angle)
        else:  # z-axis (default)
            rot_trans[0, 0] = wp.cos(angle)
            rot_trans[0, 1] = -wp.sin(angle)
            rot_trans[1, 0] = wp.sin(angle)
            rot_trans[1, 1] = wp.cos(angle)
        T = rot_trans @ trans
    transforms[idx] = wp_inverse_rigid_transform(T)

@wp.kernel
def random_mesh_sample(
    mesh: wp.uint64,
    nt: wp.int32,
    cumweight: wp.array(dtype=wp.float32),
    seed: wp.int32,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
):
    id = wp.tid()
    rng = wp.rand_init(seed, id)
    scale = cumweight[nt - 1]
    rand_num = scale * wp.randf(rng)

    face = wp.lower_bound(cumweight, rand_num)
    uv = wp.sample_triangle(rng)
    x = wp.mesh_eval_position(mesh, face, uv[0], uv[1])
    n = wp.mesh_eval_face_normal(mesh, face)

    points[id] = x
    normals[id] = n

@wp.kernel
def set_is_success_kernel(
    src_start_idx: wp.int32,
    dst_start_idx: wp.int32,
    is_success_src: wp.array(dtype=wp.bool),
    is_success_dst: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    is_success_dst[tid+dst_start_idx] = wp.int32(is_success_src[tid+src_start_idx])


@wp.kernel
def get_joint_pos_kernel(
    src_start_idx: wp.int32,
    dst_start_idx: wp.int32,
    cspace_values: wp.array2d(dtype=wp.float32),
    cspace_joint_indices: wp.array(dtype=wp.int32),
    joint_pos: wp.array2d(dtype=wp.float32),
):
    env_id, j_idx = wp.tid()
    joint_idx = cspace_joint_indices[j_idx]
    joint_pos[dst_start_idx+env_id, joint_idx] = cspace_values[env_id+src_start_idx, j_idx]

@wp.kernel
def get_joint_pos_kernel(
    src_start_idx: wp.int32,
    dst_start_idx: wp.int32,
    in_joint_pos: wp.array2d(dtype=wp.float32),
    cspace_joint_indices: wp.array(dtype=wp.int32),
    joint_pos: wp.array2d(dtype=wp.float32),
):
    env_id, j_idx = wp.tid()
    joint_idx = cspace_joint_indices[j_idx]
    joint_pos[dst_start_idx+env_id, j_idx] = in_joint_pos[env_id+src_start_idx, joint_idx]

@wp.kernel
def get_body_pos_kernel(
    src_start_idx: wp.int32,
    dst_start_idx: wp.int32,
    body_idx: wp.int32,
    bite_point: wp.vec3,
    src_pos_w: wp.array2d(dtype=wp.vec3),
    src_pos_root: wp.array(dtype=wp.vec3),
    bite_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    bite_points[tid+dst_start_idx] = src_pos_w[tid+src_start_idx, body_idx] - src_pos_root[tid+src_start_idx] + bite_point

@wp.kernel
def transform_points_kernel(
    points: wp.array(dtype=wp.vec3),
    xform: wp.transform,
    scale: wp.vec3
):
    """Transform and scale points in-place using a warp transform.
    
    Args:
        points: Array of 3D points to transform
        xform: Transform to apply to the points
        scale: Scale factor to apply to the points
    """
    tid = wp.tid()
    # Scale and transform in-place
    points[tid] = wp.transform_point(xform, wp.cw_mul(points[tid], scale))

@wp.kernel
def transform_to_mat44_kernel(
    src: wp.array(dtype=wp.transform),
    dst: wp.array(dtype=wp.mat44)
):
    tid = wp.tid()
    dst[tid] = transform_to_mat44(src[tid])

@wp.kernel
def transform_to_mat44_kernel2d(
    src: wp.array2d(dtype=wp.transform),
    dst: wp.array2d(dtype=wp.mat44)
):
    b_idx, env_idx = wp.tid()
    dst[b_idx, env_idx] = transform_to_mat44(src[b_idx, env_idx])

@wp.kernel
def triangle_area(indices: wp.array(dtype=wp.int32),
                  points: wp.array(dtype=wp.vec3),
                  area: wp.array(dtype=wp.float32)):  # I size array
    id = wp.tid()
    a = points[indices[id * 3 + 0]]
    b = points[indices[id * 3 + 1]]
    c = points[indices[id * 3 + 2]]
    area[id] = 0.5 * wp.length(wp.cross(b - a, c - a))

@wp.kernel
def world_to_object_force_kernel(
    world_quat: wp.array2d(dtype=wp.float32),  # in the IsaacLab format of [w,x,y,z]
    force: wp.vec3,  # a constant force to apply
    local_force: wp.array2d(dtype=wp.vec3)  # size of (num_envs, 1)
):
    tid = wp.tid()
    il_quat = world_quat[tid]
    # inverse the quat
    quat = wp.quat(il_quat[1], il_quat[2], il_quat[3], -il_quat[0])
    local_force[tid, 0] = wp.quat_rotate(quat, force)

@wp.kernel
def transform_inverse_kernel(
    src_start_idx: int,
    dst_start_idx: int,
    src: wp.array(dtype=wp.transform),
    dst: wp.array(dtype=wp.transform),
    output_is_lab: bool,
):
    tid = wp.tid()
    inv = wp.transform_inverse(src[tid+src_start_idx])
    if output_is_lab:
        dst[tid+dst_start_idx] = wp.transform(wp.vec3(inv[0], inv[1], inv[2]), wp.quat(inv[6], inv[3], inv[4], inv[5]))
    else:
        dst[tid+dst_start_idx] = inv

@wp.kernel
def lab_to_warp_transform_kernel(
    src_p: wp.array(dtype=wp.vec3),
    src_q: wp.array(dtype=wp.quat), # quat input as (qw, qx, qy, qz)
    dst: wp.array(dtype=wp.transform), # transform output as (x, y, z, qx, qy, qz, qw)
):
    tid = wp.tid()
    dst[tid] = wp.transform(src_p[tid], wp.quat(src_q[tid][1], src_q[tid][2], src_q[tid][3], src_q[tid][0]))

@wp.kernel
def compute_relative_pos_and_rot_kernel(
    src_start_idx: wp.int32,
    dst_start_idx: wp.int32,
    a_p: wp.array(dtype=wp.vec3),
    a_q: wp.array(dtype=wp.quat), # quat input as (qw, qx, qy, qz)
    b_p: wp.array(dtype=wp.vec3),
    b_q: wp.array(dtype=wp.quat), # quat input as (qw, qx, qy, qz)
    dst: wp.array(dtype=wp.transform),
):
    # remember, warp quat is (qx, qy, qz, qw)
    tid = wp.tid()
    src_tid = tid + src_start_idx
    a = wp.transform(a_p[src_tid], wp.quat(a_q[src_tid][1], a_q[src_tid][2], a_q[src_tid][3], a_q[src_tid][0]))
    b = wp.transform(b_p[src_tid], wp.quat(b_q[src_tid][1], b_q[src_tid][2], b_q[src_tid][3], b_q[src_tid][0]))

    dst[tid+dst_start_idx] = wp.transform_inverse(a) * b

@wp.kernel
def ingest_grasp_guess_data_kernel(
    transforms: wp.array(dtype=wp.transform),
    offsets: wp.array(dtype=wp.int32),
    pregrasp_offsets: wp.array(dtype=wp.int32),
    is_invalid: wp.array(dtype=wp.int32),
    idx_map: wp.array(dtype=wp.int32),
    max_succ: int,
    max_fail: int,
    succ_num_grasps: wp.array(dtype=wp.int32),
    succ_transforms: wp.array(dtype=wp.transform),
    succ_offsets: wp.array(dtype=wp.int32),
    succ_pregrasp_offsets: wp.array(dtype=wp.int32),
    succ_idx_map: wp.array(dtype=wp.int32),
    fail_num_grasps: wp.array(dtype=wp.int32),
    fail_transforms: wp.array(dtype=wp.transform),
    fail_offsets: wp.array(dtype=wp.int32),
    fail_pregrasp_offsets: wp.array(dtype=wp.int32),
    fail_idx_map: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    is_invalid_val = is_invalid[tid]
    if is_invalid_val == GraspState.VALID:
        if max_succ > 0:
            idx_succ = wp.atomic_add(succ_num_grasps, 0, 1)
            if idx_succ < max_succ:
                succ_transforms[idx_succ] = transforms[tid]
                succ_offsets[idx_succ] = offsets[tid]
                succ_pregrasp_offsets[idx_succ] = pregrasp_offsets[tid]
                succ_idx_map[idx_succ] = idx_map[tid]
    elif max_fail > 0:
        idx_fail = wp.atomic_add(fail_num_grasps, 0, 1)
        if idx_fail < max_fail:
            fail_transforms[idx_fail] = transforms[tid]
            fail_offsets[idx_fail] = offsets[tid]
            fail_pregrasp_offsets[idx_fail] = pregrasp_offsets[tid]
            fail_idx_map[idx_fail] = idx_map[tid]

@wp.kernel
def get_cspace_positions_kernel(
    offsets: wp.array(dtype=wp.int32), # num_grasps
    cspace_joint_indices: wp.array(dtype=wp.int32), # num_cspace_joint_names
    joint_cspace_pos: wp.array2d(dtype=wp.float32), # num_grasps x num_cspace_joints
    cspace_positions: wp.array2d(dtype=wp.float32), # num_grasps x num_cspace_joint_names
):
    grasp_idx, joint_idx = wp.tid()
    offset = offsets[grasp_idx]
    cspace_joint_idx = cspace_joint_indices[joint_idx]
    cspace_positions[grasp_idx, joint_idx] = joint_cspace_pos[grasp_idx, cspace_joint_idx]

@wp.kernel
def get_bite_points_kernel(
    offsets: wp.array(dtype=wp.int32), # num_grasps
    offset_bite_points: wp.array(dtype=wp.vec3), # max(offsets), num of oppening positions in created gripper
    bite_points: wp.array(dtype=wp.vec3), # num_grasps
):
    grasp_idx = wp.tid()
    offset = offsets[grasp_idx]
    bite_points[grasp_idx] = offset_bite_points[offset]

@wp.kernel
def get_default_root_state_kernel(
    default_root_state: wp.array2d(dtype=wp.float32), # num_envs x 13
    root_state_pose: wp.array2d(dtype=wp.float32), # num_envs x 7
    root_state_velocity: wp.array2d(dtype=wp.float32), # num_envs x 6
):
    env_idx = wp.tid()
    root_state_pose[env_idx][0] = default_root_state[env_idx][0]
    root_state_pose[env_idx][1] = default_root_state[env_idx][1]
    root_state_pose[env_idx][2] = default_root_state[env_idx][2]
    root_state_pose[env_idx][3] = default_root_state[env_idx][3]
    root_state_pose[env_idx][4] = default_root_state[env_idx][4]
    root_state_pose[env_idx][5] = default_root_state[env_idx][5]
    root_state_pose[env_idx][6] = default_root_state[env_idx][6]
    root_state_velocity[env_idx][0] = default_root_state[env_idx][7]
    root_state_velocity[env_idx][1] = default_root_state[env_idx][8]
    root_state_velocity[env_idx][2] = default_root_state[env_idx][9]
    root_state_velocity[env_idx][3] = default_root_state[env_idx][10]
    root_state_velocity[env_idx][4] = default_root_state[env_idx][11]
    root_state_velocity[env_idx][5] = default_root_state[env_idx][12]

@wp.kernel
def add_isaaclab_translation_kernel(
    root_state_pose: wp.array2d(dtype=wp.float32), # num_envs x 7
    translation: wp.array(dtype=wp.vec3), # 3
):
    env_idx = wp.tid()
    root_state_pose[env_idx][0] += translation[env_idx][0]
    root_state_pose[env_idx][1] += translation[env_idx][1]
    root_state_pose[env_idx][2] += translation[env_idx][2]

@wp.kernel
def transform_inverse_isaaclab_kernel(
    src_start_idx: int,
    dst_start_idx: int,
    src: wp.array(dtype=wp.transform),
    dst: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    inv = wp.transform_inverse(src[tid+src_start_idx])
    dst[tid+dst_start_idx][0] = inv[0]
    dst[tid+dst_start_idx][1] = inv[1]
    dst[tid+dst_start_idx][2] = inv[2]
    dst[tid+dst_start_idx][3] = inv[6]
    dst[tid+dst_start_idx][4] = inv[3]
    dst[tid+dst_start_idx][5] = inv[4]
    dst[tid+dst_start_idx][6] = inv[5]
   