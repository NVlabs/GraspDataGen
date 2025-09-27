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

"""Warp functions for mesh processing and transformations."""
import warp as wp

@wp.func
def compute_offset_along_negative_normal(offset: float, xform: wp.mat44, open_axis: int):
    return wp.vec3(xform[0,open_axis], xform[1,open_axis], xform[2,open_axis]) * -offset

@wp.func
def cw_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))

@wp.func
def cw_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))

@wp.func
def mat44_to_transform(src: wp.mat44):
    rot33 = wp.mat33(src[0,0], src[0,1], src[0,2],
                     src[1,0], src[1,1], src[1,2],
                     src[2,0], src[2,1], src[2,2])
    tns = wp.vec3(src[0,3], src[1,3], src[2,3])
    quat = wp.quat_from_matrix(rot33)
    return wp.transform(tns, quat)

@wp.func
def transform_to_mat44(src: wp.transform):
    xform44 = wp.mat44(1.0, 0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 1.0)
    rot = wp.quat_to_matrix(wp.transform_get_rotation(src))
    tns = wp.transform_get_translation(src)
    xform44[0,0] = rot[0,0]
    xform44[0,1] = rot[0,1]
    xform44[0,2] = rot[0,2]
    xform44[1,0] = rot[1,0]
    xform44[1,1] = rot[1,1]
    xform44[1,2] = rot[1,2]
    xform44[2,0] = rot[2,0]
    xform44[2,1] = rot[2,1]
    xform44[2,2] = rot[2,2]
    xform44[0,3] = tns[0]
    xform44[1,3] = tns[1]
    xform44[2,3] = tns[2]
    return xform44

@wp.func
def triangle_mesh_intersect(face: int, xform: wp.mat44,
                            verts0: wp.array(dtype=wp.vec3), tris0: wp.array(dtype=wp.int32),
                            mesh1: wp.uint64, verts1: wp.array(dtype=wp.vec3), tris1: wp.array(dtype=wp.int32)):
    v0 = wp.transform_point(xform, verts0[tris0[face*3 + 0]])
    v1 = wp.transform_point(xform, verts0[tris0[face*3 + 1]])
    v2 = wp.transform_point(xform, verts0[tris0[face*3 + 2]])
    #if 0 and offset_idx == 0 and idx == 0:
    #    # Use this debug output with parse_mesh.py to visualize the meshes in the collision check
    #    # make sure not to early out if in_collision is true
    #    #str_out = "0:" + "0"
    #    wp.printf("0:v:%i: %f,%f,%f\n", tris0[face*3 + 0], v0[0], v0[1], v0[2])
    #    wp.printf("0:v:%i: %f,%f,%f\n", tris0[face*3 + 1], v1[0], v1[1], v1[2])
    #    wp.printf("0:v:%i: %f,%f,%f\n", tris0[face*3 + 2], v2[0], v2[1], v2[2])
    #    if face == 0:
    #        for f0idx in range(num_faces):
    #            wp.printf("0:f: %i: %i, %i, %i\n", f0idx, tris0[f0idx*3 + 0], tris0[f0idx*3 + 1], tris0[f0idx*3 + 2])
    #        for f1idx in range(num_faces1):
    #            wp.printf("1:f: %i: %i, %i, %i\n", f1idx, tris1[f1idx*3 + 0], tris1[f1idx*3 + 1], tris1[f1idx*3 + 2])
    #        for v1idx in range(num_verts1):
    #            wp.printf("1:v:%i: %f,%f,%f\n", v1idx, verts1[v1idx][0], verts1[v1idx][1], verts1[v1idx][2])

    # compute bounds of the query triangle
    lower = cw_min(cw_min(v0, v1), v2)
    upper = cw_max(cw_max(v0, v1), v2)

    query = wp.mesh_query_aabb(mesh1, lower, upper)

    for f in query:
        u0 = verts1[tris1[f*3+0]]
        u1 = verts1[tris1[f*3+1]]
        u2 = verts1[tris1[f*3+2]]

        # test for triangle intersection
        s = 100.0
        i = wp.intersect_tri_tri(s*v0, s*v1, s*v2, s*u0, s*u1, s*u2)

        if i > 0:
            return True
        
    return False

@wp.func
def wp_inverse_rigid_transform(T: wp.mat44):
    rot = wp.mat33(T[0,0],T[1,0],T[2,0],
                   T[0,1],T[1,1],T[2,1],
                   T[0,2],T[1,2],T[2,2])
    trans = -(rot * wp.vec3(T[0,3],T[1,3],T[2,3]))
    return wp.mat44(rot[0,0],rot[0,1],rot[0,2],trans[0],
                    rot[1,0],rot[1,1],rot[1,2],trans[1],
                    rot[2,0],rot[2,1],rot[2,2],trans[2],
                    0.0,0.0,0.0,1.0)

@wp.func
def wp_plane_transform(origin: wp.vec3, normal: wp.vec3):
    """Specialization of transform = align_vectors(normal, [0, 0, 1])
    followed by getting the origin translation... transform[:3, 3] = -np.dot(transform, np.append(origin, 1))[:3]
    """
    T = wp.mat44(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)#eye(4)
    
    U = wp_svd_normal(normal)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    # perform equivalet of bu.dot(U) where bu is svd([0,0,1].T)
    T[0, 0] = -U[0, 2]
    T[0, 1] = -U[1, 2]
    T[0, 2] = -U[2, 2]

    T[1, 0] = U[0, 1]
    T[1, 1] = U[1, 1]
    T[1, 2] = U[2, 1]

    T[2, 0] = U[0, 0]
    T[2, 1] = U[1, 0]
    T[2, 2] = U[2, 0]

    # transform[:3, 3] = -np.dot(transform, np.append(origin, 1))[:3]
    T[0, 3] = -(T[0, 0]*origin[0] + T[0, 1]*origin[1] + T[0, 2]*origin[2])
    T[1, 3] = -(T[1, 0]*origin[0] + T[1, 1]*origin[1] + T[1, 2]*origin[2])
    T[2, 3] = -(T[2, 0]*origin[0] + T[2, 1]*origin[1] + T[2, 2]*origin[2])
    
    return T

@wp.func
def wp_plane_transform_axis(origin: wp.vec3, normal: wp.vec3, axis: int):
    """
    Parameters
    ----------
    origin : wp.vec3
        The origin point
    normal : wp.vec3
        The normal vector
    axis : int
        The axis to align the normal to
    """
    T = wp.mat44(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)#eye(4)
    
    U = wp_svd_normal(normal)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if axis == 0:
        # perform equivalent of bu.dot(U) where bu is svd([1,0,0].T)
        T[0, 0] = U[0, 0]
        T[0, 1] = U[1, 0]
        T[0, 2] = U[2, 0]

        T[1, 0] = U[0, 1]
        T[1, 1] = U[1, 1]
        T[1, 2] = U[2, 1]

        T[2, 0] = U[0, 2]
        T[2, 1] = U[1, 2]
        T[2, 2] = U[2, 2]

    if axis == 1:
        # perform equivalent of bu.dot(U) where bu is svd([0,1,0].T)
        T[0, 0] = -U[0, 1]
        T[0, 1] = -U[1, 1]
        T[0, 2] = -U[2, 1]

        T[1, 0] = U[0, 0]
        T[1, 1] = U[1, 0]
        T[1, 2] = U[2, 0]

        T[2, 0] = U[0, 2]
        T[2, 1] = U[1, 2]
        T[2, 2] = U[2, 2]

    if axis == 2:
        # perform equivalent of bu.dot(U) where bu is svd([0,0,1].T)
        T[0, 0] = -U[0, 2]
        T[0, 1] = -U[1, 2]
        T[0, 2] = -U[2, 2]

        T[1, 0] = U[0, 1]
        T[1, 1] = U[1, 1]
        T[1, 2] = U[2, 1]

        T[2, 0] = U[0, 0]
        T[2, 1] = U[1, 0]
        T[2, 2] = U[2, 0]

    # transform[:3, 3] = -np.dot(transform, np.append(origin, 1))[:3]
    T[0, 3] = -(T[0, 0]*origin[0] + T[0, 1]*origin[1] + T[0, 2]*origin[2])
    T[1, 3] = -(T[1, 0]*origin[0] + T[1, 1]*origin[1] + T[1, 2]*origin[2])
    T[2, 3] = -(T[2, 0]*origin[0] + T[2, 1]*origin[1] + T[2, 2]*origin[2])
    
    return T

@wp.func
def wp_svd_normal(normal: wp.vec3):
    """
    Given the a normalized vec3, return the U from normal.T = U*Sigma*V.T.
    Sigma and V.T are always 1 with a normal vector, so don't return those.

    with float32 this isclose to numpy.linalg.svd within a good atol, aside from
    the last two column may be multiplied by -1.0

    Parameters
    ----------
    normal : wp.vec3
        Some unit vector

    Returns
    ---------
    U: wp.mat33
        U vector from the SVD calculation
    """
    a = normal[0]
    b = normal[1]
    c = normal[2]
    if a >= 0.0:
        bp = -b
        cp = -c
        z = bp*c/(1.0+a)
        if bool(c):
            x = (b*z-cp)/c
        else:
            x = a
        y = a*x-bp*b
    else:
        bp = b
        cp = c
        z = b*c/(a-1.0)
        if bool(c):
            x = (z*b+c)/c
        else:
            x = -a
        y = b*b-a*x
    U = wp.mat33()
    U[0,0] = a
    U[0,1] = bp
    U[0,2] = cp
    
    U[1,0] = b
    U[1,1] = x
    U[1,2] = z
    
    U[2,0] = c
    U[2,1] = z
    U[2,2] = y

    return U
