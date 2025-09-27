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

"""Constants representing different grasp states."""

# flake8: noqa: E303, E111

# Grasp state constants
class GraspState:
    """Constants for different grasp states.
    
    These constants represent the possible states of a grasp:
    - VALID: The grasp is valid and can be executed
    - IN_COLLISION: The grasp would result in a collision
    - SIM_FAILED: The grasp simulation failed
    - TOO_CLOSED: The gripper is too closed for this grasp
    """
    VALID = 0
    IN_COLLISION = 1
    SIM_FAILED = 2
    TOO_CLOSED = 3

# Convergence constants for gripper creation
DEFAULT_MEASURE_CONVERGENCE = False
DEFAULT_CONVERGENCE_ITERATIONS = 20