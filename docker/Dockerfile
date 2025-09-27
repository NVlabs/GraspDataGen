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

FROM nvcr.io/nvidia/base/ubuntu:noble-20250619

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Disable hardware-specific library optimizations in dynamic linker for wider compatibility (legacy Debian/Ubuntu)
# Install deps
# Install runtime configuration files
RUN touch /etc/ld.so.nohwcap \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        git \
        git-lfs \
        build-essential \
        gcc-11 \
        g++-11 \
        make \
        cmake \
        python3-dev \
        python3-pip \
        python3-venv \
        libatomic1 \
        libegl1 \
        libgl1 \
        libglu1-mesa \
        libglx0 \
        libgomp1 \
        libsm6 \
        libxi6 \
        libxrandr2 \
        libxt6 \
        unzip \
        ca-certificates \
        libglib2.0-0 \
        libnghttp2-14 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200 \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/*  \
    && mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_mesa.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/50_mesa.json \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194"\n    }\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json \
    && mkdir -p /etc/vulkan/implicit_layer.d && printf '{\n    "file_format_version" : "1.0.0",\n    "layer": {\n        "name": "VK_LAYER_NV_optimus",\n        "type": "INSTANCE",\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194",\n        "implementation_version" : "1",\n        "description" : "NVIDIA Optimus layer",\n        "functions": {\n            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",\n            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"\n        },\n        "enable_environment": {\n            "__NV_PRIME_RENDER_OFFLOAD": "1"\n        },\n        "disable_environment": {\n            "DISABLE_LAYER_NV_OPTIMUS_1": ""\n        }\n    }\n}\n' > /etc/vulkan/implicit_layer.d/nvidia_layers.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# Clone Isaac Sim from GitHub
WORKDIR /isaac-sim
ENV ACCEPT_EULA=Y

# Clone Isaac Sim repository with Git LFS support
RUN git clone https://github.com/isaac-sim/IsaacSim.git . \
    && git lfs install \
    && git lfs pull \
    && mkdir -p /isaac-sim/.nvidia-omniverse/config \
    && echo "yes" | ./build.sh 
RUN ln -s /isaac-sim/_build/linux-x86_64/release /isaacsim


# Clone IsaacLab from GitHub
WORKDIR /isaaclab

RUN git clone https://github.com/isaac-sim/IsaacLab.git . \
    && git lfs install \
    && git lfs pull

# Isaac Sim root directory

# Set environment variables
ENV ISAACSIM_PATH="/isaacsim"
# Isaac Sim python executable
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
ENV ISAACLAB_PATH=/isaaclab
ENV GRASPDATAGEN_PATH=/code/GraspDataGen

# Copy the entire GraspDataGen repository
COPY . ${GRASPDATAGEN_PATH}/


# Create python wrapper script for Isaac Sim's Python
RUN echo '#!/bin/bash\n\
# Save current working directory\n\
ORIGINAL_PWD=$(pwd)\n\
# Set Isaac Sim environment\n\
export CARB_APP_PATH=/isaacsim/kit\n\
export ISAAC_PATH=/isaacsim\n\
export EXP_PATH=/isaacsim/apps\n\
cd /isaacsim\n\
source /isaacsim/setup_python_env.sh\n\
# Return to original directory and execute python\n\
cd "$ORIGINAL_PWD"\n\
exec /isaacsim/kit/python/bin/python3 "$@"' > /usr/local/bin/python \
    && chmod +x /usr/local/bin/python

# Install debugpy and visualization dependencies for debugging
RUN /isaacsim/kit/python/bin/python3 -m pip install --upgrade pip \
    && /isaacsim/kit/python/bin/python3 -m pip install debugpy \
    && /isaacsim/kit/python/bin/python3 -m pip install meshcat \
    && /isaacsim/kit/python/bin/python3 -m pip install numpy \
    && /isaacsim/kit/python/bin/python3 -m pip install trimesh

# Prepare IsaacLab for use
RUN ln -s /isaacsim _isaac_sim
# Create a modified isaaclab.sh that skips tabs command in Docker otherwise ./isaaclab.sh -i will fail
RUN sed 's/^tabs 4/# tabs 4  # Disabled in Docker/' /isaaclab/isaaclab.sh > /isaaclab/isaaclab_docker.sh \
    && chmod +x /isaaclab/isaaclab_docker.sh
RUN cp /isaaclab/isaaclab_docker.sh /isaaclab/isaaclab.sh
# This step won't cache when building TODO Fix this annoying issue because it wastes time
RUN ./isaaclab.sh -i

# Set working directory
WORKDIR ${GRASPDATAGEN_PATH}

ENTRYPOINT [  ]

