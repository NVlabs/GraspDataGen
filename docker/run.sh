#!/bin/bash

# Function to display usage information
show_usage() {
    echo "Usage: $0 [--grasp_dataset <grasp_dataset_dir>] [--object_dataset <object_dataset_dir>] [--grasp_gen_code <grasp_gen_code_dir>] [--graspdatagen_code <graspdatagen_code_dir>]"
    echo ""
    echo "Arguments:"
    echo "  --grasp_dataset <dir> (optional) Path to the grasp dataset directory (relative or absolute)"
    echo "  --object_dataset <dir> (optional) Path to the object dataset directory (relative or absolute)"
    echo "  --grasp_gen_code <dir> (optional) Path to the grasp gen code directory (relative or absolute)"
    echo "  --graspdatagen_code <dir> (optional) Path to the GraspDataGen code directory (relative or absolute)"
    echo ""
    echo "Examples:"
    echo "  # Basic run:"
    echo "  $0"
    echo ""
    echo "  # With grasp dataset:"
    echo "  $0 --grasp_dataset /path/to/grasp_dataset"
    echo "  $0 --grasp_dataset ./datasets/grasp"
    echo ""
    echo "  # With object dataset:"
    echo "  $0 --object_dataset /path/to/object_dataset"
    echo "  $0 --object_dataset ./datasets/objects"
    echo ""
    echo "  # With grasp gen code:"
    echo "  $0 --grasp_gen_code /path/to/grasp_gen"
    echo "  $0 --grasp_gen_code ./grasp_gen"
    echo ""
    echo "  # With GraspDataGen code (for development):"
    echo "  $0 --graspdatagen_code /path/to/GraspDataGen"
    echo "  $0 --graspdatagen_code ."
    echo ""
    echo "  # With all parameters:"
    echo "  $0 --grasp_dataset /path/to/grasp_dataset --object_dataset /path/to/object_dataset --grasp_gen_code /path/to/grasp_gen --graspdatagen_code /path/to/GraspDataGen"
    echo "  $0 --grasp_dataset ./datasets/grasp --object_dataset ./datasets/objects --grasp_gen_code ./grasp_gen --graspdatagen_code ."
    echo ""
    echo "Note: grasp_dataset_dir (if provided) is mounted at /grasp_dataset"
    echo "      object_dataset_dir (if provided) is mounted at /object_dataset"
    echo "      grasp_gen_code_dir (if provided) is mounted at /grasp_gen_code"
    echo "      graspdatagen_code_dir (if provided) is mounted at /code/GraspDataGen (overrides built-in code)"
    echo "      All paths are converted to absolute paths for Docker volume mounting"
    exit 1
}

# Function to convert path to absolute path
make_absolute_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        # Already absolute path
        echo "$path"
    else
        # Relative path, convert to absolute
        echo "$(realpath "$path")"
    fi
}

GRASP_DATASET_DIR=""
OBJECT_DATASET_DIR=""
GRASP_GEN_CODE_DIR=""
GRASPDATAGEN_CODE_DIR=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --grasp_dataset)
            GRASP_DATASET_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        --object_dataset)
            OBJECT_DATASET_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        --grasp_gen_code)
            GRASP_GEN_CODE_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        --graspdatagen_code)
            GRASPDATAGEN_CODE_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Build volume mount arguments
VOLUME_MOUNTS=""

# Add grasp dataset directory if provided
if [ -n "$GRASP_DATASET_DIR" ]; then
    if [ ! -d "$GRASP_DATASET_DIR" ]; then
        echo "Error: grasp_dataset_dir '$GRASP_DATASET_DIR' does not exist or is not a directory"
        exit 1
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${GRASP_DATASET_DIR}:/grasp_dataset"
fi

# Add object dataset directory if provided
if [ -n "$OBJECT_DATASET_DIR" ]; then
    if [ ! -d "$OBJECT_DATASET_DIR" ]; then
        echo "Error: object_dataset_dir '$OBJECT_DATASET_DIR' does not exist or is not a directory"
        exit 1
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${OBJECT_DATASET_DIR}:/object_dataset"
fi

# Add grasp gen code directory if provided
if [ -n "$GRASP_GEN_CODE_DIR" ]; then
    if [ ! -d "$GRASP_GEN_CODE_DIR" ]; then
        echo "Error: grasp_gen_code_dir '$GRASP_GEN_CODE_DIR' does not exist or is not a directory"
        exit 1
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${GRASP_GEN_CODE_DIR}:/grasp_gen_code"
fi

# Add GraspDataGen code directory if provided
if [ -n "$GRASPDATAGEN_CODE_DIR" ]; then
    if [ ! -d "$GRASPDATAGEN_CODE_DIR" ]; then
        echo "Error: graspdatagen_code_dir '$GRASPDATAGEN_CODE_DIR' does not exist or is not a directory"
        exit 1
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${GRASPDATAGEN_CODE_DIR}:/code/GraspDataGen"
fi


echo "Starting GraspDataGen Docker container with:"
if [ -n "$GRASP_DATASET_DIR" ]; then
    echo "  Grasp dataset directory: $GRASP_DATASET_DIR -> /grasp_dataset"
fi
if [ -n "$OBJECT_DATASET_DIR" ]; then
    echo "  Object dataset directory: $OBJECT_DATASET_DIR -> /object_dataset"
fi
if [ -n "$GRASP_GEN_CODE_DIR" ]; then
    echo "  Grasp gen code directory: $GRASP_GEN_CODE_DIR -> /grasp_gen_code"
fi
if [ -n "$GRASPDATAGEN_CODE_DIR" ]; then
    echo "  GraspDataGen code directory: $GRASPDATAGEN_CODE_DIR -> /code/GraspDataGen (overrides built-in code)"
fi
if [ -z "$GRASP_DATASET_DIR" ] && [ -z "$OBJECT_DATASET_DIR" ] && [ -z "$GRASP_GEN_CODE_DIR" ] && [ -z "$GRASPDATAGEN_CODE_DIR" ]; then
    echo "  No additional volumes mounted"
fi
echo ""

set -e



docker run \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --security-opt apparmor=unconfined \
  --privileged \
  --runtime nvidia \
  --gpus all \
  --net host \
  -e "GRASP_DATASET_DIR=/grasp_dataset" \
  -e "OBJECT_DATASET_DIR=/object_dataset" \
  -e "GRASP_GEN_CODE_DIR=/grasp_gen_code" \
  -it \
  --rm \
  -e "ACCEPT_EULA=Y" \
  $VOLUME_MOUNTS \
  graspdatagen \
  /bin/bash
  