# GraspGen Workflow

This workflow guides you through generating grasp data specifically formatted for [GraspGen](https://github.com/NVlabs/GraspGen) training. This script takes some specialized input and creates all the output data needed to train a new gripper with GraspGen.

## Inputs
 - **Objects and Scales JSON file**: The `--object_scales_json` parameter is a dictionary of objects to generate grasps on, and scales at which to do so.  The relative location will be the `/object_dataset` folder in Docker, which the user maps when they use the `docker/run.sh` command.  You can also override the value of the object location folder with the `--object_dataset` command-line argument.
 - **Objects to generate grasps for**: Grasps will be generated for all objects in the `object_scales_json` dictionary, and `object_dataset` is concatenated onto the location of the object.


## Outputs
 - **Location**: All outputs will be in the `/grasp_dataset` folder.
 - **Successful and failed grasps**: GraspGen requires both successful and failed grasps to train its discriminator, making the data generation process slightly different from standard grasp simulation.  Failed grasps, based on collision with the gripper, are generated during the guess phase.  When the verification-by-simulation phase is run, these fails are not tested, but the fails from the simulation will be mixed with these in order to make sure the user has enough failed results.
 - **GraspGen gripper description files**: In the `<GraspGen>/config/grippers` folder, two training config files must exist for GraspGen training to succeed: `<gripper_name>.yaml` and `<gripper_name>.py`.  These files are described in the [GraspGen documentation](https://github.com/NVlabs/GraspGen/blob/main/docs/GRIPPER_DESCRIPTION.md).
 - **Object name to grasp file map**: For every set of objects, the `/grasp_dataset/grasp_data/<gripper_name>/map_uuid_to_path.json` file must exist.  This file maps the relative object locations to the grasp file that they generate.
 - **Training and validation splits**: In the `/grasp_dataset/splits/<gripper_name>` folder, the `train.txt` and `valid.txt` files must exist to define which objects to use for training and which to use for validation.  Users can create their own `train.txt` and `valid.txt` files and they will not be overwritten, but if they don't exist then `graspgen.py` will create them.

## Walkthrough

It is strongly recommended you run the `graspgen.py` example in a [Docker container](../installation.md#method-1-docker-recommended), and this walkthrough will assume that you will.

### Preparing Inputs

There are multiple ways to create the `<object_scales_json>`.  For example, you could write a quick script to extract the object name and scale from the [**Grasp Dataset**](https://github.com/NVlabs/GraspGen/tree/main?tab=readme-ov-file#dataset) described in the GraspGen repo and referenced as `<path_to_grasp_dataset>` therein.  This example `<object_scales_json>` file can be found in this repo at `objects/full_graspgen_object_scales.json`, and can be used to generate the same type of training set used in the GraspGen paper.

### Starting The Docker Container

All the parameters for the `docker/run.sh` script are optional, but they are very useful in this `graspgen.py` example.  If you download the object dataset described in the [**GraspGen documentation**](https://github.com/NVlabs/GraspGen/tree/main?tab=readme-ov-file#dataset) to the `/path/to/object_dataset`, have your GraspGen source code in `/path/to/grasp_gen`, and you want to output to the `/path/to/grasp_dataset` folder, then you would use the following command to start the container from the GraspDataGen code folder:

```bash
bash docker/run.sh --grasp_gen_code /path/to/grasp_gen --grasp_dataset /path/to/grasp_dataset --object_dataset /path/to/object_dataset
```

Once started, you will be at a bash terminal in the container.

### Run the graspgen.py script

The `graspgen.py` script can use all the [GraspDataGen parameters](../api/parameter-system.md) and accepts numerous arguments to control the grasp generation process on top of them. Here are the main argument categories:

#### GraspGen-Specific Arguments

The input and output locations are controlled with these three arguments:

- **`--object_scales_json`**: Path to JSON file containing object scales in format `{"path/to/object.obj": scale, ...}` (default: `objects/graspgen_example.json`)
- **`--object_dataset`**: Root folder path for the objects referenced in the JSON file (default: `objects`)
- **`--grasp_dataset`**: Root folder path for the grasp data output (default: `results`)

When requesting grasps, you generally want to start with guesses that are collision-free, but in the interest of making sure there are enough failed grasps for various reasons, you can also request guesses that failed because the gripper was in collision with the object.  This is a quick way to create failed grasps even without simulation validation.  If you have already generated some guesses you have not validated, or wish to validate them again with new simulation settings, you can load the guesses from existing grasp files and re-test them.  The arguments that control the number of grasps requested or loaded are:

- **`--num_collision_free_grasps`**: Number of collision-free grasps to generate and test (default: 64)
- **`--num_colliding_grasps`**: Number of colliding grasps to generate and test (default: 16)
- **`--guess_only`**: Only generate grasp guess data and skip validation with simulation (default: false)
- **`--load_guesses`**: Load grasps from existing grasp files instead of generating new ones (default: false)

There are also several arguments for controlling the output:
 
- **`--overwrite_grasps`**: Overwrite existing grasp data files instead of skipping them (default: false)
- **`--fill_grasps`**: Fill existing grasp files with additional grasps if needed (default: false).  If this is true, then grasp files with fewer than `num_collision_free_grasps` successes or `num_colliding_grasps` fails will be appended with the results of another round of guessing and validation.  This will not overwrite the existing grasps unless more grasps than necessary are generated; then a random set of only the number requested will be saved.
- **`--overwrite_gripper_config`**: Overwrite existing gripper configuration files in the `<grasp_gen_code>/config/grippers` folder (default: false)
- **`--valid_set_percentage`**: Size of the validation set as a fraction of total grasps (default: 0.2).  The `<grasp_dataset>/splits/train.txt` and `valid.txt` files are created only if they don't already exist, or if the existing files don't contain all current objects from the object scales JSON.
- **`--graspgen_source`**: Path to the GraspGen source code directory (default: `~/github/GraspGen`).  If this is provided then the necessary gripper config files will be placed in the `<grasp_gen_code>/config/grippers` folder.

### Iterative Grasp Generation Workflow

It can take several days on a single GPU to generate enough grasps for the 8,000+ Objaverse objects in the GraspGen training set.  For large-scale grasp generation, it's often beneficial to use an iterative approach.

First, you could generate grasps with `--guess_only` to quickly create geometrically plausible (non-colliding) initial grasp candidates without simulation validation. You could use those grasps to start optimizing and tuning your training network while you send them out again to be validated through simulation. Then use the [analyze_grasp_data.py](../tools/utility-tools.md#analyze-grasp-data) tool to check which files have insufficient successful or failed grasps. Files that don't meet your criteria can be rerun with `--fill_grasps` to add more grasps, or with `--load_guesses` to validate existing guesses with different simulation parameters.

> **Note:** The current version of IsaacLab has a memory leak that can take a few MB per grasp validation simulation, or a few dozen MB if you want to simulate 2048 grasps.  The memory leak is CPU-side, and not a GPU memory problem, but it would be wise to restart the script every hundred grasp file generations.  Without `--fill_grasps` the next run will very quickly iterate over the already finished grasp files, so you don't lose much time.


### Example Usage

```bash
python graspgen.py \
    --graspgen_source /path/to/GraspGen \
    --object_scales_json objects/my_objects.json \
    --object_dataset /path/to/objects \
    --grasp_dataset /path/to/output \
    --gripper_config robotiq_2f_85 \
    --num_collision_free_grasps 2048 \
    --num_colliding_grasps 1024
```

This command would generate grasp data using the Robotiq 2F-85 gripper configuration, process objects defined in `my_objects.json`, and validate the grasps with simulation.

### Training with GraspGen

This section outlines a minimal, end-to-end path from generating grasps with GraspDataGen to training a GraspGen model. It assumes you have the GraspGen source available locally and that you will use Docker for both steps.

#### 1) Generate data with GraspDataGen (robotiq_2f_85)

Run the generator inside the GraspDataGen container, mounting three folders so paths remain consistent across both containers:

```bash
./docker/run.sh \
  --grasp_dataset /path/to/grasp_dataset \
  --object_dataset /path/to/object_dataset \
  --grasp_gen_code /path/to/GraspGen

# Inside the container:
python scripts/graspgen/graspgen.py \
  --gripper_config robotiq_2f_85 \
  --object_scales_json objects/full_graspgen_object_scales.json \
  --object_dataset /object_dataset \
  --grasp_dataset /grasp_dataset
```

What this produces/updates:

- `</path/to/GraspGen>/config/grippers/robotiq_2f_85.yaml` and `.py` created if missing
- `/grasp_dataset/grasp_data/robotiq_2f_85/*.grasps.json`
- `/grasp_dataset/grasp_data/robotiq_2f_85/map_uuid_to_path.json`
- `/grasp_dataset/splits/robotiq_2f_85/{train.txt,valid.txt}` (created if missing or out-of-date)

Keep these mount points stable so the same absolute in-container paths are used by both containers: `/grasp_dataset`, `/object_dataset`, and the GraspGen code directory.

#### 2) Create Training Configuration Files

GraspGen requires separate training scripts for the generator and discriminator models. You can create these by adapting existing configurations from the GraspGen repository.

**Creating Generator Training Script (`train_graspgen_<gripper_name>_gen.sh`):**

1. Copy an existing generator script (e.g., `train_graspgen_robotiq_2f_140_gen.sh`)
2. Update the gripper name: `export GRIPPER_NAME="robotiq_2f_85"`
3. Adjust save frequency if needed: `export SAVE_FREQ=100` (vs 2000 for 140)
4. Ensure paths match your mounted directories:
   - `export GRASP_DATASET_DIR="$GRASP_DIR/grasp_data/$GRIPPER_NAME"`
   - `export SPLIT_DATASET_DIR="$GRASP_DIR/splits/$GRIPPER_NAME"`

**Creating Discriminator Training Script (`train_graspgen_<gripper_name>_dis.sh`):**

1. Copy an existing discriminator script (e.g., `train_graspgen_robotiq_2f_140_dis.sh`)
2. Update the gripper name: `export GRIPPER_NAME="robotiq_2f_85"`
3. Change the ratio to `export RATIO="[0.250,0.250,0.00,0.0,0.00,0.250,0.250]"`
4. Ensure the same path consistency as the generator script

**Key Changes from robotiq_2f_140 to robotiq_2f_85:**
- `GRIPPER_NAME`: `"robotiq_2f_140"` → `"robotiq_2f_85"`
- All other parameters remain the same (batch size, epochs, learning rate, etc.)

Place these scripts in your GraspGen repository's `runs/` directory and make them executable:
```bash
chmod +x runs/train_graspgen_robotiq_2f_85_gen.sh
chmod +x runs/train_graspgen_robotiq_2f_85_dis.sh
```

#### 3) Train in the GraspGen container

Start the GraspGen container (follow the instructions in the GraspGen repo), mounting the same two data folders and the GraspGen code. Ensure the runs/scripts point at:

- Gripper config: `config/grippers/robotiq_2f_85.yaml` and `.py`
- Data root: `/grasp_dataset`
- Splits: `/grasp_dataset/splits/robotiq_2f_85/{train.txt,valid.txt}`

Example (generator then discriminator; adjust to your GraspGen repo’s runs/scripts):

```bash
# From inside the GraspGen container at your repo root
bash runs/train_graspgen_robotiq_2f_85_gen.sh
bash runs/train_graspgen_robotiq_2f_85_dis.sh
```


While training, you can use the logs with TensorBoard to see the progress:

```bash
tensorboard --logdir results/logs
```

Notes:

- Using the same mount points in both containers avoids copying data between them.
- If you regenerate grasps, do not move or rename the `/grasp_dataset` or GraspGen `config/grippers` folders; the runs can continue from the updated data.

#### 4) Inference with Trained Models

Once you have trained GraspGen models, you can use them to predict grasps for new objects. While GraspGen supports multiple input types (object meshes, object point clouds, and scene point clouds), this workflow focuses on **object mesh inference** since that's the data format we generate with GraspDataGen, and the objects are readily available.

**Prerequisites for Inference:**

1. **Model Checkpoints**: Instead of using the [GraspGenModels from HuggingFace](https://huggingface.co/adithyamurali/GraspGenModels), you will use the `last.pth` checkpoint files created in the `logs/robotiq_2f_85_dis/` and `logs/robotiq_2f_85_gen/` folder of your training.

2. **Inference Config File**: The inference config file will be a combination of the `logs/robotiq_2f_85_dis/config.yaml` and the `logs/robotiq_2f_85_gen/config.yaml`, and should be kept in the `logs/` folder so it can be accessible with the <path_to_models_repo> when you run the Docker container (step 4), and the `--gripper_config` argument when you use the `python scripts/demo_object_mesh.py` command in the container (step 5).

3. **Start MeshCat Server**: For visualization, start a MeshCat server in a new terminal:
   ```bash
   meshcat-server
   # Or use the dedicated Docker container:
   bash docker/run_meshcat.sh
   ```

4. **Start GraspGen Container**: Mount the GraspGen code and models:
   ```bash
   bash docker/run.sh . --models results/logs
   ```

5. **Object Mesh Inference**:
   This is the primary inference method for GraspDataGen workflows since we generate grasp data from object meshes (OBJ/STL/PLY files). The inference script will visualize results in your MeshCat browser window, showing the object and predicted grasps colored by quality score.

   ```bash
    cd /code/ && python scripts/demo_object_mesh.py \
        --mesh_file /path/to/your/object.obj \
        --mesh_scale 1.0 \
        --gripper_config /models/robotiq_2f_85_config.yaml
    ```

## Related Documentation

- [GraspGen Repo](https://github.com/NVlabs/GraspGen)
- [DataGen Workflow](datagen.md) - General data generation workflow
- [Grasp Guess Component](../components/grasp-guess.md) - Grasp generation details
- [Grasp Simulation Component](../components/grasp-sim.md) - Simulation details
- [Convert YAML to JSON Tool](../tools/utility-tools.md#convert-yaml-to-json) - Conversion tool documentation