#!/usr/bin/env python3
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

import argparse
import json
import os
import random
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path to import graspgen_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graspgen_utils import print_blue, print_purple, print_yellow, print_green  # noqa: E402


def balance_grasp_data(input_folder, output_folder, total_grasps=2048, overwrite=False, seed=None):
    """
    Load grasp data files from input folder and create balanced files in output folder.

    For files that contain grasp data (with 'grasps.transforms' and 'grasps.object_in_gripper'),
    this function will balance the positive/negative grasps to the specified total.

    For other JSON files, they will be copied unchanged to the output folder.

    Args:
        input_folder (str): Path to folder containing input JSON files
        output_folder (str): Path to folder where balanced grasp files will be saved
        total_grasps (int): Total number of grasps in output files (default: 2048)
        overwrite (bool): Whether to overwrite existing output files
        seed (int): Random seed for reproducible results (optional)
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate target positive/negative split (50/50 by default)
    target_positive = total_grasps // 2
    target_negative = total_grasps - target_positive

    print_blue("Balancing grasp data:")
    print_blue(f"  Input folder: {input_folder}")
    print_blue(f"  Output folder: {output_folder}")
    print_blue(f"  Target total grasps: {total_grasps}")
    print_blue(f"  Target positive grasps: {target_positive}")
    print_blue(f"  Target negative grasps: {target_negative}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Find all JSON files in input folder
    input_path = Path(input_folder)
    json_files = list(input_path.glob("*.json"))

    if not json_files:
        print_yellow(f"No JSON files found in {input_folder}")
        return

    print_blue(f"Found {len(json_files)} JSON files to process")

    processed_count = 0
    skipped_count = 0

    for json_file in json_files:
        output_file = Path(output_folder) / json_file.name

        # Check if output file exists and overwrite is False
        if not overwrite and output_file.exists():
            print_blue(f"Skipping {json_file.name} - output file already exists")
            skipped_count += 1
            continue

        print_purple(f"Processing {json_file.name}...")

        try:
            # Load the JSON data
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            # Check if this is a grasp file by looking for the expected structure
            transforms = json_data.get('grasps', {}).get('transforms', [])
            object_in_gripper = json_data.get('grasps', {}).get('object_in_gripper', [])

            # If this is not a grasp file, just copy it over unchanged
            if not transforms or not object_in_gripper:
                print_blue(f"Copying non-grasp file: {json_file.name}")
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                print_green(f"  Copied {json_file.name} unchanged")
                processed_count += 1
                continue

            if len(transforms) != len(object_in_gripper):
                print_yellow(f"Warning: Mismatch between transforms ({len(transforms)}) and object_in_gripper ({len(object_in_gripper)}) in {json_file.name}")
                continue

            # Separate positive and negative grasps
            positive_indices = [i for i, success in enumerate(object_in_gripper) if success]
            negative_indices = [i for i, success in enumerate(object_in_gripper) if not success]

            print_blue(f"  Original: {len(positive_indices)} positive, {len(negative_indices)} negative grasps")

            # Sample grasps to meet target counts
            balanced_transforms = []
            balanced_object_in_gripper = []

            # Sample positive grasps
            if len(positive_indices) >= target_positive:
                # Enough positive grasps, sample randomly
                selected_positive = random.sample(positive_indices, target_positive)
            else:
                # Not enough positive grasps, use all available
                selected_positive = positive_indices.copy()
                # Fill remaining slots with negative grasps
                remaining_slots = target_positive - len(selected_positive)
                if len(negative_indices) >= remaining_slots:
                    selected_positive.extend(random.sample(negative_indices, remaining_slots))
                else:
                    selected_positive.extend(negative_indices)

            # Sample negative grasps
            remaining_negative_slots = target_negative
            if len(negative_indices) >= remaining_negative_slots:
                # Enough negative grasps, sample randomly
                selected_negative = random.sample(negative_indices, remaining_negative_slots)
            else:
                # Not enough negative grasps, use all available
                selected_negative = negative_indices.copy()
                # Fill remaining slots with positive grasps
                remaining_slots = remaining_negative_slots - len(selected_negative)
                if len(positive_indices) >= remaining_slots:
                    selected_negative.extend(random.sample(positive_indices, remaining_slots))
                else:
                    selected_negative.extend(positive_indices)

            # Combine selected grasps
            all_selected_indices = selected_positive + selected_negative
            random.shuffle(all_selected_indices)  # Randomize order

            # Extract the selected grasps
            for idx in all_selected_indices:
                balanced_transforms.append(transforms[idx])
                balanced_object_in_gripper.append(object_in_gripper[idx])

            # Count final positive/negative
            final_positive = sum(balanced_object_in_gripper)
            final_negative = len(balanced_object_in_gripper) - final_positive

            print_green(f"  Balanced: {final_positive} positive, {final_negative} negative grasps")

            # Create balanced grasp data
            balanced_data = json_data.copy()
            balanced_data['grasps']['transforms'] = balanced_transforms
            balanced_data['grasps']['object_in_gripper'] = balanced_object_in_gripper

            # Save balanced data
            with open(output_file, 'w') as f:
                json.dump(balanced_data, f, indent=2)

            print_green(f"  Saved balanced data to {output_file.name}")
            processed_count += 1

        except Exception as e:
            print_yellow(f"Error processing {json_file.name}: {e}")
            continue

    print_blue("\nProcessing complete:")
    print_blue(f"  Processed: {processed_count} files")
    print_blue(f"  Skipped: {skipped_count} files")


def make_parser():
    parser = argparse.ArgumentParser(description="Balance grasp data files with equal positive/negative grasps. Non-grasp JSON files are copied unchanged.")

    parser.add_argument("input_folder", type=str, help="Path to folder containing input JSON files")
    parser.add_argument("output_folder", type=str, help="Path to folder where balanced grasp files will be saved")
    parser.add_argument("--total_grasps", type=int, default=2048, help="Total number of grasps in output files (default: 2048)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results")

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # Expand user paths
    input_folder = os.path.expanduser(args.input_folder)
    output_folder = os.path.expanduser(args.output_folder)

    # Validate input folder exists
    if not os.path.exists(input_folder):
        print_yellow(f"Error: Input folder {input_folder} does not exist")
        return

    balance_grasp_data(
        input_folder=input_folder,
        output_folder=output_folder,
        total_grasps=args.total_grasps,
        overwrite=args.overwrite,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
