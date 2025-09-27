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

#!/usr/bin/env python3
"""
Compare two grasp simulation YAML files and analyze confidence values.

This script compares grasp simulation results between two YAML files that were
run with the same pregrasp parameters and object (or reports if objects differ).
The gripper will typically be different between the files.

Usage:
    python compare_grasp_sims.py <file1.yaml> <file2.yaml>
"""

import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Tuple


def load_grasp_data(file_path: str) -> Tuple[Dict, str, str]:
    """Load grasp data from YAML file and return grasps dict, object file, and gripper file."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        grasps = data.get('grasps', {})
        object_file = data.get('object_file', 'unknown')
        gripper_file = data.get('gripper_file', 'unknown')
        
        return grasps, object_file, gripper_file
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)


def extract_confidence_values(grasps: Dict) -> Dict[str, float]:
    """Extract grasp names and their confidence values."""
    confidence_data = {}
    for grasp_name, grasp_data in grasps.items():
        if isinstance(grasp_data, dict) and 'confidence' in grasp_data:
            confidence_data[grasp_name] = grasp_data['confidence']
    return confidence_data


def analyze_confidence_comparison(conf1: Dict[str, float], conf2: Dict[str, float]) -> Dict:
    """Analyze confidence values between two grasp sets."""
    # Get all unique grasp names
    all_grasps = set(conf1.keys()) | set(conf2.keys())
    
    # Categorize grasps
    both_1_0 = set()  # Both have confidence 1.0
    both_0_0 = set()  # Both have confidence 0.0
    file1_1_file2_0 = set()  # File1 has 1.0, File2 has 0.0
    file1_0_file2_1 = set()  # File1 has 0.0, File2 has 1.0
    only_in_file1 = set()  # Only in file1
    only_in_file2 = set()  # Only in file2
    
    for grasp in all_grasps:
        conf1_val = conf1.get(grasp, None)
        conf2_val = conf2.get(grasp, None)
        
        if conf1_val is not None and conf2_val is not None:
            # Grasp exists in both files
            if conf1_val == 1.0 and conf2_val == 1.0:
                both_1_0.add(grasp)
            elif conf1_val == 0.0 and conf2_val == 0.0:
                both_0_0.add(grasp)
            elif conf1_val == 1.0 and conf2_val == 0.0:
                file1_1_file2_0.add(grasp)
            elif conf1_val == 0.0 and conf2_val == 1.0:
                file1_0_file2_1.add(grasp)
        elif conf1_val is not None:
            only_in_file1.add(grasp)
        elif conf2_val is not None:
            only_in_file2.add(grasp)
    
    return {
        'both_1_0': both_1_0,
        'both_0_0': both_0_0,
        'file1_1_file2_0': file1_1_file2_0,
        'file1_0_file2_1': file1_0_file2_1,
        'only_in_file1': only_in_file1,
        'only_in_file2': only_in_file2,
        'total_grasps': len(all_grasps)
    }


def extract_mismatched_grasps(grasps1: Dict, grasps2: Dict, analysis: Dict) -> Dict:
    """Extract mismatched grasps from both files for output."""
    mismatched_grasps = {}
    
    # Add grasps where file1 is confident but file2 failed
    for grasp_name in analysis['file1_1_file2_0']:
        if grasp_name in grasps1:
            mismatched_grasps[f"{grasp_name}_file1_confident"] = grasps1[grasp_name]
    
    # Add grasps where file1 failed but file2 is confident
    for grasp_name in analysis['file1_0_file2_1']:
        if grasp_name in grasps2:
            mismatched_grasps[f"{grasp_name}_file2_confident"] = grasps2[grasp_name]
    
    return mismatched_grasps


def save_mismatched_grasps(grasps1: Dict, grasps2: Dict, analysis: Dict, output_file: str,
                           file1_name: str, file2_name: str, file1_data: Dict, file2_data: Dict):
    """Save mismatched grasps to a YAML file."""
    mismatched_grasps = extract_mismatched_grasps(grasps1, grasps2, analysis)
    
    # Create output data structure with header from first file
    output_data = {
        'format': file1_data.get('format', 'isaac_grasp'),
        'format_version': file1_data.get('format_version', '1.0'),
        'created_with': 'compare_grasp_sims',
        'description': f'Mismatched grasps between {file1_name} and {file2_name}',
        'file1_name': file1_name,
        'file2_name': file2_name,
        'object_file': file1_data.get('object_file', 'unknown'),
        'object_file_scale': file1_data.get('object_file_scale', 1.0),
        'gripper_file': file1_data.get('gripper_file', 'unknown'),
        'gripper_frame_link': file1_data.get('gripper_frame_link', 'base_link'),
        'open_limit': file1_data.get('open_limit', 'lower'),
        'finger_colliders': file1_data.get('finger_colliders', []),
        'base_length': file1_data.get('base_length', 0.0),
        'approach_axis': file1_data.get('approach_axis', 2),
        'bite_point': file1_data.get('bite_point', [0.0, 0.0, 0.0]),
        'bite_body_idx': file1_data.get('bite_body_idx', 0),
        'grasps': mismatched_grasps
    }
    
    try:
        with open(output_file, 'w') as f:
            # Write fields in the desired order
            f.write(f"format: {output_data['format']}\n")
            f.write(f"format_version: {output_data['format_version']}\n")
            f.write(f"created_with: {output_data['created_with']}\n")
            f.write(f"description: {output_data['description']}\n")
            f.write(f"file1_name: {output_data['file1_name']}\n")
            f.write(f"file2_name: {output_data['file2_name']}\n")
            f.write(f"object_file: {output_data['object_file']}\n")
            f.write(f"object_file_scale: {output_data['object_file_scale']}\n")
            f.write(f"gripper_file: {output_data['gripper_file']}\n")
            # Add commented gripper_file from second file right below
            gripper2 = file2_data.get('gripper_file', 'unknown')
            if gripper2 != 'unknown':
                f.write(f"# gripper_file: {gripper2}\n")
            f.write(f"gripper_frame_link: {output_data['gripper_frame_link']}\n")
            f.write(f"open_limit: {output_data['open_limit']}\n")
            f.write("finger_colliders:\n")
            for collider in output_data['finger_colliders']:
                f.write(f"- {collider}\n")
            f.write(f"base_length: {output_data['base_length']}\n")
            f.write(f"approach_axis: {output_data['approach_axis']}\n")
            f.write("bite_point:\n")
            for point in output_data['bite_point']:
                f.write(f"- {point}\n")
            f.write(f"bite_body_idx: {output_data['bite_body_idx']}\n")
            
            # Write grasps using yaml.dump for the complex structure
            grasps_data = {'grasps': output_data['grasps']}
            yaml.dump(grasps_data, f, default_flow_style=False, indent=2, sort_keys=False)
        
        print(f"\n💾 Saved {len(mismatched_grasps)} mismatched grasps to: {output_file}")
        print(f"   - {len(analysis['file1_1_file2_0'])} grasps where {file1_name} confident, {file2_name} failed")
        print(f"   - {len(analysis['file1_0_file2_1'])} grasps where {file1_name} failed, {file2_name} confident")
    except Exception as e:
        print(f"Error saving mismatched grasps to {output_file}: {e}")


def print_ascii_art_comparison(analysis: Dict, file1_name: str, file2_name: str):
    """Print ASCII art visualization of the comparison."""
    print("\n" + "="*80)
    print("                    GRASP SIMULATION COMPARISON")
    print("="*80)
    
    # File information
    print(f"\n📁 File 1: {file1_name}")
    print(f"📁 File 2: {file2_name}")
    
    # Summary statistics
    total = analysis['total_grasps']
    both_1_0_count = len(analysis['both_1_0'])
    both_0_0_count = len(analysis['both_0_0'])
    file1_1_file2_0_count = len(analysis['file1_1_file2_0'])
    file1_0_file2_1_count = len(analysis['file1_0_file2_1'])
    only_file1_count = len(analysis['only_in_file1'])
    only_file2_count = len(analysis['only_in_file2'])
    
    print(f"\n📊 TOTAL GRASPS: {total}")
    
    # ASCII Art Visualization
    print("\n" + "─"*60)
    print("                    CONFIDENCE COMPARISON")
    print("─"*60)
    
    # Both confident (1.0)
    print(f"\n✅ BOTH CONFIDENT (1.0): {both_1_0_count}")
    if both_1_0_count > 0:
        percentage = (both_1_0_count / total) * 100
        bar_length = int((both_1_0_count / total) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {bar} {percentage:.1f}%")
    
    # Both failed (0.0)
    print(f"\n❌ BOTH FAILED (0.0): {both_0_0_count}")
    if both_0_0_count > 0:
        percentage = (both_0_0_count / total) * 100
        bar_length = int((both_0_0_count / total) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {bar} {percentage:.1f}%")
    
    # Mismatched - File1 confident, File2 failed
    print(f"\n⚠️  MISMATCHED - {file1_name} confident, {file2_name} failed: {file1_1_file2_0_count}")
    if file1_1_file2_0_count > 0:
        percentage = (file1_1_file2_0_count / total) * 100
        bar_length = int((file1_1_file2_0_count / total) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {bar} {percentage:.1f}%")
    
    # Mismatched - File1 failed, File2 confident
    print(f"\n⚠️  MISMATCHED - {file1_name} failed, {file2_name} confident: {file1_0_file2_1_count}")
    if file1_0_file2_1_count > 0:
        percentage = (file1_0_file2_1_count / total) * 100
        bar_length = int((file1_0_file2_1_count / total) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {bar} {percentage:.1f}%")
    
    # Only in one file
    if only_file1_count > 0:
        print(f"\n📄 ONLY IN {file1_name}: {only_file1_count}")
        percentage = (only_file1_count / total) * 100
        bar_length = int((only_file1_count / total) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {bar} {percentage:.1f}%")
    
    if only_file2_count > 0:
        print(f"\n📄 ONLY IN {file2_name}: {only_file2_count}")
        percentage = (only_file2_count / total) * 100
        bar_length = int((only_file2_count / total) * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {bar} {percentage:.1f}%")
    
    # Summary table
    print("\n" + "─"*60)
    print("                    SUMMARY TABLE")
    print("─"*60)
    print(f"{'Category':<35} {'Count':<8} {'Percentage':<12}")
    print("─" * 60)
    print(f"{'Both Confident (1.0)':<35} {both_1_0_count:<8} {(both_1_0_count/total*100):<12.1f}%")
    print(f"{'Both Failed (0.0)':<35} {both_0_0_count:<8} {(both_0_0_count/total*100):<12.1f}%")
    print(f"{'File1 1.0, File2 0.0':<35} {file1_1_file2_0_count:<8} {(file1_1_file2_0_count/total*100):<12.1f}%")
    print(f"{'File1 0.0, File2 1.0':<35} {file1_0_file2_1_count:<8} {(file1_0_file2_1_count/total*100):<12.1f}%")
    if only_file1_count > 0:
        print(f"{'Only in File1':<35} {only_file1_count:<8} {(only_file1_count/total*100):<12.1f}%")
    if only_file2_count > 0:
        print(f"{'Only in File2':<35} {only_file2_count:<8} {(only_file2_count/total*100):<12.1f}%")
    
    # Agreement percentage
    agreement_count = both_1_0_count + both_0_0_count
    agreement_percentage = (agreement_count / total) * 100
    print("─" * 60)
    print(f"{'TOTAL AGREEMENT':<35} {agreement_count:<8} {agreement_percentage:<12.1f}%")
    print(f"{'TOTAL DISAGREEMENT':<35} {(total-agreement_count):<8} {((total-agreement_count)/total*100):<12.1f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two grasp simulation YAML files and analyze confidence values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_grasp_sims.py grasp_sim_data/robotiq_2f_85/cuda.banana.yaml \\
                              grasp_sim_data/Robotiq_2F_85_msJul21/cuda.banana.yaml
  
  # Save mismatched grasps for visualization/simulation
  python compare_grasp_sims.py grasp_sim_data/robotiq_2f_85/cuda.banana.yaml \\
                              grasp_sim_data/Robotiq_2F_85_msJul21/cuda.banana.yaml \\
                              -o mismatched_grasps.yaml
        """
    )
    parser.add_argument('file1', help='First grasp simulation YAML file')
    parser.add_argument('file2', help='Second grasp simulation YAML file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed grasp names in each category')
    parser.add_argument('--output-mismatches', '-o', type=str, metavar='FILE',
                        help='Output mismatched grasps to YAML file for visualization/simulation')
    
    args = parser.parse_args()
    
    # Load data from both files
    print(f"Loading grasp data from {args.file1}...")
    grasps1, object1, gripper1 = load_grasp_data(args.file1)
    
    print(f"Loading grasp data from {args.file2}...")
    grasps2, object2, gripper2 = load_grasp_data(args.file2)
    
    # Load full data for header information
    with open(args.file1, 'r') as f:
        file1_data = yaml.safe_load(f)
    with open(args.file2, 'r') as f:
        file2_data = yaml.safe_load(f)
    
    # Check if objects are the same
    if object1 != object2:
        print("\n⚠️  WARNING: Objects are different!")
        print(f"   File1 object: {object1}")
        print(f"   File2 object: {object2}")
        print("   Comparison may not be meaningful.\n")
    
    # Extract confidence values
    conf1 = extract_confidence_values(grasps1)
    conf2 = extract_confidence_values(grasps2)
    
    print(f"Found {len(conf1)} grasps in {args.file1}")
    print(f"Found {len(conf2)} grasps in {args.file2}")
    
    # Calculate success/fail counts for each file
    file1_success = sum(1 for conf in conf1.values() if conf == 1.0)
    file1_fails = sum(1 for conf in conf1.values() if conf == 0.0)
    file2_success = sum(1 for conf in conf2.values() if conf == 1.0)
    file2_fails = sum(1 for conf in conf2.values() if conf == 0.0)
    
    file1_name = Path(args.file1).parent.name
    file2_name = Path(args.file2).parent.name
    
    print(f"\n📊 {file1_name}: {file1_success} successful, {file1_fails} failed ({file1_success/len(conf1)*100:.1f}% success rate)")
    print(f"📊 {file2_name}: {file2_success} successful, {file2_fails} failed ({file2_success/len(conf2)*100:.1f}% success rate)")
    
    # Analyze comparison
    analysis = analyze_confidence_comparison(conf1, conf2)
    
    # Print results
    print_ascii_art_comparison(analysis, file1_name, file2_name)
    
    # Save mismatched grasps if requested
    if args.output_mismatches:
        save_mismatched_grasps(grasps1, grasps2, analysis, args.output_mismatches, file1_name, file2_name, file1_data, file2_data)
    
    # Verbose output if requested
    if args.verbose:
        print("\n" + "="*80)
        print("                    DETAILED BREAKDOWN")
        print("="*80)
        
        if analysis['both_1_0']:
            print(f"\n✅ BOTH CONFIDENT (1.0) - {len(analysis['both_1_0'])} grasps:")
            for grasp in sorted(analysis['both_1_0']):
                print(f"   {grasp}")
        
        if analysis['both_0_0']:
            print(f"\n❌ BOTH FAILED (0.0) - {len(analysis['both_0_0'])} grasps:")
            for grasp in sorted(analysis['both_0_0']):
                print(f"   {grasp}")
        
        if analysis['file1_1_file2_0']:
            print(f"\n⚠️  {file1_name} confident, {file2_name} failed - {len(analysis['file1_1_file2_0'])} grasps:")
            for grasp in sorted(analysis['file1_1_file2_0']):
                print(f"   {grasp}")
        
        if analysis['file1_0_file2_1']:
            print(f"\n⚠️  {file1_name} failed, {file2_name} confident - {len(analysis['file1_0_file2_1'])} grasps:")
            for grasp in sorted(analysis['file1_0_file2_1']):
                print(f"   {grasp}")
        
        if analysis['only_in_file1']:
            print(f"\n📄 Only in {file1_name} - {len(analysis['only_in_file1'])} grasps:")
            for grasp in sorted(analysis['only_in_file1']):
                print(f"   {grasp}")
        
        if analysis['only_in_file2']:
            print(f"\n📄 Only in {file2_name} - {len(analysis['only_in_file2'])} grasps:")
            for grasp in sorted(analysis['only_in_file2']):
                print(f"   {grasp}")


if __name__ == "__main__":
    main() 