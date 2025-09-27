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

import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import json

def load_npz_file(filepath: str) -> Dict[str, Any]:
    """Load NPZ file and return as dictionary."""
    try:
        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.keys()}
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def format_value(value: Any, max_elements: int = 10) -> str:
    """Format a value for display, truncating large arrays."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            # Scalar array
            return f"{value.item()} (scalar array, dtype: {value.dtype})"
        elif value.size == 0:
            return f"[] (empty array, shape: {value.shape}, dtype: {value.dtype})"
        elif value.size <= max_elements:
            if value.ndim == 1:
                return f"{value.tolist()} (shape: {value.shape}, dtype: {value.dtype})"
            else:
                return f"Array with shape {value.shape}, dtype: {value.dtype}:\n{str(value)}"
        else:
            flat = value.flatten()
            preview = flat[:max_elements].tolist()
            return f"[{', '.join(map(str, preview))}, ...] (shape: {value.shape}, dtype: {value.dtype}, {value.size} total elements)"
    elif isinstance(value, dict):
        if len(value) <= 5:
            return json.dumps(value, indent=2, default=str)
        else:
            keys = list(value.keys())[:5]
            preview = {k: value[k] for k in keys}
            return f"{json.dumps(preview, indent=2, default=str)}\n... ({len(value)} total keys)"
    elif isinstance(value, (list, tuple)) and len(value) > max_elements:
        preview = value[:max_elements]
        return f"{preview} ... ({len(value)} total elements)"
    else:
        return str(value)

def arrays_equal(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two arrays are equal, handling different dtypes and shapes."""
    if a.shape != b.shape:
        return False
    
    try:
        if a.dtype != b.dtype:
            # Try to compare after type conversion if possible
            if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':  # Both float/complex
                return np.allclose(a.astype(float), b.astype(float), rtol=rtol, atol=atol)
            elif a.dtype.kind in 'ui' and b.dtype.kind in 'ui':  # Both integer
                return np.array_equal(a, b)
            else:
                # For strings or other types, convert to string and compare
                return np.array_equal(a.astype(str), b.astype(str))
        
        if a.dtype.kind in 'fc':  # Float or complex
            return np.allclose(a, b, rtol=rtol, atol=atol)
        else:
            return np.array_equal(a, b)
    except Exception:
        # Fallback comparison
        try:
            return bool((a == b).all())
        except Exception:
            return False

def compare_values(key: str, val1: Any, val2: Any) -> Tuple[bool, str]:
    """Compare two values and return (is_equal, difference_description)."""
    
    # Handle None values
    if val1 is None and val2 is None:
        return True, ""
    elif val1 is None or val2 is None:
        return False, f"One is None: {format_value(val1)} vs {format_value(val2)}"
    
    # Handle numpy arrays
    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        if arrays_equal(val1, val2):
            return True, ""
        else:
            return False, f"Arrays differ:\n  File 1: {format_value(val1)}\n  File 2: {format_value(val2)}"
    
    # Handle dictionaries
    elif isinstance(val1, dict) and isinstance(val2, dict):
        if val1 == val2:
            return True, ""
        else:
            diff_keys = set(val1.keys()) ^ set(val2.keys())
            common_keys = set(val1.keys()) & set(val2.keys())
            
            differences = []
            if diff_keys:
                differences.append(f"Different keys: {diff_keys}")
            
            for k in common_keys:
                if val1[k] != val2[k]:
                    differences.append(f"  {k}: {val1[k]} vs {val2[k]}")
            
            return False, "Dictionary differences:\n" + "\n".join(differences)
    
    # Handle other types
    else:
        try:
            if val1 == val2:
                return True, ""
            else:
                return False, f"{format_value(val1)} vs {format_value(val2)}"
        except ValueError:
            # Handle cases where == doesn't work (like arrays)
            return False, f"Cannot compare: {format_value(val1)} vs {format_value(val2)}"

def compare_npz_files(file1: str, file2: str, verbose: bool = False) -> None:
    """Compare two NPZ files and display differences."""
    
    print(f"Comparing NPZ files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print("=" * 80)
    
    data1 = load_npz_file(file1)
    data2 = load_npz_file(file2)
    
    # Get all keys from both files
    all_keys = set(data1.keys()) | set(data2.keys())
    keys1_only = set(data1.keys()) - set(data2.keys())
    keys2_only = set(data2.keys()) - set(data1.keys())
    common_keys = set(data1.keys()) & set(data2.keys())
    
    print(f"Summary:")
    print(f"  Total unique keys: {len(all_keys)}")
    print(f"  Keys in both files: {len(common_keys)}")
    print(f"  Keys only in file 1: {len(keys1_only)}")
    print(f"  Keys only in file 2: {len(keys2_only)}")
    print()
    
    # Report keys only in one file
    if keys1_only:
        print(f"Keys only in {file1}:")
        for key in sorted(keys1_only):
            print(f"  {key}: {format_value(data1[key])}")
        print()
    
    if keys2_only:
        print(f"Keys only in {file2}:")
        for key in sorted(keys2_only):
            print(f"  {key}: {format_value(data2[key])}")
        print()
    
    # Compare common keys
    differences = []
    identical = []
    
    for key in sorted(common_keys):
        is_equal, diff_description = compare_values(key, data1[key], data2[key])
        
        if is_equal:
            identical.append(key)
            if verbose:
                print(f"✓ {key}: IDENTICAL")
                print(f"  Value: {format_value(data1[key])}")
        else:
            differences.append((key, diff_description))
            print(f"✗ {key}: DIFFERENT")
            print(f"  {diff_description}")
        print()
    
    # Summary
    print("=" * 80)
    print(f"COMPARISON SUMMARY:")
    print(f"  Identical fields: {len(identical)}")
    print(f"  Different fields: {len(differences)}")
    print(f"  Fields only in file 1: {len(keys1_only)}")
    print(f"  Fields only in file 2: {len(keys2_only)}")
    
    if identical and verbose:
        print(f"\nIdentical fields: {', '.join(identical)}")
    
    if differences:
        print(f"\nDifferent fields: {', '.join([d[0] for d in differences])}")
    
    if len(differences) == 0 and len(keys1_only) == 0 and len(keys2_only) == 0:
        print(f"\n✓ FILES ARE IDENTICAL!")
    else:
        print(f"\n✗ FILES HAVE DIFFERENCES!")

def main():
    parser = argparse.ArgumentParser(description="Compare NPZ gripper files")
    parser.add_argument("file1", default="bots/Robotiq_2F_85.npz", help="First NPZ file to compare")
    parser.add_argument("file2", default="bots/Robotiq_2F_85_z.npz", help="Second NPZ file to compare")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Show details for identical fields too")
    parser.add_argument("--rtol", type=float, default=1e-5,
                       help="Relative tolerance for floating point comparison")
    parser.add_argument("--atol", type=float, default=1e-8,
                       help="Absolute tolerance for floating point comparison")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for comparison")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.file1).exists():
        print(f"Error: File {args.file1} does not exist")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"Error: File {args.file2} does not exist")
        sys.exit(1)
    
    compare_npz_files(args.file1, args.file2, args.verbose)

if __name__ == "__main__":
    main() 