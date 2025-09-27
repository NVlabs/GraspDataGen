#!/usr/bin/env python3
"""
Analyze grasp data files and provide comprehensive statistics.

This tool analyzes grasp data files (both YAML and JSON formats) and provides
detailed statistics including success/failure rates, confidence distributions,
and quality metrics. It can analyze individual files or entire directories.

Usage:
    python analyze_grasp_data.py <file_or_directory> [options]
    
Examples:
    # Analyze a single YAML file
    python analyze_grasp_data.py grasp_sim_data/onrobot_rg6/banana.yaml
    
    # Analyze a single JSON file
    python analyze_grasp_data.py graspgen_data/banana.json
    
    # Analyze all files in a directory
    python analyze_grasp_data.py grasp_sim_data/onrobot_rg6/
    
    # Analyze with detailed output and save results
    python analyze_grasp_data.py graspgen_data/ --detailed --output results.json
"""

import argparse
import json
import yaml
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np
from collections import defaultdict, Counter
import statistics

# Add the parent directory to the path to import graspgen_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graspgen_utils import print_blue, print_purple, print_yellow, print_green, print_red


def load_grasp_file(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Load a grasp file (YAML or JSON) and return the data and format type.
    
    Args:
        file_path: Path to the grasp file
        
    Returns:
        Tuple of (data_dict, format_type)
        
    Raises:
        ValueError: If file format is not supported or file is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            try:
                data = yaml.safe_load(f)
                return data, 'yaml'
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML file {file_path}: {e}")
        elif file_path.suffix.lower() == '.json':
            try:
                data = json.load(f)
                return data, 'json'
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file {file_path}: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


def extract_grasp_data_yaml(data: Dict[str, Any]) -> Tuple[List[float], List[bool], Dict[str, Any]]:
    """
    Extract grasp data from Isaac Lab YAML format.
    
    Args:
        data: YAML data dictionary
        
    Returns:
        Tuple of (confidence_values, success_flags, metadata)
    """
    grasps = data.get('grasps', {})
    confidence_values = []
    success_flags = []
    
    for grasp_name, grasp_data in grasps.items():
        confidence = grasp_data.get('confidence', 1.0)
        confidence_values.append(confidence)
        # Isaac Lab uses confidence > 0.0 as success
        success_flags.append(confidence > 0.0)
    
    metadata = {
        'object_file': data.get('object_file', 'unknown'),
        'object_scale': data.get('object_scale', 1.0),
        'gripper_file': data.get('gripper_file', 'unknown'),
        'format_version': data.get('format_version', 'unknown'),
        'base_length': data.get('base_length', None),
        'approach_axis': data.get('approach_axis', None),
        'bite_point': data.get('bite_point', None)
    }
    
    return confidence_values, success_flags, metadata


def extract_grasp_data_json(data: Dict[str, Any]) -> Tuple[List[float], List[bool], Dict[str, Any]]:
    """
    Extract grasp data from GraspGen JSON format.
    
    Args:
        data: JSON data dictionary
        
    Returns:
        Tuple of (confidence_values, success_flags, metadata)
    """
    grasps = data.get('grasps', {})
    transforms = grasps.get('transforms', [])
    object_in_gripper = grasps.get('object_in_gripper', [])
    
    if len(transforms) != len(object_in_gripper):
        raise ValueError(f"Mismatch between transforms ({len(transforms)}) and object_in_gripper ({len(object_in_gripper)})")
    
    # Convert boolean success flags to confidence values (1.0 for success, 0.0 for failure)
    confidence_values = [1.0 if success else 0.0 for success in object_in_gripper]
    success_flags = object_in_gripper
    
    metadata = {
        'object_file': data.get('object', {}).get('file', 'unknown'),
        'object_scale': data.get('object', {}).get('scale', 1.0),
        'gripper_name': data.get('gripper', {}).get('name', 'unknown'),
        'gripper_file': data.get('gripper', {}).get('file_name', 'unknown'),
        'gripper_width': data.get('gripper', {}).get('width', None),
        'gripper_depth': data.get('gripper', {}).get('depth', None)
    }
    
    return confidence_values, success_flags, metadata


def calculate_statistics(confidence_values: List[float], success_flags: List[bool]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from grasp data.
    
    Args:
        confidence_values: List of confidence values
        success_flags: List of success flags
        
    Returns:
        Dictionary containing various statistics
    """
    if not confidence_values:
        return {}
    
    total_grasps = len(confidence_values)
    successful_grasps = sum(success_flags)
    failed_grasps = total_grasps - successful_grasps
    success_rate = successful_grasps / total_grasps if total_grasps > 0 else 0.0
    
    # Confidence statistics
    conf_array = np.array(confidence_values)
    confidence_stats = {
        'mean': float(np.mean(conf_array)),
        'median': float(np.median(conf_array)),
        'std': float(np.std(conf_array)),
        'min': float(np.min(conf_array)),
        'max': float(np.max(conf_array)),
        'q25': float(np.percentile(conf_array, 25)),
        'q75': float(np.percentile(conf_array, 75))
    }
    
    # Confidence distribution
    confidence_distribution = Counter(confidence_values)
    
    # Quality metrics
    high_confidence = sum(1 for conf in confidence_values if conf >= 0.8)
    medium_confidence = sum(1 for conf in confidence_values if 0.3 <= conf < 0.8)
    low_confidence = sum(1 for conf in confidence_values if conf < 0.3)
    
    return {
        'total_grasps': total_grasps,
        'successful_grasps': successful_grasps,
        'failed_grasps': failed_grasps,
        'success_rate': success_rate,
        'confidence_stats': confidence_stats,
        'confidence_distribution': dict(confidence_distribution),
        'quality_metrics': {
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'high_confidence_rate': high_confidence / total_grasps if total_grasps > 0 else 0.0
        }
    }


def analyze_single_file(file_path: Path, detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze a single grasp file.
    
    Args:
        file_path: Path to the grasp file
        detailed: Whether to include detailed output
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        data, format_type = load_grasp_file(file_path)
        
        if format_type == 'yaml':
            confidence_values, success_flags, metadata = extract_grasp_data_yaml(data)
        elif format_type == 'json':
            confidence_values, success_flags, metadata = extract_grasp_data_json(data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        stats = calculate_statistics(confidence_values, success_flags)
        
        result = {
            'file_path': str(file_path),
            'format': format_type,
            'metadata': metadata,
            'statistics': stats
        }
        
        if detailed:
            result['raw_data'] = {
                'confidence_values': confidence_values,
                'success_flags': success_flags
            }
        
        return result
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': str(e),
            'statistics': {}
        }


def print_file_analysis(result: Dict[str, Any], detailed: bool = False):
    """
    Print analysis results for a single file.
    
    Args:
        result: Analysis result dictionary
        detailed: Whether to print detailed information
    """
    file_path = result['file_path']
    
    if 'error' in result:
        print_red(f"❌ Error analyzing {file_path}: {result['error']}")
        return
    
    stats = result['statistics']
    metadata = result['metadata']
    format_type = result['format']
    
    print_blue(f"\n📊 Analysis: {Path(file_path).name}")
    print_blue(f"   Format: {format_type.upper()}")
    print_blue(f"   Object: {metadata.get('object_file', 'unknown')}")
    print_blue(f"   Scale: {metadata.get('object_scale', 'unknown')}")
    
    if format_type == 'yaml':
        print_blue(f"   Gripper: {metadata.get('gripper_file', 'unknown')}")
    elif format_type == 'json':
        print_blue(f"   Gripper: {metadata.get('gripper_name', 'unknown')}")
    
    if stats:
        print_green(f"   Total Grasps: {stats['total_grasps']}")
        print_green(f"   Successful: {stats['successful_grasps']} ({stats['success_rate']:.1%})")
        print_green(f"   Failed: {stats['failed_grasps']}")
        
        conf_stats = stats['confidence_stats']
        print_purple(f"   Confidence - Mean: {conf_stats['mean']:.3f}, Std: {conf_stats['std']:.3f}")
        print_purple(f"   Confidence - Min: {conf_stats['min']:.3f}, Max: {conf_stats['max']:.3f}")
        
        quality = stats['quality_metrics']
        print_yellow(f"   Quality - High: {quality['high_confidence']}, Medium: {quality['medium_confidence']}, Low: {quality['low_confidence']}")
        
        if detailed:
            print_blue(f"   Confidence Distribution: {stats['confidence_distribution']}")


def analyze_criteria(files: List[Dict[str, Any]], min_success: int = None, min_failed: int = None, min_total: int = None) -> Dict[str, Any]:
    """
    Analyze files against specified criteria.
    
    Args:
        files: List of file analysis results
        min_success: Minimum successful grasps required
        min_failed: Minimum failed grasps required  
        min_total: Minimum total grasps required
        
    Returns:
        Dictionary containing criteria analysis results
    """
    if not any([min_success, min_failed, min_total]):
        return {}
    
    criteria = {
        'min_success': min_success,
        'min_failed': min_failed,
        'min_total': min_total
    }
    
    meets_criteria = []
    fails_criteria = []
    
    # Track totals for completion statistics
    total_success_current = 0
    total_success_needed = 0
    total_failed_current = 0
    total_failed_needed = 0
    total_grasps_current = 0
    total_grasps_needed = 0
    
    for result in files:
        if 'error' in result or not result['statistics']:
            continue
            
        stats = result['statistics']
        filename = Path(result['file_path']).name
        
        reasons = []
        meets = True
        
        # Check success criteria
        if min_success and stats['successful_grasps'] < min_success:
            meets = False
            reasons.append(f"<{min_success} successful")
        
        # Check failed criteria
        if min_failed and stats['failed_grasps'] < min_failed:
            meets = False
            reasons.append(f"<{min_failed} failed")
        
        # Check total criteria
        if min_total and stats['total_grasps'] < min_total:
            meets = False
            reasons.append(f"<{min_total} total")
        
        # Accumulate totals for all files (not just failing ones)
        if min_success:
            # Only count up to the minimum requirement per file
            success_contribution = min(stats['successful_grasps'], min_success)
            total_success_current += success_contribution
            if stats['successful_grasps'] < min_success:
                total_success_needed += (min_success - stats['successful_grasps'])
        
        if min_failed:
            # Only count up to the minimum requirement per file
            failed_contribution = min(stats['failed_grasps'], min_failed)
            total_failed_current += failed_contribution
            if stats['failed_grasps'] < min_failed:
                total_failed_needed += (min_failed - stats['failed_grasps'])
        
        if min_total:
            # Only count up to the minimum requirement per file
            total_contribution = min(stats['total_grasps'], min_total)
            total_grasps_current += total_contribution
            if stats['total_grasps'] < min_total:
                total_grasps_needed += (min_total - stats['total_grasps'])
        
        if meets:
            meets_criteria.append(result)
        else:
            fails_criteria.append({
                'filename': filename,
                'reasons': reasons,
                'statistics': stats
            })
    
    # Calculate completion statistics
    files_needing_more = {}
    if total_success_needed > 0:
        # Calculate what percentage of the requirement has been met
        total_required = total_success_current + total_success_needed
        completion_pct = (total_success_current / total_required) * 100 if total_required > 0 else 0
        files_needing_more['success'] = {
            'current': total_success_current,
            'needed': total_success_needed,
            'completion_percentage': min(100.0, completion_pct)
        }
    
    if total_failed_needed > 0:
        # Calculate what percentage of the requirement has been met
        total_required = total_failed_current + total_failed_needed
        completion_pct = (total_failed_current / total_required) * 100 if total_required > 0 else 0
        files_needing_more['failed'] = {
            'current': total_failed_current,
            'needed': total_failed_needed,
            'completion_percentage': min(100.0, completion_pct)
        }
    
    if total_grasps_needed > 0:
        # Calculate what percentage of the requirement has been met
        total_required = total_grasps_current + total_grasps_needed
        completion_pct = (total_grasps_current / total_required) * 100 if total_required > 0 else 0
        files_needing_more['total'] = {
            'current': total_grasps_current,
            'needed': total_grasps_needed,
            'completion_percentage': min(100.0, completion_pct)
        }
    
    total_files = len(meets_criteria) + len(fails_criteria)
    
    return {
        'criteria': criteria,
        'statistics': {
            'files_meeting_criteria': len(meets_criteria),
            'files_failing_criteria': len(fails_criteria),
            'criteria_percentage': len(meets_criteria) / total_files if total_files > 0 else 0,
            'files_needing_more': files_needing_more
        },
        'meets_criteria': meets_criteria,
        'fails_criteria': fails_criteria
    }


def print_progress_bar(current: int, total: int, start_time: float, prefix: str = "Progress"):
    """
    Print a progress bar with percentage and estimated time remaining.
    
    Args:
        current: Current progress (number of items processed)
        total: Total number of items to process
        start_time: Start time of the operation
        prefix: Prefix text for the progress bar
    """
    if total == 0:
        return
    
    percent = (current / total) * 100
    elapsed_time = time.time() - start_time
    
    if current > 0:
        estimated_total_time = elapsed_time * total / current
        remaining_time = estimated_total_time - elapsed_time
        remaining_str = f"ETA: {remaining_time:.0f}s"
    else:
        remaining_str = "ETA: calculating..."
    
    # Create progress bar
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Print progress bar
    print(f"\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total}) {remaining_str}", end='', flush=True)
    
    if current == total:
        print()  # New line when complete


def analyze_directory(directory_path: Path, detailed: bool = False, min_success: int = None, min_failed: int = None, min_total: int = None, show_progress: bool = True) -> Dict[str, Any]:
    """
    Analyze all grasp files in a directory.
    
    Args:
        directory_path: Path to the directory
        detailed: Whether to include detailed output
        
    Returns:
        Dictionary containing analysis results for all files
    """
    results = []
    total_stats = defaultdict(int)
    all_confidence_values = []
    all_success_flags = []
    
    # Per-file statistics for averages
    per_file_stats = {
        'total_grasps': [],
        'successful_grasps': [],
        'failed_grasps': [],
        'success_rates': []
    }
    
    # Find all YAML and JSON files
    yaml_files = list(directory_path.glob("*.yaml")) + list(directory_path.glob("*.yml"))
    json_files = list(directory_path.glob("*.json"))
    all_files = yaml_files + json_files
    
    if not all_files:
        print_yellow(f"No grasp files found in {directory_path}")
        return {'files': [], 'summary': {}}
    
    if show_progress:
        print_blue(f"Found {len(all_files)} grasp files to analyze")
        start_time = time.time()
    else:
        print_blue(f"Found {len(all_files)} grasp files to analyze")
    
    for i, file_path in enumerate(all_files):
        result = analyze_single_file(file_path, detailed)
        results.append(result)
        
        if show_progress and (i + 1) % 50 == 0 or i == len(all_files) - 1:
            print_progress_bar(i + 1, len(all_files), start_time, "Analyzing")
        
        if 'error' not in result and result['statistics']:
            stats = result['statistics']
            total_stats['total_grasps'] += stats['total_grasps']
            total_stats['successful_grasps'] += stats['successful_grasps']
            total_stats['failed_grasps'] += stats['failed_grasps']
            
            # Collect per-file statistics
            per_file_stats['total_grasps'].append(stats['total_grasps'])
            per_file_stats['successful_grasps'].append(stats['successful_grasps'])
            per_file_stats['failed_grasps'].append(stats['failed_grasps'])
            per_file_stats['success_rates'].append(stats['success_rate'])
            
            if detailed and 'raw_data' in result:
                all_confidence_values.extend(result['raw_data']['confidence_values'])
                all_success_flags.extend(result['raw_data']['success_flags'])
    
    # Calculate overall statistics
    overall_stats = {}
    valid_files = len([r for r in results if 'error' not in r])
    
    if total_stats['total_grasps'] > 0 and valid_files > 0:
        overall_stats = {
            'total_grasps': total_stats['total_grasps'],
            'successful_grasps': total_stats['successful_grasps'],
            'failed_grasps': total_stats['failed_grasps'],
            'success_rate': total_stats['successful_grasps'] / total_stats['total_grasps'],
            'files_analyzed': valid_files,
            'files_with_errors': len(results) - valid_files
        }
        
        # Calculate per-file averages
        if per_file_stats['total_grasps']:
            overall_stats['per_file_averages'] = {
                'avg_grasps_per_file': float(np.mean(per_file_stats['total_grasps'])),
                'avg_successful_per_file': float(np.mean(per_file_stats['successful_grasps'])),
                'avg_failed_per_file': float(np.mean(per_file_stats['failed_grasps'])),
                'avg_success_rate_per_file': float(np.mean(per_file_stats['success_rates'])),
                'std_grasps_per_file': float(np.std(per_file_stats['total_grasps'])),
                'std_successful_per_file': float(np.std(per_file_stats['successful_grasps'])),
                'std_failed_per_file': float(np.std(per_file_stats['failed_grasps'])),
                'std_success_rate_per_file': float(np.std(per_file_stats['success_rates'])),
                'min_grasps_per_file': int(np.min(per_file_stats['total_grasps'])),
                'max_grasps_per_file': int(np.max(per_file_stats['total_grasps'])),
                'min_successful_per_file': int(np.min(per_file_stats['successful_grasps'])),
                'max_successful_per_file': int(np.max(per_file_stats['successful_grasps'])),
                'min_failed_per_file': int(np.min(per_file_stats['failed_grasps'])),
                'max_failed_per_file': int(np.max(per_file_stats['failed_grasps']))
            }
        
        if detailed and all_confidence_values:
            conf_array = np.array(all_confidence_values)
            overall_stats['confidence_stats'] = {
                'mean': float(np.mean(conf_array)),
                'median': float(np.median(conf_array)),
                'std': float(np.std(conf_array)),
                'min': float(np.min(conf_array)),
                'max': float(np.max(conf_array))
            }
    
    # Perform criteria analysis if criteria are specified
    criteria_analysis = analyze_criteria(results, min_success, min_failed, min_total)
    
    return {
        'files': results,
        'summary': overall_stats,
        'directory': str(directory_path),
        'criteria_analysis': criteria_analysis
    }


def print_criteria_analysis(criteria_analysis: Dict[str, Any]):
    """
    Print criteria analysis results.
    
    Args:
        criteria_analysis: Criteria analysis results
    """
    if not criteria_analysis:
        return
    
    criteria = criteria_analysis['criteria']
    stats = criteria_analysis['statistics']
    meets = criteria_analysis['meets_criteria']
    fails = criteria_analysis['fails_criteria']
    
    print_blue(f"\n🎯 Criteria Analysis:")
    
    # Print criteria
    criteria_parts = []
    if criteria['min_success']:
        criteria_parts.append(f"≥{criteria['min_success']} successful")
    if criteria['min_failed']:
        criteria_parts.append(f"≥{criteria['min_failed']} failed")
    if criteria['min_total']:
        criteria_parts.append(f"≥{criteria['min_total']} total")
    
    print_blue(f"   Criteria: {', '.join(criteria_parts)}")
    
    # Print summary
    print_green(f"   Files Meeting Criteria: {stats['files_meeting_criteria']} ({stats['criteria_percentage']:.1%})")
    print_red(f"   Files Failing Criteria: {stats['files_failing_criteria']}")
    
    # Print completion statistics for files that need more
    needs_more = stats['files_needing_more']
    if needs_more.get('success'):
        success_info = needs_more['success']
        total_required = success_info['current'] + success_info['needed']
        percentage = (success_info['current'] / total_required) * 100 if total_required > 0 else 0
        print_yellow(f"   Success Progress: {success_info['current']}/{total_required} ({percentage:.1f}%)")
    
    if needs_more.get('failed'):
        failed_info = needs_more['failed']
        total_required = failed_info['current'] + failed_info['needed']
        percentage = (failed_info['current'] / total_required) * 100 if total_required > 0 else 0
        print_yellow(f"   Failed Progress: {failed_info['current']}/{total_required} ({percentage:.1f}%)")
    
    if needs_more.get('total'):
        total_info = needs_more['total']
        total_required = total_info['current'] + total_info['needed']
        percentage = (total_info['current'] / total_required) * 100 if total_required > 0 else 0
        print_yellow(f"   Total Progress: {total_info['current']}/{total_required} ({percentage:.1f}%)")
    
    # Print files that fail criteria
    if fails:
        print_red(f"\n❌ Files Failing Criteria: {len(fails)} files")
        # Only show first few examples if there are many failures
        if len(fails) <= 5:
            for file_info in fails:
                filename = file_info['filename']
                reasons = ', '.join(file_info['reasons'])
                stats = file_info['statistics']
                print_red(f"   {filename}: {reasons} (S:{stats['successful_grasps']}, F:{stats['failed_grasps']}, T:{stats['total_grasps']})")
        else:
            # Show first 3 examples
            for file_info in fails[:3]:
                filename = file_info['filename']
                reasons = ', '.join(file_info['reasons'])
                stats = file_info['statistics']
                print_red(f"   {filename}: {reasons} (S:{stats['successful_grasps']}, F:{stats['failed_grasps']}, T:{stats['total_grasps']})")
            print_red(f"   ... and {len(fails) - 3} more files")


def print_directory_summary(analysis: Dict[str, Any]):
    """
    Print summary statistics for directory analysis.
    
    Args:
        analysis: Directory analysis results
    """
    summary = analysis['summary']
    files = analysis['files']
    
    if not summary:
        print_yellow("No valid files found for analysis")
        return
    
    print_blue(f"\n📈 Directory Summary: {Path(analysis['directory']).name}")
    print_green(f"   Files Analyzed: {summary['files_analyzed']}")
    if 'files_with_errors' in summary and summary['files_with_errors'] > 0:
        print_red(f"   Files with Errors: {summary['files_with_errors']}")
    
    print_green(f"   Total Grasps: {summary['total_grasps']}")
    print_green(f"   Successful: {summary['successful_grasps']} ({summary['success_rate']:.1%})")
    print_green(f"   Failed: {summary['failed_grasps']}")
    
    if 'confidence_stats' in summary:
        conf_stats = summary['confidence_stats']
        print_purple(f"   Overall Confidence - Mean: {conf_stats['mean']:.3f}, Std: {conf_stats['std']:.3f}")
    
    # Display per-file averages
    if 'per_file_averages' in summary:
        avg_stats = summary['per_file_averages']
        print_yellow(f"\n📊 Per-File Averages:")
        print_yellow(f"   Avg Grasps per File: {avg_stats['avg_grasps_per_file']:.1f} ± {avg_stats['std_grasps_per_file']:.1f}")
        print_yellow(f"   Avg Successful per File: {avg_stats['avg_successful_per_file']:.1f} ± {avg_stats['std_successful_per_file']:.1f}")
        print_yellow(f"   Avg Failed per File: {avg_stats['avg_failed_per_file']:.1f} ± {avg_stats['std_failed_per_file']:.1f}")
        print_yellow(f"   Avg Success Rate per File: {avg_stats['avg_success_rate_per_file']:.1%} ± {avg_stats['std_success_rate_per_file']:.1%}")
        
        print_yellow(f"\n📊 Per-File Ranges:")
        print_yellow(f"   Grasps per File: {avg_stats['min_grasps_per_file']} - {avg_stats['max_grasps_per_file']}")
        print_yellow(f"   Successful per File: {avg_stats['min_successful_per_file']} - {avg_stats['max_successful_per_file']}")
        print_yellow(f"   Failed per File: {avg_stats['min_failed_per_file']} - {avg_stats['max_failed_per_file']}")
    
    # Show individual file results (only if not too many)
    if len(files) <= 10:
        print_blue(f"\n📋 Individual File Results:")
        for result in files:
            if 'error' not in result:
                stats = result['statistics']
                if stats:
                    filename = Path(result['file_path']).name
                    print_green(f"   {filename}: {stats['total_grasps']} grasps, {stats['success_rate']:.1%} success")
    else:
        print_blue(f"\n📋 Individual File Results: {len([r for r in files if 'error' not in r and r['statistics']])} files analyzed")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grasp data files and provide comprehensive statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single YAML file
  python analyze_grasp_data.py grasp_sim_data/onrobot_rg6/banana.yaml
  
  # Analyze a single JSON file  
  python analyze_grasp_data.py graspgen_data/banana.json
  
  # Analyze all files in a directory
  python analyze_grasp_data.py grasp_sim_data/onrobot_rg6/
  
  # Analyze with detailed output and save results
  python analyze_grasp_data.py graspgen_data/ --detailed --output results.json
  
  # Analyze with criteria (minimum 1000 successful, 500 failed grasps per file)
  python analyze_grasp_data.py graspgen_data/ --min-success 1000 --min-failed 500
  
  # Analyze with minimum total grasps requirement
  python analyze_grasp_data.py graspgen_data/ --min-total 2000
  
  # Analyze with progress bar disabled (for scripts)
  python analyze_grasp_data.py graspgen_data/ --no-progress
        """
    )
    
    parser.add_argument(
        'path',
        help='Path to grasp file or directory containing grasp files'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Include detailed output with raw data'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        help='Save analysis results to JSON file'
    )
    
    parser.add_argument(
        '--min-success',
        type=int,
        help='Minimum number of successful grasps required per file'
    )
    
    parser.add_argument(
        '--min-failed',
        type=int,
        help='Minimum number of failed grasps required per file'
    )
    
    parser.add_argument(
        '--min-total',
        type=int,
        help='Minimum total number of grasps required per file'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output (useful when saving to file)'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar (useful for automated scripts)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print_red(f"Path does not exist: {path}")
        sys.exit(1)
    
    # Analyze file or directory
    if path.is_file():
        analysis = analyze_single_file(path, args.detailed)
        if not args.quiet:
            print_file_analysis(analysis, args.detailed)
    elif path.is_dir():
        show_progress = not args.no_progress and not args.quiet
        analysis = analyze_directory(path, args.detailed, args.min_success, args.min_failed, args.min_total, show_progress)
        if not args.quiet:
            # Print summary (no individual file details for directories)
            print_directory_summary(analysis)
            # Print criteria analysis if criteria were specified
            if analysis.get('criteria_analysis'):
                print_criteria_analysis(analysis['criteria_analysis'])
    else:
        print_red(f"Path is neither a file nor directory: {path}")
        sys.exit(1)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        if not args.quiet:
            print_green(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
