#!/usr/bin/env python3
"""
Results folder analysis
Detailed analysis before cleanup
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results_structure(results_dir="results"):
    """Detailed analysis of the results structure"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results folder {results_dir} not found!")
        return
    
    print("🔍 Analyzing results folder...")
    print("=" * 50)
    
    # Collect statistics
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'size_mb': 0,
        'file_types': defaultdict(int),
        'model_dirs': [],
        'important_files': [],
        'large_files': [],
        'duplicate_names': defaultdict(list)
    }
    
    # Walk through all files
    for root, dirs, files in os.walk(results_path):
        stats['total_dirs'] += len(dirs)
        stats['total_files'] += len(files)
        
        for file in files:
            file_path = Path(root) / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                stats['size_mb'] += size_mb
                
                # File type
                ext = file_path.suffix.lower()
                stats['file_types'][ext] += 1
                
                # Large files (>1MB)
                if size_mb > 1:
                    stats['large_files'].append({
                        'path': str(file_path),
                        'size_mb': size_mb
                    })
                
                # Important files
                if any(keyword in file.lower() for keyword in ['metrics', 'comparison', 'final', 'best', 'summary']):
                    stats['important_files'].append(str(file_path))
                
                # Duplicates by name
                stats['duplicate_names'][file].append(str(file_path))
    
    # Identify model folders
    for item in results_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            stats['model_dirs'].append(item.name)
    
    # Output
    print(f"📊 Overall statistics:")
    print(f"   Files: {stats['total_files']}")
    print(f"   Folders: {stats['total_dirs']}")
    print(f"   Size: {stats['size_mb']:.2f} MB")
    print()
    
    print(f"📁 Model folders ({len(stats['model_dirs'])}):")
    for dir_name in sorted(stats['model_dirs']):
        print(f"   - {dir_name}")
    print()
    
    print(f"📄 File types:")
    for ext, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {ext}: {count}")
    print()
    
    print(f"⭐ Important files ({len(stats['important_files'])}):")
    for file_path in sorted(stats['important_files']):
        rel_path = Path(file_path).relative_to(results_path)
        print(f"   - {rel_path}")
    print()
    
    print(f"💾 Large files (>1MB):")
    for file_info in sorted(stats['large_files'], key=lambda x: x['size_mb'], reverse=True):
        rel_path = Path(file_info['path']).relative_to(results_path)
        print(f"   - {rel_path} ({file_info['size_mb']:.2f} MB)")
    print()
    
    # Find duplicates
    duplicates = {name: paths for name, paths in stats['duplicate_names'].items() if len(paths) > 1}
    if duplicates:
        print(f"⚠️  Potential duplicates:")
        for name, paths in list(duplicates.items())[:10]:  # Show only the first 10
            print(f"   {name}:")
            for path in paths:
                rel_path = Path(path).relative_to(results_path)
                print(f"     - {rel_path}")
        print()
    
    # Save detailed analysis
    analysis_file = results_path / 'detailed_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"📋 Detailed analysis saved: {analysis_file}")
    
    return stats

def analyze_metrics_files(results_dir="results"):
    """Analyze all metrics files"""
    
    results_path = Path(results_dir)
    metrics_files = []
    
    # Find all CSV files
    for csv_file in results_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            metrics_files.append({
                'path': str(csv_file),
                'rows': len(df),
                'columns': list(df.columns),
                'size_mb': csv_file.stat().st_size / (1024 * 1024)
            })
        except Exception as e:
            print(f"⚠️  Could not read {csv_file}: {e}")
    
    print(f"📈 Metrics files found: {len(metrics_files)}")
    print()
    
    for file_info in sorted(metrics_files, key=lambda x: x['size_mb'], reverse=True):
        rel_path = Path(file_info['path']).relative_to(results_path)
        print(f"📊 {rel_path}")
        print(f"   Size: {file_info['size_mb']:.2f} MB")
        print(f"   Rows: {file_info['rows']}")
        print(f"   Columns: {file_info['columns']}")
        print()
    
    return metrics_files

def generate_cleanup_recommendations(stats):
    """Generate cleanup recommendations"""
    
    print("🎯 Cleanup recommendations:")
    print("=" * 50)
    
    # Recommendations based on analysis
    recommendations = []
    
    # Test folders
    test_dirs = [d for d in stats['model_dirs'] if 'test' in d.lower()]
    if test_dirs:
        recommendations.append(f"📁 Move {len(test_dirs)} test folders to archive: {', '.join(test_dirs)}")
    
    # Large files
    large_files = [f for f in stats['large_files'] if f['size_mb'] > 5]
    if large_files:
        recommendations.append(f"💾 Check {len(large_files)} very large files (>5MB)")
    
    # Duplicates
    duplicates = {name: paths for name, paths in stats['duplicate_names'].items() if len(paths) > 1}
    if duplicates:
        recommendations.append(f"⚠️  {len(duplicates)} files with identical names - check for duplicates")
    
    # Preserve important files
    if stats['important_files']:
        recommendations.append(f"⭐ Preserve {len(stats['important_files'])} important files in the final_results folder")
    
    # New structure
    recommendations.append("📂 Create new structure: final_results/, experiments/, archive/, analysis/")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print()
    print("✅ Recommendation: Use the cleanup_results.py script with --dry-run first!")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze results folder')
    parser.add_argument('--results-dir', default='results', help='Path to results folder')
    parser.add_argument('--metrics-only', action='store_true', help='Analyze only metrics files')
    
    args = parser.parse_args()
    
    if args.metrics_only:
        analyze_metrics_files(args.results_dir)
    else:
        stats = analyze_results_structure(args.results_dir)
        analyze_metrics_files(args.results_dir)
        generate_cleanup_recommendations(stats)
    
    print("🎉 Analysis complete!")

if __name__ == "__main__":
    main() 