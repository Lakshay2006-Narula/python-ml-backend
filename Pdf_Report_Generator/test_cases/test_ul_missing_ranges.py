"""
UL Throughput Missing Ranges Analysis
======================================
Analyzes the 46% missing values in UL throughput that fall outside DB-defined ranges.
Specifically checks for values > 20 and provides detailed distribution.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_data_db import load_project_data
from src.threshold_resolver import resolve_kpi_ranges
import pandas as pd


def analyze_ul_missing_ranges(project_id, user_id):
    """Analyze UL throughput values outside defined ranges"""
    
    print("=" * 80)
    print("UL THROUGHPUT - MISSING RANGES ANALYSIS")
    print("=" * 80)
    print(f"Project ID: {project_id}")
    print(f"User ID: {user_id}")
    print()
    
    # Load data
    print("Loading data from database...")
    raw_df, filtered_df, project_meta = load_project_data(project_id)
    
    print(f"Total Filtered Samples: {len(filtered_df)}")
    print()
    
    # Get UL column
    ul_column = "ul_tpt"
    
    if ul_column not in filtered_df.columns:
        print(f"❌ Column '{ul_column}' not found!")
        return
    
    # Convert to numeric
    ul_values = pd.to_numeric(filtered_df[ul_column], errors='coerce').dropna()
    total_valid = len(ul_values)
    
    print(f"Valid UL values: {total_valid}")
    print()
    
    # Get DB ranges
    ranges = resolve_kpi_ranges("UL", user_id)
    
    print("=" * 80)
    print("DB-DEFINED RANGES")
    print("=" * 80)
    for r in ranges:
        print(f"  {r['min']} to {r['max']}")
    print()
    
    # Statistics
    print("=" * 80)
    print("UL THROUGHPUT STATISTICS")
    print("=" * 80)
    print(f"Min Value:    {ul_values.min():.2f}")
    print(f"Max Value:    {ul_values.max():.2f}")
    print(f"Mean Value:   {ul_values.mean():.2f}")
    print(f"Median Value: {ul_values.median():.2f}")
    print(f"Std Dev:      {ul_values.std():.2f}")
    print()
    
    # Count samples in each DB range
    print("=" * 80)
    print("DISTRIBUTION WITHIN DB RANGES")
    print("=" * 80)
    print(f"{'Range':<20} {'Count':<10} {'%':<10}")
    print("-" * 80)
    
    total_in_ranges = 0
    for r in ranges:
        count = ((ul_values >= r["min"]) & (ul_values <= r["max"])).sum()
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        total_in_ranges += count
        print(f"{r['min']:>5} to {r['max']:<10} {count:<10} {percentage:<10.2f}")
    
    print("-" * 80)
    print(f"{'Total in ranges:':<20} {total_in_ranges:<10} {(total_in_ranges/total_valid*100):<10.2f}")
    print()
    
    # Values outside all ranges
    outside_ranges = total_valid - total_in_ranges
    print("=" * 80)
    print("VALUES OUTSIDE ALL DB RANGES")
    print("=" * 80)
    print(f"Count:      {outside_ranges}")
    print(f"Percentage: {(outside_ranges/total_valid*100):.2f}%")
    print()
    
    # Find max range limit
    max_range_limit = max([r["max"] for r in ranges])
    min_range_limit = min([r["min"] for r in ranges])
    
    print(f"Max DB Range Limit: {max_range_limit}")
    print(f"Min DB Range Limit: {min_range_limit}")
    print()
    
    # Analyze values above max range
    above_max = ul_values[ul_values > max_range_limit]
    below_min = ul_values[ul_values < min_range_limit]
    
    print("=" * 80)
    print(f"VALUES ABOVE {max_range_limit} (OUTSIDE UPPER BOUND)")
    print("=" * 80)
    print(f"Count:      {len(above_max)}")
    print(f"Percentage: {(len(above_max)/total_valid*100):.2f}%")
    
    if len(above_max) > 0:
        print(f"Min:        {above_max.min():.2f}")
        print(f"Max:        {above_max.max():.2f}")
        print(f"Mean:       {above_max.mean():.2f}")
        print(f"Median:     {above_max.median():.2f}")
        print()
        
        # Custom ranges for values > 20
        print("-" * 80)
        print("CUSTOM RANGE BREAKDOWN (VALUES > 20)")
        print("-" * 80)
        print(f"{'Range':<20} {'Count':<10} {'%':<10}")
        print("-" * 80)
        
        custom_ranges = [
            (20, 25),
            (25, 30),
            (30, 40),
            (40, 50),
            (50, 100),
            (100, 1000),
        ]
        
        for min_val, max_val in custom_ranges:
            count = ((ul_values >= min_val) & (ul_values <= max_val)).sum()
            percentage = (count / total_valid * 100) if total_valid > 0 else 0
            if count > 0:
                print(f"{min_val:>5} to {max_val:<10} {count:<10} {percentage:<10.2f}")
    
    print()
    
    # Analyze values below min range
    if len(below_min) > 0:
        print("=" * 80)
        print(f"VALUES BELOW {min_range_limit} (OUTSIDE LOWER BOUND)")
        print("=" * 80)
        print(f"Count:      {len(below_min)}")
        print(f"Percentage: {(len(below_min)/total_valid*100):.2f}%")
        print(f"Min:        {below_min.min():.2f}")
        print(f"Max:        {below_min.max():.2f}")
        print(f"Mean:       {below_min.mean():.2f}")
        print()
    
    # Gap analysis - values between defined ranges
    print("=" * 80)
    print("GAP ANALYSIS (VALUES BETWEEN DEFINED RANGES)")
    print("=" * 80)
    
    # Sort ranges by min value
    sorted_ranges = sorted(ranges, key=lambda x: x["min"])
    
    gaps_found = False
    for i in range(len(sorted_ranges) - 1):
        current_max = sorted_ranges[i]["max"]
        next_min = sorted_ranges[i + 1]["min"]
        
        if next_min > current_max:
            # There's a gap
            gap_values = ul_values[(ul_values > current_max) & (ul_values < next_min)]
            if len(gap_values) > 0:
                gaps_found = True
                print(f"Gap: {current_max} to {next_min}")
                print(f"  Count: {len(gap_values)}")
                print(f"  Percentage: {(len(gap_values)/total_valid*100):.2f}%")
                print()
    
    if not gaps_found:
        print("No gaps found between defined ranges.")
        print()
    
    # Histogram-style distribution
    print("=" * 80)
    print("COMPLETE DISTRIBUTION (5-unit buckets)")
    print("=" * 80)
    print(f"{'Range':<15} {'Count':<10} {'%':<10} {'Bar'}")
    print("-" * 80)
    
    min_val = int(ul_values.min())
    max_val = int(ul_values.max()) + 1
    bucket_size = 5
    
    for bucket_start in range(min_val, max_val, bucket_size):
        bucket_end = bucket_start + bucket_size
        count = ((ul_values >= bucket_start) & (ul_values < bucket_end)).sum()
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        
        if count > 0:
            bar = "#" * int(percentage / 2)  # Scale for visibility
            print(f"{bucket_start:>3}-{bucket_end:<10} {count:<10} {percentage:<10.2f} {bar}")
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("RECOMMENDATION:")
    print(f"  Add range: 20 to 30 (covers {((ul_values >= 20) & (ul_values <= 30)).sum()} samples)")
    print(f"  Add range: 30 to 50 (covers {((ul_values >= 30) & (ul_values <= 50)).sum()} samples)")
    print()


if __name__ == "__main__":
    PROJECT_ID = 149
    USER_ID = 13
    
    analyze_ul_missing_ranges(PROJECT_ID, USER_ID)
