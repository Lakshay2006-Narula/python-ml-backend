"""
Pre-Visualization Data Validation Test Case
============================================
Validates filtered data distribution immediately after DB load,
before any map generation or visualization logic.

This test prints:
- Total filtered samples per KPI
- Valid (non-null) value counts
- Distribution across all DB-defined ranges/categories
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_data_db import load_project_data
from src.kpi_config import KPI_CONFIG
from src.threshold_resolver import resolve_kpi_ranges
import pandas as pd


def print_separator(char="=", length=80):
    print(char * length)


def print_header(title):
    print_separator("=")
    print(f" {title}")
    print_separator("=")


def validate_range_kpi(df, kpi_name, column, user_id):
    """Validate range-based KPI (RSRP, RSRQ, SINR, DL, UL, MOS)"""
    
    print(f"\n{'='*80}")
    print(f"KPI: {kpi_name}")
    print(f"Column: {column}")
    print(f"{'='*80}")
    
    # Total samples in filtered dataframe
    total_samples = len(df)
    print(f"Total Filtered Samples: {total_samples}")
    
    # Check if column exists
    if column not in df.columns:
        print(f"❌ Column '{column}' NOT FOUND in dataframe")
        return
    
    # Valid (non-null) values
    valid_values = df[column].notna().sum()
    null_values = df[column].isna().sum()
    
    print(f"Valid (non-null) Values: {valid_values}")
    print(f"Null Values: {null_values}")
    
    if valid_values == 0:
        print(f"⚠️ No valid data for {kpi_name}")
        return
    
    # Get DB-defined threshold ranges
    try:
        ranges = resolve_kpi_ranges(kpi_name, user_id)
        print(f"\nDB-Defined Ranges: {len(ranges)} ranges")
        print("-" * 80)
    except Exception as e:
        print(f"❌ Error resolving ranges: {e}")
        return
    
    # Convert to numeric for comparison
    numeric_values = pd.to_numeric(df[column], errors='coerce').dropna()
    
    print(f"\n{'Range':<30} {'Min':<10} {'Max':<10} {'Count':<10} {'%':<10}")
    print("-" * 80)
    
    total_in_ranges = 0
    for r in ranges:
        range_label = r.get("range") or f"{r['min']} to {r['max']}"
        min_val = r["min"]
        max_val = r["max"]
        
        # Count samples in this range
        count = ((numeric_values >= min_val) & (numeric_values <= max_val)).sum()
        percentage = (count / valid_values * 100) if valid_values > 0 else 0
        
        total_in_ranges += count
        
        print(f"{range_label:<30} {min_val:<10} {max_val:<10} {count:<10} {percentage:<10.2f}")
    
    # Check for values outside all ranges
    outside_ranges = valid_values - total_in_ranges
    if outside_ranges > 0:
        print("-" * 80)
        print(f"⚠️ Values Outside All Ranges: {outside_ranges} ({outside_ranges/valid_values*100:.2f}%)")
    
    # Statistics
    print("-" * 80)
    print(f"Min Value: {numeric_values.min():.2f}")
    print(f"Max Value: {numeric_values.max():.2f}")
    print(f"Mean Value: {numeric_values.mean():.2f}")
    print(f"Median Value: {numeric_values.median():.2f}")


def validate_categorical_kpi(df, kpi_name, column):
    """Validate categorical KPI (Band, PCI, Technology)"""
    
    print(f"\n{'='*80}")
    print(f"KPI: {kpi_name}")
    print(f"Column: {column}")
    print(f"{'='*80}")
    
    # Total samples in filtered dataframe
    total_samples = len(df)
    print(f"Total Filtered Samples: {total_samples}")
    
    # Check if column exists
    if column not in df.columns:
        print(f"❌ Column '{column}' NOT FOUND in dataframe")
        return
    
    # Valid (non-null) values
    valid_values = df[column].notna().sum()
    null_values = df[column].isna().sum()
    
    print(f"Valid (non-null) Values: {valid_values}")
    print(f"Null Values: {null_values}")
    
    if valid_values == 0:
        print(f"⚠️ No valid data for {kpi_name}")
        return
    
    # Get value counts
    value_counts = df[column].value_counts().sort_index()
    
    print(f"\nUnique Categories: {len(value_counts)}")
    print("-" * 80)
    print(f"{'Category':<30} {'Count':<15} {'%':<10}")
    print("-" * 80)
    
    for value, count in value_counts.items():
        percentage = (count / valid_values * 100) if valid_values > 0 else 0
        print(f"{str(value):<30} {count:<15} {percentage:<10.2f}")
    
    print("-" * 80)
    print(f"Total Categorized: {value_counts.sum()}")


def run_validation_test(project_id, user_id):
    """Main validation test"""
    
    print_header("PRE-VISUALIZATION DATA VALIDATION TEST")
    print(f"Project ID: {project_id}")
    print(f"User ID: {user_id}")
    print()
    
    # Load data from database
    print("Loading data from database...")
    try:
        raw_df, filtered_df, project_meta = load_project_data(project_id)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"✅ Data loaded successfully")
    print(f"   Raw samples: {len(raw_df)}")
    print(f"   Filtered samples: {len(filtered_df)}")
    print()
    
    if filtered_df.empty:
        print("❌ Filtered dataframe is EMPTY - cannot proceed")
        return
    
    print(f"Filtered DataFrame Columns: {list(filtered_df.columns)}")
    print()
    
    # Validate each KPI
    for kpi_name, config in KPI_CONFIG.items():
        kpi_type = config["type"]
        column = config["column"]
        
        if kpi_type == "range":
            validate_range_kpi(filtered_df, kpi_name, column, user_id)
        elif kpi_type == "categorical":
            validate_categorical_kpi(filtered_df, kpi_name, column)
    
    print_separator("=")
    print("VALIDATION TEST COMPLETED")
    print_separator("=")


if __name__ == "__main__":
    # Test with your project
    PROJECT_ID = 149
    USER_ID = 13
    
    run_validation_test(PROJECT_ID, USER_ID)
