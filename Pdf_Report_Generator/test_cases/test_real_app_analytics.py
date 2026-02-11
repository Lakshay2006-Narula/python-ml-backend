"""
Test app analytics with real DB data using only 'apps' column
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_data_db import load_project_data
from src.kpi_analysis import generate_app_analytics

print("=" * 70)
print("TESTING APP ANALYTICS WITH REAL DATABASE DATA")
print("Using only 'apps' column (not app_name)")
print("=" * 70)

# Load real data
raw_df, filtered_df, project_meta = load_project_data(149)

print(f"\nTotal filtered rows: {len(filtered_df)}")
print(f"'apps' column exists: {'apps' in filtered_df.columns}")

if 'apps' in filtered_df.columns:
    unique_apps = filtered_df['apps'].dropna().unique()
    print(f"Unique apps in database: {list(unique_apps)}")
    print(f"Total unique apps: {len(unique_apps)}")
    
    # Count samples per app
    print("\nSample count per app:")
    for app in unique_apps:
        count = len(filtered_df[filtered_df['apps'] == app])
        print(f"  {app}: {count} samples")

print("\n" + "-" * 70)
print("Generating app analytics...")
print("-" * 70)

generate_app_analytics(filtered_df)

print("\n✅ Done! Check data/images/kpi_analysis/ for:")
print("  - app_analytics_part1.png")
print("  - app_analytics_part2.png")
