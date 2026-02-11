"""
Detailed investigation for Project 148:
1. Check apps column content
2. Session filtering logic 
3. Area summary generation
4. Check for hardcoding in code
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_data_db import load_project_data
from src.kpi_analysis import run_kpi_analysis
from src.metadata_generator import build_metadata
import pandas as pd


print("\n" + "="*80)
print("DETAILED INVESTIGATION: Project 148")
print("="*80)

project_id = 148
user_id = 13

# Load data
raw_df, filtered_df, project_meta = load_project_data(project_id)

print(f"\n[1] APPS COLUMN DATA")
print(f"  Total rows: {len(filtered_df)}")
print(f"  'apps' column exists: {'apps' in filtered_df.columns}")

if 'apps' in filtered_df.columns:
    null_apps = filtered_df['apps'].isna().sum()
    non_null_apps = filtered_df['apps'].notna().sum()
    
    print(f"  Null apps: {null_apps} ({null_apps/len(filtered_df)*100:.2f}%)")
    print(f"  Non-null apps: {non_null_apps} ({non_null_apps/len(filtered_df)*100:.2f}%)")
    
    if non_null_apps > 0:
        print(f"\n  Unique app values (first 15):")
        unique_apps = filtered_df['apps'].dropna().unique()[:15]
        for app in unique_apps:
            count = (filtered_df['apps'] == app).sum()
            print(f"    '{app}': {count} records")
    else:
        print(f"  No app data in column")

print(f"\n[2] SESSION FILTERING ANALYSIS")
print(f"  Sessions in database: {[3247, 3250, 3281, 3286, 3287]}")
print(f"  Sessions in filtered_df:")
if not filtered_df.empty:
    unique_sessions = sorted(filtered_df['session_id'].unique())
    for sid in unique_sessions:
        count = (filtered_df['session_id'] == sid).sum()
        print(f"    Session {sid}: {count} records")

# Check if run_kpi_analysis filters sessions further
print(f"\n[3] RUN KPI ANALYSIS - Session handling")
try:
    session_ids = [int(s.strip()) for s in project_meta['ref_session_id'].split(',') if s.strip().isdigit()]
    print(f"  Input sessions: {session_ids} ({len(session_ids)} total)")
    
    kpi_metadata, drive_summary = run_kpi_analysis(
        filtered_df,
        user_id,
        __import__('src.kpi_config', fromlist=['KPI_CONFIG']).KPI_CONFIG,
        session_ids=session_ids,
        image_dir=f"data/tmp/test_148_kpi"
    )
    
    print(f"  KPI metadata keys: {list(kpi_metadata.keys())[:5]}")
    
    # Check if drive_summary has area info
    if drive_summary and 'area_summary' in drive_summary:
        print(f"\n  Area Summary:")
        area = drive_summary['area_summary']
        if isinstance(area, dict):
            print(f"    Total areas: {len(area)}")
            for area_name, count in list(area.items())[:5]:
                print(f"      {area_name}: {count}")
        else:
            print(f"    Type: {type(area)}")
            print(f"    Value: {str(area)[:100]}")
    
except Exception as e:
    print(f"  Error: {e}")

print(f"\n[4] BUILD METADATA - Session in output")
try:
    metadata = build_metadata(filtered_df, kpi_analysis_results=kpi_metadata, drive_summary_data=drive_summary)
    
    print(f"  Metadata keys: {list(metadata.keys())[:10]}")
    
    if 'sessions' in metadata:
        print(f"  Sessions in metadata: {metadata['sessions']}")
    if 'session_count' in metadata:
        print(f"  Session count in metadata: {metadata['session_count']}")
    if 'drive_summary' in metadata:
        print(f"  Drive summary sections: {list(metadata['drive_summary'].keys())}")
        
except Exception as e:
    print(f"  Error: {e}")

print(f"\n[5] CHECK FOR HARDCODING")
print(f"  This requires reading source code directly")
print(f"  Will check for hardcoded project IDs or session counts")

# Search in key functions
import inspect
from src import kpi_analysis

print(f"\n  Checking kpi_analysis.py functions:")
source_file = inspect.getsourcefile(kpi_analysis)
print(f"  Source: {source_file}")

# Check main.py
from src import main as main_module
main_source = inspect.getsourcefile(main_module)
print(f"\n  Checking main.py:")
print(f"  Source: {main_source}")

print("\n" + "="*80)
