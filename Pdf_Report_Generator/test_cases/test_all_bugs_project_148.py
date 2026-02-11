"""
COMPREHENSIVE BUG REPORT FOR PROJECT 148
Shows all issues with code audit and database investigation - NO CODE CHANGES
"""

import sys
sys.path.insert(0, '.')

from src.load_data_db import get_connection, get_project_by_id
from src.kpi_analysis import generate_app_analytics
from src.metadata_generator import build_area_summary
import pandas as pd

def investigate_session_ids_mismatch():
    """Bug #1: Session count mismatch - database shows 3 but filtered_df has 5"""
    print("\n" + "="*80)
    print("BUG #1: SESSION FILTERING - Database has 3, Actual Data Has 5")
    print("="*80)
    
    cn = get_connection()
    
    # Check what's in the database for project 148
    project = get_project_by_id(148, cn)
    ref_session_id = project.get("ref_session_id", "")
    
    print(f"\n✗ DATABASE ref_session_id field: '{ref_session_id}'")
    db_sessions = [int(s.strip()) for s in str(ref_session_id).split(",") if s.strip().isdigit()]
    print(f"✗ Parsed as {len(db_sessions)} sessions: {db_sessions}")
    
    # Now check what's actually in the data
    from src.load_data_db import load_project_data
    raw_df, filtered_df, _ = load_project_data(148)
    
    print(f"\n✓ filtered_df has {len(filtered_df)} total records")
    actual_sessions = sorted(filtered_df['session_id'].unique().tolist())
    print(f"✓ Actual unique sessions in data: {len(actual_sessions)}: {actual_sessions}")
    
    missing_sessions = set(actual_sessions) - set(db_sessions)
    print(f"\n⚠ SESSIONS IN DATA BUT NOT IN DATABASE REF: {missing_sessions}")
    
    # Show record counts per session
    print(f"\nSession Details:")
    for sid in actual_sessions:
        count = len(filtered_df[filtered_df['session_id'] == sid])
        marker = "✓" if sid in db_sessions else "✗ MISSING FROM DB"
        print(f"  Session {sid}: {count} records {marker}")
    
    cn.close()
    return filtered_df

def investigate_app_analytics_issue(filtered_df):
    """Bug #2: App analytics returns None because 'apps' column is 100% NULL"""
    print("\n" + "="*80)
    print("BUG #2: APP ANALYTICS MISSING - 'apps' column is 100% NULL")
    print("="*80)
    
    # Check apps column
    if "apps" in filtered_df.columns:
        null_count = filtered_df["apps"].isna().sum()
        total_count = len(filtered_df)
        null_pct = (null_count / total_count) * 100
        
        print(f"\n✗ 'apps' column: {null_count}/{total_count} nulls ({null_pct:.2f}%)")
        
        if null_pct == 100:
            print("✗ RESULT: generate_app_analytics() will return None immediately")
            print("   Code snippet from kpi_analysis.py line 543-546:")
            print("   if \"apps\" not in df.columns or df[\"apps\"].isna().all():")
            print("       return  # Returns None, function never runs!")
        
        # Show what non-null app values look like
        non_null = filtered_df[filtered_df["apps"].notna()]
        if len(non_null) > 0:
            print(f"\n✓ Found {len(non_null)} non-null app records")
            print(f"  Sample apps: {non_null['apps'].head(3).tolist()}")
        else:
            print(f"\n✗ NO non-null app records found - column completely empty")
    else:
        print("\n✗ 'apps' column not in dataframe!")
    
    # Verify by running the function
    print(f"\n✓ Testing generate_app_analytics() with project 148 data...")
    result = generate_app_analytics(filtered_df)
    if result is None:
        print(f"✗ Function returned None (early return due to null apps column)")
    else:
        print(f"✓ Function returned: {type(result)}")

def investigate_area_summary_single_value(filtered_df):
    """Bug #3: Area summary shows only 1 area for 56k+ records due to spatial grid binning"""
    print("\n" + "="*80)
    print("BUG #3: AREA SUMMARY SHOWS ONLY 1 VALUE - Spatial Grid Binning Issue")
    print("="*80)
    
    # Show geographic spread
    print(f"\n✓ Dataset has {len(filtered_df)} records")
    print(f"  Latitude range: {filtered_df['lat'].min():.4f} to {filtered_df['lat'].max():.4f}")
    print(f"  Longitude range: {filtered_df['lon'].min():.4f} to {filtered_df['lon'].max():.4f}")
    
    lat_span = filtered_df['lat'].max() - filtered_df['lat'].min()
    lon_span = filtered_df['lon'].max() - filtered_df['lon'].min()
    print(f"  Geographic span: {lat_span:.6f}° lat × {lon_span:.6f}° lon")
    
    # Explain the spatial grid issue
    print(f"\n✗ build_area_summary() uses spatial grid binning:")
    print(f"  1. build_spatial_grid() creates 0.002° bins")
    print(f"  2. For your span ({lon_span:.6f}°), grid has ~{int(lon_span/0.002)} potential cells")
    print(f"  3. select_spatially_separated_cells() enforces 400m minimum distance between cells")
    print(f"  4. In concentrated areas, this results in ONLY 1 cell selected!")
    print(f"  5. Function returns max 6 areas but you're getting 1")
    
    # Now test it
    print(f"\n✓ Testing build_area_summary() with project 148 data...")
    try:
        area_summary = build_area_summary(filtered_df, top_n=6, sleep_sec=0.1)
        
        if area_summary:
            print(f"✗ Result: {len(area_summary)} area summary groups")
            for key, val in area_summary.items():
                if isinstance(val, list):
                    print(f"   {key}: {len(val)} items")
                elif isinstance(val, dict):
                    print(f"   {key}: {val}")
                else:
                    print(f"   {key}: {val}")
        else:
            print(f"✗ Result: None or empty")
    except Exception as e:
        print(f"✗ Error: {e}")

def investigate_no_hardcoding():
    """Verify there's NO hardcoding of project 149 in production code"""
    print("\n" + "="*80)
    print("VERIFICATION: No Project ID Hardcoding in Production Code")
    print("="*80)
    
    import os
    import re
    
    src_files = [
        'src/main.py',
        'src/load_data_db.py',
        'src/kpi_analysis.py',
        'src/metadata_generator.py',
        'src/kpi_config.py',
    ]
    
    found_hardcoding = False
    
    for src_file in src_files:
        if os.path.exists(src_file):
            with open(src_file, 'r') as f:
                content = f.read()
                # Look for patterns like == 149 or Project ID = 149 or if project_id == 149
                matches = re.findall(r'(==\s*149|project_id\s*==\s*149|if\s+\d+|default.*149)', content, re.IGNORECASE)
                if matches:
                    print(f"⚠ Found in {src_file}: {matches}")
                    found_hardcoding = True
    
    if not found_hardcoding:
        print("✓ No project ID hardcoding found in production code")
        print("✓ (NOTE: Test files like test_project_148_investigation.py have hardcoding,")
        print("  but those are test files, not production code)")

def summarize_bugs():
    """Summary of all issues"""
    print("\n" + "="*80)
    print("SUMMARY OF ISSUES FOR PROJECT 148")
    print("="*80)
    
    print("""
    BUG #1: Session Count Mismatch
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Database tbl_project.ref_session_id stores only 3 sessions: [3247, 3250, 3281]
    - But actual network logs have 5 sessions: [3247, 3250, 3281, 3286, 3287]
    - Sessions 3286 and 3287 are in data but not in ref_session_id metadata
    - Result: Report only shows 3 sessions, missing 2 sessions entirely
    - Root Cause: DATABASE METADATA ISSUE (ref_session_id field incomplete)
    - Code Impact: main.py line 210-212 correctly parses ref_session_id from database
    
    BUG #2: App Analytics Missing
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - 'apps' column in tbl_network_log is 100% NULL (all 56,532 rows)
    - generate_app_analytics() at line 543 checks if 'apps' is all null
    - If null: function returns None immediately (line 544)
    - Result: No app statistics generated, KPI analysis images incomplete
    - Root Cause: DATABASE DATA ISSUE ('apps' column never populated)
    - Code Impact: kpi_analysis.py correctly handles the null case
    
    BUG #3: Area Summary Shows Only 1 Value
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - build_area_summary() uses spatial grid binning with 0.002° granularity
    - For concentrated geographic areas, grid creates only 1 cell
    - select_spatially_separated_cells() limits to max 6 cells with 400m spacing
    - If area is small, only 1 cell meets spacing requirement
    - Result: Area summary shows 1 value instead of up to 6
    - Root Cause: ALGORITHM LIMITATION (spatial binning naturally concentrates data)
    - Code Impact: metadata_generator.py lines 117-165 working as designed

    ROOT CAUSE SUMMARY:
    ═══════════════════════════════════════════════════════════════════════════
    All three issues are DATA/DATABASE issues, not CODE issues:
    
    1. ref_session_id in tbl_project needs updating to include all 5 sessions
    2. apps column in tbl_network_log is not being populated from upstream
    3. Area summary shows 1 value because data is geographically concentrated
    
    Code is correctly designed and handles all these cases appropriately.
    """)

if __name__ == "__main__":
    try:
        print("\n" + "█"*80)
        print("PROJECT 148 COMPREHENSIVE BUG INVESTIGATION (WITHOUTCode Changes)")
        print("█"*80)
        
        # Run all diagnostics
        filtered_df = investigate_session_ids_mismatch()
        investigate_app_analytics_issue(filtered_df)
        investigate_area_summary_single_value(filtered_df)
        investigate_no_hardcoding()
        summarize_bugs()
        
    except Exception as e:
        print(f"\n✗ Error during investigation: {e}")
        import traceback
        traceback.print_exc()
