"""
Investigate Project 148 issues:
- Session count: 5 in DB vs 3 in report
- App names: "swiggy whatsapp", "zomato whatsapp" instead of "swiggy", "zomato"
- KPI analysis images wrong
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_connection, get_project_by_id, get_network_logs_for_sessions
from src.load_data_db import load_project_data
import pandas as pd


print("\n" + "="*80)
print("INVESTIGATE: Project 148 Session & App Name Issues")
print("="*80)

project_id = 148
cn = get_connection()

try:
    # Get project
    project = get_project_by_id(project_id, cn)
    session_ids = [int(s.strip()) for s in project['ref_session_id'].split(',') if s.strip().isdigit()]
    
    print(f"\n[DATABASE SESSIONS]")
    print(f"  Ref session ID: {project['ref_session_id']}")
    print(f"  Parsed: {session_ids}")
    print(f"  Total: {len(session_ids)} sessions")
    
    # Check each session for app data
    cur = cn.cursor(dictionary=True)
    
    print(f"\n[APP DATA PER SESSION]")
    for sid in session_ids:
        query = "SELECT COUNT(*) as cnt, COUNT(DISTINCT app_name) as app_count FROM tbl_network_log WHERE session_id = %s"
        cur.execute(query, (sid,))
        row = cur.fetchone()
        print(f"  Session {sid}: {row['cnt']} records, {row['app_count']} unique apps")
        
        # Get sample app names
        query = "SELECT DISTINCT app_name FROM tbl_network_log WHERE session_id = %s LIMIT 10"
        cur.execute(query, (sid,))
        apps = cur.fetchall()
        app_names = [app['app_name'] for app in apps if app['app_name']]
        print(f"    Apps: {app_names}")
    
    cur.close()
    
    # Load raw data
    print(f"\n[RAW DATA COLUMNS]")
    raw_df, filtered_df, meta = load_project_data(project_id)
    
    print(f"  Raw records: {len(raw_df)}")
    print(f"  Filtered records: {len(filtered_df)}")
    print(f"  Columns: {list(raw_df.columns)}")
    
    # Check app_name column
    if 'app_name' in raw_df.columns:
        print(f"\n[APP NAME ANALYSIS]")
        print(f"  Total unique app names: {raw_df['app_name'].nunique()}")
        print(f"  Null app names: {raw_df['app_name'].isna().sum()}")
        
        # Show samples
        print(f"\n  Sample app names (first 20):")
        app_samples = raw_df['app_name'].dropna().unique()[:20]
        for app in app_samples:
            count = (raw_df['app_name'] == app).sum()
            print(f"    '{app}': {count} records")
        
        # Check if there are combined names (with space/colon)
        print(f"\n  Checking for combined app names (e.g., 'swiggy whatsapp'):")
        for app in raw_df['app_name'].dropna().unique():
            if ' ' in str(app) or ':' in str(app):
                count = (raw_df['app_name'] == app).sum()
                print(f"    FOUND: '{app}': {count} records")
    
    # Check sessions in filtered data
    print(f"\n[SESSION ANALYSIS - FILTERED DATA]")
    if not filtered_df.empty:
        unique_sessions = filtered_df['session_id'].unique()
        print(f"  Sessions with data: {len(unique_sessions)}")
        print(f"  Session IDs: {sorted(unique_sessions)}")
        
        for sid in sorted(unique_sessions):
            count = (filtered_df['session_id'] == sid).sum()
            app_count = filtered_df[filtered_df['session_id'] == sid]['app_name'].nunique()
            print(f"    Session {sid}: {count} records, {app_count} unique apps")
    
    # Check for null values
    print(f"\n[NULL VALUE ANALYSIS - FILTERED DATA]")
    print(f"  Total rows: {len(filtered_df)}")
    for col in ['session_id', 'lat', 'lon', 'app_name', 'band']:
        null_count = filtered_df[col].isna().sum()
        if col in filtered_df.columns:
            print(f"  {col}: {null_count} nulls ({null_count/len(filtered_df)*100:.2f}%)")
    
    # Check what happens during KPI analysis
    print(f"\n[KPI COLUMN ANALYSIS]")
    from src.kpi_config import KPI_CONFIG
    
    for kpi, cfg in list(KPI_CONFIG.items())[:5]:
        col = cfg['column']
        if col in filtered_df.columns:
            valid = filtered_df[col].notna().sum()
            null = filtered_df[col].isna().sum()
            print(f"  {kpi} ({col}): {valid} valid, {null} null")
    
    # Check if app_name column is being used in any aggregation
    print(f"\n[IS APP_NAME IN KPI_CONFIG?]")
    app_in_kpi = False
    for kpi, cfg in KPI_CONFIG.items():
        if 'app' in str(cfg).lower():
            print(f"  {kpi}: {cfg}")
            app_in_kpi = True
    
    if not app_in_kpi:
        print(f"  App names NOT in KPI_CONFIG - might be separate analysis")
    
finally:
    cn.close()

print("\n" + "="*80)
