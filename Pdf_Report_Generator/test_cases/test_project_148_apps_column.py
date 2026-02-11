"""Check the 'apps' column in Project 148 - where the real app data might be"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_connection, get_project_by_id
import pandas as pd


print("\n" + "="*80)
print("CHECK: Project 148 'apps' column (where real data might be)")
print("="*80)

project_id = 148
cn = get_connection()

try:
    project = get_project_by_id(project_id, cn)
    session_ids = [int(s.strip()) for s in project['ref_session_id'].split(',') if s.strip().isdigit()]
    
    # Check the apps column
    cur = cn.cursor(dictionary=True)
    
    print(f"\n[APPS COLUMN - Raw Data]")
    
    for sid in session_ids[:2]:  # Check first 2 sessions
        query = f"""
        SELECT DISTINCT apps FROM tbl_network_log 
        WHERE session_id = %s AND apps IS NOT NULL AND apps != ''
        LIMIT 15
        """
        cur.execute(query, (sid,))
        rows = cur.fetchall()
        
        print(f"\n  Session {sid} - Unique app values:")
        for row in rows:
            print(f"    '{row['apps']}'")
    
    # Count non-null apps
    query = "SELECT SUM(CASE WHEN apps IS NOT NULL AND apps != '' THEN 1 ELSE 0 END) as app_count FROM tbl_network_log WHERE session_id IN (%s)"
    placeholders = ",".join(["%s"] * len(session_ids))
    query = f"SELECT SUM(CASE WHEN apps IS NOT NULL AND apps != '' THEN 1 ELSE 0 END) as app_count FROM tbl_network_log WHERE session_id IN ({placeholders})"
    
    cur.execute(query, tuple(session_ids))
    result = cur.fetchone()
    
    print(f"\n[APPS COLUMN - NULL ANALYSIS]")
    print(f"  Total records: 56597")
    print(f"  Records with apps data: {result['app_count']}")
    print(f"  Records without apps: {56597 - (result['app_count'] or 0)}")
    
    cur.close()
    
finally:
    cn.close()

print("\n" + "="*80)
