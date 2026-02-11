"""
VERIFICATION: Check if app analytics data is from database (dynamic) or hardcoded
--------------------------------------------------------------------------------
This script will show:
1. What columns are in the database table
2. What unique app names exist in the actual database
3. Whether the data is coming from DB or hardcoded
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_connection
import pandas as pd

print("=" * 80)
print("DATABASE DATA VERIFICATION - APP ANALYTICS")
print("=" * 80)

# Get connection
cn = get_connection()

# Check what columns exist in tbl_network_log
print("\n[1] Checking table structure...")
query_columns = """
SELECT COLUMN_NAME 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = 'defaultdb' 
AND TABLE_NAME = 'tbl_network_log'
AND COLUMN_NAME LIKE '%app%'
"""
columns_df = pd.read_sql(query_columns, cn)
print(f"\nColumns with 'app' in name:")
print(columns_df)

# Get actual data for project 149
print("\n[2] Checking actual data for project 149...")
session_query = """
SELECT ref_session_id 
FROM tbl_project 
WHERE id = 149
"""
project_df = pd.read_sql(session_query, cn)
ref_sessions = project_df.iloc[0]['ref_session_id']
session_ids = [int(s.strip()) for s in str(ref_sessions).split(',') if s.strip().isdigit()]

print(f"Session IDs: {session_ids}")

# Check what app-related data exists
data_query = f"""
SELECT session_id,
       CASE 
           WHEN EXISTS(SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
                      WHERE TABLE_NAME = 'tbl_network_log' 
                      AND COLUMN_NAME = 'app_name') 
           THEN (SELECT app_name FROM tbl_network_log WHERE session_id IN ({','.join(map(str, session_ids))}) LIMIT 1)
           ELSE NULL
       END as has_app_name_col,
       CASE 
           WHEN EXISTS(SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
                      WHERE TABLE_NAME = 'tbl_network_log' 
                      AND COLUMN_NAME = 'apps') 
           THEN (SELECT apps FROM tbl_network_log WHERE session_id IN ({','.join(map(str, session_ids))}) LIMIT 1)
           ELSE NULL
       END as has_apps_col
FROM tbl_network_log
WHERE session_id IN ({','.join(map(str, session_ids))})
LIMIT 1
"""

# Simpler query - just get sample data
simple_query = f"""
SELECT *
FROM tbl_network_log
WHERE session_id IN ({','.join(map(str, session_ids))})
LIMIT 5
"""

sample_df = pd.read_sql(simple_query, cn)
print(f"\n[3] Sample data columns (first 5 rows):")
print(f"Total columns: {len(sample_df.columns)}")
print(f"\nColumns containing 'app' or 'category':")
app_cols = [col for col in sample_df.columns if 'app' in col.lower() or 'category' in col.lower()]
if app_cols:
    print(app_cols)
    for col in app_cols:
        unique_vals = sample_df[col].dropna().unique()
        print(f"\n  {col}: {len(unique_vals)} unique values")
        print(f"  Sample values: {list(unique_vals[:5])}")
else:
    print("  No app or category columns found!")

# Check unique app values across all data
if app_cols:
    for col in app_cols:
        full_query = f"""
        SELECT DISTINCT {col}
        FROM tbl_network_log
        WHERE session_id IN ({','.join(map(str, session_ids))})
        AND {col} IS NOT NULL
        LIMIT 20
        """
        unique_df = pd.read_sql(full_query, cn)
        print(f"\n[4] All unique values in '{col}' column:")
        print(unique_df[col].tolist())

cn.close()

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If you see app names above, they are DYNAMIC from DATABASE")
print("If no app columns exist, app analytics won't generate any output")
print("=" * 80)
