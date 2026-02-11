"""
SESSION FILTERING TRACE - Find where 5 sessions become 3
"""

import sys
sys.path.insert(0, '.')

from src.load_data_db import load_project_data, get_connection, get_project_by_id
from src.kpi_analysis import generate_drive_summary_images, get_session_data_for_drive_summary
import pandas as pd

print("""
════════════════════════════════════════════════════════════════════════════════
TRACE: Where does session count 5 → 3 happen for Project 148?
════════════════════════════════════════════════════════════════════════════════
""")

# Step 1: Load project metadata
print("\n[STEP 1] Database tbl_project.ref_session_id")
print("─" * 80)
cn = get_connection()
project = get_project_by_id(148, cn)
ref_session_str = project.get("ref_session_id", "")
print(f"  Database ref_session_id: '{ref_session_str}'")

session_ids = [int(s.strip()) for s in str(ref_session_str).split(",") if s.strip().isdigit()]
print(f"  Parsed sessions: {session_ids}")
print(f"  Count: {len(session_ids)}")

# Step 2: Load filtered data
print("\n[STEP 2] Load project data (load_project_data)")
print("─" * 80)
raw_df, filtered_df, project_meta = load_project_data(148)

unique_sessions_in_filtered = filtered_df['session_id'].nunique()
sessions_in_filtered = sorted(filtered_df['session_id'].unique().tolist())
print(f"  Sessions in filtered_df: {unique_sessions_in_filtered}")
print(f"  Which sessions: {sessions_in_filtered}")

# Step 3: Call get_session_data_for_drive_summary with 5 sessions
print("\n[STEP 3] Call get_session_data_for_drive_summary with 5 sessions")
print("─" * 80)
session_df = get_session_data_for_drive_summary(session_ids)
print(f"  Input session_ids: {session_ids}")
if session_df is not None:
    print(f"  Returned sessions: {session_df['id'].unique().tolist()}")
    print(f"  Count: {len(session_df)}")
else:
    print(f"  Returned: None")

# Step 4: Check tbl_session for all 5 sessions
print("\n[STEP 4] Check tbl_session directly for all 5 sessions")
print("─" * 80)
session_ids_str = ",".join(map(str, session_ids))
query = f"""
SELECT id, start_time, end_time, distance
FROM defaultdb.tbl_session
WHERE id IN ({session_ids_str})
ORDER BY start_time
"""
direct_session_df = pd.read_sql(query, cn)
print(f"  Query: SELECT ... WHERE id IN ({session_ids_str})")
print(f"  Results:")
for _, row in direct_session_df.iterrows():
    print(f"    Session {int(row['id'])}: {row['start_time']} to {row['end_time']}")
print(f"  Total sessions returned: {len(direct_session_df)}")

# Step 5: Check if some sessions are missing from tbl_session
print("\n[STEP 5] Check for missing sessions in tbl_session")
print("─" * 80)
found_sessions = set(direct_session_df['id'].astype(int).tolist())
expected_sessions = set(session_ids)
missing = expected_sessions - found_sessions
if missing:
    print(f"  ✗ Sessions in ref_session_id but NOT in tbl_session: {missing}")
    print(f"    These sessions won't show in drive summary!")
else:
    print(f"  ✓ All {len(session_ids)} sessions found in tbl_session")

# Step 6: Now trace through generate_drive_summary_images
print("\n[STEP 6] Call generate_drive_summary_images with 5 sessions")
print("─" * 80)
print(f"  Input session_ids: {session_ids}")
drive_summary = generate_drive_summary_images(session_ids, len(filtered_df))
if drive_summary and 'sessions' in drive_summary:
    print(f"  Output sessions: {[s['session_id'] for s in drive_summary['sessions']]}")
    print(f"  Count: {drive_summary['total_sessions']}")
else:
    print(f"  Output: None or no sessions field")

cn.close()

print("""
════════════════════════════════════════════════════════════════════════════════
ANALYSIS COMPLETE
════════════════════════════════════════════════════════════════════════════════
""")
