"""
INTEGRATED TEST CASE (FILTERED & VALIDATED)
------------------------------------------
Validates (on SAME filtered data as DB):
1. Drive summary (tbl_project → tbl_session)
2. Session table
3. RSRP threshold statistics (< -105)
4. RSRQ threshold statistics (< -14)
5. App analytics
6. Indoor vs Outdoor split

NO MAPS
NO FILE WRITES
ONLY PRINTS
"""

from datetime import timedelta
import pandas as pd

from src.db import get_connection
from src.load_data_db import load_project_data

PROJECT_ID = 149
USER_ID = 13

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def format_duration(seconds):
    if pd.isna(seconds) or seconds <= 0:
        return "NA"
    return str(timedelta(seconds=int(seconds)))

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("=" * 90)
    print("INTEGRATED DRIVE + KPI ANALYSIS TEST (FILTERED)")
    print("=" * 90)

    # --------------------------------------------------
    # 1. LOAD PROJECT DATA WITH FILTERING (FROM src/load_data_db.py)
    # --------------------------------------------------
    print("\n[1] LOADING PROJECT DATA WITH POLYGON FILTERING")
    
    try:
        raw_df, filtered_df, project_meta = load_project_data(PROJECT_ID)
    except Exception as e:
        print(f"❌ Error loading project data: {e}")
        return

    print(f"Project ID           : {PROJECT_ID}")
    print(f"Raw Network Samples  : {len(raw_df)}")
    print(f"Filtered Samples     : {len(filtered_df)}")

    if filtered_df.empty:
        print("❌ No data after polygon filtering")
        return

    # Convert numeric columns to proper types
    for col in ["rsrp", "rsrq", "sinr", "mos", "dl_tpt", "ul_tpt"]:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

    # --------------------------------------------------
    # 2. LOAD SESSION DATA FOR SUMMARY
    # --------------------------------------------------
    print("\n[2] LOADING SESSION DATA")

    cn = get_connection()
    
    ref_sessions_raw = project_meta["ref_session_id"]
    session_ids = [int(s) for s in str(ref_sessions_raw).split(",") if s.strip().isdigit()]

    session_df = pd.read_sql(
        f"""
        SELECT id AS session_id, start_time, end_time, distance
        FROM defaultdb.tbl_session
        WHERE id IN ({",".join(map(str, session_ids))})
        """,
        cn,
    )
    cn.close()

    session_df["start_time"] = pd.to_datetime(session_df["start_time"], errors="coerce")
    session_df["end_time"] = pd.to_datetime(session_df["end_time"], errors="coerce")

    session_df = session_df.dropna(subset=["start_time", "end_time"])
    session_df = session_df[session_df["distance"] > 0]

    session_df["duration_sec"] = (
        session_df["end_time"] - session_df["start_time"]
    ).dt.total_seconds()

    print(f"Total Sessions : {len(session_df)}")
    print(f"Total Distance : {session_df['distance'].sum():.2f} km")

    # --------------------------------------------------
    # 3. SESSION TABLE
    # --------------------------------------------------
    print("\n[3] SESSION TABLE")

    for _, r in session_df.iterrows():
        print(
            f"Session {r.session_id} | "
            f"{r.start_time.date()} | "
            f"{r.start_time.strftime('%H:%M')} → {r.end_time.strftime('%H:%M')} | "
            f"{format_duration(r.duration_sec)} | "
            f"{r.distance:.2f} km"
        )

    # --------------------------------------------------
    # 4. RSRP STATISTICS (< -105)
    # --------------------------------------------------
    print("\n[4] RSRP STATISTICS")

    rsrp_poor = filtered_df[filtered_df["rsrp"] < -105]

    print(f"RSRP < -105 Samples : {len(rsrp_poor)}")
    print(f"RSRP %              : {(len(rsrp_poor)/len(filtered_df))*100:.2f}%")

    # --------------------------------------------------
    # 5. RSRQ STATISTICS (< -14)
    # --------------------------------------------------
    print("\n[5] RSRQ STATISTICS")

    rsrq_poor = filtered_df[filtered_df["rsrq"] < -14]

    print(f"RSRQ < -14 Samples  : {len(rsrq_poor)}")
    print(f"RSRQ %              : {(len(rsrq_poor)/len(filtered_df))*100:.2f}%")

    # --------------------------------------------------
    # 6. APPLICATION ANALYTICS
    # --------------------------------------------------
    print("\n[6] APPLICATION ANALYTICS")

    if "app_name" in filtered_df.columns:
        for app, df in filtered_df.groupby("app_name"):
            print(
                f"{app:15} | "
                f"Samples: {len(df):5d} | "
                f"RSRP: {df['rsrp'].mean():6.1f} | "
                f"RSRQ: {df['rsrq'].mean():6.1f} | "
                f"DL: {df['dl_tpt'].mean():6.2f} | "
                f"UL: {df['ul_tpt'].mean():6.2f}"
            )

    # --------------------------------------------------
    # 7. INDOOR vs OUTDOOR
    # --------------------------------------------------
    print("\n[7] INDOOR vs OUTDOOR")

    if "indoor_outdoor" in filtered_df.columns:
        for env in ["Indoor", "Outdoor"]:
            df_env = filtered_df[filtered_df["indoor_outdoor"] == env]
            if df_env.empty:
                continue

            print(
                f"{env:8} | "
                f"RSRP: {df_env['rsrp'].mean():6.1f} | "
                f"RSRQ: {df_env['rsrq'].mean():6.1f} | "
                f"SINR: {df_env['sinr'].mean():5.2f} | "
                f"MOS: {df_env['mos'].mean():4.2f} | "
                f"DL: {df_env['dl_tpt'].mean():6.2f} | "
                f"UL: {df_env['ul_tpt'].mean():6.2f}"
            )

    print("\n" + "=" * 90)
    print("TEST COMPLETED SUCCESSFULLY (FILTERED & VALIDATED)")
    print("=" * 90)

if __name__ == "__main__":
    main()
