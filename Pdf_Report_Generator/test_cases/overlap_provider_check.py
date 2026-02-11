"""Detect overlapping sessions and compare providers during overlap windows.

Writes CSV to `data/processed/overlap_provider_report.csv` and prints a concise summary.

Run inside the project's venv:
    $env:PYTHONPATH='.'; python -m test_cases.overlap_provider_check
"""
from pathlib import Path
import pandas as pd
from src.load_data_db import load_project_data


OUT_CSV = Path("data") / "processed" / "overlap_provider_report.csv"


def find_overlaps(sessions):
    # sessions: list of tuples (session_id, start_ts, end_ts)
    overlaps = []
    sessions_sorted = sorted(sessions, key=lambda x: x[1])
    for i in range(len(sessions_sorted)):
        sid_i, start_i, end_i = sessions_sorted[i]
        for j in range(i + 1, len(sessions_sorted)):
            sid_j, start_j, end_j = sessions_sorted[j]
            # overlap if start_j <= end_i
            if start_j <= end_i:
                overlap_start = max(start_i, start_j)
                overlap_end = min(end_i, end_j)
                if overlap_end >= overlap_start:
                    overlaps.append((sid_i, sid_j, overlap_start, overlap_end))
            else:
                # since list is sorted by start, no further overlaps for sid_i
                break
    return overlaps


def analyze_project(project_id=149):
    raw_df, filtered_df, project = load_project_data(project_id)

    if raw_df is None or raw_df.empty:
        print("No raw rows found for project", project_id)
        return

    # ensure timestamp is datetime
    raw_df = raw_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(raw_df["timestamp"]):
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")

    # compute session ranges from raw data (full coverage)
    sessions = []
    for sid, g in raw_df.groupby("session_id"):
        st = g["timestamp"].min()
        et = g["timestamp"].max()
        sessions.append((int(sid), st, et))

    overlaps = find_overlaps(sessions)

    rows = []
    for sid_a, sid_b, ostart, oend in overlaps:
        # extract rows for each session in overlap window
        a_rows = raw_df[(raw_df.session_id == sid_a) & (raw_df.timestamp >= ostart) & (raw_df.timestamp <= oend)]
        b_rows = raw_df[(raw_df.session_id == sid_b) & (raw_df.timestamp >= ostart) & (raw_df.timestamp <= oend)]

        a_count = len(a_rows)
        b_count = len(b_rows)

        a_top = a_rows["m_alpha_long"].fillna("<na>").astype(str).value_counts()
        b_top = b_rows["m_alpha_long"].fillna("<na>").astype(str).value_counts()

        a_dom = a_top.index[0] if not a_top.empty else None
        b_dom = b_top.index[0] if not b_top.empty else None

        same_provider = (a_dom == b_dom) and (a_dom is not None)

        rows.append({
            "session_a": sid_a,
            "session_b": sid_b,
            "overlap_start": ostart,
            "overlap_end": oend,
            "a_count": a_count,
            "b_count": b_count,
            "a_dominant": a_dom,
            "b_dominant": b_dom,
            "same_provider": same_provider,
        })

    df_report = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(OUT_CSV, index=False)

    # print summary
    print(f"Project {project_id}: found {len(sessions)} sessions, {len(overlaps)} overlapping pairs")
    if df_report.empty:
        print("No overlaps found.")
        return

    same = df_report[df_report.same_provider == True]
    diff = df_report[df_report.same_provider == False]
    print(f"Overlaps with same dominant provider: {len(same)}")
    print(f"Overlaps with different providers: {len(diff)}")
    print("Saved report:", OUT_CSV)

    # print detailed lines for quick inspection
    pd.set_option("display.width", 200)
    print(df_report.to_string(index=False))


if __name__ == "__main__":
    analyze_project()
