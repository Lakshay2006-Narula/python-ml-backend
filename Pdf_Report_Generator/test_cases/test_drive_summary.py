"""
TEST CASE: Drive Summary Generation
====================================
Generates drive summary statistics and session table images.

Output:
- Image 1: drive_summary.png (Distance, Samples, Sessions, Days)
- Image 2: session_table.png (Session details table)
- Location: data/images/kpi_analysis/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from src.db import get_connection
from src.load_data_db import load_project_data


# Configuration
PROJECT_ID = 149
OUTPUT_DIR = "data/images/kpi_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_table_image(df, title, filename):
    """Save dataframe as image table"""
    fig, ax = plt.subplots(figsize=(12, max(2, len(df) * 0.4)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.title(title, pad=10, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")


def format_duration(seconds):
    """Format duration from seconds to readable format"""
    if pd.isna(seconds) or seconds <= 0:
        return "NA"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if hours > 0:
        if minutes > 0:
            return f"{hours} hours {minutes} min"
        return f"{hours} hours"
    return f"{minutes} min"


def get_session_data(session_ids: list):
    """Fetch session data from database"""
    cn = get_connection()
    
    session_ids_str = ",".join(map(str, session_ids))
    
    query = f"""
    SELECT id, start_time, end_time, distance
    FROM defaultdb.tbl_session
    WHERE id IN ({session_ids_str})
    ORDER BY start_time
    """
    
    df = pd.read_sql(query, cn)
    cn.close()
    
    return df


def generate_drive_summary_image(session_df, total_samples):
    """Generate Image 1: Drive Summary Statistics"""
    
    # Calculate statistics
    total_distance = session_df["distance"].sum()
    total_sessions = len(session_df)
    
    # Calculate number of unique days
    session_df["date"] = pd.to_datetime(session_df["start_time"]).dt.date
    unique_days = session_df["date"].nunique()
    
    # Create summary dataframe
    summary_data = {
        "Metric": [
            "Distance Covered",
            "Total Samples",
            "Total Sessions",
            "Number of Days"
        ],
        "Value": [
            f"{total_distance:.2f} KM",
            f"{total_samples:,}",
            f"{total_sessions}",
            f"{unique_days} Days"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as image
    save_table_image(summary_df, "Drive Summary", "drive_summary.png")
    
    return {
        "distance_covered": round(total_distance, 2),
        "total_samples": total_samples,
        "total_sessions": total_sessions,
        "number_of_days": unique_days
    }


def generate_session_table_image(session_df):
    """Generate Image 2: Session Details Table"""
    
    # Convert to datetime
    session_df["start_time"] = pd.to_datetime(session_df["start_time"], errors='coerce')
    session_df["end_time"] = pd.to_datetime(session_df["end_time"], errors='coerce')
    
    # Filter out rows with invalid timestamps
    session_df = session_df.dropna(subset=["start_time", "end_time"])
    
    # Calculate duration in seconds
    session_df["duration_sec"] = (
        session_df["end_time"] - session_df["start_time"]
    ).dt.total_seconds()
    
    # Format for display
    display_df = pd.DataFrame({
        "Session": session_df["id"],
        "Date": session_df["start_time"].dt.strftime("%d-%m-%Y"),
        "Start": session_df["start_time"].dt.strftime("%H:%M"),
        "End": session_df["end_time"].dt.strftime("%H:%M"),
        "Duration": session_df["duration_sec"].apply(format_duration),
        "Distance": session_df["distance"].apply(lambda x: f"{x:.6f}")
    })
    
    # Save as image
    save_table_image(display_df, "Session Details", "session_table.png")
    
    # Return metadata-friendly format
    session_list = []
    for _, row in session_df.iterrows():
        session_list.append({
            "session_id": int(row["id"]),
            "date": row["start_time"].strftime("%d-%m-%Y"),
            "start_time": row["start_time"].strftime("%H:%M"),
            "end_time": row["end_time"].strftime("%H:%M"),
            "duration": format_duration(row["duration_sec"]),
            "distance": round(row["distance"], 6)
        })
    
    return session_list


def main():
    """Main test function"""
    print("=" * 80)
    print("DRIVE SUMMARY GENERATION - TEST CASE")
    print("=" * 80)
    
    # Step 1: Load project data to get session IDs and sample count
    print(f"\n[1] Loading project data for project {PROJECT_ID}")
    raw_df, filtered_df, project_meta = load_project_data(PROJECT_ID)
    
    # Get session IDs from project metadata
    ref_session_id = project_meta.get("ref_session_id", "")
    session_ids = [
        int(s.strip())
        for s in str(ref_session_id).split(",")
        if s.strip().isdigit()
    ]
    
    print(f"   ✅ Found {len(session_ids)} sessions: {session_ids}")
    print(f"   ✅ Total samples (filtered): {len(filtered_df)}")
    
    # Step 2: Fetch session data from database
    print(f"\n[2] Fetching session data from database")
    session_df = get_session_data(session_ids)
    
    if session_df.empty:
        print("   ❌ No session data found")
        return
    
    print(f"   ✅ Fetched {len(session_df)} session records")
    
    # Step 3: Generate Image 1 - Drive Summary
    print(f"\n[3] Generating Image 1: Drive Summary")
    drive_summary_metadata = generate_drive_summary_image(
        session_df, 
        total_samples=len(filtered_df)
    )
    
    # Step 4: Generate Image 2 - Session Table
    print(f"\n[4] Generating Image 2: Session Table")
    session_list_metadata = generate_session_table_image(session_df)
    
    # Step 5: Display metadata structure
    print(f"\n[5] Metadata Structure for metadata.json")
    print("=" * 80)
    print("Drive Summary Metadata:")
    for key, value in drive_summary_metadata.items():
        print(f"   {key}: {value}")
    
    print("\nSession List Metadata (first 2 sessions):")
    for session in session_list_metadata[:2]:
        print(f"   {session}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print(f"📁 Images saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("   - drive_summary.png")
    print("   - session_table.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
