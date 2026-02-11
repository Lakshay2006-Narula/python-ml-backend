"""
TEST CASE: POOR REGION CLUSTER MAPS (TOP-5 DISTINCT REGIONS)
------------------------------------------------------------
Creates maps with:
- Actual poor points
- Top-5 dense, NON-OVERLAPPING regions
- Uses ONLY filtered_df (already polygon-filtered)

Outputs:
- data/images/test_case_images/rsrp_poor_regions.html
- data/images/test_case_images/rsrq_poor_regions.html
"""

import os
import math
import shutil
import numpy as np
import pandas as pd
import folium

from src.load_data_db import load_project_data

PROJECT_ID = 149
OUTPUT_DIR = "data/images/test_case_images"

# -----------------------------
# TUNABLE PARAMETERS
# -----------------------------
GRID_SIZE = 0.0012          # ~120m grid
TOP_REGIONS = 5
MIN_DISTANCE_METERS = 400   # minimum distance between regions
POINT_RADIUS = 2
REGION_OPACITY = 0.25


# -----------------------------
# HELPERS
# -----------------------------
def clean_output_dir():
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
    else:
        os.makedirs(OUTPUT_DIR)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def select_non_overlapping_regions(candidates):
    selected = []

    for cand in candidates:
        too_close = False
        for sel in selected:
            d = haversine_m(
                cand["lat"], cand["lon"],
                sel["lat"], sel["lon"]
            )
            if d < MIN_DISTANCE_METERS:
                too_close = True
                break

        if not too_close:
            selected.append(cand)

        if len(selected) == TOP_REGIONS:
            break

    return selected


# -----------------------------
# CORE MAP LOGIC
# -----------------------------
def generate_region_map(df, value_col, threshold, output_file, title):
    poor = df[df[value_col] < threshold].dropna(subset=["lat", "lon"])

    if poor.empty:
        print(f"❌ No poor samples for {value_col}")
        return

    # Grid binning
    poor["lat_bin"] = (poor["lat"] / GRID_SIZE).round().astype(int)
    poor["lon_bin"] = (poor["lon"] / GRID_SIZE).round().astype(int)

    grid_counts = (
        poor.groupby(["lat_bin", "lon_bin"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Build candidate regions
    candidates = []
    for _, r in grid_counts.iterrows():
        cell = poor[
            (poor.lat_bin == r.lat_bin) &
            (poor.lon_bin == r.lon_bin)
        ]

        candidates.append({
            "lat": cell.lat.mean(),
            "lon": cell.lon.mean(),
            "count": r["count"],
            "points": cell
        })

    # Select TOP-5 NON-OVERLAPPING regions
    regions = select_non_overlapping_regions(candidates)

    # Map
    fmap = folium.Map(
        location=[poor.lat.mean(), poor.lon.mean()],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    for idx, region in enumerate(regions, start=1):
        pts = region["points"]
        center_lat = region["lat"]
        center_lon = region["lon"]

        # Radius = max spread (capped)
        distances = pts.apply(
            lambda r: haversine_m(
                center_lat, center_lon, r.lat, r.lon
            ),
            axis=1
        )
        radius = min(distances.max(), 350)

        # Region circle
        folium.Circle(
            location=[center_lat, center_lon],
            radius=radius,
            color="red",
            fill=True,
            fill_opacity=REGION_OPACITY,
            popup=f"{title} | Region {idx} | Samples: {len(pts)}"
        ).add_to(fmap)

        # Points
        for _, p in pts.iterrows():
            folium.CircleMarker(
                location=[p.lat, p.lon],
                radius=POINT_RADIUS,
                color="red",
                fill=True,
                fill_opacity=0.6
            ).add_to(fmap)

    fmap.save(output_file)
    print(f"✅ Saved: {output_file}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 90)
    print("POOR REGION CLUSTER MAP TEST (NON-OVERLAPPING)")
    print("=" * 90)

    clean_output_dir()
    print("🧹 Cleared old test images")

    _, filtered_df, _ = load_project_data(PROJECT_ID)

    for col in ["rsrp", "rsrq"]:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

    print(f"Filtered Samples: {len(filtered_df)}")

    # -----------------------------
    # RSRP
    # -----------------------------
    print("\n[1] RSRP POOR REGIONS (< -105)")
    generate_region_map(
        filtered_df,
        value_col="rsrp",
        threshold=-105,
        output_file=os.path.join(OUTPUT_DIR, "rsrp_poor_regions.html"),
        title="RSRP < -105"
    )

    # -----------------------------
    # RSRQ
    # -----------------------------
    print("\n[2] RSRQ POOR REGIONS (< -14)")
    generate_region_map(
        filtered_df,
        value_col="rsrq",
        threshold=-14,
        output_file=os.path.join(OUTPUT_DIR, "rsrq_poor_regions.html"),
        title="RSRQ < -14"
    )

    print("\n" + "=" * 90)
    print("POOR REGION MAP TEST COMPLETED SUCCESSFULLY")
    print("=" * 90)


if __name__ == "__main__":
    main()
