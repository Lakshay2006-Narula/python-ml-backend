"""
Test Case: Area Summary Generation (FINAL)
=========================================
This test validates the FIXED area summary logic using:
1. Spatial grid generation
2. Spatially separated cell selection
3. Multi-level reverse geocoding
4. Final build_area_summary output

This test MUST produce multiple areas if spatial diversity exists.
"""

import sys
import os
import time
import pandas as pd
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from itertools import chain
# ------------------------------------------------------------------
# Add src to path
# ------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metadata_generator import (
    build_spatial_grid,
    build_area_summary,
    reverse_geocode_area,
    select_spatially_separated_cells,
)


# ------------------------------------------------------------------
# Distance helper (for debug)
# ------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ------------------------------------------------------------------
# MAIN TEST
# ------------------------------------------------------------------
def test_area_summary_detailed():

    print("=" * 90)
    print("AREA SUMMARY GENERATION – FINAL VALIDATION TEST")
    print("=" * 90)
    print()

    # --------------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------------
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "processed",
        "filtered_data.csv",
    )

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} total samples\n")

    df = df.dropna(subset=["lat", "lon"])
    print(f"✓ Valid lat/lon samples: {len(df)}\n")

    print("Latitude range:", df["lat"].min(), "→", df["lat"].max())
    print("Longitude range:", df["lon"].min(), "→", df["lon"].max())
    print()

    # --------------------------------------------------------------
    # STEP 2: GRID GENERATION
    # --------------------------------------------------------------
    print("=" * 90)
    print("STEP 2: SPATIAL GRID GENERATION")
    print("=" * 90)

    grid_size = 0.002  # ~200m
    grid = build_spatial_grid(df, grid_size_deg=grid_size)

    print(f"Grid size       : {grid_size}° (~200m)")
    print(f"Grid cells made : {len(grid)}")
    print(f"Samples covered : {grid['sample_count'].sum()}")
    print()

    print("Grid statistics:")
    print(f"  Min samples : {grid['sample_count'].min()}")
    print(f"  Max samples : {grid['sample_count'].max()}")
    print(f"  Mean        : {grid['sample_count'].mean():.2f}")
    print(f"  Median      : {grid['sample_count'].median():.0f}")
    print()

    # --------------------------------------------------------------
    # STEP 3: SPATIALLY SEPARATED CELL SELECTION
    # --------------------------------------------------------------
    print("=" * 90)
    print("STEP 3: SPATIALLY SEPARATED TOP CELLS")
    print("=" * 90)

    top_n = 6
    grid_sorted = grid.sort_values("sample_count", ascending=False)

    selected_cells = select_spatially_separated_cells(
        grid_sorted,
        min_distance_m=400,
        max_cells=top_n,
    )

    print(f"Selected {len(selected_cells)} spatially separated cells:\n")

    for i, cell in enumerate(selected_cells, 1):
        print(f"Cell {i}:")
        print(f"  Center        : ({cell.center_lat:.6f}, {cell.center_lon:.6f})")
        print(f"  Sample count  : {cell.sample_count}")
        print(f"  Avg speed     : {cell.avg_speed:.2f}")
        print()

    # Distance sanity check
    print("Pairwise distances (meters):")
    for i in range(len(selected_cells)):
        for j in range(i + 1, len(selected_cells)):
            d = haversine(
                selected_cells[i].center_lat,
                selected_cells[i].center_lon,
                selected_cells[j].center_lat,
                selected_cells[j].center_lon,
            )
            print(f"  Cell {i+1} ↔ Cell {j+1}: {d:.1f} m")
    print()

    # --------------------------------------------------------------
    # STEP 4: REVERSE GEOCODING (DETAILED)
    # --------------------------------------------------------------
    print("=" * 90)
    print("STEP 4: REVERSE GEOCODING (MULTI-LEVEL)")
    print("=" * 90)
    print()

    geocoded = []

    for i, cell in enumerate(selected_cells, 1):
        print(f"Cell {i}: ({cell.center_lat:.6f}, {cell.center_lon:.6f})")

        name = reverse_geocode_area(
            cell.center_lat,
            cell.center_lon,
            sleep_sec=1.0,
        )

        geocoded.append(name)
        print(f"  → Area name: {name}\n")

    print("Geocoded names:")
    print(geocoded)
    print()

    print("Name frequency:")

    flat_names = list(chain.from_iterable(geocoded))

    for name, count in Counter(flat_names).items():
        print(f"  {name}: {count}")
    print()

    # --------------------------------------------------------------
    # STEP 5: FULL build_area_summary TEST
    # --------------------------------------------------------------
    print("=" * 90)
    print("STEP 5: FINAL build_area_summary OUTPUT")
    print("=" * 90)
    print()

    start = time.time()
    summary = build_area_summary(df, top_n=6, sleep_sec=1.0)
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.2f} seconds\n")

    print("Area Summary:")
    print("-" * 40)

    for key, values in summary.items():
        print(f"{key} ({len(values)}):")
        for v in values:
            print(f"  - {v}")
        print()

    # --------------------------------------------------------------
    # STEP 6: ASSERTIONS (REAL TEST)
    # --------------------------------------------------------------
    print("=" * 90)
    print("STEP 6: ASSERTIONS")
    print("=" * 90)

    assert summary is not None, "Area summary returned None"
    assert len(summary["covered_areas"]) >= 3, (
        "FAILED: Expected multiple covered areas, got only one"
    )

    print("✓ TEST PASSED: Multiple areas detected correctly")
    print("=" * 90)


# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_area_summary_detailed()
