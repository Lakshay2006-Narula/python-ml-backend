import os
import json
import pandas as pd

from src.metadata_generator import (
    build_spatial_grid,
    select_spatially_separated_cells,
    reverse_geocode_area,
    build_area_summary,
)

# Quick debug test to isolate why area_summary is empty.
# Set MOCK_GEOCODE=1 in env to use a deterministic fallback instead of Nominatim.
USE_MOCK = os.getenv("MOCK_GEOCODE", "0") == "1"


def mock_reverse_geocode_area(lat, lon, sleep_sec=0):
    # Simple deterministic labels for debugging purposes
    if 28.62 < lat < 28.66 and 77.20 < lon < 77.25:
        return {
            "labels": ["Downtown"],
            "class": "place",
            "type": "neighbourhood",
            "address": {"city": "TestCity"}
        }
    return {
        "labels": [f"Area_{round(lat,3)}_{round(lon,3)}"],
        "class": "place",
        "type": "locality",
        "address": {}
    }


def main():
    csv_path = os.path.join("data", "processed", "filtered_data.csv")
    df = pd.read_csv(csv_path)

    print("Loaded filtered_data.csv rows:", len(df))

    grid = build_spatial_grid(df)
    print("Spatial grid cells:", len(grid))
    print(grid.sort_values("sample_count", ascending=False).head(10).to_dict(orient="records"))

    selected = select_spatially_separated_cells(grid.sort_values("sample_count", ascending=False))
    print("Selected cells for reverse-geocoding:")
    for i, cell in enumerate(selected):
        print(i, dict(cell)["center_lat"], dict(cell)["center_lon"], "avg_speed=", dict(cell)["avg_speed"])

    if USE_MOCK:
        print("Using MOCK reverse geocoder")
        # monkeypatch the real function
        import src.metadata_generator as mg
        mg.reverse_geocode_area = mock_reverse_geocode_area

    # Run reverse geocode on selected cells and show outputs
    for i, cell in enumerate(selected):
        lat = cell.center_lat
        lon = cell.center_lon
        res = reverse_geocode_area(lat, lon, sleep_sec=0)
        print(f"reverse_geocode_area {i}:", res)

    # Finally, call build_area_summary and print the produced JSON (only area_summary)
    area = build_area_summary(df, sleep_sec=0)
    print("\n=== area_summary ===")
    print(json.dumps(area, indent=2))


if __name__ == "__main__":
    main()
