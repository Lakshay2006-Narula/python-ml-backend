# test_cases/test_mos.py

import pandas as pd
from src.load_data import load_excel_data
from src.polygon_filter import apply_polygon_filter
from src.kpi_config import mos_colour_manual


def test_mos_distribution():
    df = load_excel_data("data/analytics-data-2026-01-13T13-06-46.xlsx")

    print("Total rows:", len(df))

    print("\nMOS basic stats:")
    print(df["MOS"].describe())

    print("\nMOS unique values (top 10):")
    print(df["MOS"].value_counts().head(10))

    print("\nMOS min/max:")
    print("Min:", df["MOS"].min())
    print("Max:", df["MOS"].max())


def test_mos_color_mapping():
    test_values = [1, 2.5, 3, 3.5, 4, 4.5]

    print("\nMOS → Color mapping:")
    for v in test_values:
        print(v, "→", mos_colour_manual(v))


if __name__ == "__main__":
    test_mos_distribution()
    test_mos_color_mapping()
