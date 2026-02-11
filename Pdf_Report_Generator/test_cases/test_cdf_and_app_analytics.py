"""
TEST: CDF Column in KPI Range Tables and Enhanced App Analytics
----------------------------------------------------------------
Tests:
1. CDF column in KPI range tables
2. Enhanced app analytics with all columns
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.kpi_analysis import generate_kpi_range_tables, generate_app_analytics

print("=" * 70)
print("TESTING CDF COLUMN AND APP ANALYTICS")
print("=" * 70)

# Create test data
np.random.seed(42)
test_data = {
    'rsrp': np.random.uniform(-110, -70, 1000),
    'rsrq': np.random.uniform(-15, -5, 1000),
    'sinr': np.random.uniform(0, 15, 1000),
    'dl_tpt': np.random.uniform(5, 50, 1000),
    'ul_tpt': np.random.uniform(1, 15, 1000),
    'mos': np.random.choice([3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2], 1000),
    'latency': np.random.uniform(20, 100, 1000),
    'jitter': np.random.uniform(5, 25, 1000),
    'packet_loss': np.random.uniform(0, 5, 1000),
    'apps': np.random.choice(['Facebook', 'WhatsApp', 'YT', 'Chrome'], 1000),
    'category': np.random.choice(['Social', 'Messaging', 'Other', 'Browser'], 1000),
    'session_id': np.random.choice([1, 2, 3, 4], 1000),
    'timestamp': pd.date_range('2026-01-22 10:00:00', periods=1000, freq='1s'),
    'lat': np.random.uniform(28.6, 28.7, 1000),
    'lon': np.random.uniform(77.2, 77.3, 1000),
    'band': np.random.choice(['n78', 'n3', 'b40'], 1000),
    'pci': np.random.choice([1, 2, 3, 4, 5], 1000)
}

df = pd.DataFrame(test_data)

print("\n[1] Testing KPI Range Tables with CDF column...")
print("Generating range tables for all KPIs...")
generate_kpi_range_tables(df, user_id=13)
print("✓ KPI range tables generated with CDF column")

print("\n[2] Testing Enhanced App Analytics...")
print("Generating app analytics with all columns...")
generate_app_analytics(df)
print("✓ App analytics generated with Category, Sessions, Duration, SINR, MOS, Latency, Jitter, Loss%")

print("\n" + "=" * 70)
print("TEST COMPLETED - Check data/images/kpi_analysis/")
print("Look for:")
print("  - *_range_table.png (should have CDF column)")
print("  - app_analytics_part1.png (with Category, Sessions, Duration)")
print("  - app_analytics_part2.png (with UL, MOS, Latency, Jitter, Loss%)")
print("=" * 70)
