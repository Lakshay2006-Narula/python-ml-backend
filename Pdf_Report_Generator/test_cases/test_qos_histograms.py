"""
TEST: QoS Histograms with Fixed Domain-Specific Ranges
-------------------------------------------------------
Tests the new histogram generation with:
- Fixed ranges (Speed, Latency, Jitter, Packet Loss)
- Embedded legends with quality labels
- Sample count on bars
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.kpi_analysis import generate_qos_metrics

# Generate test data
np.random.seed(42)

test_data = {
    'speed': np.random.choice([3, 15, 30, 60, 90, 120], size=1000),
    'latency': np.random.choice([10, 30, 70, 120, 180], size=1000),
    'jitter': np.random.choice([3, 7, 15, 25, 40], size=1000),
    'packet_loss': np.random.choice([0.5, 2, 4, 7, 12], size=1000)
}

df = pd.DataFrame(test_data)

print("=" * 70)
print("TESTING QoS HISTOGRAM GENERATION")
print("=" * 70)

# Test Speed
print("\n[1] Generating Speed histogram...")
generate_qos_metrics(df, 'speed', 'Speed', 'speed')
print("✓ Speed histogram generated")

# Test Latency
print("\n[2] Generating Latency histogram...")
generate_qos_metrics(df, 'latency', 'Latency', 'latency')
print("✓ Latency histogram generated")

# Test Jitter
print("\n[3] Generating Jitter histogram...")
generate_qos_metrics(df, 'jitter', 'Jitter', 'jitter')
print("✓ Jitter histogram generated")

# Test Packet Loss
print("\n[4] Generating Packet Loss histogram...")
generate_qos_metrics(df, 'packet_loss', 'Packet Loss', 'packet_loss')
print("✓ Packet Loss histogram generated")

print("\n" + "=" * 70)
print("TEST COMPLETED - Check data/images/kpi_analysis/ for outputs")
print("=" * 70)
