"""
Test case to debug range boundary issue with UL and other KPIs
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

# Test the value_in_range function
def value_in_range(value, range_dict, is_last_range):
    """
    Check if value belongs to range using half-open intervals.
    - All ranges except last: min <= value < max
    - Last range: min <= value <= max
    """
    if is_last_range:
        return range_dict["min"] <= value <= range_dict["max"]
    else:
        return range_dict["min"] <= value < range_dict["max"]


def test_ul_ranges():
    """Test UL range boundaries with real-world scenario"""
    
    print("=" * 70)
    print("TEST: UL Range Boundary Issue")
    print("=" * 70)
    
    # Simulate UL ranges from DB + auto-generated
    ranges = [
        {"min": 0.0, "max": 2, "color": "#aaa", "range": "< 2", "source": "auto"},
        {"min": 2, "max": 5, "color": "#yellow", "range": "2 to 5", "source": "db"},
        {"min": 5, "max": 10, "color": "#blue", "range": "5 to 10", "source": "db"},
        {"min": 10, "max": 15, "color": "#green", "range": "10 to 15", "source": "db"},
        {"min": 15, "max": 100, "color": "#gray", "range": "> 15", "source": "auto"},
    ]
    
    # Simulate UL data
    np.random.seed(42)
    ul_values = pd.Series([
        0.5, 1.0, 1.5,           # < 2
        2.0, 3.0, 4.0, 4.9,      # 2 to 5
        5.0, 7.0, 9.0, 9.9,      # 5 to 10
        10.0, 12.0, 14.9, 15.0,  # 10 to 15 and boundary
        16.0, 20.0, 50.0         # > 15
    ])
    
    print(f"\nTotal values: {len(ul_values)}")
    print(f"Min: {ul_values.min()}, Max: {ul_values.max()}")
    
    print("\n" + "-" * 70)
    print("RANGE ASSIGNMENTS:")
    print("-" * 70)
    
    total_assigned = 0
    
    for idx, r in enumerate(ranges):
        is_last = (idx == len(ranges) - 1)
        
        # Count how many values fall in this range
        mask = ul_values.apply(lambda v: value_in_range(v, r, is_last))
        count = mask.sum()
        values_in_range = ul_values[mask].tolist()
        
        total_assigned += count
        
        print(f"\n[{idx}] {r['range']} ({r['source']})")
        print(f"    Bounds: [{r['min']}, {r['max']}{']' if is_last else ')'}")
        print(f"    Is Last: {is_last}")
        print(f"    Count: {count}")
        if count > 0 and count < 10:
            print(f"    Values: {values_in_range}")
    
    print("\n" + "-" * 70)
    print(f"SUMMARY:")
    print(f"  Total values: {len(ul_values)}")
    print(f"  Total assigned: {total_assigned}")
    print(f"  Difference: {len(ul_values) - total_assigned}")
    
    if len(ul_values) == total_assigned:
        print("  ✅ PASS: All values assigned to exactly one range")
    else:
        print("  ❌ FAIL: Values missing or double-counted")
    
    # Test boundary values specifically
    print("\n" + "-" * 70)
    print("BOUNDARY VALUE TESTS:")
    print("-" * 70)
    
    boundary_tests = [
        (2.0, "Boundary between auto and first DB range"),
        (5.0, "Boundary between first and second DB range"),
        (10.0, "Boundary between second and third DB range"),
        (15.0, "Boundary between last DB and auto range"),
    ]
    
    for test_val, description in boundary_tests:
        print(f"\nValue: {test_val} - {description}")
        assignments = []
        for idx, r in enumerate(ranges):
            is_last = (idx == len(ranges) - 1)
            if value_in_range(test_val, r, is_last):
                assignments.append(f"[{idx}] {r['range']}")
        
        if len(assignments) == 1:
            print(f"  ✅ Assigned to: {assignments[0]}")
        elif len(assignments) == 0:
            print(f"  ❌ NOT ASSIGNED TO ANY RANGE!")
        else:
            print(f"  ❌ ASSIGNED TO MULTIPLE RANGES: {assignments}")
    
    print("\n" + "=" * 70)


def test_real_scenario_from_image():
    """Test the exact scenario from the user's image"""
    
    print("\n\n" + "=" * 70)
    print("TEST: Real Scenario (> 0.0: 9349 issue)")
    print("=" * 70)
    
    # This appears to be what's happening based on the image
    ranges = [
        {"min": 0.0, "max": 2, "color": "#aaa", "range": "> 0.0", "source": "auto"},  # PROBLEM!
        {"min": 2, "max": 5, "color": "#yellow", "range": "2 to 5", "source": "db"},
        {"min": 5, "max": 10, "color": "#blue", "range": "5 to 10", "source": "db"},
        {"min": 10, "max": 15, "color": "#green", "range": "10 to 15", "source": "db"},
    ]
    
    # The problem: if there's a lower-bound auto range with label "> 0.0"
    # it's confusing and catches everything
    
    print("\nPROBLEM: Auto-range label says '> 0.0' but should be '< 2'")
    print("This makes it look like it contains everything!")
    
    # Simulate 9346 values
    np.random.seed(42)
    ul_values = pd.Series(np.random.uniform(0.1, 20, 9346))
    
    print(f"\nTotal values: {len(ul_values)}")
    
    for idx, r in enumerate(ranges):
        is_last = (idx == len(ranges) - 1)
        mask = ul_values.apply(lambda v: value_in_range(v, r, is_last))
        count = mask.sum()
        print(f"  {r['range']}: {count}")
    
    # Check if range labels match actual bounds
    print("\n" + "-" * 70)
    print("LABEL vs ACTUAL BOUNDS CHECK:")
    print("-" * 70)
    for idx, r in enumerate(ranges):
        is_last = (idx == len(ranges) - 1)
        bound_str = f"[{r['min']}, {r['max']}{']]' if is_last else ')'}"
        print(f"  Label: '{r['range']}' | Actual: {bound_str}")
        
        # Check if label makes sense
        if r['range'].startswith('>') and r['min'] == 0.0:
            print(f"    ⚠️  WARNING: Label '> 0.0' is misleading for range {bound_str}")


def test_threshold_resolver_output():
    """Test what threshold_resolver actually returns"""
    
    print("\n\n" + "=" * 70)
    print("TEST: Threshold Resolver Auto-Range Generation")
    print("=" * 70)
    
    # Simulate what happens in threshold_resolver when data goes below DB min
    db_ranges = [
        {"min": 2, "max": 5, "color": "#yellow", "range": "2 to 5"},
        {"min": 5, "max": 10, "color": "#blue", "range": "5 to 10"},
        {"min": 10, "max": 15, "color": "#green", "range": "10 to 15"},
    ]
    
    data_min = 0.1
    data_max = 20.0
    
    print(f"DB ranges: {len(db_ranges)}")
    print(f"DB min: {db_ranges[0]['min']}")
    print(f"Data min: {data_min}")
    
    final_ranges = []
    
    # Lower-bound auto range (this is the problem!)
    if data_min < db_ranges[0]["min"]:
        auto_range = {
            "min": data_min,
            "max": db_ranges[0]["min"],
            "color": "#999999",
            "label": "< Min (Auto)",
            "range": f"< {db_ranges[0]['min']}",  # This creates "< 2"
            "source": "auto",
        }
        final_ranges.append(auto_range)
        print(f"\n✅ Lower auto-range created:")
        print(f"   Label: '{auto_range['range']}'")
        print(f"   Actual: [{auto_range['min']}, {auto_range['max']})")
    
    # Add DB ranges
    final_ranges.extend(db_ranges)
    
    # Upper-bound auto range
    if data_max > db_ranges[-1]["max"]:
        auto_range = {
            "min": db_ranges[-1]["max"],
            "max": data_max,
            "color": "#777777",
            "label": "> Max (Auto)",
            "range": f"> {db_ranges[-1]['max']}",
            "source": "auto",
        }
        final_ranges.append(auto_range)
        print(f"\n✅ Upper auto-range created:")
        print(f"   Label: '{auto_range['range']}'")
        print(f"   Actual: [{auto_range['min']}, {auto_range['max']}]")
    
    print(f"\n" + "-" * 70)
    print("ISSUE FOUND:")
    print("-" * 70)
    print("The label format in threshold_resolver.py is CONFUSING!")
    print(f"  Lower auto: '< {db_ranges[0]['min']}' looks like it means 'less than X'")
    print(f"  Upper auto: '> {db_ranges[-1]['max']}' looks like it means 'greater than X'")
    print("")
    print("But they create ranges with actual min/max bounds!")
    print("This doesn't cause double-counting, but makes the legend CONFUSING.")
    
    # The actual issue might be in how ranges are displayed
    print("\n" + "-" * 70)
    print("POTENTIAL ROOT CAUSE:")
    print("-" * 70)
    print("Check if the label '> 0.0' is being generated incorrectly")
    print("It should be '< 2' but maybe the code is using data_min instead?")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "RANGE BOUNDARY DEBUG TEST" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")
    
    test_ul_ranges()
    test_real_scenario_from_image()
    test_threshold_resolver_output()
    
    print("\n\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("Run this test to see where the issue is before making changes!")
    print("=" * 70)
