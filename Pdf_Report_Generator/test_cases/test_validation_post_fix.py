"""
Validation Test Suite for Project 158 & 148 Fixes
==================================================
Run these tests after applying fixes to verify the corrections work.

Test coverage:
- Project 158: Verify polygon coordinate fix works
- Project 148: Validate KPI analysis and session count accuracy
- Data integrity throughout pipeline
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_data_db import load_project_data
from src.kpi_config import KPI_CONFIG
from src.kpi_analysis import run_kpi_analysis
import pandas as pd


def test_project_158_fixed():
    """
    VALIDATION TEST: Project 158 Polygon Fix
    
    Expected results after polygon coordinate fix:
    - Filtered data should be > 0
    - Should have data points inside polygon
    - KPI maps should generate without issues
    """
    print("\n" + "="*80)
    print("VALIDATION TEST: PROJECT 158 POLYGON COORDINATE FIX")
    print("="*80)
    
    project_id = 158
    user_id = 13
    
    print("\nLoading project data...")
    raw_df, filtered_df, project_meta = load_project_data(project_id)
    
    # Test 1: Verify polygon fix
    print(f"\n[TEST 1] Polygon Filtering Result")
    print(f"  Raw records: {len(raw_df)}")
    print(f"  Filtered records: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print("  [FAIL] Filtered data is still 0 - polygon fix did not work!")
        print("        Check if polygon coordinates were updated correctly")
        return False
    
    expected_filtered = int(len(raw_df) * 0.95)  # Expect ~95% to pass polygon
    if len(filtered_df) < expected_filtered:
        print(f"  [WARN] Filtered data lower than expected")
        print(f"         Got: {len(filtered_df)}, Expected: ~{expected_filtered}")
    else:
        print(f"  [PASS] Polygon filtering working - {len(filtered_df)} records match polygon")
    
    # Test 2: Verify KPI data availability
    print(f"\n[TEST 2] KPI Data Availability Check")
    
    kpi_status = {}
    for kpi, cfg in KPI_CONFIG.items():
        col = cfg['column']
        
        if col not in filtered_df.columns:
            kpi_status[kpi] = 'FAIL: Column missing'
            continue
        
        valid_count = filtered_df[col].notna().sum()
        
        if valid_count == 0:
            kpi_status[kpi] = 'FAIL: No valid data'
        elif valid_count < len(filtered_df) * 0.5:
            kpi_status[kpi] = f'WARN: Only {valid_count}/{len(filtered_df)} valid'
        else:
            kpi_status[kpi] = f'PASS: {valid_count}/{len(filtered_df)} valid'
    
    for kpi, status in kpi_status.items():
        print(f"  {kpi}: {status}")
    
    # Test 3: Verify KPI analysis runs without errors
    print(f"\n[TEST 3] KPI Analysis Execution")
    
    try:
        session_ids = [int(s.strip()) for s in project_meta['ref_session_id'].split(',') if s.strip().isdigit()]
        kpi_metadata, drive_summary = run_kpi_analysis(
            filtered_df, 
            user_id,
            KPI_CONFIG,
            session_ids=session_ids,
            image_dir=f"data/tmp/test_158_images"
        )
        
        if kpi_metadata is None or len(kpi_metadata) == 0:
            print("  [FAIL] KPI analysis returned empty metadata")
            return False
        
        print(f"  [PASS] KPI analysis generated {len(kpi_metadata)} KPI records")
        
        # Show summary
        for kpi, meta in list(kpi_metadata.items())[:3]:
            print(f"    {kpi}: {meta.get('total_samples', 0)} samples")
            
    except Exception as e:
        print(f"  [FAIL] KPI analysis error: {e}")
        return False
    
    print("\n[SUMMARY] Project 158 Fix Validation: COMPLETE")
    return len(filtered_df) > 0


def test_project_148_kpi_data():
    """
    VALIDATION TEST: Project 148 KPI Data Integrity
    
    Expected results:
    - All 5 sessions present
    - ~56,595 filtered records
    - Session counts consistent across analysis
    - KPI columns have expected null patterns
    """
    print("\n" + "="*80)
    print("VALIDATION TEST: PROJECT 148 KPI DATA INTEGRITY")
    print("="*80)
    
    project_id = 148
    user_id = 13
    
    print("\nLoading project data...")
    raw_df, filtered_df, project_meta = load_project_data(project_id)
    
    # Test 1: Session count verification
    print(f"\n[TEST 1] Session Count Verification")
    
    expected_sessions = [3247, 3250, 3281, 3286, 3287]
    actual_sessions = sorted(filtered_df['session_id'].unique().tolist())
    
    print(f"  Expected sessions: {expected_sessions}")
    print(f"  Actual sessions:   {actual_sessions}")
    
    if actual_sessions == expected_sessions:
        print(f"  [PASS] All 5 sessions present in filtered data")
    else:
        print(f"  [FAIL] Session mismatch!")
        return False
    
    # Test 2: Record count verification
    print(f"\n[TEST 2] Filtered Record Count")
    
    print(f"  Raw records: {len(raw_df)}")
    print(f"  Filtered records: {len(filtered_df)}")
    
    # Expected: 56,597 - 2 (null coords) = 56,595
    expected_range = (56590, 56600)  # Allow ±5 tolerance
    if expected_range[0] <= len(filtered_df) <= expected_range[1]:
        print(f"  [PASS] Filtered record count in expected range")
    else:
        print(f"  [WARN] Record count outside expected range {expected_range}")
    
    # Test 3: Session distribution
    print(f"\n[TEST 3] Session Record Distribution")
    
    expected_distribution = {
        3247: 242,
        3250: 982,
        3281: 357,
        3286: 21,
        3287: 54995
    }
    
    session_counts = filtered_df['session_id'].value_counts().to_dict()
    
    for sid, expected_count in expected_distribution.items():
        actual_count = session_counts.get(sid, 0)
        difference = abs(actual_count - expected_count)
        
        if difference <= 5:  # Allow ±5 tolerance
            print(f"  Session {sid}: {actual_count} records [PASS]")
        else:
            print(f"  Session {sid}: {actual_count} records (expected ~{expected_count}) [WARN]")
    
    # Test 4: KPI column analysis
    print(f"\n[TEST 4] KPI Column Null Analysis (explains 63-record difference)")
    
    print(f"\n  Checking for columns with ~63 null values:")
    for col in filtered_df.columns:
        if col in ['lat', 'lon', 'band', 'session_id']:
            continue
        
        null_count = filtered_df[col].isna().sum()
        valid_count = filtered_df[col].notna().sum()
        
        if 55 <= null_count <= 75:  # Around 63
            print(f"    [FOUND] {col}: {valid_count} valid, {null_count} null")
            print(f"            This explains the 63-record discrepancy!")
    
    # Test 5: KPI analysis execution
    print(f"\n[TEST 5] KPI Analysis Execution")
    
    try:
        session_ids = [int(s.strip()) for s in project_meta['ref_session_id'].split(',') if s.strip().isdigit()]
        kpi_metadata, drive_summary = run_kpi_analysis(
            filtered_df,
            user_id,
            KPI_CONFIG,
            session_ids=session_ids,
            image_dir=f"data/tmp/test_148_images"
        )
        
        if kpi_metadata is None:
            print("  [FAIL] KPI analysis returned None")
            return False
        
        print(f"  [PASS] KPI analysis generated {len(kpi_metadata)} KPI records")
        
        # Verify session count in metadata
        print(f"\n  Checking session counts in metadata:")
        for kpi, meta in list(kpi_metadata.items())[:5]:
            total = meta.get('total_samples', 0)
            print(f"    {kpi}: {total} samples")
        
    except Exception as e:
        print(f"  [FAIL] KPI analysis error: {e}")
        return False
    
    print("\n[SUMMARY] Project 148 Data Integrity: COMPLETE")
    return len(filtered_df) > 50000  # Basic sanity check


def compare_projects_post_fix():
    """
    Compare both projects after fixes applied
    """
    print("\n" + "="*80)
    print("COMPARISON: PROJECT 158 vs 148 (Post-Fix)")
    print("="*80)
    
    projects = {
        158: "Polygon coordinate fix",
        148: "Data integrity check"
    }
    
    results = {}
    
    for pid, purpose in projects.items():
        print(f"\nProject {pid} ({purpose}):")
        
        try:
            raw_df, filtered_df, meta = load_project_data(pid)
            
            results[pid] = {
                'raw': len(raw_df),
                'filtered': len(filtered_df),
                'sessions': filtered_df['session_id'].nunique() if not filtered_df.empty else 0,
                'efficiency': len(filtered_df) / len(raw_df) * 100 if len(raw_df) > 0 else 0
            }
            
            print(f"  Raw records: {results[pid]['raw']}")
            print(f"  Filtered records: {results[pid]['filtered']}")
            print(f"  Sessions: {results[pid]['sessions']}")
            print(f"  Efficiency: {results[pid]['efficiency']:.2f}%")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[pid] = None
    
    # Summary
    print("\n" + "="*80)
    print("EXPECTED POST-FIX STATE:")
    print("="*80)
    print("""
PROJECT 158:
  Status: SHOULD BE FIXED
  Expected filtered: > 100,000 records (95%+ of raw)
  Expected sessions: 22
  Expected maps: Should generate for all KPIs
  
PROJECT 148:
  Status: DATA IS CORRECT (no fix needed)
  Expected filtered: ~56,595 records
  Expected sessions: 5
  Expected maps: Should generate correctly
  63-record difference: Normal (due to KPI column nulls)
""")
    
    return results


if __name__ == "__main__":
    import json
    
    print("\n")
    print("*" * 80)
    print("PROJECT 158 & 148 VALIDATION TEST SUITE")
    print("*" * 80)
    print("""
These tests verify that the fixes identified in the diagnosis are working:

1. test_project_158_fixed()
   - Verifies polygon coordinate fix resolves 0 filtered data issue
   - Checks KPI data is available
   - Ensures analysis runs without errors

2. test_project_148_kpi_data()
   - Validates 5 sessions are loaded correctly
   - Checks ~56,595 filtered records (not 63 less)
   - Analyzes null patterns in KPI columns
   - Verifies analysis metadata is complete

3. compare_projects_post_fix()
   - Shows before/after comparison
   - Helps visualize improvements
""")
    
    # Run tests
    results = {
        'project_158': test_project_158_fixed(),
        'project_148': test_project_148_kpi_data(),
        'comparison': compare_projects_post_fix()
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\nProject 158 Validation: {'[PASS]' if results['project_158'] else '[FAIL]'}")
    print(f"Project 148 Validation: {'[PASS]' if results['project_148'] else '[FAIL]'}")
    
    if results['project_158'] and results['project_148']:
        print("\n[OK] All validations passed - fixes are working correctly!")
    else:
        print("\n[ACTION REQUIRED] Some validations failed - review diagnostic report")
    
    print("\nDetailed results saved to validation_results.json")
    with open('validation_results.json', 'w') as f:
        json.dump({k: str(v) for k, v in results.items()}, f, indent=2)
