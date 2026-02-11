"""
Project 158 & 148 Diagnosis Test
================================
Identifies root causes for data discrepancies in projects 158 and 148:
- Project 158: 0 filtered data (polygon no-match issue?)
- Project 148: 63 less filtered values, session count mismatch (data loss issue?)

Tests validate:
1. Raw data loading
2. Session ID parsing & count
3. Polygon loading & filtering logic
4. Data consistency at each pipeline stage
5. KPI calculations with correct data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_data_db import load_project_data, _parse_session_ids, _parse_polygons
from src.db import get_connection, get_project_by_id, get_network_logs_for_sessions, get_project_regions
from src.kpi_config import KPI_CONFIG
from src.threshold_resolver import resolve_kpi_ranges
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
import pandas as pd


def print_separator(char="=", length=100):
    """Print formatted separator"""
    print(char * length)


def print_section(title, char="="):
    """Print formatted section header"""
    print("\n")
    print_separator(char)
    print(f" {title}")
    print_separator(char)


def test_project_158():
    """
    Test Project 158: Debug 0 filtered data issue
    """
    print_section("PROJECT 158 DIAGNOSIS: 0 FILTERED DATA ISSUE", "=")
    project_id = 158
    
    cn = get_connection()
    try:
        # Step 1: Get project metadata
        print_section("STEP 1: Load Project Metadata", "-")
        project = get_project_by_id(project_id, cn)
        
        if not project:
            print(f"❌ PROJECT NOT FOUND: {project_id}")
            return
        
        print(f"✓ Project ID: {project['id']}")
        print(f"  Name: {project.get('name', 'N/A')}")
        print(f"  Ref Session ID: {project['ref_session_id']}")
        
        # Step 2: Parse session IDs
        print_section("STEP 2: Parse Session IDs", "-")
        ref_session_id = project["ref_session_id"]
        session_ids = _parse_session_ids(ref_session_id)
        
        print(f"Session IDs found: {session_ids}")
        print(f"Total sessions: {len(session_ids)}")
        
        if not session_ids:
            print("❌ NO VALID SESSION IDS - This is the problem!")
            return
        
        # Step 3: Load raw data
        print_section("STEP 3: Load Raw Network Data", "-")
        raw_df = get_network_logs_for_sessions(session_ids, cn)
        
        print(f"✓ Raw Data Loaded")
        print(f"  Total rows: {len(raw_df)}")
        print(f"  Columns: {list(raw_df.columns)}")
        
        # Check data integrity
        print(f"\n  Data Integrity Check:")
        print(f"    - Non-null lat: {raw_df['lat'].notna().sum()}")
        print(f"    - Non-null lon: {raw_df['lon'].notna().sum()}")
        print(f"    - Valid coordinates: {((raw_df['lat'].notna()) & (raw_df['lon'].notna())).sum()}")
        
        # Show sample data
        if not raw_df.empty:
            print(f"\n  Sample data (first 3 rows):")
            print(raw_df[['session_id', 'lat', 'lon', 'band']].head(3).to_string())
        
        # Step 4: Get project regions
        print_section("STEP 4: Load Polygon Boundaries", "-")
        region_rows = get_project_regions(project_id, cn)
        
        if not region_rows:
            print(f"⚠️  NO POLYGONS DEFINED FOR PROJECT {project_id}")
            print("   This is likely the root cause of 0 filtered data!")
            print("   Action: Define polygon boundaries for this project")
            return
        
        print(f"✓ Polygons Found: {len(region_rows)}")
        
        for idx, region in enumerate(region_rows):
            print(f"\n  Polygon {idx+1}:")
            print(f"    Name: {region.get('name', 'N/A')}")
            print(f"    WKT: {region['region_wkt'][:100]}...")
            
            # Parse polygon
            try:
                polygon = load_wkt(region['region_wkt'])
                print(f"    Type: {polygon.geom_type}")
                print(f"    Bounds: {polygon.bounds}")
                print(f"    Is Valid: {polygon.is_valid}")
                
                # Count points inside
                points_inside = 0
                for _, row in raw_df.iterrows():
                    if pd.notna(row['lat']) and pd.notna(row['lon']):
                        point = Point(row['lon'], row['lat'])
                        if polygon.contains(point):
                            points_inside += 1
                
                print(f"    Data points inside: {points_inside}/{len(raw_df)}")
                
            except Exception as e:
                print(f"    ❌ Error parsing polygon: {e}")
        
        # Step 5: Manual polygon filtering
        print_section("STEP 5: Manual Polygon Filtering", "-")
        
        polygons = []
        for region in region_rows:
            try:
                polygon = load_wkt(region['region_wkt'])
                if polygon.is_valid:
                    polygons.append(polygon)
            except:
                pass
        
        if not polygons:
            print("❌ NO VALID POLYGONS - Cannot filter data!")
            return
        
        # Filter manually
        mask = []
        for _, row in raw_df.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                mask.append(False)
            else:
                point = Point(row['lon'], row['lat'])
                inside = any(poly.contains(point) for poly in polygons)
                mask.append(inside)
        
        filtered_df = raw_df.loc[mask].reset_index(drop=True)
        
        print(f"Filtered Data Result:")
        print(f"  Raw rows: {len(raw_df)}")
        print(f"  Points inside polygon: {len(filtered_df)}")
        print(f"  Filtering efficiency: {len(filtered_df)/len(raw_df)*100:.2f}%")
        
        if len(filtered_df) == 0:
            print(f"\n❌ DIAGNOSIS: POLYGON DOES NOT CONTAIN ANY DATA POINTS")
            print(f"   Polygon bounds: {[poly.bounds for poly in polygons]}")
            print(f"   Data bounds (lat): {[raw_df['lat'].min(), raw_df['lat'].max()]}")
            print(f"   Data bounds (lon): {[raw_df['lon'].min(), raw_df['lon'].max()]}")
            print(f"\n   SOLUTION: Update polygon boundary to include data points")
        else:
            print(f"\n✓ Polygon filtering working correctly")
            print(f"  Filtered sample (first 3 rows):")
            print(filtered_df[['session_id', 'lat', 'lon', 'band']].head(3).to_string())
        
        # Step 6: Use load_project_data function
        print_section("STEP 6: Validate with load_project_data()", "-")
        raw_df2, filtered_df2, project_meta = load_project_data(project_id)
        
        print(f"Using load_project_data():")
        print(f"  Raw: {len(raw_df2)} rows")
        print(f"  Filtered: {len(filtered_df2)} rows")
        print(f"  Polygon WKT provided: {'Yes' if project_meta.get('region') else 'No'}")
        
    finally:
        cn.close()


def test_project_148():
    """
    Test Project 148: Debug session count and data mismatch issue
    """
    print_section("PROJECT 148 DIAGNOSIS: SESSION & DATA MISMATCH ISSUE", "=")
    project_id = 148
    
    cn = get_connection()
    try:
        # Step 1: Get project metadata
        print_section("STEP 1: Load Project Metadata", "-")
        project = get_project_by_id(project_id, cn)
        
        if not project:
            print(f"❌ PROJECT NOT FOUND: {project_id}")
            return
        
        print(f"✓ Project ID: {project['id']}")
        print(f"  Name: {project.get('name', 'N/A')}")
        print(f"  Ref Session ID: {project['ref_session_id']}")
        
        # Step 2: Parse session IDs
        print_section("STEP 2: Parse & Count Session IDs", "-")
        ref_session_id = project["ref_session_id"]
        session_ids = _parse_session_ids(ref_session_id)
        
        print(f"Session IDs: {session_ids}")
        print(f"Total sessions: {len(session_ids)}")
        print(f"Expected in report: {len(session_ids)}")
        
        if not session_ids:
            print("❌ NO VALID SESSION IDS")
            return
        
        # Step 3: Count records per session
        print_section("STEP 3: Record Count Per Session", "-")
        
        cur = cn.cursor(dictionary=True)
        for sid in session_ids:
            query = "SELECT COUNT(*) as cnt FROM tbl_network_log WHERE session_id = %s"
            cur.execute(query, (sid,))
            row = cur.fetchone()
            count = row['cnt'] if row else 0
            print(f"  Session {sid}: {count} records")
        cur.close()
        
        # Step 4: Load raw data
        print_section("STEP 4: Load Raw Network Data", "-")
        raw_df = get_network_logs_for_sessions(session_ids, cn)
        
        print(f"✓ Raw Data Loaded")
        print(f"  Total rows: {len(raw_df)}")
        print(f"  Unique sessions: {raw_df['session_id'].nunique()}")
        print(f"  Sessions in raw data: {sorted(raw_df['session_id'].unique().tolist())}")
        
        # Detailed session breakdown
        print(f"\n  Records per session in raw_df:")
        session_counts = raw_df['session_id'].value_counts().sort_index()
        for sid, count in session_counts.items():
            print(f"    Session {sid}: {count} records")
        
        print(f"\n  Data Integrity Check:")
        print(f"    - Total rows: {len(raw_df)}")
        print(f"    - Non-null lat: {raw_df['lat'].notna().sum()}")
        print(f"    - Non-null lon: {raw_df['lon'].notna().sum()}")
        print(f"    - Valid coords: {((raw_df['lat'].notna()) & (raw_df['lon'].notna())).sum()}")
        print(f"    - Null/Invalid: {len(raw_df) - ((raw_df['lat'].notna()) & (raw_df['lon'].notna())).sum()}")
        
        # Step 5: Get polygon info
        print_section("STEP 5: Load Polygon Boundaries", "-")
        region_rows = get_project_regions(project_id, cn)
        
        if not region_rows:
            print(f"⚠️  NO POLYGONS DEFINED FOR PROJECT {project_id}")
        else:
            print(f"✓ Polygons Found: {len(region_rows)}")
            for idx, region in enumerate(region_rows):
                print(f"\n  Polygon {idx+1}: {region.get('name', 'N/A')}")
                print(f"    WKT: {region['region_wkt'][:80]}...")
        
        # Step 6: Manual polygon filtering
        print_section("STEP 6: Apply Polygon Filtering", "-")
        
        if region_rows:
            polygons = []
            for region in region_rows:
                try:
                    polygon = load_wkt(region['region_wkt'])
                    if polygon.is_valid:
                        polygons.append(polygon)
                except:
                    pass
            
            if polygons:
                mask = []
                for _, row in raw_df.iterrows():
                    if pd.isna(row['lat']) or pd.isna(row['lon']):
                        mask.append(False)
                    else:
                        point = Point(row['lon'], row['lat'])
                        inside = any(poly.contains(point) for poly in polygons)
                        mask.append(inside)
                
                filtered_df = raw_df.loc[mask].reset_index(drop=True)
                
                print(f"✓ Polygon Filtering Applied")
                print(f"  Raw rows: {len(raw_df)}")
                print(f"  Filtered rows: {len(filtered_df)}")
                print(f"  Rows removed: {len(raw_df) - len(filtered_df)}")
                print(f"  Filtering %: {len(filtered_df)/len(raw_df)*100:.2f}%")
                
                # Check session counts after filtering
                print(f"\n  Records per session AFTER filtering:")
                if not filtered_df.empty:
                    session_counts_filtered = filtered_df['session_id'].value_counts().sort_index()
                    for sid, count in session_counts_filtered.items():
                        original_count = (raw_df['session_id'] == sid).sum()
                        lost = original_count - count
                        print(f"    Session {sid}: {count} (lost {lost} from {original_count})")
                
            else:
                print("❌ No valid polygons to filter with")
        else:
            filtered_df = raw_df.copy()
            print("No polygons - using all raw data as filtered")
        
        # Step 7: Use load_project_data
        print_section("STEP 7: Validate with load_project_data()", "-")
        raw_df2, filtered_df2, project_meta = load_project_data(project_id)
        
        print(f"Using load_project_data():")
        print(f"  Raw: {len(raw_df2)} rows")
        print(f"  Filtered: {len(filtered_df2)} rows")
        
        # Compare with manual load
        print(f"\nComparison with Manual Load:")
        print(f"  Raw matches: {len(raw_df2) == len(raw_df)}")
        print(f"  Filtered matches: {len(filtered_df2) == len(filtered_df)}")
        
        if len(raw_df2) != len(raw_df):
            print(f"  ⚠️  Raw data mismatch: {len(raw_df2)} vs {len(raw_df)}")
        
        if len(filtered_df2) != len(filtered_df):
            print(f"  ⚠️  Filtered data mismatch: {len(filtered_df2)} vs {len(filtered_df)}")
        
        # Step 8: Check KPI data integrity
        print_section("STEP 8: KPI Data Integrity Check", "-")
        
        for kpi, cfg in list(KPI_CONFIG.items())[:3]:  # Check first 3 KPIs
            col = cfg['column']
            
            if col not in filtered_df2.columns:
                print(f"❌ {kpi} column '{col}' not found")
                continue
            
            valid_count = filtered_df2[col].notna().sum()
            total_count = len(filtered_df2)
            
            print(f"\n  {kpi} ({col}):")
            print(f"    Valid values: {valid_count}/{total_count}")
            print(f"    Missing: {total_count - valid_count}")
            
            if valid_count > 0:
                values = pd.to_numeric(filtered_df2[col], errors='coerce').dropna()
                print(f"    Min: {values.min():.2f}, Max: {values.max():.2f}, Mean: {values.mean():.2f}")
        
        # Step 9: Check if data is being modified during pipeline
        print_section("STEP 9: Data Consistency Check", "-")
        
        print("Checking if filtered_df is being modified or re-filtered...")
        print(f"  Initial filtered_df2 length: {len(filtered_df2)}")
        
        # Simulate KPI map generation (from main.py)
        from src.kpi_config import KPI_CONFIG
        
        kpi_name = "RSRP"
        cfg = KPI_CONFIG[kpi_name]
        col = cfg['column']
        
        # This is how main.py filters for each KPI map
        df_kpi = filtered_df2[
            filtered_df2[col].notna() & 
            filtered_df2["lat"].notna() & 
            filtered_df2["lon"].notna()
        ]
        
        print(f"\n  After KPI-specific filtering (RSRP):")
        print(f"    df_kpi rows: {len(df_kpi)}")
        print(f"    Rows removed for missing data: {len(filtered_df2) - len(df_kpi)}")
        
    finally:
        cn.close()


def compare_projects():
    """Compare data characteristics between 158 and 148"""
    print_section("COMPARISON: PROJECT 158 vs 148", "=")
    
    projects = [158, 148]
    comparison_data = {}
    
    for pid in projects:
        try:
            raw_df, filtered_df, project_meta = load_project_data(pid)
            comparison_data[pid] = {
                'raw': len(raw_df),
                'filtered': len(filtered_df),
                'has_polygon': project_meta.get('region') is not None,
                'efficiency': len(filtered_df) / len(raw_df) * 100 if len(raw_df) > 0 else 0
            }
        except Exception as e:
            print(f"Error loading project {pid}: {e}")
            comparison_data[pid] = {'error': str(e)}
    
    print(f"\n{'Metric':<25} {'Project 158':<20} {'Project 148':<20}")
    print("-" * 65)
    
    for metric in ['raw', 'filtered', 'has_polygon', 'efficiency']:
        val_158 = comparison_data[158].get(metric, 'N/A')
        val_148 = comparison_data[148].get(metric, 'N/A')
        
        if isinstance(val_158, bool):
            val_158 = "Yes" if val_158 else "No"
        elif isinstance(val_158, float):
            val_158 = f"{val_158:.2f}%"
        
        if isinstance(val_148, bool):
            val_148 = "Yes" if val_148 else "No"
        elif isinstance(val_148, float):
            val_148 = f"{val_148:.2f}%"
        
        print(f"{metric:<25} {str(val_158):<20} {str(val_148):<20}")


if __name__ == "__main__":
    
    # Run diagnostics for both projects
    test_project_158()
    print("\n\n")
    test_project_148()
    print("\n\n")
    compare_projects()
    
    print("\n")
    print_section("DIAGNOSIS COMPLETE", "=")
    print("\nNext steps:")
    print("1. If Project 158 shows 0 filtered: Check if polygon is defined and contains data points")
    print("2. If Project 148 shows data loss: Check if polygon filtering is too aggressive")
    print("3. Review session counts in both raw and filtered data")
    print("4. Verify KPI columns exist and have valid data")
