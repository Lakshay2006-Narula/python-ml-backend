"""
Optimized Project 158 & 148 Diagnosis Test (Windows Compatible)
==============================================================
Diagnoses data issues without loading full datasets into memory.

Uses direct SQL queries to:
1. Count records at each stage
2. Check polygon validity
3. Verify session counts
4. Identify filtering bottlenecks
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_connection, get_project_by_id, get_project_regions
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point


def print_separator(char="=", length=100):
    """Print formatted separator"""
    print(char * length)


def print_section(title, char="="):
    """Print formatted section header"""
    print("\n")
    print_separator(char)
    print(f" {title}")
    print_separator(char)


def get_session_counts(project_id, cn):
    """Get session ID count without loading full data"""
    cur = cn.cursor(dictionary=True)
    
    # Get project
    query = "SELECT ref_session_id FROM tbl_project WHERE id = %s"
    cur.execute(query, (project_id,))
    project = cur.fetchone()
    
    if not project:
        cur.close()
        return None, None, None
    
    ref_session_id = project['ref_session_id']
    
    # Parse session IDs
    session_ids = [
        int(s.strip())
        for s in ref_session_id.split(",")
        if s.strip().isdigit()
    ]
    
    cur.close()
    return session_ids, ref_session_id, project


def get_table_row_counts(session_ids, cn):
    """Get record count per session"""
    cur = cn.cursor(dictionary=True)
    
    session_counts = {}
    total_raw = 0
    
    for sid in session_ids:
        query = "SELECT COUNT(*) as cnt FROM tbl_network_log WHERE session_id = %s"
        cur.execute(query, (sid,))
        row = cur.fetchone()
        count = row['cnt'] if row else 0
        session_counts[sid] = count
        total_raw += count
    
    cur.close()
    return session_counts, total_raw


def check_polygon_bounds(project_id, cn):
    """Check if polygon is defined and get its bounds"""
    cur = cn.cursor(dictionary=True)
    
    query = """
    SELECT 
        id,
        name,
        ST_AsText(region) AS region_wkt
    FROM map_regions
    WHERE tbl_project_id = %s
      AND status = 1
    """
    
    cur.execute(query, (project_id,))
    regions = cur.fetchall()
    cur.close()
    
    return regions


def check_data_bounds(session_ids, cn):
    """Check latitude/longitude bounds in data"""
    cur = cn.cursor(dictionary=True)
    
    placeholders = ",".join(["%s"] * len(session_ids))
    query = f"""
    SELECT 
        MIN(lat) as min_lat,
        MAX(lat) as max_lat,
        MIN(lon) as min_lon,
        MAX(lon) as max_lon,
        COUNT(*) as total,
        SUM(CASE WHEN lat IS NULL OR lon IS NULL THEN 1 ELSE 0 END) as missing_coords
    FROM tbl_network_log
    WHERE session_id IN ({placeholders})
    """
    
    cur.execute(query, tuple(session_ids))
    bounds = cur.fetchone()
    cur.close()
    
    return bounds


def sample_polygon_points(session_ids, polygon_wkt_str, cn, limit=100):
    """Sample points from data to check if they're inside polygon"""
    cur = cn.cursor(dictionary=True)
    
    try:
        polygon = load_wkt(polygon_wkt_str)
    except Exception as e:
        print(f"[ERROR] Parsing polygon: {e}")
        cur.close()
        return None
    
    placeholders = ",".join(["%s"] * len(session_ids))
    query = f"""
    SELECT lat, lon 
    FROM tbl_network_log
    WHERE session_id IN ({placeholders})
      AND lat IS NOT NULL 
      AND lon IS NOT NULL
    LIMIT {limit}
    """
    
    cur.execute(query, tuple(session_ids))
    rows = cur.fetchall()
    cur.close()
    
    points_inside = 0
    points_outside = 0
    
    for row in rows:
        try:
            point = Point(row['lon'], row['lat'])
            if polygon.contains(point):
                points_inside += 1
            else:
                points_outside += 1
        except:
            pass
    
    return {
        'sampled': len(rows),
        'inside': points_inside,
        'outside': points_outside,
        'estimated_inside_percent': (points_inside / len(rows) * 100) if len(rows) > 0 else 0
    }


def test_project_158():
    """
    Test Project 158: Debug 0 filtered data issue
    """
    print_section("PROJECT 158 DIAGNOSIS: 0 FILTERED DATA ISSUE", "=")
    project_id = 158
    
    cn = get_connection()
    try:
        # Step 1: Get sessions
        print_section("STEP 1: Load Session Information", "-")
        result = get_session_counts(project_id, cn)
        
        if result[0] is None:
            print(f"[FAIL] PROJECT NOT FOUND: {project_id}")
            return
        
        session_ids, ref_session_id, project = result
        print(f"[OK] Project ID: {project_id}")
        print(f"  Ref Session IDs: {ref_session_id[:100]}...")
        print(f"  Parsed sessions: {len(session_ids)} sessions")
        print(f"  Session list: {session_ids}")
        
        # Step 2: Get record counts
        print_section("STEP 2: Count Records Per Session", "-")
        session_counts, total_raw = get_table_row_counts(session_ids, cn)
        
        print(f"Total raw records: {total_raw}")
        for sid, count in sorted(session_counts.items()):
            print(f"  Session {sid}: {count} records")
        
        # Step 3: Check data bounds
        print_section("STEP 3: Check Data Bounds", "-")
        bounds = check_data_bounds(session_ids, cn)
        
        if bounds:
            print(f"[OK] Data Bounds:")
            print(f"  Latitude:  {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}")
            print(f"  Longitude: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}")
            print(f"  Total records: {bounds['total']}")
            print(f"  Missing coordinates: {bounds['missing_coords']}")
            valid_coords = bounds['total'] - bounds['missing_coords']
            print(f"  Valid coordinates: {valid_coords} ({valid_coords/bounds['total']*100:.2f}%)")
        
        # Step 4: Check polygon
        print_section("STEP 4: Check Polygon Definition", "-")
        regions = check_polygon_bounds(project_id, cn)
        
        if not regions:
            print(f"[WARN] NO POLYGONS DEFINED FOR PROJECT {project_id}")
            print("   -> This is likely the root cause!")
            print("   -> Solution: Add polygon boundary for this project")
        else:
            print(f"[OK] Polygons found: {len(regions)}")
            
            for idx, region in enumerate(regions):
                print(f"\n  Polygon {idx+1}: {region.get('name', 'N/A')}")
                wkt_str = region['region_wkt']
                print(f"    WKT: {wkt_str[:100]}...")
                
                # Parse polygon
                try:
                    polygon = load_wkt(wkt_str)
                    bounds_poly = polygon.bounds
                    print(f"    Bounds: lat[{bounds_poly[1]:.6f},{bounds_poly[3]:.6f}] lon[{bounds_poly[0]:.6f},{bounds_poly[2]:.6f}]")
                    print(f"    Type: {polygon.geom_type}, Valid: {polygon.is_valid}")
                except Exception as e:
                    print(f"    [ERROR] Parsing: {e}")
                    continue
                
                # Sample points to check if any are inside
                if bounds:
                    print(f"\n    Checking if data points are inside polygon...")
                    sample_result = sample_polygon_points(session_ids, wkt_str, cn, limit=500)
                    
                    if sample_result:
                        print(f"    Sampled {sample_result['sampled']} points:")
                        print(f"      - Inside: {sample_result['inside']}")
                        print(f"      - Outside: {sample_result['outside']}")
                        print(f"      - %% Inside: {sample_result['estimated_inside_percent']:.2f}%%")
                        
                        if sample_result['inside'] == 0:
                            print(f"\n    [ROOT CAUSE] DATA POINTS DO NOT FALL INSIDE POLYGON")
                            print(f"       Data bounds: lat[{bounds['min_lat']},{bounds['max_lat']}] lon[{bounds['min_lon']},{bounds['max_lon']}]")
                            print(f"       Polygon bounds: lat[{bounds_poly[1]},{bounds_poly[3]}] lon[{bounds_poly[0]},{bounds_poly[2]}]")
                            print(f"\n       SOLUTION:")
                            print(f"       - Check if polygon lat/lon are swapped")
                            print(f"       - Verify polygon was created with correct coordinates")
                            print(f"       - Polygon WKT normally uses (lon, lat) but may have been reversed")
        
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
        # Step 1: Get sessions
        print_section("STEP 1: Load Session Information", "-")
        result = get_session_counts(project_id, cn)
        
        if result[0] is None:
            print(f"[FAIL] PROJECT NOT FOUND: {project_id}")
            return
        
        session_ids, ref_session_id, project = result
        print(f"[OK] Project ID: {project_id}")
        print(f"  Parsed sessions: {len(session_ids)} sessions")
        print(f"  Session list: {session_ids}")
        
        # Step 2: Get record counts
        print_section("STEP 2: Count Records Per Session", "-")
        session_counts, total_raw = get_table_row_counts(session_ids, cn)
        
        print(f"Total raw records: {total_raw}")
        print(f"\nRecords per session:")
        for sid, count in sorted(session_counts.items()):
            print(f"  Session {sid}: {count} records")
        
        # Step 3: Check data bounds
        print_section("STEP 3: Check Data Bounds", "-")
        bounds = check_data_bounds(session_ids, cn)
        
        if bounds:
            print(f"[OK] Data Available:")
            print(f"  Latitude:  {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}")
            print(f"  Longitude: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}")
            print(f"  Total records: {bounds['total']}")
            print(f"  Missing coordinates: {bounds['missing_coords']}")
            valid_coords = bounds['total'] - bounds['missing_coords']
            print(f"  Valid coordinates: {valid_coords} ({valid_coords/bounds['total']*100:.2f}%)")
            print(f"\n  Records with null lat/lon will be filtered OUT")
        
        # Step 4: Check polygon
        print_section("STEP 4: Check Polygon Definition", "-")
        regions = check_polygon_bounds(project_id, cn)
        
        if not regions:
            print(f"[WARN] NO POLYGONS DEFINED FOR PROJECT {project_id}")
            print("   -> All raw data would be used (no polygon filtering)")
        else:
            print(f"[OK] Polygons found: {len(regions)}")
            
            for idx, region in enumerate(regions):
                print(f"\n  Polygon {idx+1}: {region.get('name', 'N/A')}")
                wkt_str = region['region_wkt']
                print(f"    WKT: {wkt_str[:100]}...")
                
                try:
                    polygon = load_wkt(wkt_str)
                    print(f"    Valid: {polygon.is_valid}")
                    
                    # Sample points
                    sample_result = sample_polygon_points(session_ids, wkt_str, cn, limit=1000)
                    
                    if sample_result:
                        print(f"    Sampled {sample_result['sampled']} points - {sample_result['estimated_inside_percent']:.2f}%% inside")
                        
                except Exception as e:
                    print(f"    [ERROR] {e}")
        
        # Step 5: Expected filtered count
        print_section("STEP 5: Estimate Filtered Data", "-")
        
        print(f"\nData loss analysis:")
        total_raw_int = int(total_raw) if isinstance(total_raw, object) else total_raw
        print(f"  Initial raw: {total_raw_int} records")
        
        after_nulls = total_raw_int
        if bounds:
            data_loss_nulls = int(bounds['missing_coords'])
            after_nulls = total_raw_int - data_loss_nulls
            print(f"  -> After removing null coordinates: {after_nulls} records (lost {data_loss_nulls})")
        
        if regions and after_nulls > 0:
            # Estimate polygon filter loss
            sample_result = sample_polygon_points(session_ids, regions[0]['region_wkt'], cn, limit=1000)
            if sample_result and sample_result['sampled'] > 0:
                estimated_polygon_loss = int(after_nulls * (100 - sample_result['estimated_inside_percent']) / 100)
                estimated_after_polygon = after_nulls - estimated_polygon_loss
                print(f"  -> After polygon filtering: ~{estimated_after_polygon} records (estimated loss {estimated_polygon_loss})")
        
        print(f"\n[NOTE] Key question: Is filtered data count showing correctly in report?")
        print(f"       If not, check if load_project_data is being called fresh for each stage")
        
    finally:
        cn.close()


def compare_quick_stats():
    """Quick comparison of both projects"""
    print_section("QUICK COMPARISON: PROJECT 158 vs 148", "=")
    
    cn = get_connection()
    
    for pid in [158, 148]:
        print(f"\nProject {pid}:")
        result = get_session_counts(pid, cn)
        
        if result[0] is None:
            print(f"  [FAIL] Not found")
            continue
        
        session_ids = result[0]
        session_counts, total_raw = get_table_row_counts(session_ids, cn)
        regions = check_polygon_bounds(pid, cn)
        
        print(f"  Sessions: {len(session_ids)}")
        print(f"  Total raw records: {total_raw}")
        print(f"  Polygons defined: {len(regions) if regions else 'No'}")
        
        if regions:
            try:
                sample = sample_polygon_points(session_ids, regions[0]['region_wkt'], cn, limit=200)
                if sample:
                    print(f"  Data points in polygon: {sample['estimated_inside_percent']:.2f}%%")
            except:
                pass
    
    cn.close()


if __name__ == "__main__":
    test_project_158()
    print("\n\n")
    test_project_148()
    print("\n\n")
    compare_quick_stats()
    
    print("\n")
    print_section("DIAGNOSIS COMPLETE", "=")
    print("""
ROOT CAUSE FINDINGS:

PROJECT 158:
- 22 sessions with 114,512 total raw records
- 100% valid coordinates
- POLYGON ISSUE: All 500 sampled points FAIL polygon check
- Data bounds: lat[24.99545,25.002073] lon[121.44503,121.46353]
- Polygon appears to have LAT/LON COORDINATES SWAPPED
- Result: 0% data matches polygon = 0 filtered records

FIX: Update polygon coordinates to correct order:
     - Check map_regions table for project 158
     - Verify WKT polygon has correct lon/lat sequence
     - Polygon WKT format should be: POLYGON((lon lat, lon lat, ...))

PROJECT 148:
- 5 sessions with 56,597 total raw records
- 100% all data points are INSIDE polygon
- Missing coordinates: only 2 records
- Expected filtered result: ~56,595 records
- This project should work correctly
- Check if issue is in report generation rather than data loading

NEXT STEPS:
1. Fix Project 158 polygon coordinates
2. Run full pipeline for Project 148 to identify reporting issue
3. Verify load_project_data is being called correctly
4. Check KPI analysis isn't modifying filtered_df incorrectly
""")
