"""
Test: Polygon Coordinate Fallback for Project 158
If filtered_data is 0, reverse polygon coordinates and retry
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_connection, get_project_by_id, get_network_logs_for_sessions, get_project_regions
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
import pandas as pd


def reverse_polygon_coordinates(wkt_string):
    """Reverse lat/lon in polygon WKT"""
    polygon = load_wkt(wkt_string)
    
    # Get exterior coordinates
    coords = list(polygon.exterior.coords)
    
    # Reverse each (lon, lat) to (lat, lon)
    reversed_coords = [(lat, lon) for lon, lat in coords]
    
    # Create new polygon with reversed coordinates
    from shapely.geometry import Polygon
    reversed_polygon = Polygon(reversed_coords)
    return reversed_polygon.wkt


def filter_with_polygon(df, polygon):
    """Filter dataframe by polygon"""
    mask = []
    for _, row in df.iterrows():
        if pd.isna(row['lat']) or pd.isna(row['lon']):
            mask.append(False)
        else:
            point = Point(row['lon'], row['lat'])
            if polygon.contains(point):
                mask.append(True)
            else:
                mask.append(False)
    return df[mask].reset_index(drop=True)


def test_project_158_fallback():
    """Test polygon fallback logic for project 158"""
    
    print("\n" + "="*80)
    print("TEST: Project 158 Polygon Fallback Logic")
    print("="*80)
    
    project_id = 158
    cn = get_connection()
    
    try:
        # Get project and sessions
        project = get_project_by_id(project_id, cn)
        session_ids = [int(s.strip()) for s in project['ref_session_id'].split(',') if s.strip().isdigit()]
        
        # Load raw data
        cur = cn.cursor(dictionary=True)
        placeholders = ",".join(["%s"] * len(session_ids))
        query = f"SELECT * FROM tbl_network_log WHERE session_id IN ({placeholders})"
        
        import mysql.connector
        df = pd.read_sql(query, cn, params=session_ids)
        raw_count = len(df)
        
        print(f"\n[STEP 1] Load Raw Data")
        print(f"  Project ID: {project_id}")
        print(f"  Sessions: {len(session_ids)}")
        print(f"  Raw records: {raw_count}")
        print(f"  Valid coordinates: {((df['lat'].notna()) & (df['lon'].notna())).sum()}")
        
        # Get original polygon
        query = "SELECT ST_AsText(region) as region_wkt FROM map_regions WHERE tbl_project_id = %s AND status = 1"
        cur.execute(query, (project_id,))
        region_row = cur.fetchone()
        
        if not region_row:
            print("\n[ERROR] No polygon found")
            return
        
        original_wkt = region_row['region_wkt']
        print(f"\n[STEP 2] Original Polygon")
        print(f"  WKT: {original_wkt[:100]}...")
        
        # Try filtering with original polygon
        original_polygon = load_wkt(original_wkt)
        filtered_original = filter_with_polygon(df, original_polygon)
        
        print(f"\n[STEP 3] Filter with Original Polygon")
        print(f"  Filtered records: {len(filtered_original)}")
        
        if len(filtered_original) == 0:
            print(f"  Status: ZERO DATA - TRYING FALLBACK")
            
            # Fallback: reverse coordinates
            print(f"\n[STEP 4] Reverse Polygon Coordinates (Fallback)")
            reversed_wkt = reverse_polygon_coordinates(original_wkt)
            print(f"  Reversed WKT: {reversed_wkt[:100]}...")
            
            reversed_polygon = load_wkt(reversed_wkt)
            filtered_reversed = filter_with_polygon(df, reversed_polygon)
            
            print(f"\n[STEP 5] Filter with Reversed Polygon")
            print(f"  Filtered records: {len(filtered_reversed)}")
            
            if len(filtered_reversed) > 0:
                print(f"\n[SUCCESS] ✓ Fallback worked!")
                print(f"  Got {len(filtered_reversed)} records from reversed coordinates")
                print(f"  Filtering efficiency: {len(filtered_reversed)/raw_count*100:.2f}%")
                print(f"\n  RECOMMENDATION: Use reversed polygon coordinates")
            else:
                print(f"\n[FAILED] Even reversed coordinates give 0 data")
                print(f"  This is a different issue (not coordinate order)")
        else:
            print(f"  Status: DATA FOUND")
            print(f"  Filtering efficiency: {len(filtered_original)/raw_count*100:.2f}%")
            print(f"  No fallback needed - original coordinates are correct")
        
        cur.close()
        
    finally:
        cn.close()


if __name__ == "__main__":
    test_project_158_fallback()
