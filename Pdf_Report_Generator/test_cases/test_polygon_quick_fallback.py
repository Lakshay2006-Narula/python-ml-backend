"""Quick test: Polygon fallback for project 158 (without full data load)"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_connection, get_project_by_id, get_project_regions
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt


def reverse_polygon_coordinates(wkt_string):
    """Reverse lat/lon in polygon WKT"""
    polygon = load_wkt(wkt_string)
    coords = list(polygon.exterior.coords)
    reversed_coords = [(lat, lon) for lon, lat in coords]
    from shapely.geometry import Polygon
    reversed_polygon = Polygon(reversed_coords)
    return reversed_polygon.wkt


def sample_points_in_polygon(session_ids, polygon, cn, limit=200):
    """Sample and test points without loading full dataframe"""
    cur = cn.cursor(dictionary=True)
    placeholders = ",".join(["%s"] * len(session_ids))
    query = f"SELECT lat, lon FROM tbl_network_log WHERE session_id IN ({placeholders}) AND lat IS NOT NULL AND lon IS NOT NULL LIMIT {limit}"
    
    cur.execute(query, tuple(session_ids))
    rows = cur.fetchall()
    cur.close()
    
    points_inside = 0
    for row in rows:
        try:
            point = Point(row['lon'], row['lat'])
            if polygon.contains(point):
                points_inside += 1
        except:
            pass
    
    return len(rows), points_inside


print("\n" + "="*80)
print("QUICK TEST: Project 158 Polygon Fallback")
print("="*80)

project_id = 158
cn = get_connection()

try:
    project = get_project_by_id(project_id, cn)
    session_ids = [int(s.strip()) for s in project['ref_session_id'].split(',') if s.strip().isdigit()]
    
    regions = get_project_regions(project_id, cn)
    
    if not regions:
        print("[ERROR] No polygon found")
    else:
        original_wkt = regions[0]['region_wkt']
        original_polygon = load_wkt(original_wkt)
        
        # Test original
        sampled, inside = sample_points_in_polygon(session_ids, original_polygon, cn, limit=500)
        
        print(f"\n[ORIGINAL POLYGON]")
        print(f"  WKT: {original_wkt[:80]}...")
        print(f"  Sampled {sampled} points: {inside} inside")
        print(f"  Match rate: {inside/sampled*100:.2f}%")
        
        if inside == 0:
            print(f"\n  Status: 0 matched - TRYING FALLBACK")
            
            # Test reversed
            reversed_wkt = reverse_polygon_coordinates(original_wkt)
            reversed_polygon = load_wkt(reversed_wkt)
            
            sampled_rev, inside_rev = sample_points_in_polygon(session_ids, reversed_polygon, cn, limit=500)
            
            print(f"\n[REVERSED POLYGON]")
            print(f"  WKT: {reversed_wkt[:80]}...")
            print(f"  Sampled {sampled_rev} points: {inside_rev} inside")
            print(f"  Match rate: {inside_rev/sampled_rev*100:.2f}%")
            
            if inside_rev > 0:
                print(f"\n[SUCCESS] Fallback worked!")
                print(f"  Reversed coordinates: {inside_rev}/{sampled_rev} points matched")
                print(f"  -> Use reversed polygon in code")
            else:
                print(f"\n[FAILED] Even reversed coordinates gave 0 matches")
                print(f"  -> Different problem (not coordinate order)")
        else:
            print(f"\n  Status: Data found - no fallback needed")
            
finally:
    cn.close()

print("\n" + "="*80)
