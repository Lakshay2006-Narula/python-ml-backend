from flask import current_app
import osmnx as ox
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
import json
import os

# Configure OSMnx globally
ox.settings.timeout = int(os.getenv('OSM_TIMEOUT', 180))
ox.settings.use_cache = os.getenv('OSM_USE_CACHE', 'true').lower() == 'true'

class BuildingService:
    
    def parse_geometry(self, data):
        """Parse WKT geometry"""
        if 'wkt' in data or 'WKT' in data:
            wkt = data.get('wkt') or data.get('WKT')
            current_app.logger.info(f"Parsing WKT: {wkt[:100]}...")
            return wkt_loads(wkt)
        raise ValueError("No valid geometry found in request")
    
    def fetch_buildings(self, polygon):
        """Fetch buildings from OSM"""
        if not polygon.is_valid:
            current_app.logger.warning("Invalid polygon, fixing...")
            polygon = polygon.buffer(0)
        
        current_app.logger.info(f"Fetching buildings for bounds: {polygon.bounds}")
        
        try:
            buildings = ox.features_from_polygon(polygon, tags={"building": True})
            buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
            
            if buildings.empty:
                current_app.logger.warning("No buildings found")
                return None, 0
            
            current_app.logger.info(f"Found {len(buildings)} buildings")
            return json.loads(buildings.to_json()), len(buildings)
            
        except Exception as e:
            if "No matching features" in str(e) or "InsufficientResponseError" in str(type(e).__name__):
                current_app.logger.warning("No buildings in OSM for this area")
                return None, 0
            raise
    
    def extract_buildings(self, data):
        """Main extraction method"""
        polygon = self.parse_geometry(data)
        geojson, count = self.fetch_buildings(polygon)
        
        if count == 0 or geojson is None:
            return {
                'Status': 0,
                'Message': 'No buildings found in this area',
                'Data': {'type': 'FeatureCollection', 'features': []},
                'Stats': {'total_buildings': 0}
            }
        
        return {
            'Status': 1,
            'Message': f'Successfully fetched {count} buildings',
            'Data': geojson,
            'Stats': {
                'total_buildings': count,
                'area_sq_degrees': polygon.area,
                'bounds': list(polygon.bounds)
            }
        }