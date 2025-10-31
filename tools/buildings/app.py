from flask import Flask, request, jsonify
from flask_cors import CORS
import osmnx as ox
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
import json
import logging
import traceback

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ox.settings.timeout = 180
ox.settings.use_cache = True

def parse_geometry(data):
    """Parse geometry from request data"""
    logger.info(f"📥 Received data keys: {list(data.keys())}")
    
    if 'wkt' in data or 'WKT' in data:
        wkt = data.get('wkt') or data.get('WKT')
        logger.info(f"Parsing as WKT: {wkt[:100]}...")
        return wkt_loads(wkt)
    else:
        raise ValueError("No valid geometry found")

def fetch_buildings(polygon):
    """Fetch buildings from OpenStreetMap"""
    
    if not polygon.is_valid:
        logger.warning("Invalid polygon, attempting to fix...")
        polygon = polygon.buffer(0)
    
    logger.info(f"🌍 Polygon bounds: {polygon.bounds}")
    logger.info(f"🌍 Polygon area: {polygon.area} square degrees")
    
    # Calculate approximate size in meters
    # Rough conversion: 1 degree ≈ 111 km
    bounds = polygon.bounds
    width_deg = bounds[2] - bounds[0]
    height_deg = bounds[3] - bounds[1]
    width_m = width_deg * 111000  # approximate meters
    height_m = height_deg * 111000
    
    logger.info(f"📏 Approximate size: {width_m:.1f}m × {height_m:.1f}m")
    
    logger.info("🔍 Fetching from OpenStreetMap...")
    
    try:
        buildings = ox.features_from_polygon(polygon, tags={"building": True})
        logger.info(f"📦 Fetched {len(buildings)} features from OSM")
        
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
        logger.info(f"🏠 Filtered to {len(buildings)} building polygons")
        
        if buildings.empty:
            logger.warning("⚠️ No buildings found")
            return None, 0
        
        geojson_str = buildings.to_json()
        geojson = json.loads(geojson_str)
        
        return geojson, len(buildings)
        
    except Exception as e:
        # If OSM returns "no features found", treat it as empty result, not error
        if "No matching features" in str(e) or "InsufficientResponseError" in str(type(e).__name__):
            logger.warning(f"⚠️ No buildings found in OpenStreetMap for this area")
            return None, 0
        else:
            # Real error - re-raise it
            raise

@app.route('/', methods=['GET'])
def home():
    return jsonify({'service': 'Building Extraction Service', 'status': 'running', 'version': '1.0.0'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/generate-buildings', methods=['POST'])
def generate_buildings():
    """Main endpoint to generate buildings"""
    
    logger.info("=" * 60)
    logger.info("🚀 NEW REQUEST: /api/generate-buildings")
    logger.info("=" * 60)
    
    try:
        data = request.get_json()
        logger.info(f"📊 Request data type: {type(data)}")
        logger.info(f"📊 Request data: {data}")
        
        if not data:
            logger.error("❌ No data provided")
            return jsonify({'Status': 0, 'Message': 'No data provided'}), 400
        
        # Parse geometry
        try:
            logger.info("🔄 Parsing geometry...")
            polygon = parse_geometry(data)
            logger.info(f"✅ Geometry parsed: {polygon.geom_type}")
        except Exception as e:
            logger.error(f"❌ Geometry parsing error: {str(e)}")
            return jsonify({'Status': 0, 'Message': f'Invalid geometry: {str(e)}'}), 400
        
        # Fetch buildings
        logger.info("🏗️ Fetching buildings from OpenStreetMap...")
        
        try:
            geojson, count = fetch_buildings(polygon)
            
            # Handle "no buildings found" as a valid result, not an error
            if count == 0 or geojson is None:
                logger.warning("⚠️ No buildings found in this area")
                
                # Return empty GeoJSON with helpful message
                empty_geojson = {
                    'type': 'FeatureCollection',
                    'features': [],
                    'properties': {
                        'message': 'No buildings found in OpenStreetMap for this area',
                        'suggestions': [
                            'Try a larger area',
                            'Check if coordinates are correct',
                            'This area might not have buildings mapped in OSM'
                        ]
                    }
                }
                
                return jsonify({
                    'Status': 0,  # 0 means no data, but not an error
                    'Message': 'No buildings found in OpenStreetMap for this area. Try a larger area or different location.',
                    'Data': empty_geojson,
                    'Stats': {
                        'total_buildings': 0,
                        'area_sq_degrees': polygon.area,
                        'bounds': polygon.bounds
                    }
                }), 200  # Return 200, not 404
            
            logger.info(f"✅ Successfully fetched {count} buildings")
            
            return jsonify({
                'Status': 1,
                'Message': f'Successfully fetched {count} buildings',
                'Data': geojson,
                'Stats': {
                    'total_buildings': count,
                    'area_sq_degrees': polygon.area,
                    'bounds': polygon.bounds
                }
            }), 200
            
        except Exception as e:
            logger.error(f"❌ OSM fetch error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'Status': 0,
                'Message': f'Error fetching buildings: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"💥 UNEXPECTED ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'Status': 0,
            'Message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/test-polygon', methods=['GET'])
def test():
    """Test endpoint with known working polygon"""
    # Small area in Delhi with buildings
    sample = {
        "WKT": "POLYGON((77.2090 28.6139, 77.2100 28.6139, 77.2100 28.6149, 77.2090 28.6149, 77.2090 28.6139))"
    }
    
    try:
        polygon = parse_geometry(sample)
        geojson, count = fetch_buildings(polygon)
        
        return jsonify({
            'Status': 1,
            'Message': f'Test successful - {count} buildings found',
            'Data': geojson or {'type': 'FeatureCollection', 'features': []}
        })
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        return jsonify({'Status': 0, 'Message': f'Test failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Building Extraction Service")
    print("📍 Running on: http://localhost:5001")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)